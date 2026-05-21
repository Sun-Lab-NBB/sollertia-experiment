"""Provides the assets for preprocessing the data acquired by the Mesoscope-VR data acquisition system during a
session's runtime and moving it to the long-term storage destinations.
"""

import os
import json
import shutil as sh
from typing import TYPE_CHECKING, Any
from pathlib import Path
from datetime import UTC, datetime
from functools import partial
from itertools import chain
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import tifffile
from natsort_rs import natsort as natsorted  # type: ignore[import-untyped]
from sollertia_shared_assets import (
    SessionData,
    SurgeryData,
    SessionTypes,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    get_google_credentials_path,
)

from .configuration import MesoscopeGoogleSheets
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import (
    delete_directory,
    transfer_directory,
    assemble_log_archives,
    calculate_directory_checksum,
)

from .tools import MesoscopeData, mesoscope_vr_sessions, get_system_configuration
from ..shared_components import WaterLog, SurgeryLog

if TYPE_CHECKING:
    from numpy.typing import NDArray

_METADATA_SCHEMA = {
    "frameNumbers": (np.int32, int),
    "acquisitionNumbers": (np.int32, int),
    "frameNumberAcquisition": (np.int32, int),
    "frameTimestamps_sec": (np.float64, float),
    "acqTriggerTimestamps_sec": (np.float64, float),
    "nextFileMarkerTimestamps_sec": (np.float64, float),
    "endOfAcquisition": (np.int32, int),
    "endOfAcquisitionMode": (np.int32, int),
    "dcOverVoltage": (np.int32, int),
}
"""Defines the schema for the frame-variant ScanImage metadata expected by the _process_stack() function
when parsing mesoscope-generated metadata. This schema is statically written to match the ScanImage version currently 
used by the Mesoscope-VR system."""

_IGNORED_METADATA_FIELDS = {"auxTrigger0", "auxTrigger1", "auxTrigger2", "auxTrigger3", "I2CData"}
"""Stores the frame-invariant ScanImage metadata fields that are currently not used by the Mesoscope-VR system."""


def _verify_and_get_stack_size(file: Path) -> int:
    """Reads the header of the specified TIFF file, and, if the file is a valid mesoscope frame stack, extracts and
    returns its size in frames.

    Args:
        file: The path to the TIFF file to evaluate.

    Returns:
        If the file is a valid mesoscope frame stack, returns the number of frames (pages) in the stack. Otherwise,
        returns 0 to indicate that the file is not a valid mesoscope stack.
    """
    with tifffile.TiffFile(file) as tiff:
        # Gets the number of pages (frames) from the tiff file's header
        n_frames = len(tiff.pages)

        # Considers all files with more than one page, a 2-dimensional (monochrome) image layout, and ScanImage metadata
        # a candidate stack for further processing. For these stacks, returns the discovered stack size
        # (number of frames).
        if n_frames > 1 and len(tiff.pages[0].shape) == 2 and tiff.scanimage_metadata is not None:  # noqa: PLR2004
            return n_frames
        # Otherwise, returns 0 to indicate that the file is not a valid mesoscope frame stack.
        return 0


def _process_stack(
    tiff_path: Path, first_frame_number: int, output_directory: Path, batch_size: int = 100
) -> dict[str, Any]:
    """Recompresses the target mesoscope frame stack TIFF file using the Limited Error Raster Coding (LERC) scheme and
    extracts its frame-variant ScanImage metadata.

    Notes:
        This function is designed to be parallelized to work on multiple TIFF files at the same time.

        As part of its runtime, the function strips the extracted metadata from the recompressed frame stack to reduce
        its size.

    Raises:
        NotImplementedError: If the extracted frame-variant ScanImage metadata cannot be processed due to a mismatch
            between the ScanImage version and the version of the sollertia-experiment library.

    Args:
        tiff_path: The path to the TIFF file that stores the stack of the mesoscope-acquired frames to process.
        first_frame_number: The position (number) of the first frame stored in the stack, relative to the overall
            sequence of frames acquired during the data acquisition session's runtime.
        output_directory: The path to the directory where to save the recompressed stacks.
        batch_size: The number of frames to process at the same time.

    Returns:
        A dictionary containing the extracted frame-variant ScanImage metadata for the processed mesoscope frame stack.
    """
    # Generates the file handle for the current stack
    with tifffile.TiffFile(tiff_path) as stack:
        # Determines the size of the stack
        stack_size = len(stack.pages)

        # Initializes arrays for storing the extracted metadata using the schema
        arrays = {key: np.zeros(stack_size, dtype=dtype) for key, (dtype, _) in _METADATA_SCHEMA.items()}

        # Also initializes the array for storing the converted frame acquisition timestamps.
        arrays["epochTimestamps_us"] = np.zeros(stack_size, dtype=np.uint64)

        # Loops over each page in the stack and extracts the metadata associated with each frame
        for i, page in enumerate(stack.pages):
            metadata = page.tags["ImageDescription"].value  # type: ignore[union-attr]

            # The metadata is returned as a 'newline'-delimited string of key=value pairs. This preprocessing header
            # splits the string into separate key=value pairs. Then, each pair is further separated and processed as
            # necessary
            for line in metadata.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Raises errors if the metadata field is unexpected (unsupported).
                if key in _METADATA_SCHEMA:  # Expected data fields
                    # Use the schema to parse and convert the value
                    _, converter = _METADATA_SCHEMA[key]
                    arrays[key][i] = converter(value)
                elif key == "epoch":  # Epoch data is converted to the Sollertia platform's timestamp format.
                    # Parses the epoch [year month day hour minute second.microsecond] as microseconds elapsed since
                    # the UTC onset.
                    epoch_vals = [float(x) for x in value[1:-1].split()]
                    timestamp = int(
                        datetime(
                            int(epoch_vals[0]),
                            int(epoch_vals[1]),
                            int(epoch_vals[2]),
                            int(epoch_vals[3]),
                            int(epoch_vals[4]),
                            int(epoch_vals[5]),
                            int((epoch_vals[5] % 1) * 1_000_000),
                            tzinfo=UTC,
                        ).timestamp()
                        * 1_000_000
                    )  # Converts to microseconds
                    arrays["epochTimestamps_us"][i] = timestamp
                elif key in _IGNORED_METADATA_FIELDS:
                    # These fields are known but not currently used by the system. This section ensures these fields are
                    # empty to prevent accidental data loss.
                    if len(value) > 2:  # noqa: PLR2004
                        message = (
                            f"Non-empty unsupported field '{key}' found in the frame-variant ScanImage metadata "
                            f"associated with the tiff file {tiff_path}. Update the _process_stack() function with the "
                            f"logic for parsing the data associated with this field."
                        )
                        console.error(message=message, error=NotImplementedError)
                else:
                    # Unknown field - raise error to ensure schema is updated
                    message = (
                        f"Unknown field '{key}' found in the frame-variant ScanImage metadata associated with the tiff "
                        f"file {tiff_path}. Update the _process_stack() function with the logic for parsing the data "
                        f"associated with this field."
                    )
                    console.error(message=message, error=NotImplementedError)

        # Computes the starting and ending frame numbers
        start_frame = first_frame_number
        end_frame = first_frame_number + stack_size - 1  # The ending frame number is length - 1 + start

        # Creates the output path for the compressed stack. Uses configured digit padding for frame numbering
        output_path = output_directory.joinpath(f"mesoscope_{str(start_frame).zfill(6)}_{str(end_frame).zfill(6)}.tiff")

        # Calculates the total number of batches required to fully process the stack
        num_batches = int(np.ceil(stack_size / batch_size))

        # Creates a TiffWriter to iteratively process and append each batch to the output file. Note, if the file
        # already exists, it will be overwritten.
        with tifffile.TiffWriter(output_path, bigtiff=False) as writer:
            for batch_idx in range(num_batches):
                # Calculates start and end indices for this batch
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, stack_size)

                # Reads a batch of original frames
                original_batch = np.array([stack.pages[i].asarray() for i in range(start_idx, end_idx)])

                # Writes the entire batch to the output file using LERC compression
                writer.write(
                    original_batch,
                    compression="lerc",
                    compressionargs={"level": 0.0},  # Lossless compression
                    predictor=True,
                )

    # Returns extracted metadata dictionary to caller (all keys from the arrays' dict)
    return arrays


def _process_invariant_metadata(frame_stack_path: Path, ops_path: Path, metadata_path: Path) -> None:
    """Extracts the frame-invariant ScanImage metadata from the target mesoscope frame stack TIFF file and uses it to
    generate the metadata.json and suite2p_parameters.json files.

    Args:
        frame_stack_path: The path to the TIFF file that stores a stack of the mesoscope-acquired frames.
        ops_path: The path to the suite2p_parameters.json file to be created.
        metadata_path: The path to the metadata.json file to be created.
    """
    # Reads the frame-invariant metadata from the first page (frame) of the stack. This metadata is the same across
    # all frames and stacks.
    with tifffile.TiffFile(frame_stack_path) as tiff:
        metadata = tiff.scanimage_metadata
        # Loads the data for the first frame in the stack to generate suite2p_parameters.json
        frame_data = tiff.asarray(key=0)

    # Writes the metadata as a JSON file.
    with metadata_path.open(mode="w") as json_file:
        # noinspection PyTypeChecker
        json.dump(metadata, json_file, separators=(",", ":"), indent=None)  # Maximizes data compression

    # Extracts the mesoscope frame_rate from metadata.
    frame_rate = float(metadata["FrameData"]["SI.hRoiManager.scanVolumeRate"])  # type: ignore[index]
    plane_number = int(metadata["FrameData"]["SI.hStackManager.actualNumSlices"])  # type: ignore[index]
    channel_number = int(metadata["FrameData"]["SI.hChannels.channelsActive"])  # type: ignore[index]
    si_rois: list[dict[str, Any]] | dict[str, Any]
    si_rois = metadata["RoiGroups"]["imagingRoiGroup"]["rois"]  # type: ignore[index]

    # If the acquisition only uses a single ROI, si_rois is a single dictionary. Converts it to a list for the code
    # below to work for this acquisition mode.
    rois = [si_rois] if isinstance(si_rois, dict) else si_rois

    # Extracts the ROI dimensions for each ROI.
    roi_number = len(rois)
    roi_heights = np.array([roi["scanfields"]["pixelResolutionXY"][1] for roi in rois])
    roi_widths = np.array([roi["scanfields"]["pixelResolutionXY"][0] for roi in rois])
    roi_centers = np.array([roi["scanfields"]["centerXY"][::-1] for roi in rois])
    roi_sizes = np.array([roi["scanfields"]["sizeXY"][::-1] for roi in rois])

    # Transforms ROI coordinates into pixel-units, while maintaining accurate relative positions for each ROI.
    roi_centers -= roi_sizes / 2  # Shifts ROI coordinates to mark the top left corner
    roi_centers -= np.min(roi_centers, axis=0)  # Normalizes ROI coordinates to leftmost/topmost ROI
    # Calculates pixels-per-unit scaling factor from ROI dimensions
    scale_factor = np.median(np.column_stack([roi_heights, roi_widths]) / roi_sizes, axis=0)
    min_positions = np.ceil(roi_centers * scale_factor)  # Converts ROI positions to pixel coordinates

    # Calculates the total number of rows across all ROIs (rows of pixels acquired while imaging ROIs)
    total_rows = np.sum(roi_heights)

    # Calculates the number of flyback pixels between ROIs. These are the pixels acquired when the galvos are moving
    # between frames.
    n_flyback = (frame_data.shape[0] - total_rows) // max(1, (roi_number - 1))  # Uses integer division

    # Creates an array that stores the start and end row indices for each ROI
    roi_rows = np.zeros(shape=(2, roi_number), dtype=np.int32)
    # noinspection PyTypeChecker
    temp = np.concatenate([[0], np.cumsum(roi_heights + n_flyback)])
    roi_rows[0] = temp[:-1]  # Stores the first line index for each ROI
    roi_rows[1] = roi_rows[0] + roi_heights  # Stores the last line index for each ROI

    # Extracts the invariant data necessary for the suite2p processing pipeline to be able to load and work with the
    # stack.
    data: dict[str, int | float | list[Any]] = {
        "frame_rate": frame_rate,
        "plane_number": plane_number,
        "channel_number": channel_number,
        "roi_number": roi_rows.shape[1],
        "roi_x_coordinates": [round(min_positions[i, 1]) for i in range(roi_number)],
        "roi_y_coordinates": [round(min_positions[i, 0]) for i in range(roi_number)],
        "roi_lines": [list(range(int(roi_rows[0, i]), int(roi_rows[1, i]))) for i in range(roi_number)],
    }

    # Saves the generated config as a JSON file (suite2p_parameters)
    with ops_path.open(mode="w") as f:
        # noinspection PyTypeChecker
        json.dump(data, f, separators=(",", ":"), indent=None)  # Maximizes data compression


def _preprocess_video_names(session_data: SessionData) -> None:
    """Renames the .MP4 video files generated during the processed data acquisition session's runtime to use
    human-friendly names instead of the source ID codes.

    Args:
        session_data: The SessionData instance that defines the processed session.
    """
    # Resolves the path to the camera frame directory
    camera_frame_directory = session_data.raw_data.camera_data_path
    session_name = session_data.session_name

    # Renames the video files to use human-friendly names. Assumes the standard data acquisition configuration with 2
    # cameras and predefined camera IDs.
    if camera_frame_directory.joinpath("051.mp4").exists():
        os.renames(
            old=camera_frame_directory.joinpath("051.mp4"),
            new=camera_frame_directory.joinpath(f"{session_name}_face_camera.mp4"),
        )
    if camera_frame_directory.joinpath("062.mp4").exists():
        os.renames(
            old=camera_frame_directory.joinpath("062.mp4"),
            new=camera_frame_directory.joinpath(f"{session_name}_body_camera.mp4"),
        )


def _pull_mesoscope_data(session_data: SessionData, mesoscope_data: MesoscopeData, threads: int = 30) -> None:
    """Pulls the target session's data acquired by the mesoscope from the ScanImagePC to the VRPC.

    Notes:
        It is safe to call this function for sessions that did not acquire mesoscope frames. It is designed to
        abort early if it cannot discover the cached mesoscope frames data for the target session on the ScanImagePC.

        This function expects that the data acquisition runtime renames the generic mesoscope_frames ScanImagePC
        directory that stores the session's data to include the session name.

    Args:
        session_data: The SessionData instance that defines the processed session.
        mesoscope_data: The MesoscopeData instance that defines the session-specific filesystem layout of the
            Mesoscope-VR data acquisition system.
        threads: The number of parallel threads to use for transferring the data.
    """
    # Determines the source directory that stores the session's data on the ScanImagePC.
    session_name = session_data.session_name
    source = mesoscope_data.scanimagepc_data.session_specific_path

    # If the source directory does not exist, the mesoscope data has already been transferred to the VRPC. In this case,
    # aborts the runtime early.
    if not source.exists():
        return

    # Ensures that the VRPC's destination directory exists.
    destination = session_data.raw_data.raw_data_path.joinpath("raw_mesoscope_frames")
    ensure_directory_exists(destination)

    # Defines the set of extensions and filenames to look for when verifying source directory contents.
    _extensions = {"*.me", "*.tiff", "*.tif", "*.roi"}
    _required_mesoscope_files = {"MotionEstimator.me", "fov.roi", "zstack.tiff"}

    # Verifies that all required files are present in the source directory.

    # Extracts the names of files stored in the source directory.
    files: tuple[Path, ...] = tuple(path for ext in _extensions for path in source.glob(ext))
    file_names: set[str] = {file.name for file in files}

    # Checks which required files are missing.
    missing_files = _required_mesoscope_files - file_names

    # Raises a runtime error if any required files are missing.
    if missing_files:
        missing_files_str = ", ".join(sorted(missing_files))
        message = (
            f"Unable to pull the mesoscope-acquired data from the ScanImagePC to the VRPC. The "
            f"'mesoscope_frames' ScanImage PC directory for the session {session_name} is missing the "
            f"following required files: {missing_files_str}. Ensure that all required files are stored in the "
            f"session-specific 'mesoscope_frames' directory on the ScanImagePC and rerun the command that caused this "
            f"error."
        )
        console.error(message=message, error=RuntimeError)

    # Removes all binary files from the source directory before transferring. This ensures that the directory
    # does not contain any marker files used during runtime.
    for bin_file in source.glob("*.bin"):
        bin_file.unlink(missing_ok=True)

    # Transfers the mesoscope frames data from the ScanImagePC to the local machine and removes the source directory
    # after the transfer is complete.
    transfer_directory(
        source=source,
        destination=destination,
        num_threads=threads,
        verify_integrity=False,
        remove_source=True,
        progress=True,
    )


def _preprocess_mesoscope_directory(
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
    processes: int,
) -> None:
    """Recompresses all mesoscope-acquired .TIFF frame stack files using the Limited Error Raster Compression (LERC)
    scheme and extracts their frame-variant and frame-invariant ScanImage metadata.

    Notes:
        This function is specifically calibrated to work with the data produced by the ScanImage matlab software and
        expects specific file formatting and metadata fields to be present in each processed .TIFF file.

        To optimize runtime efficiency, this function employs multiple processes to work with multiple TIFFs at the
        same time.

        This function is purposefully designed to combine the data from multiple acquisitions stored inside the same
        directory into the same output volume. This implementation supports processing sessions that feature mesoscope
        data acquisition interruptions.

    Args:
        session_data: The SessionData instance that defines the processed session.
        mesoscope_data: The MesoscopeData instance that defines the session-specific filesystem layout of the
            Mesoscope-VR data acquisition system.
        processes: The number of processes to use while processing the directory.
    """
    # Resolves the path to the temporary directory that stores unprocessed mesoscope-acquired data pulled to the
    # VRPC.
    image_directory = session_data.raw_data.raw_data_path.joinpath("raw_mesoscope_frames")

    # If the raw_mesoscope_frames directory does not exist, aborts processing early.
    if not image_directory.exists():
        return

    # Handles special files that need to be processed differently to the TIFF stacks.
    motion_estimator_file = image_directory.joinpath("MotionEstimator.me")
    fov_roi_file = image_directory.joinpath("fov.roi")
    zstack_file = image_directory.joinpath("zstack.tiff")

    # If necessary, persists the MotionEstimator and the fov.roi files to the 'persistent data' directory of the
    # processed animal on the ScanImagePC.
    if not mesoscope_data.scanimagepc_data.roi_path.exists():
        sh.copy2(fov_roi_file, mesoscope_data.scanimagepc_data.roi_path)
    if not mesoscope_data.scanimagepc_data.motion_estimator_path.exists():
        sh.copy2(motion_estimator_file, mesoscope_data.scanimagepc_data.motion_estimator_path)

    # Copies all files to the session's mesoscope_data (preprocessed) directory.
    output_directory = session_data.raw_data.mesoscope_data_path
    ensure_directory_exists(output_directory)
    sh.copy2(motion_estimator_file, output_directory.joinpath("MotionEstimator.me"))
    sh.copy2(fov_roi_file, output_directory.joinpath("fov.roi"))
    sh.copy2(zstack_file, output_directory.joinpath("zstack.tiff"))

    # Resolves the paths to the output directories and files used during mesoscope frame stack processing.
    frame_invariant_metadata_path = output_directory.joinpath("frame_invariant_metadata.json")
    frame_variant_metadata_path = output_directory.joinpath("frame_variant_metadata.npz")
    ops_path = output_directory.joinpath("suite2p_parameters.json")

    # Pre-creates the dictionary to store frame-variant metadata extracted from all TIFF frames.
    all_metadata: defaultdict[str, list[NDArray[Any]]] = defaultdict(list)

    # Finds all TIFF files in the input directory (deliberately non-recursive).
    tiff_files = list(chain(image_directory.glob("*.tif"), image_directory.glob("*.tiff")))

    # Sorts files naturally. Since all files use the _acquisition#_stack# format, this procedure should naturally
    # sort the data in the order of acquisition.
    tiff_files = natsorted(tiff_files, key=lambda p: p.name)

    # Validates and prepares TIFF stacks for processing. Filters out invalid files and determines frame numbering.
    valid_stacks: list[tuple[Path, int]] = []  # List of (file_path, starting_frame_number) tuples
    starting_frame = 1

    for file in tiff_files:
        # All valid mesoscope data files acquired in the lab are named with the 'session' marker.
        if "session" not in file.name:
            continue
        stack_size = _verify_and_get_stack_size(file)
        if stack_size > 0:
            # Records the file and its starting frame number
            valid_stacks.append((file, starting_frame))
            starting_frame += stack_size

    # Ends the runtime early if there are no valid TIFF files to process
    if not valid_stacks:
        delete_directory(directory_path=image_directory)
        return

    # Extracts the frame invariant metadata using the first frame of the first TIFF stack. Since this metadata is the
    # same for all stacks, it is safe to use any available stack.
    first_tiff_file = valid_stacks[0][0]
    _process_invariant_metadata(
        frame_stack_path=first_tiff_file, ops_path=ops_path, metadata_path=frame_invariant_metadata_path
    )

    # Uses partial to bind the constant arguments to the processing function.
    process_func = partial(
        _process_stack,
        output_directory=output_directory,
        batch_size=100,
    )

    # Processes each tiff stack in parallel.
    with ProcessPoolExecutor(max_workers=processes) as executor:
        # Submits all tasks and tracks futures.
        # noinspection PyTypeChecker
        futures = {executor.submit(process_func, tiff_file, frame_number) for tiff_file, frame_number in valid_stacks}

        # Displays a progress bar that tracks the frame processing.
        progress_path = Path(*image_directory.parts[-6:])
        with tqdm(
            total=len(valid_stacks),
            desc=f"Processing TIFF stacks for {progress_path}",
            unit="stack",
        ) as pbar:
            for future in as_completed(futures):
                for key, value in future.result().items():
                    all_metadata[key].append(value)
                pbar.update(1)

    # Saves concatenated metadata as an uncompressed numpy archive.
    metadata_dict = {key: np.concatenate(value) for key, value in all_metadata.items()}
    np.savez(frame_variant_metadata_path, **metadata_dict)

    # Removes the now-redundant directory that stores unprocessed files.
    delete_directory(directory_path=image_directory)


def _preprocess_log_directory(session_data: SessionData, processes: int) -> None:
    """Assembles all .NPY log entries stored in the behavior data directory into .NPZ archives, one for each data
    source recorded during the session's runtime.

    Args:
        session_data: The SessionData instance that defines the processed session.
        processes: The number of processes to use while processing the directory.

    Raises:
        RuntimeError: If the target log directory contains both unprocessed and processed log entries.
    """
    # Resolves the path to the temporary log directory generated during runtime.
    log_directory = session_data.raw_data.raw_data_path.joinpath("behavior_data_log")

    # Aborts early if the log directory does not exist.
    if not log_directory.exists():
        return

    # Searches for processed and unprocessed files inside the log directory.
    archives = list(log_directory.glob("*.npz"))
    unarchived_entries = list(log_directory.glob("*.npy"))

    # If there are no unprocessed log entry files, ends the runtime early.
    if not unarchived_entries:
        return

    # If both processed and unprocessed log files exist in the same directory, aborts with an error.
    if archives and unarchived_entries:
        message = (
            f"The temporary log directory for the session {session_data.session_name} contains both unprocessed .npy "
            f"log files and processed .npz archives. Since log archiving overwrites existing .npz archives, it is "
            f"unsafe to proceed with unsupervised log archiving. Manually back up the existing .npz files, "
            f"remove them from the log directory, and retry the processing."
        )
        console.error(message, error=RuntimeError)

    # If the input directory contains unarchived .npy log entries and no archived log entries, archives all log
    # entries in the directory.
    assemble_log_archives(
        log_directory=log_directory,
        remove_sources=True,
        memory_mapping=False,
        verbose=True,
        verify_integrity=False,
        max_workers=processes,
    )

    # Renames the processed directory to behavior_data. Since behavior_data might already exist dues to SessionData
    # directory generation, removes any existing behavior_data directories before renaming the log directory.
    behavior_data_path = Path(session_data.raw_data.behavior_data_path)

    if behavior_data_path.exists():
        console.echo(
            message=f"Removing existing behavior_data directory at {behavior_data_path} before renaming the processed "
            f"log directory.",
            level=LogLevel.WARNING,
        )
        sh.rmtree(behavior_data_path)

    log_directory.rename(target=Path(session_data.raw_data.behavior_data_path))


def _preprocess_google_sheet_data(session_data: SessionData, sheets_data: MesoscopeGoogleSheets) -> None:
    """Updates the water restriction log to include the processed session's data and adds the animal's
    surgical intervention record to the session's data directory as the surgery_data.yaml file.

    Args:
        session_data: The SessionData instance that defines the processed session.
        sheets_data: The MesoscopeGoogleSheets that stores the Google Sheets configuration parameters for the
            Mesoscope-VR data acquisition system.

    Raises:
        ValueError: If the session_type attribute of the input SessionData instance is not one of the supported options.
    """
    # Resolves the animal's unique identifier code and common Google Sheets parameters.
    animal_id = int(session_data.animal_id)
    credentials_path = get_google_credentials_path()
    descriptor_path = session_data.raw_data.session_descriptor_path

    # Loads the session's descriptor file based on the session's type.
    session_type = session_data.session_type
    is_window_checking = session_type == SessionTypes.WINDOW_CHECKING

    # Defines the dispatch pattern for loading the descriptors.
    descriptor_loaders = {
        SessionTypes.LICK_TRAINING: LickTrainingDescriptor,
        SessionTypes.RUN_TRAINING: RunTrainingDescriptor,
        SessionTypes.MESOSCOPE_EXPERIMENT: MesoscopeExperimentDescriptor,
        SessionTypes.WINDOW_CHECKING: WindowCheckingDescriptor,
    }

    if session_type not in descriptor_loaders:
        message = (
            f"Unable to extract the water restriction data from the {session_data.session_name} session's descriptor "
            f"file, as the session's type {session_type} is not one of the valid Mesoscope-VR sessions: "
            f"{', '.join(mesoscope_vr_sessions)}."
        )
        console.error(message, error=ValueError)
        # Fallback to appease mypy, should not be reachable.
        raise ValueError(message)  # pragma: no cover

    # Loads the session's descriptor data.
    descriptor_class = descriptor_loaders[session_type]  # type: ignore[index]
    descriptor = descriptor_class.from_yaml(descriptor_path)  # type: ignore[attr-defined]

    # Caches a copy of the animal's surgery log entry to the session's directory as a surgery_metadata.yaml file.
    sl_sheet = SurgeryLog(
        project_name=session_data.project_name,
        animal_id=animal_id,
        credentials_path=credentials_path,
        sheet_id=sheets_data.surgery_sheet_id,
    )
    data: SurgeryData = sl_sheet.extract_animal_data()
    data.to_yaml(session_data.raw_data.surgery_metadata_path)
    message = "Surgery data snapshot: Saved."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Handles window checking sessions differently - updates surgery quality instead of the water restriction log.
    if is_window_checking:
        # Ensures that the quality is always between 0 and 3 inclusive.
        quality = max(0, min(3, int(descriptor.surgery_quality)))
        sl_sheet.update_surgery_quality(quality=quality)
        message = "Surgery quality: Updated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    # For non-window-checking sessions, updates the water restriction log.
    else:
        # Calculates the total volume of water, in ml, the animal received during and after the session's runtime.
        training_water = round(descriptor.dispensed_water_volume_ml, ndigits=3)
        experimenter_water = round(descriptor.experimenter_given_water_volume_ml, ndigits=3)
        total_water = training_water + experimenter_water

        # Updates the Water Restriction log to reflect the processed session's data.
        wr_sheet = WaterLog(
            session_date=session_data.session_name,
            animal_id=animal_id,
            credentials_path=credentials_path,
            sheet_id=sheets_data.water_log_sheet_id,
        )
        wr_sheet.update_water_log(
            weight=descriptor.animal_weight_g,
            water_ml=total_water,
            experimenter_id=descriptor.experimenter,
            session_type=session_data.session_type,
        )
        message = "Water restriction log entry: Written."
        console.echo(message=message, level=LogLevel.SUCCESS)


def _push_data(
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
    threads: int,
) -> None:
    """Moves the preprocessed session's data from the VRPC to the long-term storage infrastructure.

    Notes:
        Currently, the long-term storage infrastructure consists of the NAS (cold storage) and the BioHPC compute
        server (hot storage).

        This function removes the local copy of the data stored on the host-machine after transferring it to the
        long-term storage destinations.

    Args:
        session_data: The SessionData instance that defines the processed session.
        mesoscope_data: The MesoscopeData instance that defines the session-specific filesystem layout of the
            Mesoscope-VR data acquisition system.
        threads: Determines the number of worker threads used by each transfer process to parallelize data processing.
    """
    # Resolves the source and destination directories.
    source = session_data.raw_data.raw_data_path
    destinations = (
        mesoscope_data.destinations.nas_data_path.joinpath("raw_data"),
        mesoscope_data.destinations.server_data_path.joinpath("raw_data"),
    )

    # Ensures all destination directories exist before starting transfers.
    for destination in destinations:
        ensure_directory_exists(destination)

    # Computes the xxHash3-128 checksum for the source directory before moving it to the destination directories.
    calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True, progress=True)

    # Parallelizes the data transfer to fully saturate the communication channels to the destination machines.
    with ProcessPoolExecutor(max_workers=len(destinations)) as executor:
        futures = {
            executor.submit(
                transfer_directory,
                source=source,
                destination=destination,
                num_threads=threads,
                progress=True,
                remove_source=False,  # Does not remove the directory as part of the transfer to avoid race conditions.
            ): destination
            for destination in destinations
        }
        for future in as_completed(futures):
            # Propagates any exceptions from the transfers.
            future.result()

        console.echo(
            message="All transfers completed successfully. Removing the now-redundant source directory...",
            level=LogLevel.INFO,
        )
        delete_directory(directory_path=source.parent)  # Removes the session's directory.


def rename_mesoscope_directory(mesoscope_data: MesoscopeData) -> None:
    """Renames the shared 'mesoscope_data' ScanImagePC directory to include the target session's name.

    Args:
        mesoscope_data: The MesoscopeData instance that defines the session-specific filesystem layout of the
            Mesoscope-VR data acquisition system.
    """
    # If necessary, renames the 'shared' mesoscope_data directory to use the name specific to the preprocessed session.
    # It is essential that this is done before preprocessing, as the preprocessing pipeline uses this semantic for
    # finding and pulling the mesoscope data for the processed session.
    general_path = mesoscope_data.scanimagepc_data.mesoscope_data_path
    session_specific_path = mesoscope_data.scanimagepc_data.session_specific_path

    # Note, the renaming only happens if the session-specific cache does not exist, the general mesoscope_frames cache
    # exists, and it is not empty (has files inside).
    if not session_specific_path.exists() and general_path.exists() and len(list(general_path.glob("*"))) > 0:
        general_path.rename(session_specific_path)
        # Generates a new empty mesoscope_frames directory to support future runtimes.
        ensure_directory_exists(general_path)


def preprocess_session_data(session_data: SessionData) -> None:
    """Aggregates all session's data on VRPC, compresses it for efficient network transmission, transfers the data to
    the BioHPC server and the Synology NAS, and removes the local data copy from the VRPC.

    Args:
        session_data: The SessionData instance that defines the processed session.
    """
    message = f"Initializing session {session_data.session_name} data preprocessing..."
    console.echo(message=message, level=LogLevel.INFO)

    # Resolves the configuration parameters for the Mesoscope-VR data acquisition system.
    system_configuration = get_system_configuration()

    # Resolves the filesystem configuration for the Mesoscope-VR data acquisition system.
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # If necessary, ensures that the mesoscope_data ScanImagePC directory is renamed to include the processed session
    # name.
    rename_mesoscope_directory(mesoscope_data=mesoscope_data)

    # Assembles all log .npy entries into archive .npz files.
    _preprocess_log_directory(session_data=session_data, processes=31)

    # Renames all videos to use human-friendly names.
    _preprocess_video_names(session_data=session_data)

    # Pulls mesoscope-acquired data from the ScanImagePC to the VRPC.
    _pull_mesoscope_data(
        session_data=session_data,
        mesoscope_data=mesoscope_data,
        threads=31,
    )

    # Compresses all mesoscope-acquired frames and extracts their metadata.
    _preprocess_mesoscope_directory(
        session_data=session_data,
        mesoscope_data=mesoscope_data,
        processes=31,
    )

    # Extracts and saves the animal's surgery data to the session's data directory and updates the water restriction
    # log to reflect the processed session.
    _preprocess_google_sheet_data(session_data=session_data, sheets_data=system_configuration.sheets)

    # Sends preprocessed data to the NAS and the BioHPC server
    _push_data(
        session_data=session_data,
        mesoscope_data=mesoscope_data,
        threads=15,
    )

    message = f"Session {session_data.session_name} data preprocessing: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def purge_session(session_data: SessionData) -> None:
    """Removes all data and directories associated with the input session from all Mesoscope-VR system machines and
    long-term storage destinations.

    Notes:
        This function is extremely dangerous and should be used with caution. It is designed to remove all data from
        failed or no longer necessary sessions from all storage locations. Never use this function on sessions that
        contain valid scientific data.

    Args:
        session_data: The SessionData instance that defines the session whose data needs to be removed.
    """
    # If a session does not contain the nk.bin marker, this suggests that it was able to successfully initialize the
    # runtime and likely contains valid data. In this case, asks the user to confirm they intend to proceed with the
    # deletion. Sessions with nk.bin markers are considered safe for removal with no further confirmation.
    if not session_data.raw_data.nk_path.exists():
        message = (
            f"Preparing to remove all data for the session {session_data.session_name} performed by the animal "
            f"{session_data.animal_id}. Warning, this process is NOT reversible and removes ALL session data!"
        )
        console.echo(message=message, level=LogLevel.WARNING)

        # Locks and waits for user response
        while True:
            answer = input("Enter 'yes' (to proceed) or 'no' (to abort): ")

            # Continues with the deletion
            if answer.lower() == "yes":
                break

            # Aborts without deleting
            if answer.lower() == "no":
                message = f"Session {session_data.session_name} data purging: Aborted"
                console.echo(message=message, level=LogLevel.SUCCESS)
                return

    # Resolves the configuration parameters for the Mesoscope-VR data acquisition system.
    system_configuration = get_system_configuration()

    # Resolves the filesystem configuration for the Mesoscope-VR data acquisition system.
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Uses MesoscopeData to query the paths to all known session data directories. This includes the directories on the
    # NAS and the BioHPC server.
    deletion_candidates = [
        session_data.raw_data.raw_data_path.parent,
        mesoscope_data.destinations.nas_data_path,
        mesoscope_data.destinations.server_data_path,
        mesoscope_data.scanimagepc_data.session_specific_path,
    ]

    # Removes all session-specific data directories from all destinations.
    for candidate in tqdm(deletion_candidates, desc="Deleting session directories", unit="directory"):
        delete_directory(directory_path=candidate)

    # Ensures that the mesoscope_data directory is reset, in case it has any lingering from the purged runtime.
    for file in mesoscope_data.scanimagepc_data.mesoscope_data_path.glob("*"):
        file.unlink(missing_ok=True)

    message = "Session data purging: Complete"
    console.echo(message=message, level=LogLevel.SUCCESS)


def migrate_animal_between_projects(animal: str, source_project: str, target_project: str) -> None:
    """Transfers all sessions performed by the specified animal from the source project to the target project across
    all storage locations.

    Args:
        animal: The animal for which to migrate the data.
        source_project: The name of the project from which to migrate the data.
        target_project: The name of the project to which the data should be migrated.
    """
    console.echo(f"Migrating the animal {animal} from project {source_project} to project {target_project}...")

    # Queries the system configuration parameters, which includes the filesystem configuration.
    system_configuration = get_system_configuration()

    # Resolves the paths to the key root directories used in the migration process.
    source_server_root = system_configuration.filesystem.server_directory.joinpath(source_project, animal)
    destination_local_root = system_configuration.filesystem.root_directory.joinpath(target_project, animal)
    source_local_root = system_configuration.filesystem.root_directory.joinpath(source_project, animal)

    # If the target project does not exist, aborts with an error.
    if not destination_local_root.parent.exists():
        message = (
            f"Unable to migrate the animal {animal} from project {source_project} to project {target_project}. The "
            f"target project does not exist. Use the 'sle configure project' command to create the project before "
            f"migrating animals to this project."
        )
        console.error(message=message, error=FileNotFoundError)

    # Ensures that the root directory for the processed animal exists on the local machine.
    ensure_directory_exists(destination_local_root)

    # Ensures that all locally stored sessions have been processed and moved to the BioHPC server for storage. This is
    # a prerequisite to ensure that all data is properly migrated from the source project to the target project.
    local_sessions = [file.parents[1] for file in source_local_root.rglob("*session_data.yaml")]
    if len(local_sessions) > 0:
        message = (
            f"Unable to migrate the animal {animal} from project {source_project} to project {target_project}. The "
            f"source project directory on the VRPC contains non-preprocessed session data. "
            f"Preprocess all locally stored sessions before starting the migration process."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loops over all sessions stored on the server and processes them sequentially
    sessions = [file.parents[1] for file in source_server_root.rglob("*session_data.yaml")]
    for session in sessions:
        console.echo(f"Migrating session {session.name}...")
        local_session_path = destination_local_root.joinpath(session.name)
        remote_session_path = source_server_root.joinpath(session.name)

        # Pulls the session to the local machine. The data is pulled into the target project's directory structure.
        ensure_directory_exists(destination_local_root)
        transfer_directory(
            source=remote_session_path, destination=local_session_path, num_threads=30, verify_integrity=False
        )

        # Copies the session_data.yaml file from the pulled directory to the old project's session-specific VRPC
        # directory. This is then used to remove old session data from all destinations.
        new_sd_path = local_session_path.joinpath("raw_data", "session_data.yaml")
        old_sd_path = source_local_root.joinpath(session.name, "raw_data", "session_data.yaml")
        ensure_directory_exists(old_sd_path)  # Since preprocessing removes the raw_data directory, this recreates it.
        sh.copy2(src=new_sd_path, dst=old_sd_path)

        # Modifies the SessionData instance for the pulled session to use the new project name.
        session_data = SessionData.load(session_path=local_session_path)
        session_data.project_name = target_project
        session_data.save()

        # Reloads session data to apply the filesystem changes resulting from changing the session's project name.
        session_data = SessionData.load(session_path=local_session_path)

        # Runs preprocessing on the session's data again, which regenerates the checksum and transfers the data to
        # the long-term storage destinations (including the NAS).
        preprocess_session_data(session_data=session_data)

        # Removes now-obsolete server, NAS, and VRPC directories. To do so, first marks the old session for
        # deletion by creating the 'nk.bin' marker and then calls the purge pipeline on that session.
        old_session_data = SessionData.load(session_path=old_sd_path.parents[1])
        old_session_data.raw_data.nk_path.touch()
        purge_session(old_session_data)

    console.echo("Migrating persistent data directories...")
    # Moves ScanImagePC persistent data for the animal between projects This preserves existing MotionEstimator and ROI
    # data, if any was resolved for any processed session.
    old = system_configuration.filesystem.mesoscope_directory.joinpath(source_project, animal)
    new = system_configuration.filesystem.mesoscope_directory.joinpath(target_project, animal)
    sh.rmtree(new)
    sh.move(src=old, dst=new)

    # Also moves the VRPC persistent data for the animal between projects.
    old = source_local_root.joinpath("persistent_data")
    new = destination_local_root.joinpath("persistent_data")
    sh.rmtree(new)
    sh.move(src=old, dst=new)

    # Removes the old animal directory from all destinations. This also removes any lingering data not moved during
    # the migration process. This ensures that each animal is found under at most a single project directory on all
    # destinations.
    deletion_candidates = [
        system_configuration.filesystem.mesoscope_directory.joinpath(source_project, animal),
        system_configuration.filesystem.nas_directory.joinpath(source_project, animal),
        system_configuration.filesystem.root_directory.joinpath(source_project, animal),
        system_configuration.filesystem.server_directory.joinpath(source_project, animal),
    ]
    for candidate in tqdm(deletion_candidates, desc="Deleting redundant animal directories", unit="directory"):
        delete_directory(directory_path=candidate)

    console.echo("Migration: Complete.", level=LogLevel.SUCCESS)
