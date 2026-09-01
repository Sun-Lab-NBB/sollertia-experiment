"""Provides the assets for preprocessing the data acquired by the Mesoscope-VR data acquisition system during a
session's runtime and moving it to the long-term storage destinations.
"""

from __future__ import annotations

import os
import json
import shutil
import signal
from typing import TYPE_CHECKING, Any, cast
from pathlib import Path
from datetime import UTC, datetime
import tempfile
from functools import partial
from itertools import chain
import contextlib
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from natsort import natsorted
import tifffile
from ataraxis_time import TimeUnits, convert_time
from ataraxis_base_utilities import LogLevel, console, chunk_iterable, resolve_worker_count, ensure_directory_exists
from sollertia_shared_assets import (
    RAW_DATA_DIRECTORY,
    DESCRIPTOR_REGISTRY,
    AnimalData,
    SessionData,
    RawDataFiles,
    SessionTypes,
    CredentialsTypes,
    get_data_root,
    get_credentials,
    iter_animal_sessions,
)
from ataraxis_data_structures import (
    direct_write,
    delete_directory,
    transfer_directory,
    limit_worker_threads,
    initialize_worker_threads,
)

from .system import MESOSCOPE_VR_SESSIONS, MesoscopeData, get_system_configuration
from ..cross_system import (
    WaterLog,
    push_session_data,
    assemble_session_logs,
    rename_session_videos,
    snapshot_surgery_data,
    migrate_session_directory,
    delete_session_directories,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sollertia_shared_assets import (
        RunTrainingDescriptor,
        LickTrainingDescriptor,
        WindowCheckingDescriptor,
        MesoscopeExperimentDescriptor,
    )

    from .system import MesoscopeGoogleSheets, MesoscopeVideoTracking
    from ..cross_system import SurgeryLog

_METADATA_SCHEMA: dict[str, tuple[type, type]] = {
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

_IGNORED_METADATA_FIELDS: set[str] = {"auxTrigger0", "auxTrigger1", "auxTrigger2", "auxTrigger3", "I2CData"}
"""Stores the frame-variant ScanImage metadata fields that are currently not used by the Mesoscope-VR system."""

_PREPROCESSING_WORKER_COUNT: int = resolve_worker_count(reserved_cores=1)
"""The number of parallel processes and threads used for the compute- and disk-bound preprocessing steps (log
assembly, mesoscope data pulling, and mesoscope frame compression). Reserves one logical core for the host system, so
the count tracks the acquisition machine the preprocessing runs on."""

_STORAGE_TRANSFER_THREAD_COUNT: int = 15
"""The number of parallel threads used to push the preprocessed session data to the configured long-term storage
destinations."""

_FACE_CAMERA_NAME: str = "face_camera"
"""The colloquial name of the camera whose video is analyzed by the DeepLabCut eye-tracking model. It matches the name
assigned to the camera during acquisition, which rename_session_videos uses to build the video's final filename."""

EYE_TRACKING_PROJECT_NAME: str = "eye_tracking"
"""The DeepLabCut project name token that has to appear in the eye-tracking prediction filename. DeepLabCut names each
prediction after the analyzed video's stem followed by its scorer string, and the scorer embeds the project's 'Task'
field verbatim, so the project the acquisition system points at has to carry this name for sollertia-forgery's
locate_mesoscope_pose_predictions to resolve the prediction downstream."""

_INFERENCE_LOG_TAIL_CHARACTERS: int = 2000
"""The number of trailing characters of the eye-tracking inference log surfaced when reporting an inference failure."""

_FACE_TRACKING_TERMINATION_TIMEOUT: float = 30.0
"""The number of seconds the eye-tracking inference subprocess is given to exit after it is signaled to terminate, past
which it is killed outright. The grace period accommodates DeepLabCut releasing the GPU and closing its prediction
files during shutdown."""


def preprocess_session_data(session_data: SessionData) -> None:
    """Aggregates all session's data on VRPC, compresses it for efficient network transmission, transfers the data to
    all configured long-term storage destinations, and removes the local data copy from the VRPC.

    Notes:
        If no long-term storage destinations are configured for the host-machine, the data transfer and the local data
        removal are skipped, and the preprocessing is limited to on-premises data conversion and aggregation steps.

        A session that still carries the 'nk.bin' uninitialized-session marker never finished initialization and holds
        no valid data. Such a session is purged from all storage locations instead of being preprocessed.

    Args:
        session_data: The SessionData instance that defines the processed session.
    """
    # A session that still carries the 'nk.bin' marker never finished initialization and holds no valid data. Reaching
    # preprocessing with the marker present means a crash skipped the runtime's cleanup path, leaving the session
    # orphaned. Purges it instead of promoting worthless data to long-term storage. purge_session removes a marked
    # session without requiring confirmation.
    if session_data.raw_data.nk_path.exists():
        message = (
            f"Session {session_data.session_name} still carries the uninitialized-session marker, so it holds no valid "
            f"data. Purging the session instead of preprocessing it."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        purge_session(session_data=session_data)
        return

    message = f"Initializing session {session_data.session_name} data preprocessing..."
    console.echo(message=message, level=LogLevel.INFO)

    # Resolves the configuration parameters for the Mesoscope-VR data acquisition system.
    system_configuration = get_system_configuration()

    # Resolves the filesystem configuration for the Mesoscope-VR data acquisition system.
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Warns about any long-term storage destinations that are not configured for the host-machine. The session's data
    # is not backed up to these destinations during the data transfer step.
    for destination_name in mesoscope_data.unconfigured_destinations:
        message = (
            f"The {destination_name} long-term storage destination is not configured for the host-machine. The "
            f"session {session_data.session_name} data is not backed up to this destination."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    # If necessary, ensures that the mesoscope_data ScanImagePC directory is renamed to include the processed session
    # name.
    rename_mesoscope_directory(mesoscope_data=mesoscope_data)

    # Assembles all log .npy entries into archive .npz files.
    assemble_session_logs(session_data=session_data, processes=_PREPROCESSING_WORKER_COUNT)

    # Renames all videos to use human-friendly names.
    rename_session_videos(session_data=session_data)

    # Launches asynchronous face-camera eye-tracking inference on the rig's otherwise-idle GPU so that it overlaps the
    # CPU- and disk-bound preprocessing stages below and is done by the time the session is transferred. Only experiment
    # sessions acquire the brain-activity data that these eye-tracking predictions accompany, so inference is limited to
    # them. The predictions are joined and verified before the transfer step further below.
    face_tracking_process: subprocess.Popen[bytes] | None = None
    if session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
        face_tracking_process = _launch_face_tracking(
            session_data=session_data, configuration=system_configuration.video_tracking
        )

    # Groups the preprocessing steps that run while the inference is in flight, so an abort that skips the join below
    # still reaps the inference child that holds the rig's GPU.
    try:
        # Pulls mesoscope-acquired data from the ScanImagePC to the VRPC.
        _pull_mesoscope_data(
            session_data=session_data,
            mesoscope_data=mesoscope_data,
            threads=_PREPROCESSING_WORKER_COUNT,
        )

        # Compresses all mesoscope-acquired frames and extracts their metadata.
        _preprocess_mesoscope_directory(
            session_data=session_data,
            mesoscope_data=mesoscope_data,
            processes=_PREPROCESSING_WORKER_COUNT,
        )

        # Extracts and saves the animal's surgery data to the session's data directory and updates the water
        # restriction log to reflect the processed session.
        _preprocess_google_sheet_data(session_data=session_data, sheets_data=system_configuration.sheets)

        # Window checking runs the face camera only as a live monitor of the animal, so it does not intentionally
        # acquire any camera or behavior data. Removes the stub camera_data and behavior_data directories the
        # acquisition camera stack leaves behind, before the session is checksummed and pushed, so the removal reaches
        # long-term storage.
        if session_data.session_type == SessionTypes.WINDOW_CHECKING:
            _purge_window_checking_behavior_data(session_data=session_data)
    except BaseException:
        # An abandoned child keeps holding the GPU and the prediction file handles, so a retry of the preprocessing
        # would start a second writer targeting the same prediction paths.
        if face_tracking_process is not None:
            _terminate_face_tracking(process=face_tracking_process, session_data=session_data)
        raise

    # Waits for the asynchronous face-camera inference to finish and verifies it produced predictions before the session
    # is checksummed and pushed. A failed or missing inference aborts the transfer and retains the local session copy
    # for a manual retry, since the predictions must be part of the raw data shipped to long-term storage.
    if face_tracking_process is not None:
        _join_face_tracking(process=face_tracking_process, session_data=session_data)

    # Sends preprocessed data to all configured long-term storage destinations.
    push_session_data(
        session_data=session_data,
        destinations=mesoscope_data.destinations,
        threads=_STORAGE_TRANSFER_THREAD_COUNT,
    )

    message = f"Session {session_data.session_name} data preprocessing: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


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

    # Note, the renaming only happens if the session-specific cache does not exist, the general mesoscope_data cache
    # exists, and it is not empty (has files inside).
    if not session_specific_path.exists() and general_path.exists() and list(general_path.glob("*")):
        general_path.rename(session_specific_path)
        # Generates a new empty mesoscope_data directory to support future runtimes.
        ensure_directory_exists(general_path)


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
    # Resolves the configuration parameters for the Mesoscope-VR data acquisition system.
    system_configuration = get_system_configuration()

    # Resolves the filesystem configuration for the Mesoscope-VR data acquisition system.
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Queries the paths to all known session data directories, including the long-term storage destinations.
    deletion_candidates = [session_data.raw_data_path.parent]
    deletion_candidates.extend(destination.session_path for destination in mesoscope_data.destinations.destinations)
    deletion_candidates.append(mesoscope_data.scanimagepc_data.session_specific_path)

    # Sessions without the nk.bin marker successfully initialized their runtime and likely contain valid data, so the
    # deletion requires explicit user confirmation. Sessions with the nk.bin marker are considered safe to remove.
    deleted = delete_session_directories(
        candidates=tuple(deletion_candidates),
        session_name=session_data.session_name,
        require_confirmation=not session_data.raw_data.nk_path.exists(),
    )

    # Aborts without further changes if the user declined the deletion.
    if not deleted:
        return

    # Ensures that the mesoscope_data directory is reset, in case it has any lingering files from the purged runtime.
    for file in mesoscope_data.scanimagepc_data.mesoscope_data_path.glob("*"):
        file.unlink(missing_ok=True)

    message = "Session data purging: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def migrate_animal_between_projects(animal: str, source_project: str, target_project: str) -> None:
    """Transfers all sessions performed by the specified animal from the source project to the target project across
    all storage locations.

    Notes:
        The migration strategy depends on whether the host-machine is configured with any long-term storage
        destinations. Systems with at least one configured destination treat the preferred (first configured)
        destination as the source of truth. They preprocess any sessions that still reside only on the host machine,
        then pull, re-preprocess, and purge each session. Systems without any configured destination keep all data on
        the acquisition host machine, so the migration reduces to an on-premises operation that relocates each locally
        stored session to the target project directory and reassigns it. The persistent data relocation and the cleanup
        of redundant directories apply to both modes.

        The migration fails with an error if any session cannot be preprocessed or migrated. In the destination-backed
        mode, each session is handled as an isolated unit that removes its in-flight source-project directory on
        failure, so re-running the migration after resolving the error resumes from the failed session without manual
        intervention. The on-premises mode relocates each session with a move that is not rolled back, so a failure
        after the move leaves that session under the target project while its record still names the source project,
        and a re-run no longer sees it. That case requires manual repair.

    Args:
        animal: The animal for which to migrate the data.
        source_project: The name of the project from which to migrate the data.
        target_project: The name of the project to which the data should be migrated.

    Raises:
        FileNotFoundError: If the target project does not exist on the host machine.
    """
    console.echo(message=f"Migrating the animal {animal} from project {source_project} to project {target_project}...")

    # Queries the system configuration parameters, which includes the filesystem configuration.
    system_configuration = get_system_configuration()
    filesystem = system_configuration.filesystem

    # Resolves the local data root and the per-project animal directories used in the migration process. The data
    # root is owned by the Sollertia platform, not by the Mesoscope-VR filesystem configuration.
    data_root = get_data_root()
    source_animal = AnimalData(root=data_root, project_name=source_project, animal_id=animal)
    destination_animal = AnimalData(root=data_root, project_name=target_project, animal_id=animal)
    destination_local_root = destination_animal.path

    # If the target project does not exist, aborts with an error.
    if not destination_local_root.parent.exists():
        message = (
            f"Unable to migrate the animal {animal} from project {source_project} to project {target_project}. The "
            f"target project does not exist. Use the 'slsa configure project' command to create the project before "
            f"migrating animals to this project."
        )
        console.error(message=message, error=FileNotFoundError)

    # Ensures that the root directory for the processed animal exists on the local machine.
    ensure_directory_exists(destination_local_root)

    # Resolves the configured long-term storage destinations in preference order (configuration order). The first
    # configured destination is treated as the source of truth from which the session data is pulled.
    configured_destinations = [(name, root) for name, root in filesystem.storage_directories.items() if root != Path()]

    # Systems without any configured long-term storage destination keep all data on the acquisition host machine, so
    # the session migration reduces to a local relocation. Systems with at least one destination pull from the
    # preferred (first configured) destination.
    if not configured_destinations:
        _migrate_sessions_on_premises(
            source_animal=source_animal,
            destination_animal=destination_animal,
            target_project=target_project,
        )
    else:
        preferred_name, preferred_root = configured_destinations[0]
        _migrate_sessions_via_destination(
            destination_name=preferred_name,
            storage_animal=source_animal.for_root(root=preferred_root),
            source_animal=source_animal,
            destination_animal=destination_animal,
            target_project=target_project,
        )

    console.echo(message="Migrating persistent data directories...")
    # Moves ScanImagePC persistent data for the animal between projects. This preserves existing MotionEstimator and ROI
    # data, if any was resolved for any processed session. Skips the move when the mesoscope directory is unconfigured
    # (an unset root resolves to an unsafe relative path, matching the deletion guard below) or when the source
    # directory was never created, which is the case for animals that never ran a mesoscope-imaging session or for an
    # accidental re-run after a prior migration already moved the data.
    old_path = filesystem.mesoscope_directory.joinpath(source_project, animal)
    new_path = filesystem.mesoscope_directory.joinpath(target_project, animal)
    if filesystem.mesoscope_directory != Path() and old_path.exists():
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.move(src=old_path, dst=new_path)

    # Also moves the VRPC persistent data for the animal between projects. Skips the move when the source persistent
    # directory was never created for this animal.
    old_path = source_animal.persistent_data_path
    new_path = destination_animal.persistent_data_path
    if old_path.exists():
        if new_path.exists():
            shutil.rmtree(new_path)
        shutil.move(src=old_path, dst=new_path)

    # Removes the old animal directory from the acquisition host machine and all configured long-term storage
    # destinations. This also removes any lingering data not moved during the migration process, ensuring that each
    # animal is found under at most a single project directory everywhere. Unconfigured destinations are skipped, since
    # their unset roots resolve to relative paths that are unsafe to delete.
    deletion_root_candidates = [
        filesystem.mesoscope_directory,
        data_root,
        *filesystem.storage_directories.values(),
    ]
    deletion_candidates = [root.joinpath(source_project, animal) for root in deletion_root_candidates if root != Path()]
    for candidate in console.track(
        iterable=deletion_candidates, description="Deleting redundant animal directories", unit="directory"
    ):
        delete_directory(directory_path=candidate)

    console.echo(message="Migration: Complete.", level=LogLevel.SUCCESS)


def _launch_face_tracking(
    session_data: SessionData, configuration: MesoscopeVideoTracking
) -> subprocess.Popen[bytes] | None:
    """Starts DeepLabCut face-camera eye-tracking inference as a background subprocess, or does nothing when it is not
    configured.

    Notes:
        Inference runs asynchronously so it overlaps the CPU- and disk-bound preprocessing stages while using the rig's
        otherwise-idle GPU. It is invoked through 'conda run' because DeepLabCut requires a separate Python environment
        that cannot be imported into the acquisition process. Predictions are written beside the face-camera video in
        the raw camera_data directory (the 'slvt infer' default when no output directory is passed), so they are
        checksummed and shipped to long-term storage as part of the raw data. The subprocess output is redirected to a
        transient log file rather than a pipe, so a long-running child cannot deadlock on a full pipe buffer.

    Args:
        session_data: The SessionData instance that defines the processed session.
        configuration: The Mesoscope-VR video-tracking configuration that provides the conda environment, DeepLabCut
            project, and inference parameters.

    Returns:
        The running inference subprocess to join later, or None when inference is not configured or the face-camera
        video is missing.
    """
    # Face-camera inference is opt-in: it runs only when the host machine configures both the conda environment and the
    # DeepLabCut project. An unset (empty) value disables it, matching the configuration section idiom.
    if not configuration.conda_environment or configuration.dlc_project_path == Path():
        return None

    face_video = session_data.raw_data.camera_data_path.joinpath(f"{session_data.session_name}_{_FACE_CAMERA_NAME}.mp4")
    if not face_video.exists():
        message = (
            f"Unable to start face-camera eye-tracking inference for session {session_data.session_name}: the expected "
            f"face-camera video does not exist at {face_video}. Skipping inference."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return None

    command = [
        "conda",
        "run",
        "-n",
        configuration.conda_environment,
        "slvt",
        "infer",
        "--config-path",
        str(configuration.dlc_project_path),
        "--videos",
        str(face_video),
        "--shuffle",
        str(configuration.shuffle),
        "--device",
        "cuda",
        "--gpus",
        "0",
        "--batch-size",
        str(configuration.batch_size),
        "--chunks",
        str(configuration.chunks),
        "--compile-model",
        "on" if configuration.compile_model else "off",
        "--no-progress",
    ]
    if configuration.crop:
        command.extend(("--crop", configuration.crop))

    message = (
        f"Starting asynchronous face-camera eye-tracking inference for session {session_data.session_name} on the "
        f"acquisition rig's GPU..."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # Redirects the subprocess output to a transient log file. A file rather than a pipe avoids a deadlock when the
    # long-running child fills an unread pipe buffer, and preserves DeepLabCut's output for diagnosing a failure.
    log_file = Path(tempfile.gettempdir()).joinpath(f"slvt_infer_{session_data.session_name}.log").open("wb")
    try:
        # Places the child in its own process group, so an aborted preprocessing run signals the whole 'conda run' ->
        # 'slvt' -> inference worker tree rather than the wrapper process alone.
        return subprocess.Popen(args=command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        # The child process holds its own duplicated descriptor, so the parent's handle is no longer needed.
        log_file.close()


def _join_face_tracking(process: subprocess.Popen[bytes], session_data: SessionData) -> None:
    """Waits for the face-camera eye-tracking inference to finish and verifies it produced predictions.

    Notes:
        This runs immediately before the session is checksummed and transferred, so a failed or missing inference
        aborts the transfer and leaves the local session copy intact for a manual retry, rather than shipping the
        session without its eye-tracking predictions. A successful run leaves the DeepLabCut '.h5' (and its companion
        pickle files) beside the face-camera video in raw camera_data, where the checksum and transfer then capture
        them. The transient inference log is removed on success and retained on failure for inspection.

        The prediction is accepted only when its name carries the EYE_TRACKING_PROJECT_NAME token, which is the token
        the downstream sollertia-forgery locator matches on. A run driven by a DeepLabCut project under a different
        name therefore fails here, rather than shipping a session whose predictions the downstream pipeline skips.

    Args:
        process: The running inference subprocess returned by _launch_face_tracking.
        session_data: The SessionData instance that defines the processed session.

    Raises:
        RuntimeError: If the inference subprocess exits with a non-zero status or writes no prediction file named
            after the eye-tracking project.
    """
    return_code = process.wait()
    face_video = session_data.raw_data.camera_data_path.joinpath(f"{session_data.session_name}_{_FACE_CAMERA_NAME}.mp4")
    predictions = list(face_video.parent.glob(f"{face_video.stem}*{EYE_TRACKING_PROJECT_NAME}*.h5"))
    log_path = Path(tempfile.gettempdir()).joinpath(f"slvt_infer_{session_data.session_name}.log")

    if return_code != 0 or not predictions:
        message = (
            f"Face-camera eye-tracking inference failed for session {session_data.session_name} (exit code "
            f"{return_code}, {len(predictions)} prediction file(s) written). Aborting the transfer to long-term "
            f"storage and retaining the local session copy for a manual retry. Inference log tail:\n"
            f"{_read_inference_log_tail(log_path=log_path)}"
        )
        console.error(message=message, error=RuntimeError)

    log_path.unlink(missing_ok=True)
    message = f"Face-camera eye-tracking inference for session {session_data.session_name}: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _terminate_face_tracking(process: subprocess.Popen[bytes], session_data: SessionData) -> None:
    """Stops the face-camera eye-tracking inference subprocess and waits for it to exit.

    Notes:
        The 'conda run' wrapper the inference launches under does not forward signals to the tool it wraps, so the
        signal goes to the whole process group the child heads. The group is interrupted rather than terminated,
        because slvt reaps its GPU worker processes from the KeyboardInterrupt handler of its inference pipeline.

        The wait on the interrupted child is bounded and escalates to a kill of the same group, so an unresponsive
        child never stalls the abort that triggers this cleanup.

    Args:
        process: The inference subprocess returned by _launch_face_tracking.
        session_data: The SessionData instance that defines the processed session.
    """
    if process.poll() is not None:
        return

    message = (
        f"Preprocessing of the session {session_data.session_name} aborted while the face-camera eye-tracking "
        f"inference was still running. Terminating the inference subprocess."
    )
    console.echo(message=message, level=LogLevel.WARNING)

    # Suppresses the lookup error for the race where the group exits between the poll above and the signal below.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid=process.pid), signal.SIGINT)
    try:
        process.wait(timeout=_FACE_TRACKING_TERMINATION_TIMEOUT)
    except subprocess.TimeoutExpired:
        # A group that ignores the interrupt keeps the GPU reserved, so it is killed and the child is reaped outright.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid=process.pid), signal.SIGKILL)
        process.wait()


def _read_inference_log_tail(log_path: Path) -> str:
    """Returns the trailing characters of the inference log for a failure report, or a placeholder when unavailable.

    Args:
        log_path: The path to the transient inference log file.

    Returns:
        The last _INFERENCE_LOG_TAIL_CHARACTERS characters of the log, or a placeholder when the log cannot be read.
    """
    with contextlib.suppress(OSError):
        return log_path.read_text(encoding="utf-8", errors="replace")[-_INFERENCE_LOG_TAIL_CHARACTERS:]
    return "<no inference log was captured>"


def _purge_window_checking_behavior_data(session_data: SessionData) -> None:
    """Removes the stub camera_data and behavior_data directories from a window checking session's raw_data.

    Notes:
        Window checking uses the face camera solely as a live monitor of the animal and does not intentionally acquire
        any camera or behavior data. The acquisition camera stack nonetheless leaves behind a truncated video, an
        acquisition-onset log, and a camera manifest inside the camera_data and behavior_data directories. This
        function deletes both directories so the session retains only its metadata and mesoscope reference snapshot.
        ``preprocess_session_data`` invokes it before ``push_session_data`` so the removal is reflected in the data
        integrity checksum and propagated to every long-term storage destination.

    Args:
        session_data: The SessionData instance that defines the processed window checking session.
    """
    for directory in (session_data.raw_data.behavior_data_path, session_data.raw_data.camera_data_path):
        if directory.exists():
            delete_directory(directory_path=directory)
            message = f"Window checking {directory.name} directory: Removed."
            console.echo(message=message, level=LogLevel.SUCCESS)


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
        frame_count = len(tiff.pages)

        # Considers all files with more than one page, a 2-dimensional (monochrome) image layout, and ScanImage metadata
        # a candidate stack for further processing. For these stacks, returns the discovered stack size
        # (number of frames).
        if frame_count > 1 and len(tiff.pages[0].shape) == 2 and tiff.scanimage_metadata is not None:  # noqa: PLR2004
            return frame_count
        # Otherwise, returns 0 to indicate that the file is not a valid mesoscope frame stack.
        return 0


def _process_stack(
    tiff_path: Path, first_frame_number: int, output_directory: Path, batch_size: int = 100
) -> dict[str, Any]:
    """Recompresses the target mesoscope frame stack TIFF file using the Limited Error Raster Compression (LERC)
    scheme and extracts its frame-variant ScanImage metadata.

    Notes:
        This function is designed to be parallelized to work on multiple TIFF files at the same time.

        As part of its runtime, the function strips the extracted metadata from the recompressed frame stack to reduce
        its size.

    Args:
        tiff_path: The path to the TIFF file that stores the stack of the mesoscope-acquired frames to process.
        first_frame_number: The position (number) of the first frame stored in the stack, relative to the overall
            sequence of frames acquired during the data acquisition session's runtime.
        output_directory: The path to the directory where to save the recompressed stacks.
        batch_size: The number of frames to process at the same time.

    Returns:
        A dictionary containing the extracted frame-variant ScanImage metadata for the processed mesoscope frame stack.

    Raises:
        NotImplementedError: If the extracted frame-variant ScanImage metadata cannot be processed due to a mismatch
            between the ScanImage version and the version of the sollertia-experiment library.
    """
    with tifffile.TiffFile(tiff_path) as stack:
        stack_size = len(stack.pages)

        # Initializes arrays for storing the extracted metadata using the schema.
        arrays: dict[str, NDArray[Any]] = {
            key: np.zeros(stack_size, dtype=dtype) for key, (dtype, _) in _METADATA_SCHEMA.items()
        }

        # Also initializes the array for storing the converted frame acquisition timestamps.
        arrays["epochTimestamps_us"] = np.zeros(stack_size, dtype=np.uint64)

        # Loops over each page in the stack and extracts the metadata associated with each frame.
        for frame_index, page in enumerate(stack.pages):
            metadata = page.tags["ImageDescription"].value  # type: ignore[union-attr]

            # The metadata is returned as a 'newline'-delimited string of key=value pairs. This preprocessing header
            # splits the string into separate key=value pairs. Then, each pair is further separated and processed as
            # necessary.
            for line in metadata.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Raises errors if the metadata field is unexpected (unsupported).
                if key in _METADATA_SCHEMA:
                    # Uses the schema to parse and convert the value.
                    _, converter = _METADATA_SCHEMA[key]
                    arrays[key][frame_index] = converter(value)
                elif key == "epoch":  # Epoch data is converted to the Sollertia platform's timestamp format.
                    # Parses the epoch [year month day hour minute second.microsecond] as microseconds elapsed since
                    # the UTC onset.
                    epoch_values = [float(component) for component in value[1:-1].split()]
                    epoch_seconds = datetime(
                        int(epoch_values[0]),
                        int(epoch_values[1]),
                        int(epoch_values[2]),
                        int(epoch_values[3]),
                        int(epoch_values[4]),
                        int(epoch_values[5]),
                        int((epoch_values[5] % 1) * 1_000_000),
                        tzinfo=UTC,
                    ).timestamp()
                    timestamp = int(
                        convert_time(time=epoch_seconds, from_units=TimeUnits.SECOND, to_units=TimeUnits.MICROSECOND)
                    )
                    arrays["epochTimestamps_us"][frame_index] = timestamp
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
                    # Raises an error so the schema is updated to support the new metadata field.
                    message = (
                        f"Unknown field '{key}' found in the frame-variant ScanImage metadata associated with the tiff "
                        f"file {tiff_path}. Update the _process_stack() function with the logic for parsing the data "
                        f"associated with this field."
                    )
                    console.error(message=message, error=NotImplementedError)

        # Computes the starting and ending frame numbers. The ending frame number is the stack length minus one plus
        # the starting frame number.
        start_frame = first_frame_number
        end_frame = first_frame_number + stack_size - 1

        # Creates the output path for the compressed stack, using fixed 6-digit zero-padding for frame numbering.
        output_path = output_directory.joinpath(f"mesoscope_{str(start_frame).zfill(6)}_{str(end_frame).zfill(6)}.tiff")

        # Reuses one batch buffer across every chunk and decodes each page directly into its slot. Building the batch
        # from a list of per-page arrays instead would allocate the batch twice and hold both copies at once.
        first_page = stack.pages[0]
        batch_buffer: NDArray[Any] = np.empty(shape=(batch_size, *first_page.shape), dtype=first_page.dtype)

        # Creates a TiffWriter to iteratively process and append each batch to the output file. Note, if the file
        # already exists, it will be overwritten.
        with tifffile.TiffWriter(output_path, bigtiff=False) as writer:
            for page_indices in chunk_iterable(iterable=list(range(stack_size)), chunk_size=batch_size):
                for slot, page_index in enumerate(page_indices):
                    stack.pages[page_index].asarray(out=batch_buffer[slot])

                # Writes the entire batch to the output file using LERC compression. The last chunk carries fewer
                # pages than the batch size, so the buffer is sliced to the number of pages it received.
                writer.write(
                    data=batch_buffer[: len(page_indices)],
                    compression="lerc",
                    compressionargs={"level": 0.0},  # Lossless compression
                    predictor=True,
                )

    return arrays


def _process_invariant_metadata(frame_stack_path: Path, cindra_parameters_path: Path, metadata_path: Path) -> None:
    """Extracts the frame-invariant ScanImage metadata from the target mesoscope frame stack TIFF file and uses it to
    generate the frame_invariant_metadata.json and cindra_parameters.json files.

    Args:
        frame_stack_path: The path to the TIFF file that stores a stack of the mesoscope-acquired frames.
        cindra_parameters_path: The path to the cindra_parameters.json file to be created.
        metadata_path: The path to the frame_invariant_metadata.json file to be created.
    """
    # Reads the frame-invariant metadata from the first page (frame) of the stack. This metadata is the same across
    # all frames and stacks.
    with tifffile.TiffFile(frame_stack_path) as tiff:
        metadata = tiff.scanimage_metadata
        # Loads the data for the first frame in the stack to generate cindra_parameters.json.
        frame_data = tiff.asarray(key=0)

    with direct_write(file_path=metadata_path) as json_file:
        json.dump(obj=metadata, fp=json_file, separators=(",", ":"), indent=None)  # Maximizes data compression

    # Extracts the mesoscope frame_rate from metadata.
    frame_rate = float(metadata["FrameData"]["SI.hRoiManager.scanVolumeRate"])  # type: ignore[index]
    plane_number = int(metadata["FrameData"]["SI.hStackManager.actualNumSlices"])  # type: ignore[index]
    channel_number = _resolve_active_channel_count(
        channels_active=metadata["FrameData"]["SI.hChannels.channelsActive"],  # type: ignore[index]
    )
    si_rois: list[dict[str, Any]] | dict[str, Any]
    si_rois = metadata["RoiGroups"]["imagingRoiGroup"]["rois"]  # type: ignore[index]

    # If the acquisition only uses a single ROI, si_rois is a single dictionary. Converts it to a list for the code
    # below to work for this acquisition mode.
    rois = [si_rois] if isinstance(si_rois, dict) else si_rois

    # Extracts the ROI dimensions for each ROI.
    roi_number = len(rois)
    roi_heights = np.array([roi["scanfields"]["pixelResolutionXY"][1] for roi in rois], dtype=np.int64)
    roi_widths = np.array([roi["scanfields"]["pixelResolutionXY"][0] for roi in rois], dtype=np.int64)
    roi_centers = np.array([roi["scanfields"]["centerXY"][::-1] for roi in rois], dtype=np.float64)
    roi_sizes = np.array([roi["scanfields"]["sizeXY"][::-1] for roi in rois], dtype=np.float64)

    # Transforms ROI coordinates into pixel-units, while maintaining accurate relative positions for each ROI.
    # Shifts ROI coordinates to mark the top left corner.
    roi_centers -= roi_sizes / 2
    # Normalizes ROI coordinates to the leftmost and topmost ROI.
    roi_centers -= np.min(roi_centers, axis=0)
    # Calculates the pixels-per-unit scaling factor from ROI dimensions.
    scale_factor = np.median(np.column_stack([roi_heights, roi_widths]) / roi_sizes, axis=0)
    # Converts ROI positions to pixel coordinates.
    min_positions = np.ceil(roi_centers * scale_factor)

    # Calculates the total number of rows across all ROIs (rows of pixels acquired while imaging ROIs).
    total_rows = np.sum(roi_heights)

    # Calculates the number of flyback pixels between ROIs. These are the pixels acquired when the galvos are moving
    # between consecutive ROIs within a frame.
    flyback_pixels = (frame_data.shape[0] - total_rows) // max(1, (roi_number - 1))

    # Creates an array that stores the start and end row indices for each ROI.
    roi_rows = np.zeros(shape=(2, roi_number), dtype=np.int32)
    cumulative_row_indices = np.concatenate([[0], np.cumsum(roi_heights + flyback_pixels)])
    roi_rows[0] = cumulative_row_indices[:-1]
    roi_rows[1] = roi_rows[0] + roi_heights

    # Extracts the invariant data necessary for the cindra processing pipeline to be able to load and work with the
    # stack.
    data: dict[str, int | float | list[Any]] = {
        "frame_rate": frame_rate,
        "plane_number": plane_number,
        "channel_number": channel_number,
        "roi_number": roi_rows.shape[1],
        "roi_x_coordinates": [round(min_positions[roi_index, 1]) for roi_index in range(roi_number)],
        "roi_y_coordinates": [round(min_positions[roi_index, 0]) for roi_index in range(roi_number)],
        "roi_lines": [
            list(range(int(roi_rows[0, roi_index]), int(roi_rows[1, roi_index]))) for roi_index in range(roi_number)
        ],
    }

    with direct_write(file_path=cindra_parameters_path) as parameters_file:
        json.dump(obj=data, fp=parameters_file, separators=(",", ":"), indent=None)  # Maximizes data compression


def _resolve_active_channel_count(channels_active: object) -> int:
    """Converts the value of the 'SI.hChannels.channelsActive' ScanImage metadata field into the number of imaging
    channels the mesoscope acquired.

    Notes:
        ScanImage stores the indices of the active channels rather than their count. A single active channel is
        serialized either as a scalar or as a one-element MATLAB vector, and multiple active channels as a longer
        vector. The tifffile parser resolves such a vector into a flat list for a row vector and into a list of
        one-element lists for a column vector.

    Args:
        channels_active: The raw value of the 'SI.hChannels.channelsActive' frame-invariant metadata field.

    Returns:
        The number of channels acquired for each imaged plane.

    Raises:
        ValueError: If the metadata field stores neither a channel index nor a vector of channel indices.
    """
    if isinstance(channels_active, int):
        return 1

    if isinstance(channels_active, list) and channels_active:
        # A column vector wraps each channel index in its own single-element row, so unwrapping the rows reduces both
        # vector layouts to the same flat sequence of indices.
        indices = [
            element[0] if isinstance(element, list) and len(element) == 1 else element for element in channels_active
        ]
        if all(isinstance(index, int) for index in indices):
            return len(indices)

    message = (
        f"Unable to determine the number of active mesoscope imaging channels from the frame-invariant ScanImage "
        f"metadata. The 'SI.hChannels.channelsActive' field must store a channel index or a vector of channel "
        f"indices, but got {channels_active!r}."
    )
    console.error(message=message, error=ValueError)
    # Satisfies ruff RET503. console.error() is NoReturn, so this line never executes.
    return 0  # pragma: no cover


def _pull_mesoscope_data(session_data: SessionData, mesoscope_data: MesoscopeData, threads: int = 30) -> None:
    """Pulls the target session's data acquired by the mesoscope from the ScanImagePC to the VRPC.

    Notes:
        It is safe to call this function for sessions that did not acquire mesoscope frames. It is designed to
        abort early if it cannot discover the cached mesoscope frames data for the target session on the ScanImagePC.

        This function expects that the data acquisition runtime renames the generic mesoscope_data ScanImagePC
        directory that stores the session's data to include the session name.

    Args:
        session_data: The SessionData instance that defines the processed session.
        mesoscope_data: The MesoscopeData instance that defines the session-specific filesystem layout of the
            Mesoscope-VR data acquisition system.
        threads: The number of parallel threads to use for transferring the data.

    Raises:
        RuntimeError: If any required mesoscope files (MotionEstimator.me, fov.roi, zstack.tiff) are missing from the
            session's ScanImagePC directory.
    """
    # Determines the source directory that stores the session's data on the ScanImagePC.
    session_name = session_data.session_name
    source = mesoscope_data.scanimagepc_data.session_specific_path

    # If the source directory does not exist, the mesoscope data has already been transferred to the VRPC. In this case,
    # aborts the runtime early.
    if not source.exists():
        return

    # Defines the set of extensions and filenames to look for when verifying source directory contents.
    extensions = {"*.me", "*.tiff", "*.tif", "*.roi"}
    required_mesoscope_files = {"MotionEstimator.me", "fov.roi", "zstack.tiff"}

    # Verifies that all required files are present in the source directory.

    # Extracts the names of files stored in the source directory.
    files: tuple[Path, ...] = tuple(path for extension in extensions for path in source.glob(extension))
    file_names: set[str] = {file.name for file in files}

    # Checks which required files are missing.
    missing_files = required_mesoscope_files - file_names

    # Raises a runtime error if any required files are missing.
    if missing_files:
        missing_files_listing = ", ".join(sorted(missing_files))
        message = (
            f"Unable to pull the mesoscope-acquired data from the ScanImagePC to the VRPC. The ScanImagePC directory "
            f"for the session {session_name} is missing the following required files: {missing_files_listing}. "
            f"Ensure that all required files are stored in the session-specific directory named after the session "
            f"on the ScanImagePC and rerun the command that caused this error."
        )
        console.error(message=message, error=RuntimeError)

    # Removes all binary files from the source directory before transferring. This ensures that the directory
    # does not contain any marker files used during runtime.
    for binary_file in source.glob("*.bin"):
        binary_file.unlink(missing_ok=True)

    # Creates the VRPC's destination directory only after the source directory passes verification. An empty directory
    # left behind by an aborted pull looks like a completed transfer to the frame preprocessing step below.
    destination = session_data.raw_data_path.joinpath("raw_mesoscope_frames")
    ensure_directory_exists(destination)

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

    Raises:
        RuntimeError: If the session's 'raw_mesoscope_frames' directory exists, but does not store all files
            (MotionEstimator.me, fov.roi, zstack.tiff) required to process the mesoscope-acquired data.
    """
    # Resolves the path to the temporary directory that stores unprocessed mesoscope-acquired data pulled to the
    # VRPC.
    image_directory = session_data.raw_data_path.joinpath("raw_mesoscope_frames")

    # If the raw_mesoscope_frames directory does not exist, aborts processing early.
    if not image_directory.exists():
        return

    # Handles special files that need to be processed differently to the TIFF stacks.
    motion_estimator_file = image_directory.joinpath("MotionEstimator.me")
    fov_roi_file = image_directory.joinpath("fov.roi")
    zstack_file = image_directory.joinpath("zstack.tiff")

    # An interrupted pull leaves the directory behind without the files copied unconditionally below, so the contents
    # are verified before any of them is read. Naming every missing file at once tells the operator what the wedged
    # session needs to become processable again.
    missing_files = sorted(
        file.name for file in (motion_estimator_file, fov_roi_file, zstack_file) if not file.exists()
    )
    if missing_files:
        missing_files_listing = ", ".join(missing_files)
        message = (
            f"Unable to preprocess the mesoscope-acquired data for the session {session_data.session_name}. The "
            f"session's 'raw_mesoscope_frames' directory must store the MotionEstimator.me, fov.roi, and zstack.tiff "
            f"files pulled from the ScanImagePC, but the following files are missing: {missing_files_listing}. Remove "
            f"the {image_directory} directory, restore the session-specific ScanImagePC directory named after the "
            f"session, and rerun the command that caused this error."
        )
        console.error(message=message, error=RuntimeError)

    # If necessary, persists the MotionEstimator and the fov.roi files to the 'persistent data' directory of the
    # processed animal on the ScanImagePC.
    if not mesoscope_data.scanimagepc_data.roi_path.exists():
        shutil.copy2(src=fov_roi_file, dst=mesoscope_data.scanimagepc_data.roi_path)
    if not mesoscope_data.scanimagepc_data.motion_estimator_path.exists():
        shutil.copy2(src=motion_estimator_file, dst=mesoscope_data.scanimagepc_data.motion_estimator_path)

    # Copies all files to the session's mesoscope_data (preprocessed) directory.
    output_directory = session_data.system_raw_data.mesoscope_data_path
    ensure_directory_exists(output_directory)
    shutil.copy2(src=motion_estimator_file, dst=output_directory.joinpath("MotionEstimator.me"))
    shutil.copy2(src=fov_roi_file, dst=output_directory.joinpath("fov.roi"))
    shutil.copy2(src=zstack_file, dst=output_directory.joinpath("zstack.tiff"))

    # Resolves the paths to the output directories and files used during mesoscope frame stack processing.
    frame_invariant_metadata_path = output_directory.joinpath("frame_invariant_metadata.json")
    frame_variant_metadata_path = output_directory.joinpath("frame_variant_metadata.npz")
    cindra_parameters_path = output_directory.joinpath("cindra_parameters.json")

    # Pre-creates the dictionary to store frame-variant metadata extracted from all TIFF frames.
    all_metadata: defaultdict[str, list[NDArray[Any]]] = defaultdict(list)

    # Finds all TIFF files in the input directory (deliberately non-recursive).
    tiff_files = list(chain(image_directory.glob("*.tif"), image_directory.glob("*.tiff")))

    # Sorts files naturally. Since all files use the _acquisition#_stack# format, this procedure should naturally
    # sort the data in the order of acquisition.
    tiff_files = natsorted(tiff_files, key=lambda path: path.name)

    # Validates and prepares TIFF stacks for processing. Filters out invalid files and determines frame numbering.
    valid_stacks: list[tuple[Path, int]] = []  # List of (file_path, starting_frame_number) tuples
    starting_frame = 1

    for file in tiff_files:
        # All valid mesoscope data files acquired in the lab are named with the 'session' marker.
        if "session" not in file.name:
            continue
        stack_size = _verify_and_get_stack_size(file)
        if stack_size > 0:
            # Records the file and its starting frame number.
            valid_stacks.append((file, starting_frame))
            starting_frame += stack_size

    # Ends the runtime early if there are no valid TIFF files to process.
    if not valid_stacks:
        delete_directory(directory_path=image_directory)
        return

    # Extracts the frame invariant metadata using the first frame of the first TIFF stack. Since this metadata is the
    # same for all stacks, it is safe to use any available stack.
    first_tiff_file = valid_stacks[0][0]
    _process_invariant_metadata(
        frame_stack_path=first_tiff_file,
        cindra_parameters_path=cindra_parameters_path,
        metadata_path=frame_invariant_metadata_path,
    )

    # Uses partial to bind the constant arguments to the processing function.
    process_stack_partial = partial(
        _process_stack,
        output_directory=output_directory,
        batch_size=100,
    )

    # Processes each tiff stack in parallel. The results are keyed by the stack's starting frame number, because
    # completion order is arbitrary and the metadata rows must be concatenated in acquisition order to line up with
    # the frames written to the output stacks.
    stack_metadata: dict[int, dict[str, NDArray[Any]]] = {}

    # Pins the numeric backends of each worker to a single thread. Without this, every worker opens a core-count-wide
    # decode and compression pool of its own, oversubscribing the host by the square of the core count. The limit
    # reaches a spawned worker through the environment it inherits, so it has to enclose the pool's whole lifetime,
    # and the initializer covers the backends that read their width after the worker has started.
    with (
        limit_worker_threads(),
        ProcessPoolExecutor(max_workers=processes, initializer=initialize_worker_threads) as executor,
    ):
        futures = {
            executor.submit(process_stack_partial, tiff_file, frame_number): frame_number
            for tiff_file, frame_number in valid_stacks
        }

        # Displays a progress bar that tracks the frame processing.
        progress_path = Path(*image_directory.parts[-6:])
        with console.progress(
            total=len(valid_stacks),
            description=f"Processing TIFF stacks for {progress_path}",
            unit="stack",
        ) as progress_bar:
            for future in as_completed(futures):
                stack_metadata[futures[future]] = future.result()
                progress_bar.update(1)

    for starting_frame in sorted(stack_metadata):
        for key, value in stack_metadata[starting_frame].items():
            all_metadata[key].append(value)

    metadata_dict = {key: np.concatenate(value) for key, value in all_metadata.items()}

    # ScanImage restarts its per-acquisition frame counter and elapsed-time clock every time acquisition is stopped
    # and resumed, which a single session does whenever the experimenter interrupts imaging. Renumbering the frames
    # over the concatenated sequence and making the elapsed time strictly increasing gives every frame in the
    # session a unique identifier and a monotonic timestamp, matching the order the frames are written to disk.
    frame_count = len(metadata_dict["frameNumberAcquisition"])
    elapsed_seconds = metadata_dict["frameTimestamps_sec"].astype(np.float64)

    # Resolves the restart boundaries before the renumbering below overwrites the per-acquisition frame counter that
    # marks them. The counter rises within an acquisition and falls back to one at a restart, so a non-positive step
    # isolates every boundary regardless of the increment the counter uses. Pairing it with the elapsed-time step
    # covers a counter that does not reset, keeping the detection at least as sensitive as the timestamps alone.
    restarts = np.union1d(
        np.flatnonzero(np.diff(metadata_dict["frameNumberAcquisition"]) <= 0),
        np.flatnonzero(np.diff(elapsed_seconds) < 0),
    )

    metadata_dict["frameNumberAcquisition"] = np.arange(1, frame_count + 1, dtype=np.int32)
    metadata_dict["frameNumbers"] = np.arange(1, frame_count + 1, dtype=np.int32)

    if restarts.size:
        # Each restart resumes from zero, so the elapsed time carried into it is added to every subsequent frame. The
        # interval separating an interrupted run from the run resuming it is absent from the ScanImage metadata, so
        # the frames resume one frame period after the interruption. That is the shortest interval the hardware can
        # place between two frames, which understates the pause and keeps the series strictly increasing.
        steps = np.diff(elapsed_seconds)
        within_acquisition = np.ones(steps.size, dtype=np.bool_)
        within_acquisition[restarts] = False
        frame_period = float(np.median(steps[within_acquisition])) if within_acquisition.any() else 0.0
        offsets = np.zeros(frame_count, dtype=np.float64)
        for restart_index in restarts:
            offsets[restart_index + 1 :] += (
                elapsed_seconds[restart_index] - elapsed_seconds[restart_index + 1] + frame_period
            )
        elapsed_seconds += offsets
        # Acquisition numbers are only meaningful once each interrupted run is distinguishable from its siblings.
        acquisition = np.zeros(frame_count, dtype=np.int32)
        acquisition[restarts + 1] = 1
        metadata_dict["acquisitionNumbers"] = np.cumsum(acquisition, dtype=np.int32) + 1
    metadata_dict["frameTimestamps_sec"] = elapsed_seconds

    # Saves concatenated metadata as an uncompressed numpy archive.
    np.savez(frame_variant_metadata_path, **metadata_dict)  # type: ignore[arg-type]

    # Removes the now-redundant directory that stores unprocessed files.
    delete_directory(directory_path=image_directory)


def _preprocess_google_sheet_data(session_data: SessionData, sheets_data: MesoscopeGoogleSheets) -> None:
    """Updates the water restriction log to include the processed session's data and adds the animal's
    surgical intervention record to the session's data directory as the surgery_metadata.yaml file.

    Notes:
        Google Sheets processing is optional and gated on the configured sheet identifiers. If neither sheet identifier
        is set, the function skips all Google Sheets processing with a warning and does not require Google service
        account credentials. If at least one sheet identifier is set, the host-machine must provide valid credentials,
        and a missing or invalid credentials file aborts preprocessing. When credentials are available, a sheet whose
        identifier is unset is skipped individually with a warning.

    Args:
        session_data: The SessionData instance that defines the processed session.
        sheets_data: The MesoscopeGoogleSheets that stores the Google Sheets configuration parameters for the
            Mesoscope-VR data acquisition system.

    Raises:
        ValueError: If the session_type attribute of the input SessionData instance is not one of the supported options.
        FileNotFoundError: If at least one Google Sheet is configured for the host-machine, but the Google service
            account credentials are not configured or the configured credentials file does not exist.
    """
    # Skips all Google Sheets processing without requiring credentials when neither Google Sheet is configured for the
    # host-machine. This supports systems that do not use the Google Sheets integration at all.
    if not sheets_data.surgery_sheet_id and not sheets_data.water_log_sheet_id:
        message = (
            f"No Google Sheets are configured for the host-machine. Skipping all Google Sheets processing (surgery "
            f"data snapshot and water restriction log update) for the session {session_data.session_name}."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # At least one Google Sheet is configured, so the host-machine is expected to provide valid Google service account
    # credentials. Resolving the path raises a FileNotFoundError if the credentials are missing or invalid, aborting
    # preprocessing.
    credentials_path = get_credentials(credentials=CredentialsTypes.GOOGLE)

    # Resolves the animal's unique identifier code and loads the session's descriptor file based on the session's type.
    animal_id = int(session_data.animal_id)
    descriptor_path = session_data.raw_data.session_descriptor_path
    session_type = session_data.session_type
    is_window_checking = session_type == SessionTypes.WINDOW_CHECKING

    if session_type not in MESOSCOPE_VR_SESSIONS:
        message = (
            f"Unable to extract the water restriction data from the {session_data.session_name} session's descriptor "
            f"file, as the session's type {session_type} is not one of the valid Mesoscope-VR sessions: "
            f"{', '.join(MESOSCOPE_VR_SESSIONS)}."
        )
        console.error(message=message, error=ValueError)

    # Loads the session's descriptor data through the registry that maps each session type to its descriptor class.
    descriptor_class = DESCRIPTOR_REGISTRY[SessionTypes(session_type)]
    descriptor = descriptor_class.from_yaml(file_path=descriptor_path)

    # Caches a copy of the animal's surgery log entry to the session's directory as a surgery_metadata.yaml file, if
    # the surgery log Google Sheet is configured. The returned handle reuses the established Google Sheets connection
    # for any follow-up surgery log updates.
    # Tracks both Google Sheets handles so the enclosing try/finally can close their HTTP connections deterministically.
    # Relying on garbage collection to release these connections leaves the underlying SSL sockets open until an
    # unpredictable finalization point, which surfaces as a ResourceWarning during interpreter shutdown.
    surgery_log: SurgeryLog | None = None
    water_log_sheet: WaterLog | None = None
    try:
        if sheets_data.surgery_sheet_id:
            surgery_log = snapshot_surgery_data(
                session_data=session_data,
                animal_id=animal_id,
                credentials_path=credentials_path,
                surgery_sheet_id=sheets_data.surgery_sheet_id,
            )
        else:
            message = (
                f"The surgery log Google Sheet is not configured for the host-machine. Skipping the surgery data "
                f"snapshot for the session {session_data.session_name}."
            )
            console.echo(message=message, level=LogLevel.WARNING)

        # Handles window checking sessions differently - updates surgery quality instead of the water restriction log.
        if is_window_checking:
            # Updating the surgery quality requires the surgery log Google Sheet connection established above. Skips the
            # update with a warning if the surgery log Google Sheet is not configured.
            if surgery_log is None:
                message = (
                    f"The surgery log Google Sheet is not configured for the host-machine. Skipping the surgery "
                    f"quality assessment update for the session {session_data.session_name}."
                )
                console.echo(message=message, level=LogLevel.WARNING)
                return

            # The session type resolved the descriptor class through DESCRIPTOR_REGISTRY, so this branch only ever
            # sees the window checking descriptor. The cast restores the concrete type the registry erases.
            window_descriptor = cast("WindowCheckingDescriptor", descriptor)

            # Ensures that the quality is always between 0 and 3 inclusive.
            quality = max(0, min(3, int(window_descriptor.surgery_quality)))
            surgery_log.update_surgery_quality(quality=quality)
            message = "Surgery quality: Updated."
            console.echo(message=message, level=LogLevel.SUCCESS)
            return

        # For non-window-checking sessions, updates the water restriction log, if the water restriction log Google Sheet
        # is configured. Skips the update with a warning otherwise.
        if not sheets_data.water_log_sheet_id:
            message = (
                f"The water restriction log Google Sheet is not configured for the host-machine. Skipping the water "
                f"restriction log update for the session {session_data.session_name}."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            return

        # Every remaining Mesoscope-VR session type maps to a descriptor that records the animal's weight and water
        # intake. The cast restores the concrete types DESCRIPTOR_REGISTRY erases.
        water_descriptor = cast(
            "LickTrainingDescriptor | RunTrainingDescriptor | MesoscopeExperimentDescriptor", descriptor
        )

        # Calculates the volume of water, in ml, the animal received during the active portion of the session's
        # runtime plus any water the experimenter gave afterwards. Water dispensed while the runtime was paused is
        # tracked separately and is excluded from this total.
        training_water = round(water_descriptor.dispensed_water_volume_ml, ndigits=3)
        experimenter_water = round(water_descriptor.experimenter_given_water_volume_ml, ndigits=3)
        total_water = training_water + experimenter_water

        # Updates the Water Restriction log to reflect the processed session's data.
        water_log_sheet = WaterLog(
            session_date=session_data.session_name,
            animal_id=animal_id,
            credentials_path=credentials_path,
            sheet_id=sheets_data.water_log_sheet_id,
        )
        water_log_sheet.update_water_log(
            weight=water_descriptor.animal_weight_g,
            water_ml=total_water,
            experimenter_id=water_descriptor.experimenter,
            session_type=session_data.session_type,
        )
        message = "Water restriction log entry: Written."
        console.echo(message=message, level=LogLevel.SUCCESS)
    finally:
        # Closes both Google Sheets connections regardless of which processing branch executed or whether it raised.
        if surgery_log is not None:
            surgery_log.close()
        if water_log_sheet is not None:
            water_log_sheet.close()


def _migrate_sessions_via_destination(
    destination_name: str,
    storage_animal: AnimalData,
    source_animal: AnimalData,
    destination_animal: AnimalData,
    target_project: str,
) -> None:
    """Migrates the animal's sessions from the source to the target project using a long-term storage destination as
    the source of truth.

    Notes:
        This helper supports systems that transfer their data to long-term storage destinations. It first preprocesses
        any sessions that still reside only on the host machine under the source project, which moves them to the
        long-term storage destinations so the destination holds the authoritative copy of every session. It then pulls
        each destination-stored session to the host machine, re-preprocesses it under the target project, and purges the
        obsolete source-project copies.

        Each session is migrated as an isolated unit. When migrating a session fails, the helper removes the in-flight
        source-project marker it recreated while pulling the session and re-raises the error. This keeps a failed run
        re-runnable: already-migrated sessions have been purged from the source and no longer surface during discovery,
        so the resumed migration continues from the failed session.

    Args:
        destination_name: The name of the long-term storage destination used as the source of truth, used in status
            messages.
        storage_animal: The animal view resolving the source-project directory on the long-term storage destination.
        source_animal: The animal view resolving the source-project directory on the acquisition host machine.
        destination_animal: The animal view resolving the target-project directory on the acquisition host machine.
        target_project: The name of the project to which the data should be migrated.
    """
    console.echo(
        message=f"Using the {destination_name} long-term storage destination as the migration source of truth."
    )

    # Preprocesses any sessions that still reside only on the host machine under the source project, moving them to the
    # long-term storage destinations. This guarantees the destination holds the authoritative copy of every session
    # before the migration relocates it. Preprocessing is idempotent and removes each local copy only after a
    # successful transfer, so an interrupted preprocessing run can be safely re-run.
    for session in iter_animal_sessions(animal=source_animal):
        console.echo(message=f"Preprocessing non-migrated local session {session.name}...")
        preprocess_session_data(session_data=SessionData.load(session_path=session))

    # Loops over all sessions stored on the long-term storage destination and migrates them sequentially.
    for session in iter_animal_sessions(animal=storage_animal):
        console.echo(message=f"Migrating session {session.name}...")
        local_session_path = destination_animal.session_path(session_name=session.name)
        old_session_data_path = source_animal.session_path(session_name=session.name).joinpath(
            RAW_DATA_DIRECTORY, RawDataFiles.SESSION_DATA
        )

        migrated = False
        try:
            # Pulls the session to the local machine and reassigns it to the target project.
            session_data = migrate_session_directory(
                remote_session_path=storage_animal.session_path(session_name=session.name),
                local_session_path=local_session_path,
                old_session_data_path=old_session_data_path,
                target_project=target_project,
                threads=30,
            )

            # Runs preprocessing on the session's data again, which regenerates the checksum and transfers the data to
            # all configured long-term storage destinations under the target project.
            preprocess_session_data(session_data=session_data)

            # Removes the now-obsolete long-term storage and VRPC directories. To do so, first marks the old session for
            # deletion by creating the 'nk.bin' marker and then calls the purge pipeline on that session.
            old_session_data = SessionData.load(session_path=old_session_data_path.parents[1])
            old_session_data.raw_data.nk_path.touch()
            purge_session(session_data=old_session_data)
            migrated = True
        finally:
            # On a failed or interrupted migration, removes the in-flight source-project session directory recreated
            # while pulling the session. Otherwise, that leftover directory would surface as a non-migrated local
            # session on the next run and corrupt the resumed migration. The local target copy is left as-is, since
            # pulling overwrites it on retry.
            if not migrated:
                delete_directory(directory_path=old_session_data_path.parents[1])


def _migrate_sessions_on_premises(
    source_animal: AnimalData,
    destination_animal: AnimalData,
    target_project: str,
) -> None:
    """Migrates the animal's sessions from the source to the target project entirely on the acquisition host machine.

    Notes:
        This helper supports systems that do not transfer their data to long-term storage destinations. Since the host
        machine holds the only copy of the data, the migration relocates each locally stored session directory to the
        target project and reassigns the session to the target project, without any remote data transfer.

    Args:
        source_animal: The animal view resolving the source-project directory on the acquisition host machine.
        destination_animal: The animal view resolving the target-project directory on the acquisition host machine.
        target_project: The name of the project to which the data should be migrated.
    """
    # Relocates each locally stored session directory to the target project and reassigns it. Reassigning the project
    # name and saving the SessionData instance applies the filesystem changes resulting from the project change.
    for session in iter_animal_sessions(animal=source_animal):
        console.echo(message=f"Migrating session {session.name}...")
        target_session_path = destination_animal.session_path(session_name=session.name)
        shutil.move(src=session, dst=target_session_path)

        session_data = SessionData.load(session_path=target_session_path)
        session_data.project_name = target_project
        session_data.save()
