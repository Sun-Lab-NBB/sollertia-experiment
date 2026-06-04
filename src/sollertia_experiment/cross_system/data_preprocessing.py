"""Provides the assets shared by all acquisition systems for preprocessing a session's data and moving it to the
long-term storage destinations.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import questionary
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import (
    delete_directory,
    transfer_directory,
    assemble_log_archives,
    calculate_directory_checksum,
)

from .google_sheet_tools import SurgeryLog

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SurgeryData

_LOG_DIRECTORY_NAME: str = "behavior_data_log"
"""The name of the directory generated during runtime to store the unprocessed .npy log entries and the camera
manifest (camera_manifest.yaml). During preprocessing, this directory is archived in place into .npz archives and then
renamed to behavior_data."""


@dataclass(frozen=True, slots=True)
class StorageDestination:
    """Defines a single long-term storage destination resolved for a particular data acquisition session."""

    name: str
    """The human-readable name of the storage destination, used in progress and status messages."""
    session_path: Path
    """The absolute path to the session's data directory on this storage destination."""


@dataclass(frozen=True, slots=True)
class StorageDestinations:
    """Defines the ordered collection of long-term storage destinations resolved for a data acquisition session.

    This is the system-agnostic interface used by the cross-system preprocessing utilities. Each acquisition system
    resolves its own destinations from its configuration and passes the resulting collection to the utilities, which
    only ever operate on the resolved paths.
    """

    destinations: tuple[StorageDestination, ...] = ()
    """The storage destinations to which the session's data is transferred and from which it can be removed."""


def assemble_session_logs(session_data: SessionData, processes: int) -> None:
    """Assembles all .npy log entries stored in the session's temporary log directory into .npz archives, one for each
    data source recorded during the session's runtime.

    Args:
        session_data: The SessionData instance that defines the processed session.
        processes: The number of processes to use while archiving the log entries.

    Raises:
        RuntimeError: If the target log directory contains both unprocessed and processed log entries.
    """
    log_directory = session_data.raw_data_path.joinpath(_LOG_DIRECTORY_NAME)

    if not log_directory.exists():
        return

    archives = list(log_directory.glob("*.npz"))
    unarchived_entries = list(log_directory.glob("*.npy"))

    if not unarchived_entries:
        return

    if archives and unarchived_entries:
        message = (
            f"The temporary log directory for the session {session_data.session_name} contains both unprocessed .npy "
            f"log files and processed .npz archives. Since log archiving overwrites existing .npz archives, it is "
            f"unsafe to proceed with unsupervised log archiving. Manually back up the existing .npz files, "
            f"remove them from the log directory, and retry the processing."
        )
        console.error(message=message, error=RuntimeError)

    assemble_log_archives(
        log_directory=log_directory,
        remove_sources=True,
        memory_mapping=False,
        verbose=True,
        verify_integrity=False,
        max_workers=processes,
    )

    # Renames the processed directory to behavior_data. Since behavior_data might already exist due to SessionData
    # directory generation, removes any existing behavior_data directories before renaming the log directory.
    behavior_data_path = session_data.raw_data.behavior_data_path

    if behavior_data_path.exists():
        console.echo(
            message=f"Removing existing behavior_data directory at {behavior_data_path} before renaming the processed "
            f"log directory.",
            level=LogLevel.WARNING,
        )
        shutil.rmtree(behavior_data_path)

    log_directory.rename(target=behavior_data_path)


def rename_session_videos(session_data: SessionData) -> None:
    """Renames the .MP4 video files generated during the processed session's runtime to use human-friendly names
    instead of the numeric camera source ID codes.

    Notes:
        The mapping between source IDs and human-friendly names is resolved from the camera manifest written by the
        ataraxis-video-system library during acquisition. Therefore, this function does not require any acquisition
        system to provide a static source-ID-to-name mapping.

    Args:
        session_data: The SessionData instance that defines the processed session.
    """
    camera_directory = session_data.raw_data.camera_data_path
    session_name = session_data.session_name

    # Resolves the camera manifest written by ataraxis-video-system during acquisition. The manifest resides in the
    # assembled behavior_data directory or, if the log directory has not been archived yet, the temporary log directory.
    manifest_candidates = (
        session_data.raw_data.behavior_data_path.joinpath(CAMERA_MANIFEST_FILENAME),
        session_data.raw_data_path.joinpath(_LOG_DIRECTORY_NAME, CAMERA_MANIFEST_FILENAME),
    )
    manifest_path = next((candidate for candidate in manifest_candidates if candidate.exists()), None)

    # Aborts early if the session did not register any cameras and, therefore, did not generate a camera manifest.
    if manifest_path is None:
        return

    manifest = CameraManifest.from_yaml(file_path=manifest_path)

    # Renames each acquired video file from its numeric source ID to a human-friendly name. Skips sources whose video
    # file is missing to support sessions where some configured cameras did not save frames to disk.
    for source in manifest.sources:
        video_path = camera_directory.joinpath(f"{source.id:03d}.mp4")
        if not video_path.exists():
            continue
        video_path.rename(target=camera_directory.joinpath(f"{session_name}_{source.name}.mp4"))


def snapshot_surgery_data(
    session_data: SessionData, animal_id: int, credentials_path: Path, surgery_sheet_id: str
) -> SurgeryLog:
    """Caches a copy of the animal's surgical intervention record to the session's data directory as the
    surgery_metadata.yaml file.

    Notes:
        Returns the SurgeryLog handle so callers can reuse the established Google Sheets connection for follow-up
        operations, such as updating the surgery quality assessment.

    Args:
        session_data: The SessionData instance that defines the processed session.
        animal_id: The unique identifier code of the animal that participated in the processed session.
        credentials_path: The path to the Google service account credentials file.
        surgery_sheet_id: The identifier of the Google Sheet that stores the animals' surgical intervention records.

    Returns:
        The SurgeryLog instance connected to the surgery log Google Sheet.
    """
    surgery_log = SurgeryLog(
        project_name=session_data.project_name,
        animal_id=animal_id,
        credentials_path=credentials_path,
        sheet_id=surgery_sheet_id,
    )
    surgery_data: SurgeryData = surgery_log.extract_animal_data()
    surgery_data.to_yaml(session_data.raw_data.surgery_metadata_path)
    console.echo(message="Surgery data snapshot: Saved.", level=LogLevel.SUCCESS)
    return surgery_log


def push_session_data(session_data: SessionData, destinations: StorageDestinations, threads: int) -> None:
    """Moves the preprocessed session's raw data from the acquisition host machine to all long-term storage
    destinations.

    Notes:
        This function computes the data integrity checksum before the transfer and removes the entire local session
        directory — including any processed_data not transferred — after the raw data is successfully transferred to
        all destinations.

        If the input collection contains no storage destinations, the function aborts early with a warning and leaves
        the local copy of the session's data intact, since there is no destination to back the data up to.

    Args:
        session_data: The SessionData instance that defines the processed session.
        destinations: The StorageDestinations collection that defines the long-term storage destinations resolved for
            the processed session.
        threads: The number of worker threads used by each transfer process to parallelize the data transfer.
    """
    # If no long-term storage destinations are configured, the preprocessed data has nowhere to be backed up. Aborts
    # the transfer and retains the local copy on the acquisition host machine instead of deleting it.
    if not destinations.destinations:
        message = (
            f"No long-term storage destinations are configured for the host-machine. Skipping the data transfer for "
            f"session {session_data.session_name} and retaining the local data copy on the acquisition host machine."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return

    # Resolves the source directory and the per-destination target directories. The target reuses the source's raw_data
    # directory name so that the long-term storage layout matches the host machine's layout.
    source = session_data.raw_data_path
    targets = tuple(destination.session_path.joinpath(source.name) for destination in destinations.destinations)

    for target in targets:
        ensure_directory_exists(target)

    # Computes the xxHash3-128 checksum for the source directory before moving it to the target directories.
    calculate_directory_checksum(directory=source, num_processes=None, save_checksum=True, progress=True)

    # Parallelizes the data transfer to fully saturate the communication channels to the destination machines.
    with ProcessPoolExecutor(max_workers=max(1, len(targets))) as executor:
        futures = {
            executor.submit(
                transfer_directory,
                source=source,
                destination=target,
                num_threads=threads,
                progress=True,
                # Does not remove the directory as part of the transfer to avoid race conditions.
                remove_source=False,
            ): target
            for target in targets
        }
        for future in as_completed(futures):
            # Propagates any exceptions from the transfers.
            future.result()

    console.echo(
        message="All transfers completed successfully. Removing the now-redundant source directory...",
        level=LogLevel.INFO,
    )
    # source is the raw_data directory; its parent is the session directory to remove.
    delete_directory(directory_path=source.parent)


def delete_session_directories(candidates: tuple[Path, ...], session_name: str, *, require_confirmation: bool) -> bool:
    """Removes the target session's data directories from all provided storage locations.

    Notes:
        This function is destructive and irreversible. When confirmation is requested, it locks the runtime until the
        user explicitly confirms or aborts the deletion.

    Args:
        candidates: The directories to remove. Typically, includes the session's directories on the host machine and
            all long-term storage destinations.
        session_name: The name of the session whose data is being removed, used in the confirmation prompt.
        require_confirmation: Determines whether to prompt the user to confirm the deletion before proceeding.

    Returns:
        True if the directories were removed, False if the user aborted the deletion.
    """
    if require_confirmation:
        message = (
            f"Preparing to remove all data for the session {session_name}. Warning, this process is NOT reversible "
            f"and removes ALL session data!"
        )
        console.echo(message=message, level=LogLevel.WARNING)

        if not questionary.confirm(
            f"Permanently delete all data for session {session_name}?", default=False
        ).unsafe_ask():
            console.echo(message=f"Session {session_name} data purging: Aborted", level=LogLevel.SUCCESS)
            return False

    for candidate in console.track(iterable=candidates, description="Deleting session directories", unit="directory"):
        delete_directory(directory_path=candidate)

    return True


def migrate_session_directory(
    remote_session_path: Path,
    local_session_path: Path,
    old_session_data_path: Path,
    target_project: str,
    threads: int,
) -> SessionData:
    """Pulls a single session from a remote storage destination to the host machine and reassigns it to the target
    project.

    Notes:
        This function copies the pulled session_data.yaml file to the source project's host-machine location so the
        caller can later remove the obsolete data from all storage destinations. It also recreates the source project's
        per-session raw_data directory (which preprocessing removed) so the copied session_data.yaml has a valid
        destination on the host machine.

    Args:
        remote_session_path: The path to the session's directory on the remote storage destination.
        local_session_path: The path to the session's directory on the host machine, under the target project.
        old_session_data_path: The path to the session_data.yaml file under the source project on the host machine.
        target_project: The name of the project to which the session is reassigned.
        threads: The number of worker threads used to parallelize the data transfer.

    Returns:
        The reloaded SessionData instance that reflects the session's reassignment to the target project.
    """
    # Pulls the session to the host machine. The data is pulled into the target project's directory structure.
    ensure_directory_exists(local_session_path.parent)
    transfer_directory(
        source=remote_session_path, destination=local_session_path, num_threads=threads, verify_integrity=False
    )

    # Copies the session_data.yaml file from the pulled directory to the old project's session-specific host-machine
    # directory. This is then used to remove the old session data from all destinations.
    new_session_data_path = local_session_path.joinpath("raw_data", "session_data.yaml")
    # Since preprocessing removes the raw_data directory, recreates it.
    ensure_directory_exists(old_session_data_path)
    shutil.copy2(src=new_session_data_path, dst=old_session_data_path)

    session_data = SessionData.load(session_path=local_session_path)
    session_data.project_name = target_project
    session_data.save()

    # Reloads the session data to apply the filesystem changes resulting from changing the session's project name.
    return SessionData.load(session_path=local_session_path)
