from typing import Any
from pathlib import Path
import subprocess

import numpy as np
from numpy.typing import NDArray
from sollertia_shared_assets import (
    AnimalData,
    SessionData,
    RunTrainingDescriptor as RunTrainingDescriptor,
    LickTrainingDescriptor as LickTrainingDescriptor,
    WindowCheckingDescriptor as WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor as MesoscopeExperimentDescriptor,
)

from .system import (
    MESOSCOPE_VR_SESSIONS as MESOSCOPE_VR_SESSIONS,
    MesoscopeData as MesoscopeData,
    MesoscopeGoogleSheets as MesoscopeGoogleSheets,
    MesoscopeVideoTracking as MesoscopeVideoTracking,
    get_system_configuration as get_system_configuration,
)
from ..cross_system import (
    WaterLog as WaterLog,
    SurgeryLog as SurgeryLog,
    push_session_data as push_session_data,
    assemble_session_logs as assemble_session_logs,
    rename_session_videos as rename_session_videos,
    snapshot_surgery_data as snapshot_surgery_data,
    migrate_session_directory as migrate_session_directory,
    delete_session_directories as delete_session_directories,
)

type _MetadataArray = NDArray[np.int32] | NDArray[np.float64] | NDArray[np.uint64]
_METADATA_SCHEMA: dict[str, tuple[type, type]]
_IGNORED_METADATA_FIELDS: set[str]
_MONOCHROME_PAGE_RANK: int
_EMPTY_MATLAB_VECTOR_LENGTH: int
_PREPROCESSING_WORKER_COUNT: int
_STORAGE_TRANSFER_THREAD_COUNT: int
_FACE_CAMERA_NAME: str
EYE_TRACKING_PROJECT_NAME: str
_INFERENCE_LOG_TAIL_CHARACTERS: int
_FACE_TRACKING_TERMINATION_TIMEOUT: float

def preprocess_session_data(session_data: SessionData) -> None: ...
def rename_mesoscope_directory(mesoscope_data: MesoscopeData) -> None: ...
def purge_session(session_data: SessionData) -> None: ...
def migrate_animal_between_projects(animal: str, source_project: str, target_project: str) -> None: ...
def _launch_face_tracking(
    session_data: SessionData, configuration: MesoscopeVideoTracking
) -> subprocess.Popen[bytes] | None: ...
def _join_face_tracking(process: subprocess.Popen[bytes], session_data: SessionData) -> None: ...
def _terminate_face_tracking(process: subprocess.Popen[bytes], session_data: SessionData) -> None: ...
def _read_inference_log_tail(log_path: Path) -> str: ...
def _purge_window_checking_behavior_data(session_data: SessionData) -> None: ...
def _verify_and_get_stack_size(file: Path) -> int: ...
def _process_stack(
    tiff_path: Path, first_frame_number: int, output_directory: Path, batch_size: int = 100
) -> dict[str, Any]: ...
def _process_invariant_metadata(frame_stack_path: Path, cindra_parameters_path: Path, metadata_path: Path) -> None: ...
def _resolve_active_channel_count(channels_active: object) -> int: ...
def _pull_mesoscope_data(session_data: SessionData, mesoscope_data: MesoscopeData, threads: int = 30) -> None: ...
def _preprocess_mesoscope_directory(
    session_data: SessionData, mesoscope_data: MesoscopeData, processes: int
) -> None: ...
def _preprocess_google_sheet_data(session_data: SessionData, sheets_data: MesoscopeGoogleSheets) -> None: ...
def _migrate_sessions_via_destination(
    destination_name: str,
    storage_animal: AnimalData,
    source_animal: AnimalData,
    destination_animal: AnimalData,
    target_project: str,
) -> None: ...
def _migrate_sessions_on_premises(
    source_animal: AnimalData, destination_animal: AnimalData, target_project: str
) -> None: ...
