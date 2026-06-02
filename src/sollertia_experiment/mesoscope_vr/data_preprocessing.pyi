from typing import Any
from pathlib import Path

from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import SessionData

from .system import (
    MESOSCOPE_VR_SESSIONS as MESOSCOPE_VR_SESSIONS,
    MesoscopeData as MesoscopeData,
    MesoscopeGoogleSheets as MesoscopeGoogleSheets,
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

_METADATA_SCHEMA: dict[str, tuple[type, type]]
_IGNORED_METADATA_FIELDS: set[str]

def preprocess_session_data(session_data: SessionData) -> None: ...
def rename_mesoscope_directory(mesoscope_data: MesoscopeData) -> None: ...
def purge_session(session_data: SessionData) -> None: ...
def migrate_animal_between_projects(animal: str, source_project: str, target_project: str) -> None: ...
def _verify_and_get_stack_size(file: Path) -> int: ...
def _process_stack(
    tiff_path: Path, first_frame_number: int, output_directory: Path, batch_size: int = 100
) -> dict[str, Any]: ...
def _process_invariant_metadata(frame_stack_path: Path, cindra_parameters_path: Path, metadata_path: Path) -> None: ...
def _pull_mesoscope_data(session_data: SessionData, mesoscope_data: MesoscopeData, threads: int = 30) -> None: ...
def _preprocess_mesoscope_directory(
    session_data: SessionData, mesoscope_data: MesoscopeData, processes: int
) -> None: ...
def _preprocess_google_sheet_data(session_data: SessionData, sheets_data: MesoscopeGoogleSheets) -> None: ...
def _migrate_sessions_via_destination(
    destination_name: str,
    destination_root: Path,
    source_local_root: Path,
    destination_local_root: Path,
    animal: str,
    source_project: str,
    target_project: str,
) -> None: ...
def _migrate_sessions_on_premises(
    source_local_root: Path, destination_local_root: Path, target_project: str
) -> None: ...
