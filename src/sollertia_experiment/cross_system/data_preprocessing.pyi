from pathlib import Path
from dataclasses import dataclass

from sollertia_shared_assets import (
    SessionData,
    SurgeryData as SurgeryData,
)

from .google_sheet_tools import SurgeryLog as SurgeryLog

_LOG_DIRECTORY_NAME: str

@dataclass(frozen=True, slots=True)
class StorageDestination:
    name: str
    session_path: Path

@dataclass(frozen=True, slots=True)
class StorageDestinations:
    destinations: tuple[StorageDestination, ...] = ...

def assemble_session_logs(session_data: SessionData, processes: int) -> None: ...
def rename_session_videos(session_data: SessionData) -> None: ...
def snapshot_surgery_data(
    session_data: SessionData, animal_id: int, credentials_path: Path, surgery_sheet_id: str
) -> SurgeryLog: ...
def push_session_data(session_data: SessionData, destinations: StorageDestinations, threads: int) -> None: ...
def delete_session_directories(
    candidates: tuple[Path, ...], session_name: str, *, require_confirmation: bool
) -> bool: ...
def migrate_session_directory(
    remote_session_path: Path, local_session_path: Path, old_session_data_path: Path, target_project: str, threads: int
) -> SessionData: ...
