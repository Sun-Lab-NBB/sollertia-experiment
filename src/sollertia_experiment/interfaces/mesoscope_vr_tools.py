"""Provides MCP tools wrapping the Mesoscope-VR-specific 'sle mesoscope' configuration and data-management logic."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from sollertia_shared_assets import SessionData, get_data_root

from .mcp_instance import mcp, read_yaml, serialize, probe_writable, describe_dataclass, write_yaml_validated
from ..mesoscope_vr import (
    ZaberPositions,
    MesoscopePositions,
    MesoscopeSystemConfiguration,
    purge_session,
    preprocess_session_data,
    get_system_configuration,
    get_system_configuration_path,
    migrate_animal_between_projects,
)

_ZABER_POSITIONS_FILENAME: str = "zaber_positions.yaml"
"""Canonical filename for the per-session ZaberPositions YAML."""

_MESOSCOPE_POSITIONS_FILENAME: str = "mesoscope_positions.yaml"
"""Canonical filename for the per-session MesoscopePositions YAML."""

_SESSION_SYSTEM_CONFIG_FILENAME: str = "system_configuration.yaml"
"""Canonical filename for the per-session snapshot of MesoscopeSystemConfiguration."""

_RAW_DATA_DIR: str = "raw_data"
"""Subdirectory under each session root that holds the raw data and metadata files."""


@mcp.tool()
def read_system_configuration_tool() -> dict[str, Any]:
    """Loads the Mesoscope-VR system configuration YAML from the working directory.

    Returns:
        A dictionary with ``data`` (the serialized MesoscopeSystemConfiguration payload) and ``file_path``, or
        ``{"error": ...}`` on failure.
    """
    try:
        instance = get_system_configuration()
    except (FileNotFoundError, OSError, ValueError) as exception:
        return {"error": str(exception)}
    return {"file_path": str(get_system_configuration_path()), "data": serialize(value=instance)}


@mcp.tool()
def write_system_configuration_tool(
    configuration_payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Creates or replaces the Mesoscope-VR system configuration YAML in the working directory.

    Args:
        configuration_payload: The complete MesoscopeSystemConfiguration payload.
        overwrite: Determines whether to overwrite an existing system configuration file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated configuration payload), or
        ``{"error": ...}`` on failure.
    """
    try:
        file_path = get_system_configuration_path()
    except FileNotFoundError as exception:
        return {"error": str(exception)}
    return write_yaml_validated(
        file_path=file_path,
        payload=configuration_payload,
        validator_cls=MesoscopeSystemConfiguration,
        overwrite=overwrite,
        use_save_method=True,
    )


@mcp.tool()
def validate_system_configuration_tool() -> dict[str, Any]:
    """Validates the active Mesoscope-VR system configuration and reports filesystem mount status.

    Returns:
        A dictionary with ``valid``, ``issues``, and ``paths`` (the per-path mount report), or
        ``{"error": ...}`` on failure.
    """
    try:
        configuration = get_system_configuration()
    except (FileNotFoundError, OSError, ValueError) as exception:
        return {"error": str(exception)}

    paths = _filesystem_paths_report(configuration=configuration)
    issues = [
        f"{name}: {report.get('error', 'not ok')}" for name, report in paths.items() if not report.get("ok", False)
    ]
    return {"valid": not issues, "issues": issues, "paths": paths}


@mcp.tool()
def describe_system_configuration_schema_tool() -> dict[str, Any]:
    """Returns the schema for MesoscopeSystemConfiguration and its nested hardware dataclasses.

    Returns:
        A dictionary with a single ``schema`` key holding the recursive field description of
        MesoscopeSystemConfiguration and every nested hardware dataclass.
    """
    return {"schema": describe_dataclass(cls=MesoscopeSystemConfiguration)}


@mcp.tool()
def check_system_mounts_tool() -> dict[str, Any]:
    """Verifies all filesystem paths declared in the active Mesoscope-VR system configuration.

    Returns:
        A dictionary with ``system_name``, ``paths`` (the per-path diagnostic report), and ``summary`` (the count
        of reachable and failed paths), or ``{"error": ...}`` on failure.
    """
    try:
        configuration = get_system_configuration()
    except (FileNotFoundError, OSError, ValueError) as exception:
        return {"error": str(exception)}

    paths = _filesystem_paths_report(configuration=configuration)
    summary = {
        "ok": sum(1 for report in paths.values() if report.get("ok", False)),
        "failed": sum(1 for report in paths.values() if not report.get("ok", False)),
    }
    return {"system_name": configuration.name, "paths": paths, "summary": summary}


@mcp.tool()
def read_session_zaber_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the ZaberPositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized ZaberPositions payload), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _ZABER_POSITIONS_FILENAME),  # type: ignore[union-attr]
        validator_cls=ZaberPositions,
    )


@mcp.tool()
def write_session_zaber_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the ZaberPositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.
        positions_payload: The complete ZaberPositions payload to write.
        overwrite: Determines whether to overwrite an existing positions file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated ZaberPositions payload), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return write_yaml_validated(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _ZABER_POSITIONS_FILENAME),  # type: ignore[union-attr]
        payload=positions_payload,
        validator_cls=ZaberPositions,
        overwrite=overwrite,
    )


@mcp.tool()
def read_session_mesoscope_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the MesoscopePositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized MesoscopePositions payload), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _MESOSCOPE_POSITIONS_FILENAME),  # type: ignore[union-attr]
        validator_cls=MesoscopePositions,
    )


@mcp.tool()
def write_session_mesoscope_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the MesoscopePositions YAML for a session.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.
        positions_payload: The complete MesoscopePositions payload to write.
        overwrite: Determines whether to overwrite an existing positions file.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the validated MesoscopePositions payload), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return write_yaml_validated(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _MESOSCOPE_POSITIONS_FILENAME),  # type: ignore[union-attr]
        payload=positions_payload,
        validator_cls=MesoscopePositions,
        overwrite=overwrite,
    )


@mcp.tool()
def read_session_system_configuration_tool(session_path: str) -> dict[str, Any]:
    """Loads the per-session snapshot of MesoscopeSystemConfiguration.

    Args:
        session_path: The path to the session directory or its raw_data subdirectory.

    Returns:
        A dictionary with ``file_path`` and ``data`` (the serialized MesoscopeSystemConfiguration snapshot), or
        ``{"error": ...}`` on failure.
    """
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _SESSION_SYSTEM_CONFIG_FILENAME),  # type: ignore[union-attr]
        validator_cls=MesoscopeSystemConfiguration,
    )


@mcp.tool()
def preprocess_session_tool(session_path: str) -> str:
    """Preprocesses a session's data stored on the data acquisition system's host-machine.

    Args:
        session_path: The absolute path to the session directory to preprocess. The session must be located
            inside the data root of the data acquisition system.

    Returns:
        A success message upon completion, or an error description if preprocessing fails.
    """
    try:
        path = Path(session_path)
        system_configuration = get_system_configuration()
        data_root = get_data_root()

        # Validates that the session is stored locally.
        if not path.is_relative_to(data_root):
            return (
                f"Error: Session directory must be inside the data root of the "
                f"{system_configuration.name} data acquisition system "
                f"({data_root})."
            )

        session_data = SessionData.load(session_path=path)
        preprocess_session_data(session_data)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Session preprocessed: {session_path}"


@mcp.tool()
def delete_session_tool(session_path: str, *, confirm_deletion: bool = False) -> str:
    """Removes a session's data from all storage locations accessible to the data acquisition system.

    Important:
        This operation is irreversible and removes data from all machines and long-term storage destinations.
        The AI agent MUST warn the user about the consequences of this action before calling this tool with
        confirm_deletion=True.

    Args:
        session_path: The absolute path to the session directory to delete. The session must be located
            inside the data root of the data acquisition system.
        confirm_deletion: Safety parameter that must be explicitly set to True to proceed with deletion.
            When False (the default), the tool returns a warning message instead of deleting data.

    Returns:
        A success message upon completion, a safety warning if 'confirm_deletion' is False, or an error description
        if deletion fails.
    """
    # Enforces explicit confirmation before proceeding with deletion.
    if not confirm_deletion:
        return (
            "Error: Session deletion requires explicit confirmation. Set confirm_deletion=True to proceed. "
            "WARNING: This operation permanently removes the session's data from all machines and long-term "
            "storage destinations accessible to the data acquisition system. This action cannot be undone."
        )

    try:
        path = Path(session_path)
        system_configuration = get_system_configuration()
        data_root = get_data_root()

        # Validates that the session is stored locally.
        if not path.is_relative_to(data_root):
            return (
                f"Error: Session directory must be inside the data root of the "
                f"{system_configuration.name} data acquisition system "
                f"({data_root})."
            )

        session_data = SessionData.load(session_path=path)
        purge_session(session_data)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Session deleted: {session_path}"


@mcp.tool()
def migrate_animal_tool(source_project: str, destination_project: str, animal_id: str) -> str:
    """Transfers all sessions for an animal from one project to another.

    Args:
        source_project: The name of the project from which to migrate the data.
        destination_project: The name of the project to which to migrate the data.
        animal_id: The ID of the animal whose session data to migrate.

    Returns:
        A success message upon completion, or an error description if migration fails.
    """
    try:
        migrate_animal_between_projects(
            source_project=source_project,
            target_project=destination_project,
            animal=animal_id,
        )
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Animal {animal_id} migrated: {source_project} -> {destination_project}"


def _resolve_session_root(session_path: str) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolves an input session path to its root directory (the parent of raw_data)."""
    path = Path(session_path)
    if not path.exists():
        return None, {"error": f"Session path does not exist: {path}"}
    if path.joinpath(_RAW_DATA_DIR).is_dir():
        return path, None
    if path.name == _RAW_DATA_DIR and path.is_dir():
        return path.parent, None
    return None, {"error": f"Could not locate the {_RAW_DATA_DIR} directory under {path}"}


def _check_path(path: Path) -> dict[str, Any]:
    """Returns a diagnostic report for a single filesystem path."""
    report: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        report["ok"] = False
        report["error"] = "Path does not exist"
        return report
    report["is_mount"] = path.is_mount()
    write_error = probe_writable(path=path)
    report["writable"] = write_error is None
    if write_error is not None:
        report["error"] = f"Not writable: {write_error}"
    report["ok"] = report["exists"] and report["writable"]
    return report


def _filesystem_paths_report(configuration: MesoscopeSystemConfiguration) -> dict[str, Any]:
    """Builds a per-path diagnostic report for the filesystem configuration of the Mesoscope-VR system.

    Notes:
        Long-term storage destinations whose root is left unset are reported as not configured rather than as errors,
        since configuring them is optional.

    Args:
        configuration: The Mesoscope-VR system configuration whose filesystem paths are reported on.

    Returns:
        A dictionary mapping each configuration path name to its diagnostic report.
    """
    filesystem = configuration.filesystem
    # The local data root is owned by the Sollertia platform, not the Mesoscope-VR filesystem configuration, so it
    # is resolved separately and reported as not configured when the platform data root has not been set.
    try:
        data_root_report = _check_path(path=get_data_root())
    except FileNotFoundError as exception:
        data_root_report = {"path": "", "exists": False, "ok": False, "error": str(exception)}
    paths: dict[str, Any] = {
        "data_root": data_root_report,
        "mesoscope_directory": _check_path(path=filesystem.mesoscope_directory),
    }
    for destination_name, destination_root in filesystem.storage_directories.items():
        report_key = f"storage_directory:{destination_name}"
        if destination_root == Path():
            paths[report_key] = {"path": str(destination_root), "configured": False, "ok": True}
            continue
        report = _check_path(path=destination_root)
        report["configured"] = True
        paths[report_key] = report
    return paths
