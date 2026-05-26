"""Provides MCP servers for agentic interaction with sollertia-experiment CLI functionality.

This module exposes tools from the 'sle get' and 'sle manage' subcommand groups through the Model Context Protocol
(MCP), enabling AI agents to programmatically interact with data acquisition system features.
"""

from __future__ import annotations

from enum import Enum
import uuid
from typing import TYPE_CHECKING, Any, Literal, get_type_hints
from pathlib import Path
import contextlib
from dataclasses import MISSING, fields, is_dataclass

import yaml  # type: ignore[import-untyped]
from mcp.server.fastmcp import FastMCP
from ataraxis_base_utilities import ensure_directory_exists
from sollertia_shared_assets import SessionData, get_data_root

if TYPE_CHECKING:
    from ataraxis_data_structures import YamlConfig

from ..cross_system import (
    CRCCalculator,
    get_zaber_devices_info,
    set_zaber_device_setting,
    get_zaber_device_settings,
    validate_zaber_device_configuration,
)
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

# Initializes the MCP server for 'sle get' tools.
get_mcp = FastMCP(name="sollertia-experiment-get", json_response=True)

# Initializes the MCP server for 'sle manage' tools.
manage_mcp = FastMCP(name="sollertia-experiment-manage", json_response=True)


@get_mcp.tool()
def get_zaber_devices_tool() -> str:
    """Identifies Zaber devices accessible to the data acquisition system.

    Scans all available serial ports and returns a formatted table containing port, device, and axis information
    for all discovered Zaber motor controllers.

    Notes:
        Connection errors encountered during scanning are logged at DEBUG level and do not interrupt the discovery
        process. Ports with connection errors are listed as having "No Devices".
    """
    try:
        return get_zaber_devices_info()
    except Exception as exception:
        return f"Error: {exception}"


@get_mcp.tool()
def get_checksum_tool(input_string: str) -> str:
    """Calculates the CRC32-XFER checksum for the input string.

    Args:
        input_string: The string for which to compute the checksum.

    Returns:
        The computed CRC32-XFER checksum value.
    """
    try:
        calculator = CRCCalculator()
        checksum = calculator.string_checksum(input_string)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"CRC32-XFER checksum for '{input_string}': {checksum}"


@get_mcp.tool()
def get_zaber_device_settings_tool(port: str, device_index: int) -> str:
    """Reads configuration settings from a Zaber device's non-volatile memory.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).

    Returns:
        A formatted string containing device settings including labels, positions, flags, and motion limits.
    """
    try:
        settings = get_zaber_device_settings(port=port, device_index=device_index)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return (
            f"Device: {settings.device_label or '(not set)'} | Axis: {settings.axis_label or '(not set)'} | "
            f"Checksum: {settings.checksum} | Shutdown: {settings.shutdown_flag} | Unsafe: {settings.unsafe_flag} | "
            f"Park: {settings.park_position} | Maintenance: {settings.maintenance_position} | "
            f"Mount: {settings.mount_position} | Limits: [{settings.limit_min}, {settings.limit_max}] | "
            f"Position: {settings.current_position}"
        )


@get_mcp.tool()
def set_zaber_device_setting_tool(
    port: str,
    device_index: int,
    setting: str,
    value: str,
    *,
    confirm: bool = False,
) -> str:
    """Writes a configuration setting to a Zaber device's non-volatile memory.

    Important:
        This operation modifies hardware non-volatile memory. The AI agent MUST show the user the current value
        and proposed change before calling with confirm=True.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).
        setting: Setting name. Valid options are park_position, maintenance_position, mount_position,
            unsafe_flag, shutdown_flag, device_label, and axis_label.
        value: Value to write. Use integer strings for positions and flags, regular strings for labels.
        confirm: Must be True to execute the write operation. When False, returns a preview without modifying hardware.

    Returns:
        Success message with old and new values, or a preview message if confirm is False.
    """
    if not confirm:
        try:
            settings = get_zaber_device_settings(port=port, device_index=device_index)
        except Exception as exception:
            return f"Error: {exception}"
        else:
            current_values = {
                "park_position": settings.park_position,
                "maintenance_position": settings.maintenance_position,
                "mount_position": settings.mount_position,
                "unsafe_flag": settings.unsafe_flag,
                "shutdown_flag": settings.shutdown_flag,
                "device_label": settings.device_label,
                "axis_label": settings.axis_label,
            }
            if setting not in current_values:
                return f"Error: Invalid setting '{setting}'. Valid: {', '.join(sorted(current_values.keys()))}"
            current = current_values[setting]
            return f"Preview: {setting} would change from '{current}' to '{value}'. Set confirm=True to apply."

    try:
        # Converts value to appropriate type based on setting.
        if setting in {"park_position", "maintenance_position", "mount_position", "unsafe_flag", "shutdown_flag"}:
            typed_value: int | str = int(value)
        else:
            typed_value = value

        result = set_zaber_device_setting(
            port=port,
            device_index=device_index,
            setting=setting,
            value=typed_value,
        )
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"Success: {result}"


@get_mcp.tool()
def check_mount_accessibility_tool(path: str) -> str:
    """Verifies that a filesystem path exists and is writable.

    Probes the path by writing and removing a temporary file to confirm write access. Useful for verifying
    that a mounted storage location is reachable before invoking acquisition or transfer operations.

    Args:
        path: The absolute filesystem path to verify.

    Returns:
        A formatted string reporting existence, mount status, and writability, or an error description.
    """
    target = Path(path)
    if str(target) in ("", "."):
        return f"Error: Path '{path}' is empty or relative; provide an absolute path."
    if not target.exists():
        return f"Path: {target} | Exists: False | OK: False"

    is_mount = target.is_mount()
    test_file = target.joinpath(f".mount_test_{uuid.uuid4().hex[:8]}")
    try:
        test_file.write_text("test")
        test_file.unlink()
    except PermissionError:
        return (
            f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: False | OK: False | "
            f"Error: Permission denied"
        )
    except OSError as os_error:
        return f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: False | OK: False | Error: {os_error}"

    return f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: True | OK: True"


@get_mcp.tool()
def validate_zaber_configuration_tool(port: str, device_index: int) -> str:
    """Validates a Zaber device's configuration for use with the binding library.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).

    Returns:
        A validation report including checksum verification, position bounds checking, and any errors or warnings.
    """
    try:
        result = validate_zaber_device_configuration(port=port, device_index=device_index)
        status = "VALID" if result.is_valid else "INVALID"
        parts = [
            f"Status: {status} | Checksum: {'OK' if result.checksum_valid else 'FAIL'} | "
            f"Positions: {'OK' if result.positions_valid else 'FAIL'}"
        ]

        if result.errors:
            parts.append(f"Errors: {'; '.join(result.errors)}")
        if result.warnings:
            parts.append(f"Warnings: {'; '.join(result.warnings)}")

        return " | ".join(parts)
    except Exception as exception:
        return f"Error: {exception}"


@manage_mcp.tool()
def preprocess_session_tool(session_path: str) -> str:
    """Preprocesses a session's data stored on the data acquisition system's host machine.

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


@manage_mcp.tool()
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


@manage_mcp.tool()
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


_HARDWARE_STATE_FILENAME: str = "hardware_state.yaml"
"""Canonical filename for the per-session MesoscopeHardwareState YAML."""

_ZABER_POSITIONS_FILENAME: str = "zaber_positions.yaml"
"""Canonical filename for the per-session ZaberPositions YAML."""

_MESOSCOPE_POSITIONS_FILENAME: str = "mesoscope_positions.yaml"
"""Canonical filename for the per-session MesoscopePositions YAML."""

_SESSION_SYSTEM_CONFIG_FILENAME: str = "system_configuration.yaml"
"""Canonical filename for the per-session snapshot of MesoscopeSystemConfiguration."""

_RAW_DATA_DIR: str = "raw_data"
"""Subdirectory under each session root that holds the raw data and metadata files."""


def _serialize(value: Any) -> Any:
    """Recursively converts a dataclass, Path, Enum, mapping, or sequence into JSON-friendly Python."""
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_definition.name: _serialize(value=getattr(value, field_definition.name))
            for field_definition in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize(value=item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(value=item) for item in value]
    return value


def _describe_dataclass(cls: type, *, seen: frozenset[type] | None = None) -> dict[str, Any]:
    """Returns a structured schema description of a dataclass type, recursively describing nested dataclasses."""
    seen = frozenset() if seen is None else seen
    if cls in seen:
        return {"class": cls.__name__, "recursive_reference": True}
    if not is_dataclass(cls):
        type_name = cls.__name__ if isinstance(cls, type) else str(cls)
        return {"type": type_name}

    next_seen = seen | {cls}
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    schema: dict[str, Any] = {"class": cls.__name__, "fields": {}}
    # noinspection PyDataclass
    for field_definition in fields(cls):
        type_hint = hints.get(field_definition.name, field_definition.type)
        type_name = type_hint.__name__ if isinstance(type_hint, type) else str(type_hint).replace("typing.", "")
        field_schema: dict[str, Any] = {"type": type_name}
        if field_definition.default is not MISSING:
            field_schema["default"] = _serialize(value=field_definition.default)
        elif field_definition.default_factory is not MISSING:
            try:
                field_schema["default"] = _serialize(value=field_definition.default_factory())
            except Exception:
                field_schema["required"] = True
        else:
            field_schema["required"] = True
        if isinstance(type_hint, type) and is_dataclass(type_hint):
            field_schema["nested"] = _describe_dataclass(cls=type_hint, seen=next_seen)
        schema["fields"][field_definition.name] = field_schema
    return schema


def _write_yaml_validated(
    file_path: Path,
    payload: dict[str, Any],
    validator_cls: type[YamlConfig],
    *,
    overwrite: bool = False,
    use_save_method: bool = False,
) -> dict[str, Any]:
    """Writes a payload as YAML and validates by round-tripping through ``validator_cls``."""
    if file_path.exists() and not overwrite:
        return {"error": f"File already exists: {file_path}. Pass overwrite=True to replace."}

    ensure_directory_exists(path=file_path.parent)
    temp_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        temp_path.write_text(yaml.safe_dump(data=payload, sort_keys=False))
        instance = validator_cls.from_yaml(file_path=temp_path)
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()
    except Exception as exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        return {"error": f"Validation failed for {validator_cls.__name__}: {exception}"}
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()

    try:
        if use_save_method and hasattr(instance, "save"):
            instance.save(path=file_path)
        else:
            instance.to_yaml(file_path=file_path)
    except Exception as exception:
        return {"error": f"Failed to persist {validator_cls.__name__} to {file_path}: {exception}"}

    return {"file_path": str(file_path), "data": _serialize(value=instance)}


def _read_yaml(file_path: Path, validator_cls: type[YamlConfig]) -> dict[str, Any]:
    """Loads a YAML file via ``validator_cls`` and returns its serialized form."""
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    try:
        instance = validator_cls.from_yaml(file_path=file_path)
    except Exception as exception:
        return {"error": f"Failed to load {file_path} as {validator_cls.__name__}: {exception}"}
    return {"file_path": str(file_path), "data": _serialize(value=instance)}


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
    try:
        probe = path.joinpath(f".__sollertia_experiment_probe_{uuid.uuid4().hex[:8]}")
        probe.touch()
        probe.unlink()
        report["writable"] = True
    except OSError as exception:
        report["writable"] = False
        report["error"] = f"Not writable: {exception}"
    report["ok"] = report["exists"] and report.get("writable", False)
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


@get_mcp.tool()
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
    return {"file_path": str(get_system_configuration_path()), "data": _serialize(value=instance)}


@get_mcp.tool()
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
    return _write_yaml_validated(
        file_path=file_path,
        payload=configuration_payload,
        validator_cls=MesoscopeSystemConfiguration,
        overwrite=overwrite,
        use_save_method=True,
    )


@get_mcp.tool()
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


@get_mcp.tool()
def describe_system_configuration_schema_tool() -> dict[str, Any]:
    """Returns the schema for MesoscopeSystemConfiguration and its nested hardware dataclasses."""
    return {"schema": _describe_dataclass(cls=MesoscopeSystemConfiguration)}


@get_mcp.tool()
def check_system_mounts_tool() -> dict[str, Any]:
    """Verifies all filesystem paths declared in the active Mesoscope-VR system configuration."""
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


@get_mcp.tool()
def read_session_zaber_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the ZaberPositions YAML for a session."""
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return _read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _ZABER_POSITIONS_FILENAME),  # type: ignore[union-attr]
        validator_cls=ZaberPositions,
    )


@get_mcp.tool()
def write_session_zaber_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the ZaberPositions YAML for a session."""
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return _write_yaml_validated(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _ZABER_POSITIONS_FILENAME),  # type: ignore[union-attr]
        payload=positions_payload,
        validator_cls=ZaberPositions,
        overwrite=overwrite,
    )


@get_mcp.tool()
def read_session_mesoscope_positions_tool(session_path: str) -> dict[str, Any]:
    """Loads the MesoscopePositions YAML for a session."""
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return _read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _MESOSCOPE_POSITIONS_FILENAME),  # type: ignore[union-attr]
        validator_cls=MesoscopePositions,
    )


@get_mcp.tool()
def write_session_mesoscope_positions_tool(
    session_path: str,
    positions_payload: dict[str, Any],
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Creates or replaces the MesoscopePositions YAML for a session."""
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return _write_yaml_validated(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _MESOSCOPE_POSITIONS_FILENAME),  # type: ignore[union-attr]
        payload=positions_payload,
        validator_cls=MesoscopePositions,
        overwrite=overwrite,
    )


@get_mcp.tool()
def read_session_system_configuration_tool(session_path: str) -> dict[str, Any]:
    """Loads the per-session snapshot of MesoscopeSystemConfiguration."""
    session_root, error = _resolve_session_root(session_path=session_path)
    if error is not None:
        return error
    return _read_yaml(
        file_path=session_root.joinpath(_RAW_DATA_DIR, _SESSION_SYSTEM_CONFIG_FILENAME),  # type: ignore[union-attr]
        validator_cls=MesoscopeSystemConfiguration,
    )


def run_get_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the 'sle get' MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration), 'sse' for Server-Sent Events,
            and 'streamable-http' for HTTP-based communication.
    """
    get_mcp.run(transport=transport)


def run_manage_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the 'sle manage' MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration), 'sse' for Server-Sent Events,
            and 'streamable-http' for HTTP-based communication.
    """
    manage_mcp.run(transport=transport)
