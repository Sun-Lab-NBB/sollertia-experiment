"""Provides MCP servers for agentic interaction with sl-experiment CLI functionality.

This module exposes tools from the 'sl-get' and 'sl-manage' CLI groups through the Model Context Protocol (MCP),
enabling AI agents to programmatically interact with data acquisition system features.
"""

import uuid
from typing import Literal
from pathlib import Path

from sl_shared_assets import SessionData, get_system_configuration_data
from mcp.server.fastmcp import FastMCP

from ..mesoscope_vr import (
    CRCCalculator,
    purge_session,
    get_zaber_devices_info,
    preprocess_session_data,
    set_zaber_device_setting,
    get_zaber_device_settings,
    migrate_animal_between_projects,
    validate_zaber_device_configuration,
)

# Initializes the MCP server for sl-get tools.
get_mcp = FastMCP(name="sl-experiment-get", json_response=True)

# Initializes the MCP server for sl-manage tools.
manage_mcp = FastMCP(name="sl-experiment-manage", json_response=True)


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
        return f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: False | OK: False | Error: Permission denied"
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
            inside the root directory of the data acquisition system.

    Returns:
        A success message upon completion, or an error description if preprocessing fails.
    """
    try:
        path = Path(session_path)
        system_configuration = get_system_configuration_data()

        # Validates that the session is stored locally.
        if not path.is_relative_to(system_configuration.filesystem.root_directory):
            return (
                f"Error: Session directory must be inside the root directory of the "
                f"{system_configuration.name} data acquisition system "
                f"({system_configuration.filesystem.root_directory})."
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
            inside the root directory of the data acquisition system.
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
        system_configuration = get_system_configuration_data()

        # Validates that the session is stored locally.
        if not path.is_relative_to(system_configuration.filesystem.root_directory):
            return (
                f"Error: Session directory must be inside the root directory of the "
                f"{system_configuration.name} data acquisition system "
                f"({system_configuration.filesystem.root_directory})."
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


def run_get_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the sl-get MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration), 'sse' for Server-Sent Events,
            and 'streamable-http' for HTTP-based communication.
    """
    get_mcp.run(transport=transport)


def run_manage_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the sl-manage MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration), 'sse' for Server-Sent Events,
            and 'streamable-http' for HTTP-based communication.
    """
    manage_mcp.run(transport=transport)
