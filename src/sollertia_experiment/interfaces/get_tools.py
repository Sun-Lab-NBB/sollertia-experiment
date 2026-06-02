"""Provides MCP tools wrapping the general, hardware-agnostic 'sle get' acquisition system discovery utilities."""

from __future__ import annotations

from pathlib import Path

from ..vr_task import UnityBridgeClient
from .mcp_instance import mcp, probe_writable
from ..cross_system import (
    CRCCalculator,
    get_zaber_devices_info,
    set_zaber_device_setting,
    get_zaber_device_settings,
    validate_zaber_device_configuration,
)


@mcp.tool()
def get_zaber_devices_tool() -> str:
    """Identifies Zaber devices accessible to the data acquisition system.

    Scans all available serial ports and returns a formatted table containing port, device, and axis information
    for all discovered Zaber motor controllers.

    Notes:
        Connection errors encountered during scanning are logged at DEBUG level and do not interrupt the discovery
        process. Ports with connection errors are listed as having "No Devices".

    Returns:
        A formatted table listing each scanned port together with the device and axis information discovered on it,
        or an error description on failure.
    """
    try:
        return get_zaber_devices_info()
    except Exception as exception:
        return f"Error: {exception}"


@mcp.tool()
def get_checksum_tool(input_string: str) -> str:
    """Calculates the CRC32-XFER checksum for the input string.

    Args:
        input_string: The string for which to compute the checksum.

    Returns:
        The computed CRC32-XFER checksum value.
    """
    try:
        calculator = CRCCalculator()
        checksum = calculator.string_checksum(string=input_string)
    except Exception as exception:
        return f"Error: {exception}"
    else:
        return f"CRC32-XFER checksum for '{input_string}': {checksum}"


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
    write_error = probe_writable(path=target)
    if write_error is not None:
        return f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: False | OK: False | Error: {write_error}"

    return f"Path: {target} | Exists: True | Mount: {is_mount} | Writable: True | OK: True"


@mcp.tool()
def check_unity_bridge_tool() -> str:
    """Checks whether the Unity Editor MCP Bridge is reachable for Virtual Reality task sessions.

    Probes the editor-only bridge that the Virtual Reality task driver uses to control scene activation and Play
    Mode. Use this tool during pre-flight health checks to confirm the Unity Editor is open before starting a
    Mesoscope experiment session.

    Returns:
        A status line reporting whether the bridge is reachable, and when reachable, the active scene name and the
        editor's play state.
    """
    client = UnityBridgeClient()
    try:
        return client.describe_status()
    finally:
        client.close()


@mcp.tool()
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
