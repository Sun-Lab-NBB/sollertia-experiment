from .mcp_instance import (
    mcp as mcp,
    probe_writable as probe_writable,
)
from ..cross_system import (
    CRCCalculator as CRCCalculator,
    get_zaber_devices_info as get_zaber_devices_info,
    set_zaber_device_setting as set_zaber_device_setting,
    get_zaber_device_settings as get_zaber_device_settings,
    validate_zaber_device_configuration as validate_zaber_device_configuration,
)

def get_zaber_devices_tool() -> str: ...
def get_checksum_tool(input_string: str) -> str: ...
def get_zaber_device_settings_tool(port: str, device_index: int) -> str: ...
def set_zaber_device_setting_tool(
    port: str, device_index: int, setting: str, value: str, *, confirm: bool = False
) -> str: ...
def check_mount_accessibility_tool(path: str) -> str: ...
def validate_zaber_configuration_tool(port: str, device_index: int) -> str: ...
