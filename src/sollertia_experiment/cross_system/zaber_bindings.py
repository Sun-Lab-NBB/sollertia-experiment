"""Provides interfaces for working with Zaber motor controllers and devices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import field, dataclass

from crc import Calculator, Configuration
from tabulate import tabulate
from zaber_motion import Tools
from ataraxis_time import Timeout, TimerPrecisions
from zaber_motion.ascii import Axis, Device, Connection, SettingConstants
from ataraxis_base_utilities import LogLevel, console

from .shutdown_tools import run_shutdown_step
from .terminal_prompts import request_required_confirmation

if TYPE_CHECKING:
    from collections.abc import Callable

_PARK_POSITION_TOLERANCE: float = 100.0
"""The largest deviation, in native motor units, between the resting position of a parked motor and the park position
stored in its non-volatile memory that still counts as the motor resting where it should. The window absorbs the
microstep-scale settling error left behind by the final move command, while staying far below the deviation that would
carry an unsafe motor outside the range from which it can be homed.
"""


@dataclass(frozen=True, slots=True)
class ZaberDeviceSettings:
    """Stores configuration settings read from a Zaber device's non-volatile memory."""

    device_label: str
    """The user-assigned name of the device."""
    axis_label: str
    """The user-assigned name of the axis."""
    checksum: int
    """The CRC32-XFER checksum stored in USER_DATA_0."""
    shutdown_flag: int
    """The shutdown flag stored in USER_DATA_1 (1 = proper shutdown, 0 = abnormal)."""
    unsafe_flag: int
    """The unsafe flag stored in USER_DATA_10 (1 = requires safe position for homing)."""
    park_position: int
    """The park position in native motor units stored in USER_DATA_11."""
    maintenance_position: int
    """The maintenance position in native motor units stored in USER_DATA_12."""
    mount_position: int
    """The mount position in native motor units stored in USER_DATA_13."""
    limit_min: float
    """The minimum allowed position relative to home in native motor units."""
    limit_max: float
    """The maximum allowed position relative to home in native motor units."""
    current_position: float
    """The current absolute position of the motor in native motor units."""


@dataclass(frozen=True, slots=True)
class ZaberValidationResult:
    """Stores the results of validating a Zaber device's configuration."""

    is_valid: bool
    """Determines whether the device configuration is valid for use with the binding library."""
    checksum_valid: bool
    """Determines whether the stored checksum matches the calculated checksum for the device label."""
    positions_valid: bool
    """Determines whether all predefined positions are within the device motion limits."""
    errors: tuple[str, ...]
    """Contains critical issues that prevent the device from being used with the binding library."""
    warnings: tuple[str, ...]
    """Contains non-critical issues that may affect device operation."""


@dataclass(slots=True)
class _ZaberAxisData:
    """Stores the identification data for an axis of a Zaber device."""

    axis_id: int
    """The 1-based positional number of the axis within its parent device."""
    axis_label: str
    """The user-assigned name of the axis."""


@dataclass(slots=True)
class _ZaberDeviceData:
    """Stores the identification data about a Zaber device."""

    device_number: int
    """The 1-based positional number of the device in the daisy-chain of devices connected to the same serial
    port. This equals the device_index used by the configuration functions plus one."""
    device_id: int
    """The unique identifier code of the device."""
    label: str
    """The user-assigned name of the device."""
    name: str
    """The manufacturer-assigned name of the device."""
    axes: list[_ZaberAxisData] = field(default_factory=list)
    """Stores _ZaberAxisData instances for each axis managed by this device."""


@dataclass(slots=True)
class _ZaberPortData:
    """Stores the identification data for all Zaber devices connected to a serial port."""

    port_name: str
    """The name of the USB port."""
    devices: list[_ZaberDeviceData] = field(default_factory=list)
    """Stores _ZaberDeviceData instances for each device connected to this port."""

    @property
    def has_devices(self) -> bool:
        """Returns True if any devices are connected to this port."""
        return bool(self.devices)


@dataclass(frozen=True)
class _ZaberSettings:
    """Defines the set of codes used to access Zaber settings stored in each interfaced device's non-volatile memory."""

    maximum_limit: str = SettingConstants.LIMIT_MAX
    """The maximum absolute position, in native motor units, the motor is allowed to reach during runtime, relative to
    the motor's home position."""
    minimum_limit: str = SettingConstants.LIMIT_MIN
    """The minimum absolute position, in native motor units, the motor is allowed to reach during runtime, relative to
    the motor's home position."""
    position: str = SettingConstants.POS
    """The current absolute position of the motor, in native motor units, relative to its home position."""
    checksum: str = SettingConstants.USER_DATA_0
    """The CRC32-XFER checksum that should match the checksum of the device's label, which is used to confirm that the
    device has been configured to work with the bindings exposed by this library. Uses USER_DATA 0 variable."""
    shutdown_flag: str = SettingConstants.USER_DATA_1
    """Tracks whether the device has been properly shut down during the previous runtime. Uses USER_DATA 1 variable."""
    unsafe_flag: str = SettingConstants.USER_DATA_10
    """Tracks whether the device can be positioned in a way that is not safe to home after power cycling.
    Uses USER_DATA 10 variable."""
    axis_park_position: str = SettingConstants.USER_DATA_11
    """The absolute position, in native motor units, where the motor should be moved to before parking and shutting
    down. Uses USER_DATA 11 variable."""
    axis_maintenance_position: str = SettingConstants.USER_DATA_12
    """The absolute position, in native motor units, where the motor should be moved as part of the preparation for the
    system's maintenance. Uses USER_DATA 12 variable.
    """
    axis_mount_position: str = SettingConstants.USER_DATA_13
    """The absolute position, in native motor units, where the motor should be moved before mounting the animal into the
    system's enclosure. Uses USER_DATA 13 variable.
    """


def discover_zaber_devices() -> None:
    """Scans all available serial ports and displays information about connected Zaber devices.

    Notes:
        Connection errors encountered during scanning are logged at DEBUG level and do not interrupt
        the discovery process. Ports that cannot be connected are listed as having "No Devices".
    """
    port_info_list = _scan_active_ports()
    formatted_info = _format_device_info(port_info_list=port_info_list)

    # Prints the formatted table. Uses console.echo with raw=True to bypass the console's line-wrapping and log
    # prefixing since the table is already formatted by tabulate.
    console.echo(message="Device and Axis Information:", raw=True)
    console.echo(message=formatted_info, raw=True)


def get_zaber_devices_info() -> str:
    """Scans all available serial ports for Zaber devices and returns formatted device information.

    Notes:
        Connection errors encountered during scanning are logged at DEBUG level and do not interrupt the discovery
        process.

    Returns:
        A formatted table string containing port, device, and axis information for all discovered Zaber devices.
        Ports with connection errors are listed as having "No Devices".
    """
    port_info_list = _scan_active_ports()
    return _format_device_info(port_info_list=port_info_list)


def get_zaber_device_settings(port: str, device_index: int) -> ZaberDeviceSettings:
    """Reads configuration settings from a Zaber device's non-volatile memory.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).

    Returns:
        A ZaberDeviceSettings instance containing the device configuration including labels, positions,
        flags, and motion limits.

    Raises:
        ConnectionError: If unable to connect to the specified port.
        IndexError: If device_index is out of range for the connected devices.
    """
    try:
        with Connection.open_serial_port(port_name=port, direct=False) as connection:
            devices = connection.detect_devices()

            if device_index < 0 or device_index >= len(devices):
                message = (
                    f"Unable to read settings from device at index {device_index}. The port {port} has "
                    f"{len(devices)} device(s) connected (valid indices: 0 to {len(devices) - 1})."
                )
                console.error(message=message, error=IndexError)

            device = devices[device_index]
            axis = device.get_axis(axis_number=1)

            # Reads all configuration settings from non-volatile memory.
            return ZaberDeviceSettings(
                device_label=device.label or "",
                axis_label=axis.label or "",
                checksum=int(device.settings.get(setting=SettingConstants.USER_DATA_0)),
                shutdown_flag=int(device.settings.get(setting=SettingConstants.USER_DATA_1)),
                unsafe_flag=int(device.settings.get(setting=SettingConstants.USER_DATA_10)),
                park_position=int(device.settings.get(setting=SettingConstants.USER_DATA_11)),
                maintenance_position=int(device.settings.get(setting=SettingConstants.USER_DATA_12)),
                mount_position=int(device.settings.get(setting=SettingConstants.USER_DATA_13)),
                limit_min=axis.settings.get(setting=SettingConstants.LIMIT_MIN),
                limit_max=axis.settings.get(setting=SettingConstants.LIMIT_MAX),
                current_position=axis.get_position(),
            )

    except Exception as exception:
        if isinstance(exception, IndexError):
            raise
        message = f"Unable to connect to Zaber device on port {port}: {exception}"
        console.error(message=message, error=ConnectionError)


def set_zaber_device_setting(port: str, device_index: int, setting: str, value: int | str) -> str:
    """Writes a configuration setting to a Zaber device's non-volatile memory.

    Notes:
        Position values are validated against device motion limits before writing. Device label changes
        automatically update the checksum (USER_DATA_0) to maintain device validation, and axis label changes do not.
        The checksum setting cannot be modified directly as it is managed by the binding library.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).
        setting: Setting name. Valid options are park_position, maintenance_position, mount_position,
            unsafe_flag, shutdown_flag, device_label, and axis_label.
        value: Value to write. Use integers for positions and flags, strings for labels.

    Returns:
        A success message containing the old and new values.

    Raises:
        ConnectionError: If unable to connect to the specified port.
        IndexError: If device_index is out of range for the connected devices.
        TypeError: If the value type does not match the setting (a non-string label, or a non-integer position or flag).
        ValueError: If the setting name is invalid, if the value is out of range, or if a device_label write
            succeeded but the matching USER_DATA_0 checksum write failed, leaving the label and the checksum
            possibly out of agreement.
    """
    valid_settings = {
        "park_position",
        "maintenance_position",
        "mount_position",
        "unsafe_flag",
        "shutdown_flag",
        "device_label",
        "axis_label",
    }
    protected_settings = {"checksum"}

    if setting in protected_settings:
        message = (
            f"Unable to modify the '{setting}' setting directly. This setting is managed by the binding library "
            f"and cannot be changed through this interface."
        )
        console.error(message=message, error=ValueError)

    if setting not in valid_settings:
        message = f"Unable to modify setting '{setting}'. Valid settings are: {', '.join(sorted(valid_settings))}."
        console.error(message=message, error=ValueError)

    try:
        with Connection.open_serial_port(port_name=port, direct=False) as connection:
            devices = connection.detect_devices()

            if device_index < 0 or device_index >= len(devices):
                message = (
                    f"Unable to write setting to device at index {device_index}. The port {port} has "
                    f"{len(devices)} device(s) connected (valid indices: 0 to {len(devices) - 1})."
                )
                console.error(message=message, error=IndexError)

            device = devices[device_index]
            axis = device.get_axis(axis_number=1)

            # Handles label settings.
            if setting == "device_label":
                if not isinstance(value, str):
                    message = f"Unable to set device_label. Expected a string value, but got {type(value).__name__}."
                    console.error(message=message, error=TypeError)

                old_value = device.label or ""
                device.set_label(label=value)

                # Calculates and updates the checksum to match the new label.
                calculator = CRCCalculator()
                new_checksum = calculator.string_checksum(string=value)
                try:
                    device.settings.set(setting=SettingConstants.USER_DATA_0, value=new_checksum)
                except (Exception, KeyboardInterrupt) as exception:
                    # The label and its checksum live in separate non-volatile variables written by separate device
                    # transactions, and every binding class refuses to open a device whose two variables disagree.
                    # Restoring the previous label brings the pair back into agreement, and the restore is itself
                    # best-effort so the checksum write failure remains the error the caller sees.
                    try:
                        device.set_label(label=old_value)
                    except (Exception, KeyboardInterrupt) as restore_exception:
                        restore_report = (
                            f"Restoring the previous label '{old_value}' also failed with: {restore_exception}, so "
                            f"the device now holds the label '{value}' with the checksum of '{old_value}'."
                        )
                    else:
                        restore_report = f"The previous label '{old_value}' was restored."
                    message = (
                        f"Unable to set device_label to '{value}'. The label was written to the device, but the "
                        f"matching USER_DATA_0 checksum write failed with: {exception}. {restore_report} Re-run the "
                        f"same device_label write to bring the label and the checksum back into agreement."
                    )
                    console.error(message=message, error=ValueError)

                return f"device_label: '{old_value}' -> '{value}' (checksum updated to {new_checksum})"

            if setting == "axis_label":
                if not isinstance(value, str):
                    message = f"Unable to set axis_label. Expected a string value, but got {type(value).__name__}."
                    console.error(message=message, error=TypeError)

                old_value = axis.label or ""
                axis.set_label(label=value)
                return f"axis_label: '{old_value}' -> '{value}'"

            # Handles numeric settings. Ensures the value is an integer.
            if not isinstance(value, int):
                message = f"Unable to set {setting}. Expected an integer value, but got {type(value).__name__}."
                console.error(message=message, error=TypeError)

            # Validates position values against motion limits.
            if setting in {"park_position", "maintenance_position", "mount_position"}:
                limit_min = axis.settings.get(setting=SettingConstants.LIMIT_MIN)
                limit_max = axis.settings.get(setting=SettingConstants.LIMIT_MAX)

                if value < limit_min or value > limit_max:
                    message = (
                        f"Unable to set {setting} to {value}. The value must be within the device motion limits "
                        f"[{limit_min}, {limit_max}]."
                    )
                    console.error(message=message, error=ValueError)

            # Validates flag values.
            if setting == "unsafe_flag" and value not in (0, 1):
                message = f"Unable to set unsafe_flag to {value}. The value must be 0 or 1."
                console.error(message=message, error=ValueError)

            if setting == "shutdown_flag" and value not in (0, 1):
                message = f"Unable to set shutdown_flag to {value}. The value must be 0 or 1."
                console.error(message=message, error=ValueError)

            # Maps setting names to USER_DATA constants.
            setting_map = {
                "park_position": SettingConstants.USER_DATA_11,
                "maintenance_position": SettingConstants.USER_DATA_12,
                "mount_position": SettingConstants.USER_DATA_13,
                "unsafe_flag": SettingConstants.USER_DATA_10,
                "shutdown_flag": SettingConstants.USER_DATA_1,
            }

            user_data_setting = setting_map[setting]
            old_int_value = int(device.settings.get(setting=user_data_setting))
            device.settings.set(setting=user_data_setting, value=float(value))

            return f"{setting}: {old_int_value} -> {value}"

    except Exception as exception:
        if isinstance(exception, (IndexError, TypeError, ValueError)):
            raise
        message = f"Unable to connect to Zaber device on port {port}: {exception}"
        console.error(message=message, error=ConnectionError)


def validate_zaber_device_configuration(port: str, device_index: int) -> ZaberValidationResult:
    """Validates a Zaber device's configuration for use with the binding library.

    Notes:
        Performs comprehensive validation including checksum verification against the device label, position bounds
        checking against motion limits, and configuration completeness verification.

    Args:
        port: Serial port path (e.g., "/dev/ttyUSB0").
        device_index: Zero-based index in the daisy-chain (0 = closest to USB port).

    Returns:
        A ZaberValidationResult instance containing validation status, error messages, and warnings.

    Raises:
        ConnectionError: If unable to connect to the specified port.
        IndexError: If device_index is out of range for the connected devices.
    """
    try:
        settings = get_zaber_device_settings(port=port, device_index=device_index)
    except ConnectionError, IndexError:
        raise
    except Exception as exception:
        message = f"Unable to validate device configuration: {exception}"
        console.error(message=message, error=ConnectionError)

    errors: list[str] = []
    warnings: list[str] = []

    # Validates checksum against device label. An unlabeled device is rejected before any checksum comparison, since
    # the checksum of an empty label matches the factory USER_DATA_0 value of 0 and would otherwise pass.
    if not settings.device_label:
        errors.append("Device label is not set. Set device_label before using with binding library.")
        checksum_valid = False
    else:
        calculator = CRCCalculator()
        expected_checksum = calculator.string_checksum(string=settings.device_label)
        stored_checksum = settings.checksum
        checksum_valid = expected_checksum == stored_checksum

        if not checksum_valid:
            errors.append(
                f"Checksum mismatch: stored {stored_checksum}, expected {expected_checksum} for label "
                f"'{settings.device_label}'. Update device_label to recalculate checksum."
            )

    # Validates position values against motion limits.
    limit_min = settings.limit_min
    limit_max = settings.limit_max
    positions_valid = True

    position_checks = [
        ("park_position", settings.park_position),
        ("maintenance_position", settings.maintenance_position),
        ("mount_position", settings.mount_position),
    ]
    for position_name, position_value in position_checks:
        if position_value < limit_min or position_value > limit_max:
            positions_valid = False
            errors.append(f"{position_name} ({position_value}) is outside motion limits [{limit_min}, {limit_max}].")

    # Checks for potential configuration issues.
    if settings.shutdown_flag == 0 and settings.unsafe_flag == 1:
        warnings.append(
            "Device was not properly shut down and is marked as unsafe. Manual verification may be "
            "required before homing."
        )

    is_valid = checksum_valid and positions_valid and not errors

    return ZaberValidationResult(
        is_valid=is_valid,
        checksum_valid=checksum_valid,
        positions_valid=positions_valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


class CRCCalculator:
    """Exposes methods for calculating CRC32-XFER checksums for ASCII strings.

    Attributes:
        _calculator: The configured Calculator instance used to calculate the checksums.
    """

    def __init__(self) -> None:
        # Specializes and instantiates the CRC checksum calculator.
        configuration = Configuration(
            width=32,
            polynomial=0x000000AF,
            init_value=0x00000000,
            final_xor_value=0x00000000,
            reverse_input=False,
            reverse_output=False,
        )
        self._calculator: Calculator = Calculator(configuration=configuration)

    def string_checksum(self, string: str) -> int:
        """Calculates the CRC32-XFER checksum for the input string.

        Args:
            string: The string for which to calculate the CRC checksum.

        Returns:
            The integer CRC32-XFER checksum.
        """
        return self._calculator.checksum(data=bytes(string, "ASCII"))


# Initializes a shared CRCCalculator instance used by the ZaberDevice class instances to verify the interfaced device's
# configuration.
_crc_calculator = CRCCalculator()


class ZaberAxis:
    """Interfaces with a Zaber motor (axis).

    Notes:
        This class represents the lowest level of the tri-class hierarchy used to control Zaber motors during runtime.

    Args:
        motor: The Axis class instance that interfaces with the motor's hardware.

    Attributes:
        _motor: The Axis class instance that physically controls the motor's hardware through Zaber ASCII protocol.
        _park_position: The absolute position, in native motor units, where the motor should be moved to before parking
            and shutting down.
        _maintenance_position: The absolute position, in native motor units, where the motor should be moved as part of
            the preparation for the system's maintenance.
        _mount_position: The absolute position, in native motor units, where the motor should be moved before mounting
            the animal into the system's enclosure.
        _maximum_limit: The maximum absolute position relative to the home sensor position, in native motor units,
            the motor is allowed to reach during runtime. Read from the configurable 'limit.max' axis setting.
        _minimum_limit: The minimum absolute position relative to the home sensor position, in native motor units,
            the motor is allowed to reach during runtime. Read from the configurable 'limit.min' axis setting.
        _shutdown_flag: Tracks whether the motor has been shut down.
        _pacing_guard: A Timeout class instance that is used to ensure that communication with the motor is carried
            out at a pace that does not overwhelm the connection interface with too many successive calls.

    Raises:
        ValueError: If any parameter read from the motor's non-volatile memory is outside the expected range of
            values.
    """

    _COMMUNICATION_DELAY_MS: int = 5
    """The minimum delay, in milliseconds, that must separate all consecutive interactions with the motor's
    hardware."""

    def __init__(self, motor: Axis) -> None:
        self._shutdown_flag: bool = False

        # Parses hardcoded information stored in non-volatile hardware memory:
        self._motor: Axis = motor
        self._park_position: int = int(self._motor.device.settings.get(setting=_ZaberSettings.axis_park_position))
        self._maintenance_position: int = int(
            self._motor.device.settings.get(setting=_ZaberSettings.axis_maintenance_position)
        )
        self._mount_position: int = int(self._motor.device.settings.get(setting=_ZaberSettings.axis_mount_position))
        self._maximum_limit: float = self._motor.settings.get(setting=_ZaberSettings.maximum_limit)
        self._minimum_limit: float = self._motor.settings.get(setting=_ZaberSettings.minimum_limit)

        # Verifies that all predefined axis positions fall within the axis motion limits.
        if self._park_position < self._minimum_limit or self._park_position > self._maximum_limit:
            message = (
                f"Invalid parking position hardware parameter value encountered when initializing ZaberAxis class for "
                f"{self._motor.label} axis of the Device {self._motor.device.label}. Expected a value between "
                f"{self._minimum_limit} and {self._maximum_limit}, but read {self._park_position}."
            )
            console.error(message=message, error=ValueError)
        if self._maintenance_position < self._minimum_limit or self._maintenance_position > self._maximum_limit:
            message = (
                f"Invalid system maintenance position hardware parameter value encountered when initializing ZaberAxis "
                f"class for {self._motor.label} axis of the Device {self._motor.device.label}. Expected a value "
                f"between {self._minimum_limit} and {self._maximum_limit}, but read {self._maintenance_position}."
            )
            console.error(message=message, error=ValueError)
        if self._mount_position < self._minimum_limit or self._mount_position > self._maximum_limit:
            message = (
                f"Invalid animal mounting position hardware parameter value encountered when initializing ZaberAxis "
                f"class for {self._motor.label} axis of the Device {self._motor.device.label}. Expected a value"
                f" between {self._minimum_limit} and {self._maximum_limit}, but read {self._mount_position}."
            )
            console.error(message=message, error=ValueError)

        # Initializes a timeout guard to ensure the class cannot issue commands fast enough to overwhelm the motor
        # communication interface.
        self._pacing_guard: Timeout = Timeout(
            duration=self._COMMUNICATION_DELAY_MS, precision=TimerPrecisions.MILLISECOND
        )

    def __repr__(self) -> str:
        """Returns a string representation of the ZaberAxis instance."""
        return (
            f"ZaberAxis(name={self._motor.label}, homed={self.is_homed}, parked={self.is_parked}, busy={self.is_busy}, "
            f"position={self.get_position()})."
        )

    def get_position(self) -> float:
        """Returns the current absolute position of the motor, in native motor units, relative to its home position."""
        return self._padded_method_call(method=self._motor.get_position)

    @property
    def is_homed(self) -> bool:
        """Returns True if the motor has been homed (has a motion reference point)."""
        return self._padded_method_call(method=self._motor.is_homed)

    @property
    def is_parked(self) -> bool:
        """Returns True if the motor is parked."""
        return self._padded_method_call(method=self._motor.is_parked)

    @property
    def is_busy(self) -> bool:
        """Returns True if the motor is currently executing a command (is moving)."""
        return self._padded_method_call(method=self._motor.is_busy)

    @property
    def park_position(self) -> int:
        """Returns the absolute position, in native motor units, where the motor needs to be moved as part of the
        system's shutdown procedure.
        """
        return self._park_position

    @property
    def maintenance_position(self) -> int:
        """Returns the absolute position, in native motor units, where the motor needs to be moved as part of preparing
        the system for maintenance.
        """
        return self._maintenance_position

    @property
    def mount_position(self) -> int:
        """Returns the absolute position, in native motor units, where the motor needs to be moved before mounting
        the animal into the system's enclosure.
        """
        return self._mount_position

    def home(self) -> None:
        """Homes the motor by moving it towards the home sensor position until it triggers the sensor.

        Notes:
            This method establishes a stable reference point used to execute all other motion commands.

            The method initializes the homing procedure but does not block until it is over. This feature is designed
            to support homing multiple motors in parallel.
        """
        # A parked motor cannot be homed until it is unparked. As a safety measure, this command does NOT automatically
        # override the parking state. Additionally, the motor is not allowed to execute a home command unless it is
        # idle.
        if self.is_parked or self.is_busy:
            return

        # Moves the motor towards the home sensor until it triggers the limit switch.
        self._padded_method_call(method=self._motor.home, wait_until_idle=False)

    def move(self, position: int) -> None:
        """Moves the motor to the requested absolute position.

        Notes:
            This method initiates the movement, but does not wait until it is completed. This behavior is designed to
            enable parallel operation of multiple motors.

        Args:
            position: The exact position, in native motor units, to move the motor to.
        """
        # If the motor is already executing a different command, it has to be stopped or allowed to finish the command
        # before executing a new command. Also, movement is only allowed if the motor is not parked and has been homed.
        if self.is_busy or not self.is_homed or self.is_parked:
            return

        # Ensures that the position to move the motor to is within the motor's software limits.
        if position < self._minimum_limit or position > self._maximum_limit:
            return

        self._padded_method_call(method=self._motor.move_absolute, position=position, wait_until_idle=False)

    def stop(self) -> None:
        """Decelerates and stops the motor.

        Notes:
            This method can be called to interrupt other currently running methods, which is primarily used in the case
            of an emergency.

            Calling this method once instructs the motor to decelerate and stop. Per the Zaber ASCII protocol
            manual, a second stop command issued while the motor is still decelerating halts it immediately. That
            is controller firmware behavior and is not implemented by this wrapper.

            This command does not block until the motor stops to allow stopping multiple motors (axes) in rapid
            succession.
        """
        # This is the only command that does not have a padding timer check. This design pattern is to allow calling
        # this method in the case of an emergency to shut down the managed motor.
        self._motor.stop(wait_until_idle=False)
        # Manually restarts the guard, since stop commands are not routed through the padding method.
        self._pacing_guard.kick()

    def park(self) -> None:
        """Parks the motor, making it unresponsive to motor commands, and stores the current absolute position of the
        motor in its non-volatile memory.
        """
        # The motor has to be idle to be parked.
        if self.is_busy:
            return

        self._padded_method_call(method=self._motor.park)

    def unpark(self) -> None:
        """Unparks a parked motor, which allows the motor to accept and execute motion commands."""
        if self.is_parked:
            self._padded_method_call(method=self._motor.unpark)

    def shutdown(self) -> None:
        """Prepares the motor for shutting down by seizing any ongoing movement and parking it to cache its current
        position to the non-volatile memory.
        """
        # If the shutdown flag indicates that the motor has already been shut, abort early. Also returns early if the
        # motor is already parked.
        if self._shutdown_flag or self.is_parked:
            self._shutdown_flag = True
            return

        # If the motor is moving, stops it.
        if self.is_busy:
            self._motor.stop(wait_until_idle=True)

        # Parks the motor and sets the shutdown flag.
        self.park()
        self._shutdown_flag = True

    def _padded_method_call[T](self, method: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Interacts with the motor hardware by executing the requested method with the appropriate time padding to
        prevent overwhelming the communication interface.

        Args:
            method: The method to call with timing guards.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The value returned by the specified method's call.
        """
        # Ensures that at least 5 milliseconds have elapsed since the previous interaction with the motor's hardware.
        # This design is chosen over delay() to allow instantaneous escapes if this method is called when the delay
        # has already expired.
        while not self._pacing_guard.expired:
            pass

        result = method(*args, **kwargs)

        self._pacing_guard.kick()
        return result


class ZaberDevice:
    """Manages a Zaber controller (device) that manages one or more motors (axes).

    Notes:
        This class represents the intermediate level of the tri-class hierarchy used to control Zaber motors during
        runtime.

        This class is explicitly designed to work with devices that manage a single axis (motor) and raises errors
        if it is initialized for a controller with more than a single axis.

    Args:
        device: The Device class instance that interfaces with the controller's hardware.

    Attributes:
        _controller: The Device class instance that interfaces with the Zaber controller's hardware.
        _axis: Stores the ZaberAxis class instance that interfaces with the motor managed by this instance.
        _shutdown_flag: Tracks whether the device has been shut down.

    Raises:
        ValueError: If the device checksum stored in the device's non-volatile memory does not match the CRC32-XFER
            checksum of the device's label. If the device is unsafe and was not properly shut down during the previous
            runtime as indicated by its non-volatile trackers, and the user declines the interactive confirmation
            prompt to proceed. If the device manages more than a single axis (motor).
    """

    def __init__(self, device: Device) -> None:
        # Extracts and records the necessary ID information about the device
        self._controller: Device = device

        # Ensures that the device is managing a single axis.
        if device.axis_count != 1:
            message = (
                f"Unexpected value encountered when checking the number of axes (motors) managed by the device "
                f"{self._controller.label}. Currently, ZaberDevice instances only work with devices (controllers) that "
                f"manage a single Axis (motor). Instead, the device has {device.axis_count} axes, which indicates that "
                f"it manages multiple motors."
            )
            console.error(message=message, error=ValueError)

        # Initializes the ZaberAxis class to interface with the motor managed by the Device.
        self._axis: ZaberAxis = ZaberAxis(motor=self._controller.get_axis(axis_number=1))

        # Uses the CRC calculator to generate the checksum for the device's label. It is expected that the
        # device_code (USER_DATA_0) non-volatile variable of the device is set to this checksum for any
        # correctly configured device.
        device_check: int = _crc_calculator.string_checksum(string=self._controller.label)
        device_code: int = int(device.settings.get(setting=_ZaberSettings.checksum))
        if device_code != device_check:
            message = (
                f"Unable to verify that the ZaberDevice instance for the {self._controller.label} "
                f"({self._controller.name}) device is configured to work with ZaberDevice instances. Based on the "
                f"device's label '{self._controller.label}', expected the validation checksum of {device_check}, but "
                f"read {device_code}. The non-volatile memory variable used to store this data is USER_DATA_0."
            )
            console.error(message=message, error=ValueError)

        # Verifies that the device has been properly shut down during the previous runtime. While this is not an issue
        # for most motors, certain motors must be positioned in a specific way to ensure they can be homed. These
        # motors use the 'unsafe_flag' non-volatile tracker to indicate that they require proper shutdown.
        shutdown_completed: bool = bool(self._controller.settings.get(setting=_ZaberSettings.shutdown_flag))
        homing_unsafe: bool = bool(self._controller.settings.get(setting=_ZaberSettings.unsafe_flag))
        if not shutdown_completed and homing_unsafe:
            message = (
                f"The {self._controller.label} ({self._controller.name}) device was not properly shutdown during the "
                f"previous runtime. Since the device is marked as 'unsafe,' it is not possible to reset the device "
                f"in the unsupervised mode. Ensure that the device is positioned correctly for the homing procedure "
                f"before proceeding. Do you want to proceed with initializing this motor?"
            )
            console.echo(message=message, level=LogLevel.WARNING)

            # Blocks until the user confirms or declines the unsupervised reset procedure. The prompt has no default,
            # so an accidental empty Enter cannot silently decide whether this motor is initialized.
            if not request_required_confirmation(message="Proceed with initializing this motor?"):
                message = (
                    f"Unsafe automatic reset procedure for the {self._controller.label} "
                    f"({self._controller.name}) device: Declined. Manually set the value of the shutdown tracker "
                    f"to 1 after ensuring the device is positioned correctly for homing. The non-volatile memory "
                    f"variable used to store this data is USER_DATA_1."
                )
                console.error(message=message, error=ValueError)

        # Sets the device's shutdown tracker to 0. This tracker is used to detect when a device is not properly shut
        # down, which may have implications for the use of the device, such as the ability to home the device.
        # During the proper shutdown procedure, the tracker is always set to 1, so setting it to 0 now allows
        # detecting cases where the shutdown is not carried out.
        self._controller.settings.set(setting=_ZaberSettings.shutdown_flag, value=0)
        self._shutdown_flag = False

    def __repr__(self) -> str:
        """Returns a string representation of the ZaberDevice instance."""
        return (
            f"ZaberDevice(name='{self._controller.name}', label={self._controller.label}, "
            f"id={self._controller.device_id})"
        )

    def shutdown(self) -> None:
        """Gracefully shuts down the motor (axis) managed by this controller.

        Notes:
            The shutdown tracker written to the device's non-volatile memory doubles as the marker that tells the next
            runtime whether the motor rests where the homing procedure expects it. It is therefore set to 1 only for a
            motor that is parked at the park position stored in the same memory, and to 0 for a motor resting anywhere
            else.
        """
        self._axis.shutdown()

        motor_parked = self._axis.is_parked
        current_position = self._axis.get_position()
        parked_at_park_position = (
            motor_parked and abs(current_position - self._axis.park_position) <= _PARK_POSITION_TOLERANCE
        )

        if not parked_at_park_position:
            message = (
                f"The {self._controller.label} ({self._controller.name}) device did not come to rest at its park "
                f"position during the shutdown sequence. The motor reports the position {current_position} and the "
                f"parked state of {motor_parked}, while the park position stored in its non-volatile memory is "
                f"{self._axis.park_position}. The device is recorded as improperly shut down, so the next runtime "
                f"asks for manual confirmation before initializing it if the device is marked as unsafe to home."
            )
            console.echo(message=message, level=LogLevel.WARNING)

        self._controller.settings.set(setting=_ZaberSettings.shutdown_flag, value=int(parked_at_park_position))
        self._shutdown_flag = True

    @property
    def axis(self) -> ZaberAxis:
        """Returns the ZaberAxis instance that allows interfacing with the motor (axis) managed by this Zaber
        controller.
        """
        return self._axis


class ZaberConnection:
    """Interfaces with a serial USB port and all Zaber devices (controllers) and axes (motors) available through that
    port.

    Notes:
        This class represents the highest level of the tri-class Zaber binding hierarchy.

        This class does not automatically initialize the connection with the port. Call the connect() method to
        establish connection before calling other class methods.

    Args:
        port: The name of the USB port to connect to.

    Attributes:
        _port: Stores the name of the serial port to connect to.
        _connection: The Connection class instance that manages the specified serial port and all Zaber devices using
            the port.
        _devices: The tuple of ZaberDevice instances used to interface with Zaber devices available through the
            connected port.
        _is_connected: Tracks whether the instance is currently connected to the managed serial port.

    Raises:
        TypeError: If the provided 'port' argument value is not a string.
    """

    def __init__(self, port: str) -> None:
        if not isinstance(port, str):
            message = (
                f"Invalid 'port' argument type encountered when initializing a ZaberConnection class instance. "
                f"Expected a {str.__name__}, but encountered {port} of type {type(port).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._port: str = port
        self._connection: Connection | None = None
        self._devices: tuple[ZaberDevice, ...] = ()
        self._is_connected: bool = False

    def __repr__(self) -> str:
        """Returns a string representation of the ZaberConnection instance."""
        return f"ZaberConnection(port='{self._port}', connected={self.is_connected})"

    def __del__(self) -> None:
        """Ensures that the instance shuts down all managed devices and disconnects from the managed port before it is
        garbage-collected.
        """
        if self._connection is not None and self.is_connected:
            self._release_runtime_assets()

    def connect(self) -> None:
        """Opens the serial port and detects and connects to any available Zaber devices (controllers).

        Raises:
            NoDeviceFoundException: If no Zaber devices are discovered using the target serial port.
        """
        # If the connection is already established, prevents re-establishing the connection.
        if self.is_connected:
            return

        # Establishes the connection.
        connection = Connection.open_serial_port(port_name=self._port, direct=False)
        self._connection = connection
        self._is_connected = True

        # Gets the list of all connected Zaber devices.
        devices: list[Device] = connection.detect_devices()

        # Packages each discovered Device into a ZaberDevice class instance and builds the internal device interface
        # tuple. The tuple is rebuilt after every successful construction, so a failure partway through the daisy chain
        # leaves the already-constructed devices reachable for the release below.
        initialized_devices: list[ZaberDevice] = []
        try:
            for device in devices:
                initialized_devices.append(ZaberDevice(device=device))
                self._devices = tuple(initialized_devices)
        except Exception, KeyboardInterrupt:
            # Releases the port without shutting the constructed devices down. The shutdown sequence parks the motor,
            # which commits its current position to non-volatile memory and blocks every motion command until it is
            # unparked, so a failed connect must not reach it: the operator may need the Zaber Launcher to move the
            # HeadBar and free a head-fixed animal. The devices keep their zeroed shutdown tracker, which honestly
            # records that this runtime aborted before it could shut them down.
            self._release_runtime_assets(shutdown_devices=False)
            raise

    def disconnect(self) -> None:
        """Shuts down all managed Zaber devices and closes the connection."""
        # Prevents the method from running if the connection is not established.
        if not self.is_connected:
            return

        self._release_runtime_assets()

    @property
    def is_connected(self) -> bool:
        """Returns True if the class has established connection with the managed serial port."""
        if self._connection is not None and self._is_connected:
            try:
                # Tries to detect available devices using the connection. If the connection is broken, this will
                # necessarily fail with an error.
                self._connection.detect_devices()
            except Exception:
                # Otherwise, the connection is broken.
                self._is_connected = False
            else:
                self._is_connected = True  # If device check succeeded the connection is active.
                return True
        return self._is_connected

    def get_device(self, index: int) -> ZaberDevice:
        """Returns the ZaberDevice instance for the requested Zaber controller (device).

        Args:
            index: The index of the controller for which to retrieve the interface. The controllers are indexed based
                on their position in the daisy-chain of Zaber devices relative to the USB port, with the device
                directly connected to the port having an index of 0.

        Returns:
            A ZaberDevice instance that interfaces with the specified controller.

        Raises:
            ConnectionError: If the instance is not connected to the managed serial port.
        """
        # Prevents retrieving the device data if the connection has not been established.
        if not self.is_connected:
            message = (
                f"Unable to retrieve the Zaber device at index {index} as the ZaberConnection instance has not "
                f"established the connection with the managed port ({self._port})."
            )
            console.error(message=message, error=ConnectionError)

        return self._devices[index]

    def _release_runtime_assets(self, *, shutdown_devices: bool = True) -> None:
        """Shuts down every managed Zaber device and closes the managed serial port.

        Notes:
            Each device shutdown is isolated and the port is released from a finally block, so the remaining devices
            and the serial port are still released when one controller stops responding.

        Args:
            shutdown_devices: Determines whether to run each managed device's shutdown sequence before releasing the
                port. An aborted connection attempt releases the port alone, because the shutdown sequence parks the
                motor and a motor parked at an arbitrary position refuses every motion command until it is unparked.
        """
        try:
            if shutdown_devices:
                for number, device in enumerate(self._devices):
                    run_shutdown_step(
                        description=f"shutting down the Zaber device at index {number}", step=device.shutdown
                    )
        finally:
            self._devices = ()
            self._is_connected = False
            if self._connection is not None:
                # An error raised while closing the port would replace the error that triggered the release, so the
                # close is isolated as well.
                run_shutdown_step(description="closing the managed serial port", step=self._connection.close)


def _attempt_connection(port: str) -> list[_ZaberDeviceData]:
    """Checks the specified USB port for Zaber devices and parses identification data for any discovered device.

    Args:
        port: The name of the USB port to scan for Zaber devices.

    Returns:
        A list of _ZaberDeviceData instances, one for each discovered device or an empty list if none are discovered.
    """
    # Uses 'with' to automatically close the connection at the end of the runtime. This statement opens the serial
    # port without testing it for Zaber devices. The detect_devices() call below raises NoDeviceFoundException when
    # the port carries no Zaber devices, which the caller catches.
    with Connection.open_serial_port(port_name=port, direct=False) as connection:
        # Parses each detected device and its axes into _ZaberDeviceData instances.
        return [
            _ZaberDeviceData(
                device_number=number + 1,
                device_id=device.device_id,
                label=device.label,
                name=device.name,
                axes=[
                    _ZaberAxisData(
                        axis_id=axis_number,
                        axis_label=device.get_axis(axis_number=axis_number).label or "Not Used",
                    )
                    for axis_number in range(1, device.axis_count + 1)
                ],
            )
            for number, device in enumerate(connection.detect_devices())
        ]


def _scan_active_ports() -> list[_ZaberPortData]:
    """Scans all available serial ports for Zaber devices and parses their identification data.

    Returns:
        A list of _ZaberPortData objects, one for each scanned port.
    """
    port_info_list = []

    # Gets the list of serial ports active for the current platform and scans each to determine if any Zaber devices
    # are connected to that port.
    for port in Tools.list_serial_ports():
        try:
            devices = _attempt_connection(port=port)
            port_info = _ZaberPortData(port_name=port, devices=devices)
        except Exception as exception:
            # Logs connection errors at debug level and creates empty _ZaberPortData instances.
            console.echo(message=f"Error connecting to port {port}: {exception}.", level=LogLevel.DEBUG)
            port_info = _ZaberPortData(port_name=port, devices=[])

        port_info_list.append(port_info)

    return port_info_list


def _format_device_info(port_info_list: list[_ZaberPortData]) -> str:
    """Formats the device and axis ID information discovered during port scanning as a table for display.

    Args:
        port_info_list: A list of _ZaberPortData instances containing device and axis information for each scanned port.

    Returns:
        A string containing the formatted device and axis ID information as a table.
    """
    table_data = []

    for port_info in port_info_list:
        if not port_info.has_devices:
            table_data.append([port_info.port_name, "No Devices", "", "", "", "", ""])
        else:
            for device in port_info.devices:
                device_row = [
                    port_info.port_name,
                    str(device.device_number),
                    str(device.device_id),
                    device.label,
                    device.name,
                ]
                for axis in device.axes:
                    axis_row = [*device_row, str(axis.axis_id), axis.axis_label]
                    table_data.append(axis_row)
                    device_row = [""] * 5
        table_data.append([""] * 7)  # Adds an empty row to separate port sections.

    return tabulate(
        tabular_data=table_data,
        headers=["Port", "Device Num", "ID", "Label", "Name", "Axis ID", "Axis Label"],
        tablefmt="grid",
        stralign="center",
    )
