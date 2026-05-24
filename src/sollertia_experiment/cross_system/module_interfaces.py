"""Provides the interfaces (ModuleInterface class implementations) for the hardware modules assembled and configured
according to the instructions from the Sollertia platform microcontrollers project:
https://github.com/Sun-Lab-NBB/sollertia-micro-controllers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import TimeUnits, PrecisionTimer, TimerPrecisions, convert_time
from scipy.optimize import curve_fit
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import SharedMemoryArray
from ataraxis_communication_interface import (
    ModuleData,
    ModuleState,
    ModuleInterface,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ZERO_UINT64: np.uint64 = np.uint64(0)
"""A pre-created uint64 zero, reused across the module to avoid unnecessary object recreation."""
_ZERO_FLOAT64: np.float64 = np.float64(0.0)
"""A pre-created float64 zero, reused across the module to avoid unnecessary object recreation."""
_ZERO_UINT32: np.uint32 = np.uint32(0)
"""A pre-created uint32 zero, reused across the module to avoid unnecessary object recreation."""
_FALSE: np.bool_ = np.bool_(0)
"""A pre-created boolean False, reused across the module to avoid unnecessary object recreation."""

_MAXIMUM_VALVE_PULSE_DURATION_MS: int = 400
"""The maximum pulse duration, in milliseconds, that water and gas valves can use before keepalive intervention."""
_MAXIMUM_VALVE_PULSE_DURATION_US: int = _MAXIMUM_VALVE_PULSE_DURATION_MS * 1000
"""The maximum valve pulse duration, in microseconds."""

_MAXIMUM_BRAKING_STRENGTH: np.uint8 = np.uint8(255)
"""The braking strength sent to the BrakeModule for pulse commands, set to the maximum value (255)."""


class EncoderInterface(ModuleInterface):
    """Interfaces with EncoderModule instances running on the Encoder MicroController.

    Notes:
        Type code 2.

        The encoder is constructed without a Unity scale and reports zero Unity motion until ``set_unity_scale`` is
        called. VR-experiment runtimes load the cm-per-Unity-unit conversion factor from the VR TaskTemplate and
        apply it via ``set_unity_scale`` at experiment-start.

    Args:
        encoder_ppr: The resolution of the module's quadrature encoder, in Pulses Per Revolution (PPR).
        wheel_diameter: The diameter of the running wheel attached to the encoder, in centimeters.
        polling_frequency: The frequency, in microseconds, at which to check the encoder's state when monitoring the
            encoder.

    Attributes:
        _ppr: The resolution of the managed quadrature encoder.
        _wheel_diameter: The diameter of the running wheel connected to the encoder.
        _cm_per_pulse: The conversion factor that translates encoder pulses into centimeters.
        _unity_unit_per_pulse: The conversion factor that translates encoder pulses into Unity units. Zero until
            ``set_unity_scale`` is called.
        _polling_frequency: The frequency, in microseconds, at which to check the encoder's state when monitoring the
            encoder.
        _distance_tracker: The SharedMemoryArray instance that transfers the distance data collected by the module from
            the communication process to other runtime processes.
        _check_state: The code for the CheckState module command.
        _reset_encoder: The code for the ResetEncoder module command.
        _monitoring: Tracks whether the instance is currently configured to monitor the managed encoder's state.
    """

    def __init__(
        self,
        encoder_ppr: int,
        wheel_diameter: float,
        polling_frequency: int,
    ) -> None:
        data_codes: set[np.uint8] = {np.uint8(51), np.uint8(52)}  # kRotatedCCW, kRotatedCW

        super().__init__(
            module_type=np.uint8(2),
            module_id=np.uint8(1),
            name="encoder",
            data_codes=data_codes,
            error_codes=None,
        )

        # Saves additional data to class attributes.
        self._ppr: int = encoder_ppr
        self._wheel_diameter: float = wheel_diameter

        # Computes the conversion factor to go from pulses to centimeters
        self._cm_per_pulse: np.float64 = np.round(
            a=np.float64((math.pi * self._wheel_diameter) / self._ppr),
            decimals=8,
        )

        # Defaults to zero. VR-experiment runtimes call set_unity_scale() with the cm-per-Unity-unit conversion
        # loaded from the VR TaskTemplate before consuming encoder data.
        self._unity_unit_per_pulse: np.float64 = np.float64(0.0)

        # Saves the encoder's polling frequency in microseconds.
        self._polling_frequency: np.uint32 = np.uint32(polling_frequency)

        # Pre-creates a shared memory array used to track and share the absolute distance, in centimeters, traveled by
        # the animal since class initialization and the current absolute position of the animal in centimeters relative
        # to the onset position.
        self._distance_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{int(self._module_type)}_{int(self._module_id)}_distance_tracker",
            prototype=np.zeros(shape=2, dtype=np.float64),
            exists_ok=True,
        )

        # Statically computes command code objects
        self._check_state: np.uint8 = np.uint8(1)
        self._reset_encoder: np.uint8 = np.uint8(2)

        # Tracks the current encoder monitoring status
        self._monitoring: bool = False

    def __del__(self) -> None:
        """Ensures the instance's shared memory buffer is properly cleaned up when the instance is garbage-collected."""
        self._distance_tracker.disconnect()
        self._distance_tracker.destroy()

    def initialize_local_assets(self) -> None:
        """Connects to the instance's shared memory buffer and enables buffer cleanup at shutdown."""
        self._distance_tracker.connect()
        self._distance_tracker.enable_buffer_destruction()

    def initialize_remote_assets(self) -> None:
        """Connects to the instance's shared memory buffer."""
        self._distance_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the instance's shared memory buffer."""
        self._distance_tracker.disconnect()

    def process_received_data(self, message: ModuleData) -> None:  # type: ignore[override]
        """Updates the distance data stored in the instance's shared memory buffer based on the messages received from
        the microcontroller.
        """
        # The rotation direction is encoded via the message event code. CW rotation (code 52) is interpreted as negative
        # and CCW (code 51) as positive.
        _ccw_rotation_code = 51
        sign = 1 if message.event == _ccw_rotation_code else -1

        # Translates the absolute motion into the CW / CCW vector and converts from raw pulse count to Unity units
        # using the precomputed conversion factor. Uses float64 and rounds to 8 decimal places for consistency and
        # precision.
        unity_motion = message.data_object * self._unity_unit_per_pulse * sign

        # Converts the motion into centimeters. Does not include the sign, as this value is used to compute the absolute
        # traveled distance regardless of the traveled direction.
        cm_motion = message.data_object * self._cm_per_pulse

        # Increments the total distance traveled by the animal.
        self._distance_tracker[0] += cm_motion

        # Updates the current absolute position of the animal in the VR environment (given relative to experiment onset
        # position 0).
        self._distance_tracker[1] += unity_motion

    def set_parameters(
        self,
        report_ccw: np.bool_,
        report_cw: np.bool_,
        delta_threshold: np.uint32,
    ) -> None:
        """Sets the module's PC-addressable runtime parameters to the input values.

        Args:
            report_ccw: Determines whether to report rotation in the counterclockwise (CCW; positive) direction.
            report_cw: Determines whether to report rotation in the clockwise (CW; negative) direction.
            delta_threshold: The minimum displacement change (delta) between any two consecutive readouts for reporting
                the rotation to the PC.
        """
        self.send_parameters(parameter_data=(report_ccw, report_cw, delta_threshold))

    def set_monitoring_state(self, *, state: bool) -> None:
        """Configures the module to start or stop continuously monitoring the managed sensor's state.

        Args:
            state: Determines whether to start or stop monitoring the managed sensor's state.
        """
        # If the current monitoring state matches the desired state, aborts the runtime early.
        if state == self._monitoring:
            return

        # Enables sensor monitoring
        if state:
            self.send_command(command=self._reset_encoder, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
            self.send_command(command=self._check_state, noblock=_FALSE, repetition_delay=self._polling_frequency)
            self._monitoring = True

        # Disables sensor monitoring
        else:
            self.reset_command_queue()
            self._monitoring = False

    @property
    def cm_per_pulse(self) -> np.float64:
        """Returns the conversion factor that translates the raw encoder pulse counts to traveled centimeters."""
        return self._cm_per_pulse

    @property
    def absolute_position(self) -> np.float64:
        """Returns the absolute position of the animal, in Unity units, relative to the runtime onset."""
        return self._distance_tracker[1]

    @property
    def traveled_distance(self) -> np.float64:
        """Returns the total distance, in centimeters, traveled by the animal since the runtime onset."""
        return self._distance_tracker[0]

    def reset_distance_tracker(self) -> None:
        """Resets the traveled distance trackers to zero."""
        self._distance_tracker[0] = _ZERO_FLOAT64
        self._distance_tracker[1] = _ZERO_FLOAT64

    def set_unity_scale(self, cm_per_unity_unit: float) -> None:
        """Updates the encoder pulse to Unity unit conversion factor used to report animal motion to Unity.

        Notes:
            VR-experiment runtimes call this method at experiment-start with the cm-per-Unity-unit conversion factor
            loaded from the VR TaskTemplate. The new value takes effect on the next encoder message processed by
            the interface.

        Args:
            cm_per_unity_unit: The length of one Virtual Reality environment distance unit (Unity unit) in
                centimeters.
        """
        self._unity_unit_per_pulse = np.round(
            a=np.float64((math.pi * self._wheel_diameter) / (self._ppr * cm_per_unity_unit)),
            decimals=8,
        )


class LickInterface(ModuleInterface):
    """Interfaces with LickModule instances running on the Sensor MicroController.

    Notes:
        Type code 4.

    Args:
        lick_threshold: The threshold voltage, in raw analog units measured by a 3.3 Volt 12-bit
            Analog-to-Digital-Converter module, for interpreting the signal received from the sensor as a lick event.
        polling_frequency: The frequency, in microseconds, at which to check the lick sensor's state when monitoring the
            sensor.

    Attributes:
        _lick_threshold: The threshold voltage for detecting lick events.
        _polling_frequency: The frequency, in microseconds, at which to check the lick sensor's state when monitoring
            the sensor.
        _lick_tracker: The SharedMemoryArray instance that transfers the lick data collected by the module from
            the communication process to other runtime processes.
        _previous_readout_zero: Tracks whether the previous voltage readout reported by the sensor was 0 (no contact).
        _check_state: The code for the CheckState module command.
        _monitoring: Tracks whether the instance is currently configured to monitor the managed lick sensor's state.
    """

    def __init__(self, lick_threshold: int, polling_frequency: int) -> None:
        data_codes: set[np.uint8] = {np.uint8(51)}  # kChanged

        # Initializes the subclassed ModuleInterface using the input instance data.
        super().__init__(
            module_type=np.uint8(4),
            module_id=np.uint8(1),
            name="lick",
            data_codes=data_codes,
            error_codes=None,
        )

        self._lick_threshold: np.uint16 = np.uint16(lick_threshold)
        self._polling_frequency: np.uint32 = np.uint32(polling_frequency)

        # Pre-creates a shared memory array used to track and share the total number of licks recorded by the sensor
        # since class initialization.
        self._lick_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{int(self._module_type)}_{int(self._module_id)}_lick_tracker",
            prototype=np.zeros(shape=1, dtype=np.uint64),
            exists_ok=True,
        )

        # Prevents excessive lick reporting by ensuring that lick counter is only incremented after the signal reaches
        # the zero value.
        self._previous_readout_zero: bool = False

        # Statically computes command code objects
        self._check_state: np.uint8 = np.uint8(1)

        # Tracks the current sensor monitoring status
        self._monitoring: bool = False

    def __del__(self) -> None:
        """Ensures the instance's shared memory buffer is properly cleaned up when the instance is garbage-collected."""
        self._lick_tracker.disconnect()
        self._lick_tracker.destroy()

    def initialize_local_assets(self) -> None:
        """Connects to the instance's shared memory buffer and enables buffer cleanup at shutdown."""
        self._lick_tracker.connect()
        self._lick_tracker.enable_buffer_destruction()

    def initialize_remote_assets(self) -> None:
        """Connects to the instance's shared memory buffer."""
        self._lick_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the instance's shared memory buffer."""
        self._lick_tracker.disconnect()

    def process_received_data(self, message: ModuleData) -> None:  # type: ignore[override]
        """Updates the lick event data stored in the instance's shared memory buffer based on the messages received from
        the microcontroller.
        """
        # Currently, only code 51 ModuleData messages are passed to this method. From each, extracts the detected
        # voltage level.
        # noinspection PyTypeChecker
        detected_voltage: np.uint16 = message.data_object  # type: ignore[assignment]

        # Since the sensor is pulled to 0 to indicate the lack of tongue contact, a zero-readout necessarily means no
        # lick. Sets the zero-tracker to 1 to indicate that a zero-state has been encountered.
        if detected_voltage == 0:
            self._previous_readout_zero = True
            return

        # If the voltage level exceeds the lick threshold and this is the first time the threshold is exceeded since
        # the last zero-value, classifies the current sensor's state as a lick event and increments the shared memory
        # counter.
        if detected_voltage >= self._lick_threshold and self._previous_readout_zero:
            # Increments the shared lick counter
            self._lick_tracker[0] += 1

            # Disables further reports until the sensor sends a zero-value again
            self._previous_readout_zero = False

    def set_parameters(
        self,
        signal_threshold: np.uint16,
        delta_threshold: np.uint16,
        average_pool_size: np.uint8,
    ) -> None:
        """Sets the module's PC-addressable runtime parameters to the input values.

        Args:
            signal_threshold: The minimum voltage level, in raw analog units of a 3.3 Volt 12-bit
                Analog-to-Digital-Converter (ADC), reported to the PC as a significant sensor interaction. Note:
                signals below the threshold are pulled to 0.
            delta_threshold: The minimum difference between two consecutive voltage level readouts for reporting the
                new signal value to the PC.
            average_pool_size: The number of analog pin readouts to average together when checking the sensor's state.
        """
        self.send_parameters(parameter_data=(signal_threshold, delta_threshold, average_pool_size))

    def set_monitoring_state(self, *, state: bool) -> None:
        """Configures the module to start or stop continuously monitoring the managed sensor's state.

        Args:
            state: Determines whether to start or stop monitoring the managed sensor's state.
        """
        # If the current monitoring state matches the desired state, aborts the runtime early.
        if state == self._monitoring:
            return

        # Enables sensor monitoring
        if state:
            self.send_command(command=self._check_state, noblock=_FALSE, repetition_delay=self._polling_frequency)
            self._monitoring = True

        # Disables sensor monitoring
        else:
            self.reset_command_queue()
            self._monitoring = False

    @property
    def lick_count(self) -> np.uint64:
        """Returns the total number of licks detected by the module since the runtime onset."""
        return self._lick_tracker[0]

    @property
    def lick_threshold(self) -> np.uint16:
        """Returns the voltage threshold, in raw ADC units of a 12-bit Analog-to-Digital voltage converter, interpreted
        as the animal licking at the sensor.
        """
        return self._lick_threshold


class TorqueInterface(ModuleInterface):
    """Interfaces with TorqueModule instances running on the Sensor MicroController.

    Notes:
        Type code 6.

    Args:
        baseline_voltage: The voltage level, in raw analog units measured by a 3.3 Volt 12-bit
            Analog-to-Digital-Converter module, that corresponds to no torque (0) readout.
        maximum_voltage: The voltage level, in raw analog units measured by a 3.3 Volt 12-bit
            Analog-to-Digital-Converter module, that corresponds to the absolute maximum torque detectable by the
            sensor.
        sensor_capacity: The maximum torque level, in grams centimeter (g cm) detectable by the sensor.
        polling_frequency: The frequency, in microseconds, at which to check the torque sensor's state when monitoring
            the sensor.

    Attributes:
        _polling_frequency: The frequency, in microseconds, at which to check the torque sensor's state when monitoring
            the sensor.
        _torque_per_adc_unit: The conversion factor that translates the raw analog units of a 3.3 Volt 12-bit ADC to
            torque in Newtons centimeter.
        _check_state: The code for the CheckState module command.
        _monitoring: Tracks whether the instance is currently configured to monitor the managed torque sensor's state.
    """

    def __init__(
        self, baseline_voltage: int, maximum_voltage: int, sensor_capacity: float, polling_frequency: int
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data.
        super().__init__(
            module_type=np.uint8(6),
            module_id=np.uint8(1),
            name="torque",
            data_codes=None,
            error_codes=None,
        )

        # Caches the polling frequency to an instance attribute
        self._polling_frequency: np.uint32 = np.uint32(polling_frequency)

        # Computes the conversion factor to translate the recorded raw analog readouts of the 3.3V 12-bit ADC to
        # torque in Newton centimeter. Rounds to 8 decimal places for consistency and to ensure
        # repeatability. Uses a hardcoded conversion factor to translate sensor capacity from g cm to N cm.
        self._torque_per_adc_unit: np.float64 = np.round(
            a=(np.float64(sensor_capacity) * np.float64(0.00981) / (maximum_voltage - baseline_voltage)),
            decimals=8,
        )

        # Statically computes command code objects
        self._check_state: np.uint8 = np.uint8(1)

        # Tracks the current sensor monitoring status
        self._monitoring: bool = False

    def initialize_remote_assets(self) -> None:
        """Not used."""
        return

    def terminate_remote_assets(self) -> None:
        """Not used."""
        return

    def process_received_data(self, _message: ModuleData | ModuleState) -> None:
        """Not used, as the module currently does not require real-time data processing."""
        return

    def set_parameters(
        self,
        report_ccw: np.bool_,
        report_cw: np.bool_,
        signal_threshold: np.uint16,
        delta_threshold: np.uint16,
        averaging_pool_size: np.uint8,
    ) -> None:
        """Sets the module's PC-addressable runtime parameters to the input values.

        Args:
            report_ccw: Determines whether the sensor should report torques in the counterclockwise (CCW; positive)
                direction.
            report_cw: Determines whether the sensor should report torque in the clockwise (CW; negative) direction.
            signal_threshold: The minimum torque level, in raw analog units of 12-bit Analog-to-Digital-Converter
                (ADC), reported to the PC as a significant torque signal. Note: signals below the threshold are
                pulled to 0.
            delta_threshold: The minimum difference between two consecutive torque level readouts for reporting the
                new signal value to the PC.
            averaging_pool_size: The number of analog pin readouts to average together when checking the sensor's state.
        """
        self.send_parameters(
            parameter_data=(
                report_ccw,
                report_cw,
                signal_threshold,
                delta_threshold,
                averaging_pool_size,
            )
        )

    def set_monitoring_state(self, *, state: bool) -> None:
        """Configures the module to start or stop continuously monitoring the managed sensor's state.

        Args:
            state: Determines whether to start or stop monitoring the managed sensor's state.
        """
        # If the current monitoring state matches the desired state, aborts the runtime early.
        if state == self._monitoring:
            return

        # Enables sensor monitoring
        if state:
            self.send_command(command=self._check_state, noblock=_FALSE, repetition_delay=self._polling_frequency)
            self._monitoring = True

        # Disables sensor monitoring
        else:
            self.reset_command_queue()
            self._monitoring = False

    @property
    def torque_per_adc_unit(self) -> np.float64:
        """Returns the conversion factor that translates the raw analog values recorded by the 3.3 Volt 12-bit ADC into
        torque in Newton centimeter.
        """
        return self._torque_per_adc_unit


class TTLInterface(ModuleInterface):
    """Interfaces with TTLModule instances running on the Sensor MicroController.

    Args:
        polling_frequency: The frequency, in microseconds, at which to check for incoming TTL signals when monitoring
            the TTL sensor.

    Attributes:
        _polling_frequency: The frequency, in microseconds, at which to check for incoming TTL signals when monitoring
            the TTL sensor.
        _pulse_tracker: The SharedMemoryArray instance that transfers the TTL pulse data collected by the module from
            the communication process to other runtime processes.
        _check_state: The code for the CheckState module command.
        _monitoring: Tracks whether the instance is currently configured to monitor the incoming TTL signals.
    """

    def __init__(self, polling_frequency: int) -> None:
        error_codes: set[np.uint8] = {np.uint8(53)}  # kInvalidPinMode
        data_codes: set[np.uint8] = {np.uint8(51)}  # kInputOn

        super().__init__(
            module_type=np.uint8(1),
            module_id=np.uint8(1),
            name="mesoscope_frame",
            data_codes=data_codes,
            error_codes=error_codes,
        )

        self._polling_frequency: np.uint32 = np.uint32(polling_frequency)

        # Pre-creates a SharedMemoryArray used to track and share the number of TTL pulses recorded by the instance
        # with other processes.
        self._pulse_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{int(self._module_type)}_{int(self._module_id)}_pulse_tracker",
            prototype=np.zeros(shape=1, dtype=np.uint64),
            exists_ok=True,
        )

        # Statically computes command code objects
        # Note, commands codes 1 through 3 are reserved for commands that output TTL signals. This module's
        # functionality is currently NOT used by the Mesoscope-VR system, so these command codes are not stored here to
        # avoid unnecessary clutter.
        self._check_state: np.uint8 = np.uint8(4)

        # Tracks the current sensor monitoring status
        self._monitoring: bool = False

    def __del__(self) -> None:
        """Ensures the instance's shared memory buffer is properly cleaned up when the instance is garbage-collected."""
        self._pulse_tracker.disconnect()
        self._pulse_tracker.destroy()

    def initialize_local_assets(self) -> None:
        """Connects to the instance's shared memory buffer and enables buffer cleanup at shutdown."""
        self._pulse_tracker.connect()
        self._pulse_tracker.enable_buffer_destruction()

    def initialize_remote_assets(self) -> None:
        """Connects to the instance's shared memory buffer."""
        self._pulse_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the instance's shared memory buffer."""
        self._pulse_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Updates the TTL pulse count stored in the instance's shared memory buffer based on the messages received from
        the microcontroller.
        """
        # Each time the module detects a HIGH TTL pulse edge, increments the pulse counter.
        _ttl_on_code = 51
        if message.event == _ttl_on_code:
            self._pulse_tracker[0] += 1

    def set_parameters(
        self,
        averaging_pool_size: np.uint8,
    ) -> None:
        """Sets the module's PC-addressable runtime parameters to the input values.

        Args:
            averaging_pool_size: The number of sensor readouts to average together when checking the incoming TTL
                signal state.
        """
        # Since the module is currently not used to deliver TTL pulses statically sets the pulse duration parameter
        # to zero.
        self.send_parameters(parameter_data=(_ZERO_UINT32, averaging_pool_size))

    def set_monitoring_state(self, *, state: bool) -> None:
        """Configures the module to start or stop continuously monitoring the managed sensor's state.

        Args:
            state: Determines whether to start or stop monitoring the managed sensor's state.
        """
        # If the current monitoring state matches the desired state, aborts the runtime early.
        if state == self._monitoring:
            return

        # Enables sensor monitoring
        if state:
            self.send_command(command=self._check_state, noblock=_FALSE, repetition_delay=self._polling_frequency)
            self._monitoring = True

        # Disables sensor monitoring
        else:
            self.reset_command_queue()
            self._monitoring = False

    @property
    def pulse_count(self) -> np.uint64:
        """Returns the number of received TTL pulses recorded by the module since runtime onset."""
        return self._pulse_tracker[0]

    def reset_pulse_count(self) -> None:
        """Resets the TTL pulse tracker to zero."""
        self._pulse_tracker[0] = _ZERO_UINT64


class BrakeInterface(ModuleInterface):
    """Interfaces with BrakeModule instances running on the Actor MicroController.

    Notes:
        Type code 3.

    Args:
        minimum_brake_strength: The torque, in gram centimeter, applied by the brake when it is fully disengaged.
        maximum_brake_strength: The torque, in gram centimeter, applied by the brake when it is maximally engaged.

    Attributes:
        _minimum_brake_strength: The minimum torque, in N cm, the brake delivers at minimum voltage.
        _maximum_brake_strength: The maximum torque, in N cm, the brake delivers at maximum voltage.
        _engage: The code for the EnableBrake module command.
        _disengage: The code for the DisableBrake module command.
        _pulse: The code for the SendPulse module command.
        _enabled: Tracks the current state of the managed brake.
        _previous_pulse_duration: Tracks the pulse duration used during the previous send_pulse() call.
    """

    def __init__(
        self,
        minimum_brake_strength: float,
        maximum_brake_strength: float,
    ) -> None:
        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            module_type=np.uint8(3),
            module_id=np.uint8(1),
            name="brake",
            data_codes=None,
            error_codes=None,
        )

        # Converts minimum and maximum brake strength into Newton centimeter. Uses the hardcoded conversion factor of
        # 0.00981 to translate g cm to N cm.
        self._minimum_brake_strength: np.float64 = np.round(
            a=minimum_brake_strength * 0.00981,
            decimals=8,
        )
        self._maximum_brake_strength: np.float64 = np.round(
            a=maximum_brake_strength * 0.00981,
            decimals=8,
        )

        # Statically computes command code objects
        self._engage: np.uint8 = np.uint8(1)
        self._disengage: np.uint8 = np.uint8(2)
        self._pulse: np.uint8 = np.uint8(4)

        # Tracks whether the managed brake is currently engaged. The brake starts in the normally engaged state, so it
        # is ON at class initialization.
        self._enabled: bool = True

        # Tracks the pulse duration used by the previous send_pulse() call.
        self._previous_pulse_duration: int = 0

    def initialize_remote_assets(self) -> None:
        """Not used."""
        return

    def terminate_remote_assets(self) -> None:
        """Not used."""
        return

    def process_received_data(self, _message: ModuleData | ModuleState) -> None:
        """Not used, as the module currently does not require any real-time data processing."""
        return

    def set_state(self, *, state: bool) -> None:
        """Sets the brake to the desired state.

        Args:
            state: The desired state of the brake. True means the brake is engaged; False means the brake is disengaged.
        """
        # If the requested state matches the current brake's state, aborts the runtime early.
        if state == self._enabled:
            return

        self.send_command(
            command=self._engage if state else self._disengage, noblock=_FALSE, repetition_delay=_ZERO_UINT32
        )
        self._enabled = state

    def send_pulse(self, duration_ms: int) -> None:
        """Briefly engages the brake at full strength for the specified duration then automatically disengages.

        Args:
            duration_ms: The duration, in milliseconds, to engage the brake.
        """
        # Only updates the module parameters if the pulse duration changed compared to the previous call. This ensures
        # parameters are only updated when necessary, reducing communication overhead.
        if duration_ms != self._previous_pulse_duration:
            self._previous_pulse_duration = duration_ms
            # The microcontroller expects both braking_strength (uint8) and pulse_duration (uint32) parameters.
            duration_us = np.uint32(duration_ms * 1000)
            self.send_parameters(parameter_data=(_MAXIMUM_BRAKING_STRENGTH, duration_us))

        self.send_command(command=self._pulse, noblock=_FALSE, repetition_delay=_ZERO_UINT32)

    @property
    def maximum_brake_strength(self) -> np.float64:
        """Returns the torque, in Newton centimeters, produced by the brake when it is maximally engaged."""
        return self._maximum_brake_strength

    @property
    def minimum_brake_strength(self) -> np.float64:
        """Returns the torque, in Newton centimeters, produced by the brake when it is fully disengaged."""
        return self._minimum_brake_strength


class ValveInterface(ModuleInterface):
    """Interfaces with ValveModule instances running on the Actor MicroController.

    Notes:
        Type code 5.

    Args:
        valve_calibration_data: Maps the valve open durations to delivered fluid volumes.

    Attributes:
        _calibration_count: The number of reward delivery cycles to use during calibration and referencing procedures.
        _scale_coefficient: The scale coefficient derived from fitting the power-law model to the valve's
            calibration data.
        _nonlinearity_exponent: The nonlinearity exponent derived from fitting the power-law model to the valve's
            calibration data.
        _valve_tracker: The SharedMemoryArray instance that transfers the reward data collected by the module from
            the communication process to other runtime processes.
        _reward: The code for the Pulse module command.
        _open: The code for the Open module command.
        _close: The code for the Close module command.
        _calibrate: The code for the Calibrate module command.
        _tone: The code for the Tone module command.
        _previous_module_state: Tracks the valve's state reported by the last received message sent from the
            microcontroller.
        _configured_valve_state: Tracks the current state of the valve (Open or Closed) set through this interface
            instance.
        _previous_volume: Tracks the volume of water the valve was instructed to dispense during the previous reward
            delivery.
        _previous_tone_duration: Tracks the tone duration used during the previous reward delivery or simulation.
        _cycle_timer: A PrecisionTimer instance that tracks how long the valve stays open during reward delivery.
    """

    def __init__(self, valve_calibration_data: tuple[tuple[int | float, int | float], ...]) -> None:
        error_codes: set[np.uint8] = {np.uint8(56)}  # kInvalidToneConfiguration
        data_codes: set[np.uint8] = {np.uint8(51), np.uint8(52), np.uint8(53)}  # kOpen, kClosed, kCalibrated

        super().__init__(
            module_type=np.uint8(5),
            module_id=np.uint8(1),
            name="valve",
            data_codes=data_codes,
            error_codes=error_codes,
        )

        # Statically sets the number of reward delivery cycles used during referencing and calibration procedures.
        self._calibration_count: np.uint16 = np.uint16(200)

        # Extracts pulse durations and fluid volumes into separate arrays.
        pulse_durations: NDArray[np.float64] = np.array(
            [calibration_pair[0] for calibration_pair in valve_calibration_data], dtype=np.float64
        )
        fluid_volumes: NDArray[np.float64] = np.array(
            [calibration_pair[1] for calibration_pair in valve_calibration_data], dtype=np.float64
        )

        # Fits the power-law model to the input calibration data and saves the fit parameters to instance attributes
        # noinspection PyTupleAssignmentBalance
        parameters, _ = curve_fit(
            f=_power_law_model,  # type: ignore[arg-type]
            xdata=pulse_durations,
            ydata=fluid_volumes,
        )
        scale_coefficient, nonlinearity_exponent = parameters
        self._scale_coefficient: np.float64 = np.round(a=np.float64(scale_coefficient), decimals=8)
        self._nonlinearity_exponent: np.float64 = np.round(a=np.float64(nonlinearity_exponent), decimals=8)

        # Pre-creates a shared memory array used to track and share valve state data. Index 0 tracks the total amount of
        # water dispensed by the valve during runtime. Index 1 tracks the current valve calibration state (0 -
        # calibrating, 1 - calibrated). Index 2 tracks the current valve open/close state (0 - closed, 1 - open).
        self._valve_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{int(self._module_type)}_{int(self._module_id)}_valve_tracker",
            prototype=np.zeros(shape=3, dtype=np.float64),
            exists_ok=True,
        )

        # Statically computes command code objects
        self._reward: np.uint8 = np.uint8(1)
        self._open: np.uint8 = np.uint8(2)
        self._close: np.uint8 = np.uint8(3)
        self._calibrate: np.uint8 = np.uint8(4)
        self._tone: np.uint8 = np.uint8(5)

        # Initializes additional trackers and runtime assets.
        # Reflects what the module is actually doing.
        self._previous_module_state: bool = False
        # Reflects what the interface set the module to do.
        self._configured_valve_state: bool = False
        self._previous_volume: float = 0.0
        self._previous_tone_duration: int = 0
        self._cycle_timer: PrecisionTimer | None = None

    def __del__(self) -> None:
        """Ensures the instance's shared memory buffer is properly cleaned up when the instance is garbage-collected."""
        self._valve_tracker.disconnect()
        self._valve_tracker.destroy()

    def initialize_local_assets(self) -> None:
        """Connects to the instance's shared memory buffer and enables buffer cleanup at shutdown."""
        self._valve_tracker.connect()
        self._valve_tracker.enable_buffer_destruction()

    def initialize_remote_assets(self) -> None:
        """Connects to the instance's shared memory buffer and initializes the cycle PrecisionTimer."""
        self._valve_tracker.connect()
        self._cycle_timer = PrecisionTimer(precision=TimerPrecisions.MICROSECOND)

    def terminate_remote_assets(self) -> None:
        """Disconnects from the instance's shared memory buffer."""
        self._valve_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Updates the reward data stored in the instance's shared memory buffer based on the messages received from
        the microcontroller.
        """
        _valve_open_code = 51
        _valve_closed_code = 52
        _valve_calibrated_code = 53

        if message.event == _valve_open_code and not self._previous_module_state:
            # Resets the cycle timer each time the valve transitions to an open state.
            self._previous_module_state = True
            self._valve_tracker[2] = 1  # Valve is now open
            self._cycle_timer.reset()  # type: ignore[union-attr]

        elif message.event == _valve_closed_code and self._previous_module_state:
            # Each time the valve transitions to a closed state, records the period of time the valve was open and uses
            # it to estimate the volume of fluid delivered through the valve. Accumulates the total volume in the
            # tracker array.
            self._previous_module_state = False
            self._valve_tracker[2] = 0  # Valve is now closed
            open_duration = self._cycle_timer.elapsed  # type: ignore[union-attr]

            # Accumulates delivered water volumes into the tracker.
            delivered_volume = self._scale_coefficient * np.power(open_duration, self._nonlinearity_exponent)
            self._valve_tracker[0] += delivered_volume

        # When the valve reports the completion of a calibration cycle, sets the appropriate element of the tracker
        # array to 1.
        elif message.event == _valve_calibrated_code:
            self._valve_tracker[1] = 1

    def set_state(self, *, state: bool) -> None:
        """Sets the managed valve to the desired state.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.
        """
        # If the valve is already in the desired state, aborts the runtime early.
        if state == self._configured_valve_state:
            return

        self.send_command(command=self._open if state else self._close, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        self._configured_valve_state = state

    def deliver_reward(self, volume: float = 5.0, tone_duration: int = 300) -> None:
        """Opens the valve for the duration of time necessary to deliver the requested volume of water.

        Args:
            volume: The volume of water to deliver, in microliters.
            tone_duration: The duration of the auditory tone, in milliseconds, to emit while delivering the water
                reward.
        """
        # This ensures that the valve settings are only updated if volume, tone_duration, or both changed compared to
        # the previous command runtime. This ensures that the valve settings are only updated when this is necessary,
        # reducing communication overhead.
        if volume != self._previous_volume or tone_duration != self._previous_tone_duration:
            # Parameters are cached here to use the tone_duration before it is converted to microseconds.
            self._previous_volume = volume
            self._previous_tone_duration = tone_duration

            tone_duration_us: np.uint32 = np.uint32(
                round(
                    convert_time(time=tone_duration, from_units=TimeUnits.MILLISECOND, to_units=TimeUnits.MICROSECOND)
                )
            )
            pulse_duration: np.uint32 = self.get_duration_from_volume(target_volume=volume)
            self.send_parameters(parameter_data=(pulse_duration, self._calibration_count, tone_duration_us))

        self.send_command(command=self._reward, noblock=_FALSE, repetition_delay=_ZERO_UINT32)

    def simulate_reward(self, tone_duration: int = 300) -> None:
        """Simulates delivering water reward by emitting an audible 'reward' tone without opening the valve.

        Args:
            tone_duration: The duration of the auditory tone, in milliseconds, to emit while simulating the water
                reward delivery.
        """
        # This ensures that the valve settings are only updated if tone_duration changed compared to the previous
        # command runtime. This ensures that the valve settings are only updated when this is necessary, reducing
        # communication overhead.
        if tone_duration != self._previous_tone_duration:
            # Parameters are cached here to use the tone_duration before it is converted to microseconds.
            self._previous_tone_duration = tone_duration

            # Maintains the same pulse duration.
            pulse_duration: np.uint32 = self.get_duration_from_volume(target_volume=self._previous_volume)
            tone_duration_us: np.uint32 = np.uint32(
                round(
                    convert_time(time=tone_duration, from_units=TimeUnits.MILLISECOND, to_units=TimeUnits.MICROSECOND)
                )
            )
            self.send_parameters(parameter_data=(pulse_duration, self._calibration_count, tone_duration_us))

        self.send_command(command=self._tone, noblock=_FALSE, repetition_delay=_ZERO_UINT32)

    def reference_valve(self) -> None:
        """Opens the valve 200 times for the duration necessary to deliver 5 microliters of water to verify the valve's
        calibration.

        Notes:
            A well-calibrated valve is expected to deliver 1.0 milliliter of water during this procedure.
        """
        # Always uses the same configuration: 5.0 uL and 200 pulses.
        self.send_parameters(
            parameter_data=(self.get_duration_from_volume(target_volume=5.0), self._calibration_count, _ZERO_UINT32)
        )
        self.send_command(command=self._calibrate, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        # Indicates that the valve has entered the referencing cycle.
        self._valve_tracker[1] = 0

    def calibrate_valve(self, pulse_duration: int) -> None:
        """Repeatedly opens the valve for the requested number of milliseconds to determine the volume of fluid
        dispensed through the valve during this period of time.

        Args:
            pulse_duration: The duration, in milliseconds, to keep the valve open at each calibration cycle.
        """
        # Guards against pulse durations that exceed the maximum allowed duration. Caps to the safe maximum and warns.
        if pulse_duration > _MAXIMUM_VALVE_PULSE_DURATION_MS:
            message = (
                f"The requested pulse duration of {pulse_duration} ms for ValveModule {int(self._module_id)} "
                f"exceeds the maximum allowed duration of {_MAXIMUM_VALVE_PULSE_DURATION_MS} ms. Capping to "
                f"{_MAXIMUM_VALVE_PULSE_DURATION_MS} ms."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            pulse_duration = _MAXIMUM_VALVE_PULSE_DURATION_MS

        # Converts the pulse duration to microseconds before updating the valve's parameters.
        self.send_parameters(parameter_data=(np.uint32(pulse_duration * 1000), self._calibration_count, _ZERO_UINT32))
        self.send_command(command=self._calibrate, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        # Indicates that the valve has entered the calibration cycle.
        self._valve_tracker[1] = 0

    def get_duration_from_volume(self, target_volume: float) -> np.uint32:
        """Converts the input volume of water, in microliters, to the required period of time, in microseconds, the
        managed valve must stay open to deliver the specified volume.

        Args:
            target_volume: The volume of water, in microliters, to deliver.

        Raises:
            ValueError: If the desired water volume is too small to be reliably dispensed by the valve, based on its
                calibration data.

        Returns:
            The duration, in microseconds, the valve needs to stay open to deliver the specified volume.
        """
        # Determines the minimum valid pulse duration. This is hardcoded at 10 ms as this is the lower calibration
        # boundary.
        min_dispensed_volume = self._scale_coefficient * np.power(10.0, self._nonlinearity_exponent)

        if target_volume < min_dispensed_volume:
            message = (
                f"The requested water volume of {target_volume} uL is too small to be reliably dispensed by the "
                f"ValveModule {int(self._module_id)}. The smallest volume the valve can reliably dispense is "
                f"{min_dispensed_volume} uL."
            )
            console.error(message=message, error=ValueError)

        # Inverts the power-law calibration to get the pulse duration.
        pulse_duration = (target_volume / self._scale_coefficient) ** (1.0 / self._nonlinearity_exponent)

        # Guards against pulse durations that exceed the maximum allowed duration. Caps to the safe maximum and warns.
        if pulse_duration > _MAXIMUM_VALVE_PULSE_DURATION_US:
            message = (
                f"The computed pulse duration of {pulse_duration / 1000:.1f} ms for ValveModule {int(self._module_id)} "
                f"exceeds the maximum allowed duration of {_MAXIMUM_VALVE_PULSE_DURATION_MS} ms. Capping to "
                f"{_MAXIMUM_VALVE_PULSE_DURATION_MS} ms."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            pulse_duration = np.float64(_MAXIMUM_VALVE_PULSE_DURATION_US)

        return np.uint32(np.round(pulse_duration))

    @property
    def scale_coefficient(self) -> np.float64:
        """Returns the scale coefficient (A) of the power-law model fitted to the valve's calibration data."""
        return self._scale_coefficient

    @property
    def nonlinearity_exponent(self) -> np.float64:
        """Returns the nonlinearity exponent (B) of the power-law model fitted to the valve's calibration data."""
        return self._nonlinearity_exponent

    @property
    def delivered_volume(self) -> np.float64:
        """Returns the total volume of water, in microliters, delivered by the valve since the runtime onset."""
        return self._valve_tracker[0]

    @property
    def calibrating(self) -> bool:
        """Returns True if the module is currently performing a valve calibration cycle and False otherwise."""
        return self._valve_tracker[1] == 0


class ScreenInterface(ModuleInterface):
    """Interfaces with ScreenModule instances running on the Actor MicroController.

    Notes:
        Type code 7.

        This interface expects that the managed screens are turned OFF when the interface is initialized.

    Attributes:
        _toggle: The code for the Toggle module command.
        _enabled: Tracks the current state of the managed screens.
    """

    def __init__(self) -> None:
        super().__init__(
            module_type=np.uint8(7),
            module_id=np.uint8(1),
            name="screen",
            data_codes=None,
            error_codes=None,
        )

        # Statically computes command code objects
        self._toggle: np.uint8 = np.uint8(1)

        # Tracks the state of the managed screens
        self._enabled: bool = False

    def initialize_remote_assets(self) -> None:
        """Not used."""
        return

    def terminate_remote_assets(self) -> None:
        """Not used."""
        return

    def process_received_data(self, _message: ModuleData | ModuleState) -> None:
        """Not used, as the module currently does not require any real-time data processing."""
        return

    def set_parameters(self, pulse_duration: np.uint32) -> None:
        """Sets the module's PC-addressable runtime parameters to the input values.

        Args:
            pulse_duration: The duration, in microseconds, of each emitted screen state toggle TTL pulse.
        """
        self.send_parameters(parameter_data=(pulse_duration,))

    def set_state(self, *, state: bool) -> None:
        """Sets the screens to the desired power state.

        Args:
            state: The desired screen power state. True means the screens are powered on; False means the screens are
                powered off.
        """
        # Ends the runtime early if the desired state matches the current screen power state.
        if state == self._enabled:
            return

        self.send_command(command=self._toggle, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        self._enabled = state

    @property
    def state(self) -> bool:
        """Returns True if the screens are currently powered on; False otherwise."""
        return self._enabled


class GasPuffValveInterface(ModuleInterface):
    """Interfaces with specialized ValveModule instances designed to operate gas valves.

    Notes:
        Type code 5.

        Unlike the water reward valve, gas valves do not require calibration as precise gas volume control is not
        critical. This interface provides direct duration-based control without volume conversion.

    Attributes:
        _pulse: The code for the Pulse module command.
        _open: The code for the Open module command.
        _close: The code for the Close module command.
        _configured_state: Tracks the current state of the valve (Open or Closed) set through this interface instance.
        _previous_module_state: Tracks the valve's state reported by the last received message from the microcontroller.
        _previous_duration: Tracks the pulse duration used during the previous deliver_puff() call.
        _puff_tracker: The SharedMemoryArray instance that transfers puff data from the communication process to
            other runtime processes.
    """

    def __init__(self) -> None:
        data_codes: set[np.uint8] = {np.uint8(51), np.uint8(52)}  # kOpen, kClosed

        super().__init__(
            module_type=np.uint8(5),
            module_id=np.uint8(2),
            name="gas_puff",
            data_codes=data_codes,
            error_codes=None,
        )

        # Statically computes command code objects
        self._pulse: np.uint8 = np.uint8(1)
        self._open: np.uint8 = np.uint8(2)
        self._close: np.uint8 = np.uint8(3)

        # Tracks the state of the managed valve
        self._configured_state: bool = False
        self._previous_module_state: bool = False

        # Tracks the pulse duration used by the previous deliver_puff() call
        self._previous_duration: int = 0

        # Creates a SharedMemoryArray used to track and share gas puff data. Index 0 tracks the total number of puffs
        # delivered. Index 1 tracks the current valve open/close state (0 - closed, 1 - open).
        self._puff_tracker: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{int(self._module_type)}_{int(self._module_id)}_puff_tracker",
            prototype=np.zeros(shape=2, dtype=np.uint32),
            exists_ok=True,
        )

    def __del__(self) -> None:
        """Ensures the instance's shared memory buffer is properly cleaned up when the instance is garbage-collected."""
        self._puff_tracker.disconnect()
        self._puff_tracker.destroy()

    def initialize_local_assets(self) -> None:
        """Connects to the instance's shared memory buffer and enables buffer cleanup at shutdown."""
        self._puff_tracker.connect()
        self._puff_tracker.enable_buffer_destruction()

    def initialize_remote_assets(self) -> None:
        """Connects to the instance's shared memory buffer."""
        self._puff_tracker.connect()

    def terminate_remote_assets(self) -> None:
        """Disconnects from the instance's shared memory buffer."""
        self._puff_tracker.disconnect()

    def process_received_data(self, message: ModuleData | ModuleState) -> None:
        """Updates the puff data stored in the instance's shared memory buffer based on the messages received from
        the microcontroller.
        """
        _valve_open_code = 51
        _valve_closed_code = 52

        if message.event == _valve_open_code and not self._previous_module_state:
            # Tracks valve open transitions.
            self._previous_module_state = True
            self._puff_tracker[1] = 1  # Valve is now open

        elif message.event == _valve_closed_code and self._previous_module_state:
            # Each time the valve transitions to a closed state, increments the puff count tracker.
            self._previous_module_state = False
            self._puff_tracker[1] = 0  # Valve is now closed
            self._puff_tracker[0] += 1  # Increment puff count

    def set_state(self, *, state: bool) -> None:
        """Sets the managed valve to the desired state.

        Args:
            state: The desired state of the valve. True means the valve is open; False means the valve is closed.
        """
        # If the valve is already in the desired state, aborts the runtime early.
        if state == self._configured_state:
            return

        self.send_command(command=self._open if state else self._close, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        self._configured_state = state

    def deliver_puff(self, duration_ms: int = 100) -> None:
        """Opens the valve for the specified duration to deliver a gas puff.

        Args:
            duration_ms: The duration, in milliseconds, to keep the valve open.
        """
        # Guards against pulse durations that exceed the maximum allowed duration. Caps to the safe maximum and warns.
        if duration_ms > _MAXIMUM_VALVE_PULSE_DURATION_MS:
            message = (
                f"The requested pulse duration of {duration_ms} ms for GasPuffValveModule {int(self._module_id)} "
                f"exceeds the maximum allowed duration of {_MAXIMUM_VALVE_PULSE_DURATION_MS} ms. Capping to "
                f"{_MAXIMUM_VALVE_PULSE_DURATION_MS} ms."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            duration_ms = _MAXIMUM_VALVE_PULSE_DURATION_MS

        # Only updates the module parameters if the pulse duration changed compared to the previous call. This ensures
        # parameters are only updated when necessary, reducing communication overhead.
        if duration_ms != self._previous_duration:
            self._previous_duration = duration_ms
            duration_us = np.uint32(duration_ms * 1000)
            # Parameters: pulse_duration, calibration_count (unused), tone_duration (unused)
            self.send_parameters(parameter_data=(duration_us, np.uint16(1), _ZERO_UINT32))

        self.send_command(command=self._pulse, noblock=_FALSE, repetition_delay=_ZERO_UINT32)
        # Note: Puff count is incremented in process_received_data when valve_closed event is received

    @property
    def puff_count(self) -> int:
        """Returns the total number of gas puffs delivered since runtime onset."""
        return int(self._puff_tracker[0])


def _power_law_model(pulse_duration: float | NDArray[np.float64], a: float, b: float, /) -> float | NDArray[np.float64]:
    """Defines the power-law model used during valve calibration.

    This model was empirically found to have the best fit for the water reward valve's performance data.
    """
    return a * np.power(pulse_duration, b)
