"""Exposes the high-level bindings for all Mesoscope-VR system components (cameras, microcontrollers, Zaber motors)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import TimeUnits, convert_time
from ataraxis_video_system import VideoSystem, VideoEncoders, CameraInterfaces, OutputPixelFormats
from ataraxis_base_utilities import LogLevel, console
from ataraxis_communication_interface import MicroControllerInterface

from .system import ZaberPositions
from ..cross_system import (
    ZaberAxis,
    LickInterface,
    BrakeInterface,
    ScreenInterface,
    TorqueInterface,
    ZaberConnection,
    EncoderInterface,
    WaterValveInterface,
    GasPuffValveInterface,
    MesoscopeFrameTTLInterface,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ataraxis_data_structures import DataLogger

    from .system import MesoscopeCameras, MesoscopeVRAssets, MesoscopeMicroControllers


class ZaberMotors:
    """Interfaces with Zaber controllers and motors used in the Mesoscope-VR data acquisition system.

    Notes:
        This class interfaces with the three Zaber motors groups used in the system: HeadBar, Wheel, and LickPort.

        All communication with the managed Zaber devices is performed using asynchronous threads running in the main
        runtime process.

        The class transitions the motors between a set of predefined states and should not be used directly by the user.
        Improperly using this class can damage the Mesoscope-VR hardware or harm the animals participating in the data
        acquisition sessions.

        To fine-tune the position of any Zaber motors in real time, use the main Zaber Launcher interface
        (https://software.zaber.com/zaber-launcher/download) installed on the VRPC.

    Args:
        zaber_positions: The ZaberPositions instance that stores the positions of Zaber motors used during a
            previous runtime or None if there is no previous position data to use.
        zaber_configuration: The MesoscopeVRAssets instance that stores the configuration parameters for the
            managed Zaber devices.

    Attributes:
        _headbar: The ZaberConnection instance that manages the headbar holder motor group.
        _headbar_z: The ZaberAxis instance that interfaces with the headbar's Z-axis motor.
        _headbar_pitch: The ZaberAxis instance that interfaces with the headbar's pitch-axis motor.
        _headbar_roll: The ZaberAxis instance that interfaces with the headbar's roll-axis motor.
        _wheel: The ZaberConnection instance that manages the wheel motor group.
        _wheel_x: The ZaberAxis instance that interfaces with the wheel's X-axis motor.
        _lickport: The ZaberConnection instance that manages the lickport motor group.
        _lickport_z: The ZaberAxis instance that interfaces with the lickport's Z-axis motor.
        _lickport_x: The ZaberAxis instance that interfaces with the lickport's X-axis motor.
        _lickport_y: The ZaberAxis instance that interfaces with the lickport's Y-axis motor.
        _previous_positions: A ZaberPositions instance that stores the positions of Zaber motors used during a
            previous runtime or None if there is no previous position data to use.
    """

    def __init__(self, zaber_positions: ZaberPositions | None, zaber_configuration: MesoscopeVRAssets) -> None:
        # Initializes the ZaberConnection instances for all Zaber controller groups.
        self._headbar: ZaberConnection = ZaberConnection(port=zaber_configuration.headbar_port)
        self._wheel: ZaberConnection = ZaberConnection(port=zaber_configuration.wheel_port)
        self._lickport: ZaberConnection = ZaberConnection(port=zaber_configuration.lickport_port)

        # HeadBar motor group. Assumes the following order of daisy-chaining the motors: Z-Pitch-Roll.
        self._headbar.connect()
        self._headbar_z: ZaberAxis = self._headbar.get_device(index=0).axis
        self._headbar_pitch: ZaberAxis = self._headbar.get_device(index=1).axis
        self._headbar_roll: ZaberAxis = self._headbar.get_device(index=2).axis

        # LickPort motor group. Assumes the following order of daisy-chaining the motors: Z-Y-X.
        self._lickport.connect()
        self._lickport_z: ZaberAxis = self._lickport.get_device(index=0).axis
        self._lickport_y: ZaberAxis = self._lickport.get_device(index=1).axis
        self._lickport_x: ZaberAxis = self._lickport.get_device(index=2).axis

        # Wheel motor group. Currently, this motor only uses the X-axis motor.
        self._wheel.connect()
        self._wheel_x: ZaberAxis = self._wheel.get_device(index=0).axis

        # If there is no previous zaber position data to use, displays a warning message to the user.
        self._previous_positions: ZaberPositions | None = zaber_positions
        if self._previous_positions is None:
            message = (
                "No previous runtime position data provided when initializing the ZaberMotors instance for the current "
                "runtime. Configuring all Zaber motors to use the default positions cached in the non-volatile memory "
                "of each motor controller. Proceed with caution."
            )
            console.echo(message=message, level=LogLevel.WARNING)

    def __repr__(self) -> str:
        """Returns a string representation of the ZaberMotors instance."""
        return f"ZaberMotors(is_connected={self.is_connected})"

    def restore_position(self) -> None:
        """Restores the managed Zaber motors to the positions used during the previous runtime in parallel.

        Notes:
            If previous positions are not available, the method falls back to moving the HeadBar and Wheel motors to
            their 'mounting' positions and the LickPort motors to their 'parking' positions, both saved in the
            non-volatile memory of each motor controller. These positions are designed to work for most animals and
            provide an initial position for the animal to be mounted into the Mesoscope-VR enclosure.
        """
        self.unpark_motors()

        # If previous position data is available, restores all motors to the positions used during previous sessions.
        # Otherwise, sets HeadBar and Wheel to the mounting position and the LickPort to the parking position. Note: the
        # LickPort's parking position is closer to the animal than the mounting position, but still too far to be usable
        # during runtime, requiring manual fine-tuning.
        self._headbar_z.move(
            position=self._headbar_z.mount_position
            if self._previous_positions is None
            else self._previous_positions.headbar_z,
        )
        self._headbar_pitch.move(
            position=self._headbar_pitch.mount_position
            if self._previous_positions is None
            else self._previous_positions.headbar_pitch,
        )
        self._headbar_roll.move(
            position=self._headbar_roll.mount_position
            if self._previous_positions is None
            else self._previous_positions.headbar_roll,
        )
        self._wheel_x.move(
            position=self._wheel_x.mount_position
            if self._previous_positions is None
            else self._previous_positions.wheel_x,
        )
        self._lickport_z.move(
            position=self._lickport_z.park_position
            if self._previous_positions is None
            else self._previous_positions.lickport_z,
        )
        self._lickport_x.move(
            position=self._lickport_x.park_position
            if self._previous_positions is None
            else self._previous_positions.lickport_x,
        )
        self._lickport_y.move(
            position=self._lickport_y.park_position
            if self._previous_positions is None
            else self._previous_positions.lickport_y,
        )

        self.wait_until_idle()

        self.park_motors()

    def prepare_motors(self) -> None:
        """Homes the managed Zaber motors in parallel.

        Notes:
            This method ensures that all motors have a stable reference point for executing all other methods exposed
            by this instance and must be called before any other method in most use contexts.
        """
        self.unpark_motors()

        # Homes all motors in parallel.
        self._headbar_z.home()
        self._headbar_pitch.home()
        self._headbar_roll.home()
        self._wheel_x.home()
        self._lickport_z.home()
        self._lickport_x.home()
        self._lickport_y.home()

        self.wait_until_idle()

        self.park_motors()

    def park_position(self) -> None:
        """Moves the managed Zaber motors to their parking positions in parallel.

        Notes:
            This method should be called as part of the runtime's shutdown sequence to optimally position the motors to
            support homing during the next runtime.
        """
        self.unpark_motors()

        self._headbar_z.move(position=self._headbar_z.park_position)
        self._headbar_pitch.move(position=self._headbar_pitch.park_position)
        self._headbar_roll.move(position=self._headbar_roll.park_position)
        self._wheel_x.move(position=self._wheel_x.park_position)
        self._lickport_z.move(position=self._lickport_z.park_position)
        self._lickport_x.move(position=self._lickport_x.park_position)
        self._lickport_y.move(position=self._lickport_y.park_position)

        self.wait_until_idle()

        self.park_motors()

    def maintenance_position(self) -> None:
        """Moves the managed Zaber motors to the Mesoscope-VR system maintenance position in parallel."""
        self.unpark_motors()

        self._headbar_z.move(position=self._headbar_z.maintenance_position)
        self._headbar_pitch.move(position=self._headbar_pitch.maintenance_position)
        self._headbar_roll.move(position=self._headbar_roll.maintenance_position)
        self._wheel_x.move(position=self._wheel_x.maintenance_position)
        self._lickport_z.move(position=self._lickport_z.maintenance_position)
        self._lickport_x.move(position=self._lickport_x.maintenance_position)
        self._lickport_y.move(position=self._lickport_y.maintenance_position)

        self.wait_until_idle()

        self.park_motors()

    def mount_position(self) -> None:
        """Moves the managed Zaber motors to the animal mounting position in parallel.

        This motor positioning facilitates mounting the animal into the Mesoscope-VR system enclosure.

        Notes:
            Only the LickPort motors always move to their mounting positions. When previous-runtime positions are
            available, the HeadBar and Wheel motors are instead restored to those positions, as mounting is primarily
            facilitated by moving the LickPort away from the animal.
        """
        self.unpark_motors()

        self._lickport_z.move(position=self._lickport_z.mount_position)
        self._lickport_x.move(position=self._lickport_x.mount_position)
        self._lickport_y.move(position=self._lickport_y.mount_position)

        # If previous positions are not available, moves the rest of the motors to the default mounting positions.
        if self._previous_positions is None:
            self._headbar_z.move(position=self._headbar_z.mount_position)
            self._headbar_pitch.move(position=self._headbar_pitch.mount_position)
            self._headbar_roll.move(position=self._headbar_roll.mount_position)
            self._wheel_x.move(position=self._wheel_x.mount_position)

        # If previous positions are available, restores other motors to the position used during the previous runtime.
        # This relies on the idea that mounting is primarily facilitated by moving the lickport away, while all other
        # motors can be set to the optimal runtime parameters for the animal being mounted.
        else:
            self._headbar_z.move(position=self._previous_positions.headbar_z)
            self._headbar_pitch.move(position=self._previous_positions.headbar_pitch)
            self._headbar_roll.move(position=self._previous_positions.headbar_roll)
            self._wheel_x.move(position=self._previous_positions.wheel_x)

        self.wait_until_idle()

        self.park_motors()

    def unmount_position(self) -> None:
        """Retracts the LickPort group motors back to the mount position, while maintaining the current position for all
        other managed Zaber motors.

        This motor positioning facilitates removing the animal from the Mesoscope-VR system enclosure.
        """
        self.unpark_motors()

        # Moves the LickPort back to the mount position, while keeping all other motors in their current positions.
        self._lickport_y.move(position=self._lickport_y.mount_position)
        self._lickport_z.move(position=self._lickport_z.mount_position)
        self._lickport_x.move(position=self._lickport_x.mount_position)

        self.wait_until_idle()

        self.park_motors()

    def generate_position_snapshot(self) -> ZaberPositions:
        """Queries and returns the current positions of all managed Zaber motors."""
        position_snapshot = ZaberPositions(
            headbar_z=int(self._headbar_z.get_position()),
            headbar_pitch=int(self._headbar_pitch.get_position()),
            headbar_roll=int(self._headbar_roll.get_position()),
            wheel_x=int(self._wheel_x.get_position()),
            lickport_z=int(self._lickport_z.get_position()),
            lickport_x=int(self._lickport_x.get_position()),
            lickport_y=int(self._lickport_y.get_position()),
        )
        self._previous_positions = position_snapshot
        return position_snapshot

    def unpark_motors(self) -> None:
        """Unparks all managed motor groups, allowing them to be moved via this library or the Zaber GUI."""
        self._headbar_pitch.unpark()
        self._headbar_roll.unpark()
        self._headbar_z.unpark()
        self._wheel_x.unpark()
        self._lickport_x.unpark()
        self._lickport_y.unpark()
        self._lickport_z.unpark()

    def wait_until_idle(self) -> None:
        """Blocks in-place while at least one motor in the managed motor groups is moving."""
        # Waits for the motors to finish moving. Note, motor state polling includes the built-in delay mechanism to
        # prevent overwhelming the communication interface.
        while (
            self._headbar_z.is_busy
            or self._headbar_pitch.is_busy
            or self._headbar_roll.is_busy
            or self._wheel_x.is_busy
            or self._lickport_z.is_busy
            or self._lickport_x.is_busy
            or self._lickport_y.is_busy
        ):
            pass

    def park_motors(self) -> None:
        """Parks all managed Zaber motors, preventing them from being moved via this library or Zaber GUI until
        they are unparked.
        """
        self._headbar_pitch.park()
        self._headbar_roll.park()
        self._headbar_z.park()
        self._wheel_x.park()
        self._lickport_x.park()
        self._lickport_y.park()
        self._lickport_z.park()

    def disconnect(self) -> None:
        """Shuts down all managed motors and disconnects from the motor groups."""
        self._headbar.disconnect()
        self._wheel.disconnect()
        self._lickport.disconnect()

    @property
    def is_connected(self) -> bool:
        """Returns True if all managed motor connections are active and False if at least one connection is inactive."""
        connections = [
            self._headbar.is_connected,
            self._lickport.is_connected,
            self._wheel.is_connected,
        ]
        return all(connections)


class MicroControllerInterfaces:
    """Interfaces with the Ataraxis Micro Controller (AMC) devices used in the Mesoscope-VR data acquisition system.

    Notes:
        This class interfaces with the three AMC controllers used in the system: Actor, Sensor, and Encoder.

        Calling the class initializer does not start the microcontroller communication processes.
        Use the start() method before calling other instance methods.

        The instance reserves 3 CPU cores for running the microcontroller communication processes.

    Args:
        data_logger: The DataLogger instance to use for logging the data generated by the managed microcontrollers
            during runtime.
        microcontroller_configuration: The MesoscopeMicroControllers instance that stores the configuration
            parameters for the managed microcontrollers.

    Attributes:
        _started: Tracks whether the microcontroller communication processes are currently running.
        _configuration: Stores the managed microcontrollers' configuration parameters.
        brake: The interface that controls the electromagnetic particle brake attached to the running wheel.
        valve: The interface that controls the solenoid water valve.
        gas_puff_valve: The interface that controls the gas puff valve.
        screens: The interface that controls the power state of the Virtual Reality display screens.
        _actor: The main interface for the 'Actor' Ataraxis Micro Controller (AMC) device.
        mesoscope_frame: The interface that monitors frame acquisition timestamp signals sent by the mesoscope.
        lick: The interface that monitors the animal's interactions with the lick sensor.
        torque: The interface that monitors the torque applied by the animal to the running wheel when the brakes are
            on.
        _sensor: The main interface for the 'Sensor' Ataraxis Micro Controller (AMC) device.
        wheel_encoder: The interface that monitors the distance traveled by the animal on the running wheel.
        _encoder: The main interface for the 'Encoder' Ataraxis Micro Controller (AMC) device.
    """

    def __init__(self, data_logger: DataLogger, microcontroller_configuration: MesoscopeMicroControllers) -> None:
        # Tracks whether the communication processes have been started.
        self._started: bool = False

        # Caches the microcontroller configuration parameters to the instance attribute.
        self._configuration: MesoscopeMicroControllers = microcontroller_configuration

        # Converts the sensor polling frequency from milliseconds to microseconds. This value is used below to
        # initialize most sensor interfaces.
        _sensor_polling_delay: int = round(
            convert_time(
                time=self._configuration.sensor_polling_delay_ms,
                from_units=TimeUnits.MILLISECOND,
                to_units=TimeUnits.MICROSECOND,
            ),
        )

        # ACTOR. Actor AMC controls the hardware that needs to be triggered by PC at irregular intervals. Most of such
        # hardware is designed to produce some form of an output: deliver water reward, engage wheel brake, etc.

        # Module interfaces:
        self.brake: BrakeInterface = BrakeInterface(
            minimum_brake_strength=self._configuration.minimum_brake_strength_g_cm,
            maximum_brake_strength=self._configuration.maximum_brake_strength_g_cm,
        )
        self.valve: WaterValveInterface = WaterValveInterface(
            valve_calibration_data=self._configuration.valve_calibration_data,  # type: ignore[arg-type]
        )
        self.gas_puff_valve: GasPuffValveInterface = GasPuffValveInterface()
        self.screens: ScreenInterface = ScreenInterface()

        # Main interface:
        self._actor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(101),
            buffer_size=8192,
            port=self._configuration.actor_port,
            name="actor",
            data_logger=data_logger,
            module_interfaces=(self.brake, self.valve, self.gas_puff_valve, self.screens),
            keepalive_interval=self._configuration.keepalive_interval_ms,
        )

        # SENSOR. Sensor AMC controls the hardware that collects data at regular intervals. This includes lick sensors,
        # torque sensors, and input TTL recorders. Critically, all managed hardware does not rely on hardware interrupt
        # logic to maintain the necessary precision.

        # Module interfaces:
        self.mesoscope_frame: MesoscopeFrameTTLInterface = MesoscopeFrameTTLInterface(
            polling_frequency=_sensor_polling_delay,
        )
        self.lick: LickInterface = LickInterface(
            lick_threshold=self._configuration.lick_threshold_adc,
            polling_frequency=_sensor_polling_delay,
        )
        self.torque: TorqueInterface = TorqueInterface(
            baseline_voltage=self._configuration.torque_baseline_voltage_adc,
            maximum_voltage=self._configuration.torque_maximum_voltage_adc,
            sensor_capacity=self._configuration.torque_sensor_capacity_g_cm,
            polling_frequency=_sensor_polling_delay,
        )

        # Main interface:
        self._sensor: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(152),
            buffer_size=8192,
            port=self._configuration.sensor_port,
            name="sensor",
            data_logger=data_logger,
            module_interfaces=(self.mesoscope_frame, self.lick, self.torque),
            keepalive_interval=self._configuration.keepalive_interval_ms,
        )

        # ENCODER. Encoder AMC is specifically designed to interface with a quadrature encoder connected to the running
        # wheel. The encoder uses hardware interrupt logic to maintain high precision and is isolated to a separate
        # microcontroller to ensure the highest possible throughput and sensor resolution.

        # Module interfaces:
        self.wheel_encoder: EncoderInterface = EncoderInterface(
            encoder_ppr=self._configuration.wheel_encoder_ppr,
            wheel_diameter=self._configuration.wheel_diameter_cm,
            polling_frequency=microcontroller_configuration.wheel_encoder_polling_delay_us,
        )

        # Main interface:
        self._encoder: MicroControllerInterface = MicroControllerInterface(
            controller_id=np.uint8(203),
            buffer_size=8192,
            port=self._configuration.encoder_port,
            name="encoder",
            data_logger=data_logger,
            module_interfaces=(self.wheel_encoder,),
            keepalive_interval=self._configuration.keepalive_interval_ms,
        )

    def __del__(self) -> None:
        """Ensures that all communication processes are terminated when the instance is garbage-collected."""
        self.stop()

    def __repr__(self) -> str:
        """Returns a string representation of the MicroControllerInterfaces instance."""
        return f"MicroControllerInterfaces(started={self._started})"

    def start(self) -> None:
        """Starts the communication processes for all managed microcontrollers and configures all interfaced hardware
        modules to use the runtime parameters loaded from the acquisition system's configuration file.
        """
        # Prevents executing this method if the microcontrollers are already running.
        if self._started:
            return

        message = "Initializing Ataraxis Micro Controller (AMC) Interfaces..."
        console.echo(message=message, level=LogLevel.INFO)

        self._actor.start()
        self._sensor.start()
        self._encoder.start()

        self.wheel_encoder.initialize_local_assets()
        self.valve.initialize_local_assets()
        self.gas_puff_valve.initialize_local_assets()
        self.mesoscope_frame.initialize_local_assets()
        self.lick.initialize_local_assets()

        # Wheel Encoder
        self.wheel_encoder.set_parameters(
            report_cw=np.bool_(self._configuration.wheel_encoder_report_cw),
            report_ccw=np.bool_(self._configuration.wheel_encoder_report_ccw),
            delta_threshold=np.uint32(self._configuration.wheel_encoder_delta_threshold_pulse),
        )

        # Screen Interface
        screen_pulse_duration: np.float64 = convert_time(  # type: ignore[assignment]
            time=self._configuration.screen_trigger_pulse_duration_ms,
            from_units=TimeUnits.MILLISECOND,
            to_units=TimeUnits.MICROSECOND,
        )
        self.screens.set_parameters(pulse_duration=np.uint32(round(screen_pulse_duration)))

        # Lick Sensor
        self.lick.set_parameters(
            signal_threshold=np.uint16(self._configuration.lick_signal_threshold_adc),
            delta_threshold=np.uint16(self._configuration.lick_delta_threshold_adc),
            average_pool_size=np.uint8(self._configuration.lick_averaging_pool_size),
        )

        # Torque Sensor
        self.torque.set_parameters(
            report_ccw=np.bool_(self._configuration.torque_report_ccw),
            report_cw=np.bool_(self._configuration.torque_report_cw),
            signal_threshold=np.uint16(self._configuration.torque_signal_threshold_adc),
            delta_threshold=np.uint16(self._configuration.torque_delta_threshold_adc),
            averaging_pool_size=np.uint8(self._configuration.torque_averaging_pool_size),
        )

        # Mesoscope Frame TTL Recorder
        self.mesoscope_frame.set_parameters(
            averaging_pool_size=np.uint8(self._configuration.mesoscope_frame_averaging_pool_size),
        )

        self._started = True

        message = "Ataraxis Micro Controller (AMC) Interfaces: Initialized."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops all microcontroller communication processes and releases all reserved resources."""
        # Prevents stopping already-stopped microcontroller communication processes.
        if not self._started:
            return

        message = "Terminating Ataraxis Micro Controller (AMC) Interfaces..."
        console.echo(message=message, level=LogLevel.INFO)

        self._started = False

        # Stops all microcontroller interfaces. This also shuts down and resets all managed hardware modules.
        self._actor.stop()
        self._sensor.stop()
        self._encoder.stop()

        message = "Ataraxis Micro Controller (AMC) Interfaces: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)


class VideoSystems:
    """Interfaces with the Ataraxis Video System (AVS) devices used in the Mesoscope-VR data acquisition system.

    Notes:
        This class interfaces with the two high-grade scientific (GeniCam) cameras used in the system: Face and Body
        cameras.

        Calling the class initializer does not start the camera producer and consumer processes. Call other instance
        methods to enable acquiring (grabbing) and saving (caching) the desired camera's frames.

        Once camera frame acquisition is enabled, the only way to disable frame acquisition is to call the main stop()
        method that stops all active cameras. Similarly, once frame saving is started, there is no way to disable it
        without stopping the entire instance.

        The instance reserves 4 CPU cores for running the camera consumer and producer processes (a producer and a
        consumer for each of the two cameras).

    Args:
        data_logger: The DataLogger instance to use for logging the data generated by the managed cameras during
            runtime.
        camera_configuration: The MesoscopeCameras instance that stores the configuration parameters for the managed
            cameras.
        output_directory: The path to the directory where to output the generated .MP4 video files.

    Attributes:
        _face_camera_started: Tracks whether the face camera frame acquisition is running.
        _body_camera_started: Tracks whether the body camera frame acquisition is running.
        _face_camera: The interface that captures and saves the frames acquired by the camera aimed at the animal's face
            and eye.
        _body_camera: The interface that captures and saves the frames acquired by the camera aimed at the animal's
            body.
    """

    def __init__(
        self,
        data_logger: DataLogger,
        camera_configuration: MesoscopeCameras,
        output_directory: Path,
    ) -> None:
        # Tracks whether the producer and consumer processes are initialized for the managed cameras.
        self._face_camera_started: bool = False
        self._body_camera_started: bool = False

        # Initializes the face camera system.
        self._face_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(51),
            data_logger=data_logger,
            name="face_camera",
            output_directory=output_directory,
            camera_index=camera_configuration.face_camera_index,
            camera_interface=CameraInterfaces.HARVESTERS,
            display_frame_rate=camera_configuration.face_camera_display_frame_rate,
            video_encoder=VideoEncoders.H265,
            gpu=0,
            encoder_speed_preset=camera_configuration.face_camera_preset,
            # Mesoscope-VR cameras are monochrome.
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=camera_configuration.face_camera_quantization,
        )

        # Initializes the body camera system.
        self._body_camera: VideoSystem = VideoSystem(
            system_id=np.uint8(62),
            data_logger=data_logger,
            name="body_camera",
            output_directory=output_directory,
            camera_index=camera_configuration.body_camera_index,
            camera_interface=CameraInterfaces.HARVESTERS,
            display_frame_rate=camera_configuration.body_camera_display_frame_rate,
            video_encoder=VideoEncoders.H265,
            gpu=0,
            encoder_speed_preset=camera_configuration.body_camera_preset,
            # Mesoscope-VR cameras are monochrome.
            output_pixel_format=OutputPixelFormats.YUV420,
            quantization_parameter=camera_configuration.body_camera_quantization,
        )

    def __del__(self) -> None:
        """Ensures that all consumer and producer processes are terminated when the instance is garbage-collected."""
        self.stop()

    def __repr__(self) -> str:
        """Returns a string representation of the VideoSystems instance."""
        return (
            f"VideoSystems(face_camera_started={self._face_camera_started}, "
            f"body_camera_started={self._body_camera_started})"
        )

    def start_face_camera(self) -> None:
        """Starts acquiring frames from the face camera.

        Notes:
            Does not start saving the acquired frames to disk. Call the save_face_camera_frames() method to start
            saving the acquired frames to disk.
        """
        # Prevents executing this method if the face camera frame acquisition is already running.
        if self._face_camera_started:
            return

        message = "Initializing face camera frame acquisition..."
        console.echo(message=message, level=LogLevel.INFO)

        self._face_camera.start()
        self._face_camera_started = True

        message = "Face camera frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def start_body_camera(self) -> None:
        """Starts acquiring frames from the body camera.

        Notes:
            Does not start saving the acquired frames to disk. Call the save_body_camera_frames() method to start
            saving the acquired frames to disk.
        """
        # Prevents executing this method if the body camera frame acquisition is already running.
        if self._body_camera_started:
            return

        message = "Initializing body camera frame acquisition..."
        console.echo(message=message, level=LogLevel.INFO)

        self._body_camera.start()
        self._body_camera_started = True

        message = "Body camera frame acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def save_face_camera_frames(self) -> None:
        """Starts saving the frames acquired by the face camera to disk as an .MP4 video file."""
        self._face_camera.start_frame_saving()
        message = "Face camera frame saving: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def save_body_camera_frames(self) -> None:
        """Starts saving the frames acquired by the body camera to disk as an .MP4 video file."""
        self._body_camera.start_frame_saving()
        message = "Body camera frame saving: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops acquiring and saving frames for all managed cameras."""
        # Prevents executing this method if no cameras are running.
        if not self._face_camera_started and not self._body_camera_started:
            return

        message = "Stopping camera frame acquisition and saving..."
        console.echo(message=message, level=LogLevel.INFO)

        if self._face_camera_started:
            self._face_camera.stop_frame_saving()
        if self._body_camera_started:
            self._body_camera.stop_frame_saving()

        message = "Camera frame saving: Stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        self._face_camera.stop()
        self._body_camera.stop()

        self._face_camera_started = False
        self._body_camera_started = False

        message = "Camera frame acquisition: Stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)
