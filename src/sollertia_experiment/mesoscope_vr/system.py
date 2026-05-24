"""Provides the configuration, persistent state, and filesystem-layout assets that define the Mesoscope-VR data
acquisition system.
"""

from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_video_system import EncoderSpeedPresets
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import (
    SessionData,
    SessionTypes,
    AcquisitionSystems,
    get_working_directory,
)
from ataraxis_data_structures import YamlConfig

from ..vr_task import VRTaskConfiguration

_CONFIGURATION_DIR: str = "configuration"
"""Subdirectory under the working directory that stores the Mesoscope-VR system configuration YAML."""

_SYSTEM_CONFIGURATION_FILENAME: str = "mesoscope_system_configuration.yaml"
"""Canonical filename for the Mesoscope-VR system configuration YAML."""


@dataclass(slots=True)
class MesoscopeFileSystem:
    """Stores the filesystem configuration of the Mesoscope-VR data acquisition system."""

    root_directory: Path = Path()
    """The absolute path to the directory where all projects are stored on the main data acquisition system PC."""
    server_directory: Path = Path()
    """The absolute path to the local-filesystem-mounted directory where all projects are stored on the remote compute
    server."""
    nas_directory: Path = Path()
    """The absolute path to the local-filesystem-mounted directory where all projects are stored on the NAS backup
    storage volume."""
    mesoscope_directory: Path = Path()
    """The absolute path to the local-filesystem-mounted directory where all Mesoscope-acquired data is aggregated
    during acquisition by the PC that manages the Mesoscope during runtime."""


@dataclass(slots=True)
class MesoscopeGoogleSheets:
    """Stores the identifiers for the Google Sheets used by the Mesoscope-VR data acquisition system."""

    surgery_sheet_id: str = ""
    """The identifier of the Google Sheet that stores information about surgical interventions performed on the animals
    that participate in data acquisition sessions."""
    water_log_sheet_id: str = ""
    """The identifier of the Google Sheet that stores information about water restriction and handling for all
    animals that participate in data acquisition sessions."""


@dataclass(slots=True)
class MesoscopeCameras:
    """Stores the video camera configuration of the Mesoscope-VR data acquisition system."""

    face_camera_index: int = 0
    """The index of the face camera in the list of all available Harvester-managed cameras."""
    face_camera_display_frame_rate: int = 25
    """The rate, in frames per second, at which the face camera's acquired frames are displayed in the live preview
    window. This is independent of the rate at which frames are saved to disk."""
    face_camera_quantization: int = 20
    """The quantization parameter used by the face camera to encode acquired frames as video files."""
    face_camera_preset: EncoderSpeedPresets = EncoderSpeedPresets.SLOWEST
    """The encoding speed preset used by the face camera to encode acquired frames as video files."""
    body_camera_index: int = 1
    """The index of the body camera in the list of all available Harvester-managed cameras."""
    body_camera_display_frame_rate: int = 25
    """The rate, in frames per second, at which the body camera's acquired frames are displayed in the live preview
    window. This is independent of the rate at which frames are saved to disk."""
    body_camera_quantization: int = 20
    """The quantization parameter used by the body camera to encode acquired frames as video files."""
    body_camera_preset: EncoderSpeedPresets = EncoderSpeedPresets.SLOWEST
    """The encoding speed preset used by the body camera to encode acquired frames as video files."""


@dataclass(slots=True)
class MesoscopeMicroControllers:
    """Stores the microcontroller configuration of the Mesoscope-VR data acquisition system."""

    actor_port: str = "/dev/ttyACM0"
    """The USB port used by the Actor Microcontroller."""
    sensor_port: str = "/dev/ttyACM1"
    """The USB port used by the Sensor Microcontroller."""
    encoder_port: str = "/dev/ttyACM2"
    """The USB port used by the Encoder Microcontroller."""
    keepalive_interval_ms: int = 500
    """The interval, in milliseconds, at which the microcontrollers are expected to receive and send the keepalive
    messages used to ensure that all controllers function as expected during runtime."""
    minimum_brake_strength_g_cm: float = 43.2047
    """The torque applied by the running wheel brake at the minimum operational voltage, in gram centimeter."""
    maximum_brake_strength_g_cm: float = 1152.1246
    """The torque applied by the running wheel brake at the maximum operational voltage, in gram centimeter."""
    wheel_diameter_cm: float = 15.0333
    """The diameter of the running wheel, in centimeters."""
    lick_threshold_adc: int = 600
    """The threshold voltage, in raw analog units recorded by a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC),
    interpreted as the animal's tongue contacting the lick sensor."""
    lick_signal_threshold_adc: int = 300
    """The minimum voltage, in raw analog units recorded by a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC),
    reported to the PC as a non-zero value. Voltages below this level are interpreted as 'no-lick' noise and are
    pulled to 0."""
    lick_delta_threshold_adc: int = 300
    """The minimum absolute difference between two consecutive lick sensor readouts, in raw analog units recorded by
    a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC), for the change to be reported to the PC."""
    lick_averaging_pool_size: int = 2
    """The number of lick sensor readouts to average together to produce the final lick sensor readout value."""
    torque_baseline_voltage_adc: int = 2048
    """The voltage level, in raw analog units measured by a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC) after the
    AD620 amplifier, that corresponds to no torque (0) readout."""
    torque_maximum_voltage_adc: int = 3443
    """The voltage level, in raw analog units measured by a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC)
    after the AD620 amplifier, that corresponds to the absolute maximum torque detectable by the sensor."""
    torque_sensor_capacity_g_cm: float = 720.0779
    """The maximum torque detectable by the sensor, in grams centimeter (g cm)."""
    torque_report_cw: bool = True
    """Determines whether the torque sensor should report torque in the Clockwise (CW) direction."""
    torque_report_ccw: bool = True
    """Determines whether the torque sensor should report torque in the Counter-Clockwise (CCW) direction."""
    torque_signal_threshold_adc: int = 150
    """The minimum voltage, in raw analog units recorded by a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC),
    reported to the PC as a non-zero value. Voltages below this level are interpreted as noise and are pulled to 0."""
    torque_delta_threshold_adc: int = 100
    """The minimum absolute difference between two consecutive torque sensor readouts, in raw analog units recorded by
    a 3.3 Volt 12-bit Analog-to-Digital-Converter (ADC), for the change to be reported to the PC."""
    torque_averaging_pool_size: int = 4
    """The number of torque sensor readouts to average together to produce the final torque sensor readout value."""
    wheel_encoder_ppr: int = 8192
    """The resolution of the wheel's quadrature encoder, in Pulses Per Revolution (PPR)."""
    wheel_encoder_report_cw: bool = False
    """Determines whether the encoder should report rotation in the Clockwise (CW) direction."""
    wheel_encoder_report_ccw: bool = True
    """Determines whether the encoder should report rotation in the Counter-Clockwise (CCW) direction."""
    wheel_encoder_delta_threshold_pulse: int = 15
    """The minimum absolute difference between two consecutive encoder readouts, in encoder pulse counts, for the
    change to be reported to the PC."""
    wheel_encoder_polling_delay_us: int = 500
    """The delay, in microseconds, between consecutive encoder state readouts."""
    screen_trigger_pulse_duration_ms: int = 500
    """The duration, in milliseconds, of the TTL pulse used to toggle the VR screen power state."""
    sensor_polling_delay_ms: int = 1
    """The delay, in milliseconds, between any two successive readouts of any sensor other than the encoder."""
    mesoscope_frame_averaging_pool_size: int = 0
    """The number of digital pin readouts to average together when determining the current logic level of the incoming
    TTL signal sent by the mesoscope at the onset of each frame's acquisition."""
    valve_calibration_data: dict[int | float, int | float] | tuple[tuple[int | float, int | float], ...] = (
        (15000, 1.10),
        (30000, 3.0),
        (45000, 6.25),
        (60000, 10.90),
    )
    """Maps water delivery solenoid valve open times, in microseconds, to the dispensed volumes of water, in
    microliters."""


@dataclass(slots=True)
class MesoscopeExternalAssets:
    """Stores the third-party asset configuration of the Mesoscope-VR data acquisition system."""

    headbar_port: str = "/dev/ttyUSB0"
    """The USB port used by the HeadBar Zaber motor controllers."""
    lickport_port: str = "/dev/ttyUSB1"
    """The USB port used by the LickPort Zaber motor controllers."""
    wheel_port: str = "/dev/ttyUSB2"
    """The USB port used by the Wheel Zaber motor controllers."""
    vr_task: VRTaskConfiguration = field(default_factory=VRTaskConfiguration)
    """Stores the runtime configuration used to communicate with the Unity game engine that runs the Virtual Reality
    task."""


@dataclass
class MesoscopeSystemConfiguration(YamlConfig):
    """Defines the hardware and software asset configuration for the Mesoscope-VR data acquisition system."""

    name: str = "mesoscope"
    """The descriptive name of the data acquisition system."""
    filesystem: MesoscopeFileSystem = field(default_factory=MesoscopeFileSystem)
    """Stores the filesystem configuration."""
    sheets: MesoscopeGoogleSheets = field(default_factory=MesoscopeGoogleSheets)
    """Stores the identifiers and access credentials for the Google Sheets."""
    cameras: MesoscopeCameras = field(default_factory=MesoscopeCameras)
    """Stores the video cameras configuration."""
    microcontrollers: MesoscopeMicroControllers = field(default_factory=MesoscopeMicroControllers)
    """Stores the microcontrollers configuration."""
    assets: MesoscopeExternalAssets = field(default_factory=MesoscopeExternalAssets)
    """Stores the third-party hardware and firmware assets configuration."""

    def __post_init__(self) -> None:
        """Normalizes the valve calibration data to a tuple representation and validates its shape."""
        if not isinstance(self.microcontrollers.valve_calibration_data, tuple):
            self.microcontrollers.valve_calibration_data = tuple(
                (open_time, volume) for open_time, volume in self.microcontrollers.valve_calibration_data.items()
            )

        valve_calibration_data = self.microcontrollers.valve_calibration_data
        element_count = 2
        if not all(
            isinstance(item, tuple)
            and len(item) == element_count
            and isinstance(item[0], int | float)
            and isinstance(item[1], int | float)
            for item in valve_calibration_data
        ):
            message = (
                f"Unable to initialize MesoscopeSystemConfiguration. Each item under the valve_calibration_data "
                f"field of the Mesoscope-VR acquisition system configuration .yaml file must be a tuple of two "
                f"integer or float values, but got {valve_calibration_data} with at least one incompatible "
                f"element."
            )
            console.error(message=message, error=TypeError)

    def save(self, path: Path) -> None:
        """Saves the instance's data to disk as a .yaml file.

        Notes:
            Path and Enum fields are serialized automatically by YamlConfig. The valve_calibration_data
            tuple is temporarily converted to a dict for serialization so that existing .yaml files
            retain their mapping layout for the calibration table, then restored after the write.

        Args:
            path: The path to the .yaml file to save the data to.
        """
        original_value = self.microcontrollers.valve_calibration_data
        try:
            if isinstance(original_value, tuple):
                self.microcontrollers.valve_calibration_data = dict(original_value)
            self.to_yaml(file_path=path)
        finally:
            self.microcontrollers.valve_calibration_data = original_value


@dataclass
class ZaberPositions(YamlConfig):  # pragma: no cover
    """Stores Zaber motor positions reused between data acquisition sessions that use the Mesoscope-VR system."""

    headbar_z: int = 0
    """The absolute position, in native motor units, of the HeadBar z-axis motor."""
    headbar_pitch: int = 0
    """The absolute position, in native motor units, of the HeadBar pitch-axis motor."""
    headbar_roll: int = 0
    """The absolute position, in native motor units, of the HeadBar roll-axis motor."""
    lickport_z: int = 0
    """The absolute position, in native motor units, of the LickPort z-axis motor."""
    lickport_y: int = 0
    """The absolute position, in native motor units, of the LickPort y-axis motor."""
    lickport_x: int = 0
    """The absolute position, in native motor units, of the LickPort x-axis motor."""
    wheel_x: int = 0
    """The absolute position, in native motor units, of the running wheel platform x-axis motor."""


@dataclass
class MesoscopePositions(YamlConfig):  # pragma: no cover
    """Stores the positions of real and virtual Mesoscope imaging axes reused between experiment sessions that use the
    Mesoscope-VR system.
    """

    mesoscope_x: float = 0.0
    """The Mesoscope objective's X-axis position, in micrometers."""
    mesoscope_y: float = 0.0
    """The Mesoscope objective's Y-axis position, in micrometers."""
    mesoscope_roll: float = 0.0
    """The Mesoscope objective's Roll-axis position, in degrees."""
    mesoscope_z: float = 0.0
    """The Mesoscope objective's Z-axis position, in micrometers."""
    mesoscope_fast_z: float = 0.0
    """The ScanImage's FastZ (virtual Z-axis) position, in micrometers."""
    mesoscope_tip: float = 0.0
    """The ScanImage's Tip position, in degrees."""
    mesoscope_tilt: float = 0.0
    """The ScanImage's Tilt position, in degrees."""
    laser_power_mw: float = 0.0
    """The laser excitation power at the sample, in milliwatts."""
    red_dot_alignment_z: float = 0.0
    """The Mesoscope objective's Z-axis position, in micrometers, used for the red-dot alignment procedure."""


def create_system_configuration_file(system: AcquisitionSystems | str = AcquisitionSystems.MESOSCOPE_VR) -> None:
    """Creates the .YAML configuration file for the Mesoscope-VR data acquisition system and configures the local
    machine (PC) to use this file for all future acquisition-system-related calls.

    Args:
        system: The acquisition system name. Only ``AcquisitionSystems.MESOSCOPE_VR`` is supported in this package.

    Raises:
        ValueError: If the requested acquisition system is not supported by this package.
    """
    requested = AcquisitionSystems(str(system))
    if requested is not AcquisitionSystems.MESOSCOPE_VR:
        message = (
            f"Unable to generate the system configuration file for the acquisition system '{system}'. This package "
            f"only supports '{AcquisitionSystems.MESOSCOPE_VR.value}'."
        )
        console.error(message=message, error=ValueError)

    directory = get_working_directory().joinpath(_CONFIGURATION_DIR)
    ensure_directory_exists(path=directory)

    # Removes any existing system configuration files to guarantee that exactly one remains after this call.
    for existing in tuple(directory.glob("*_system_configuration.yaml")):
        console.echo(message=f"Removing the existing configuration file {existing.name}...", level=LogLevel.INFO)
        existing.unlink()

    configuration_path = directory.joinpath(_SYSTEM_CONFIGURATION_FILENAME)
    MesoscopeSystemConfiguration().save(path=configuration_path)

    message = (
        f"Mesoscope-VR data acquisition system configuration file: Saved to {configuration_path}. Edit the default "
        f"parameters inside the configuration file to finish configuring the system."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


def get_system_configuration_path() -> Path:
    """Returns the expected path to the Mesoscope-VR system configuration YAML under the working directory."""
    return get_working_directory().joinpath(_CONFIGURATION_DIR, _SYSTEM_CONFIGURATION_FILENAME)


def get_system_configuration_data() -> MesoscopeSystemConfiguration:
    """Resolves the path to the local Mesoscope-VR system configuration file and loads the configuration data.

    Returns:
        The initialized MesoscopeSystemConfiguration instance that stores the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the local Sollertia platform working directory does not contain the expected Mesoscope-VR
            system configuration file.
    """
    config_path = get_system_configuration_path()

    if not config_path.exists():
        message = (
            f"Unable to load the Mesoscope-VR data acquisition system configuration. Expected the configuration file "
            f"at {config_path}, but it does not exist. Call the 'sle configure system' CLI command to generate a "
            f"default configuration file."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeSystemConfiguration.from_yaml(file_path=config_path)


def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Verifies that the host-machine belongs to the Mesoscope-VR data acquisition system and loads the
    system configuration data as a MesoscopeSystemConfiguration instance.

    Returns:
        The data acquisition system configuration data as a MesoscopeSystemConfiguration instance.

    Raises:
        TypeError: If the host-machine does not belong to the Mesoscope-VR data acquisition system.
    """
    system_configuration = get_system_configuration_data()
    if not isinstance(system_configuration, MesoscopeSystemConfiguration):
        message = (
            f"Unable to resolve the configuration for the Mesoscope-VR data acquisition system, as the host-machine "
            f"belongs to the {system_configuration.name} data acquisition system. Use the 'sle configure system' CLI "
            f"command to reconfigure the host-machine to belong the Mesoscope-VR data acquisition system."
        )
        console.error(message, error=TypeError)
    return system_configuration


mesoscope_vr_sessions: tuple[str, str, str, str] = (
    SessionTypes.LICK_TRAINING,
    SessionTypes.RUN_TRAINING,
    SessionTypes.MESOSCOPE_EXPERIMENT,
    SessionTypes.WINDOW_CHECKING,
)
"""Defines the data acquisition session types supported by the Mesoscope-VR data acquisition system."""


class MesoscopeData:
    """Defines the Mesoscope-VR data acquisition system's filesystem layout used to acquire and preprocess the target
    session's data.

    Args:
        session_data: The SessionData instance that defines the processed data acquisition session.

    Attributes:
        vrpc_data: Defines the layout of the session-specific VRPC's persistent data directory.
        scanimagepc_data: Defines the layout of the ScanImagePC's mesoscope data directory.
        destinations: Defines the layout of the long-term data storage infrastructure mounted to the VRPC's filesystem.
    """

    def __init__(self, system_configuration: MesoscopeSystemConfiguration, session_data: SessionData) -> None:
        # Unpacks session path nodes from the SessionData instance
        project = session_data.project_name
        animal = session_data.animal_id
        session = session_data.session_name

        # VRPC persistent data
        self.vrpc_data = _VRPCPersistentData(
            session_type=session_data.session_type,
            persistent_data_path=system_configuration.filesystem.root_directory.joinpath(
                project, animal, "persistent_data"
            ),
        )

        # ScanImagePC mesoscope data
        self.scanimagepc_data = _ScanImagePCData(
            session=session,
            meso_data_path=system_configuration.filesystem.mesoscope_directory,
            persistent_data_path=system_configuration.filesystem.mesoscope_directory.joinpath(
                project, animal, "persistent_data"
            ),
        )

        # Server and NAS (data storage)
        self.destinations = _VRPCDestinations(
            nas_data_path=system_configuration.filesystem.nas_directory.joinpath(project, animal, session),
            server_data_path=system_configuration.filesystem.server_directory.joinpath(project, animal, session),
        )


@dataclass()
class _VRPCPersistentData:
    """Defines the layout of the VRPC's 'persistent_data' directory, used to cache animal-specific runtime parameters
    between data acquisition sessions.
    """

    session_type: str
    """The type of the data acquisition session for which this instance was initialized."""
    persistent_data_path: Path
    """The path to the project- and animal-specific directory that stores the VRPC runtime parameters and data cached
    between data acquisition runtimes."""
    zaber_positions_path: Path = field(default_factory=Path, init=False)
    """The path to the .YAML file that stores Zaber motor positions used during the previous session's runtime."""
    mesoscope_positions_path: Path = field(default_factory=Path, init=False)
    """The path to the .YAML file that stores the Mesoscope's imaging axis coordinates used during the previous
    session's runtime."""
    session_descriptor_path: Path = field(default_factory=Path, init=False)
    """The path to the .YAML file that stores the data acquisition session's task parameters used during the previous
    session's runtime."""
    window_screenshot_path: Path = field(default_factory=Path, init=False)
    """The path to the .PNG file that stores the screenshot of the imaging window, the red-dot alignment state, and the
    Mesoscope's data-acquisition configuration used during the previous session's runtime."""

    def __post_init__(self) -> None:
        """Resolves the managed directory layout, creating any missing directory components."""
        # Resolves paths that can be derived from the root path.
        self.zaber_positions_path = self.persistent_data_path.joinpath("zaber_positions.yaml")
        self.mesoscope_positions_path = self.persistent_data_path.joinpath("mesoscope_positions.yaml")
        self.window_screenshot_path = self.persistent_data_path.joinpath("window_screenshot.png")

        # Resolves the session descriptor path based on the session type.
        if self.session_type == SessionTypes.LICK_TRAINING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("lick_training_descriptor.yaml")
        elif self.session_type == SessionTypes.RUN_TRAINING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("run_training_descriptor.yaml")
        elif self.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            self.session_descriptor_path = self.persistent_data_path.joinpath("mesoscope_experiment_descriptor.yaml")
        elif self.session_type == SessionTypes.WINDOW_CHECKING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("window_checking_descriptor.yaml")

        else:  # Raises an error for unsupported session types
            message = (
                f"Unsupported session type '{self.session_type}' encountered when resolving the filesystem layout for "
                f"the Mesoscope-VR data acquisition system. Currently, only the following data acquisition session "
                f"types are supported: {','.join(mesoscope_vr_sessions)}."
            )
            console.error(message, error=ValueError)

        # Ensures that the target persistent_data directory exists
        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class _ScanImagePCData:
    """Defines the layout of the ScanImagePC's 'meso_data' directory used to aggregate all Mesoscope-acquired data
    during a data acquisition session's runtime.
    """

    session: str
    """The unique identifier of the session for which this instance was initialized."""
    meso_data_path: Path
    """The path to the root ScanImagePC data-output directory."""
    persistent_data_path: Path
    """The path to the project- and animal-specific directory that stores the ScanImagePC (Mesoscope) runtime parameters
    and data cached between data acquisition runtimes."""
    mesoscope_data_path: Path = field(default_factory=Path, init=False)
    """The path to the directory used by the Mesoscope to save all acquired data during the acquisition session's
    runtime, which is shared by all data acquisition sessions."""
    session_specific_path: Path = field(default_factory=Path, init=False)
    """The path to the session-specific directory where all Mesoscope-acquired data is moved at the end of each data
    acquisition session's runtime."""
    motion_estimator_path: Path = field(default_factory=Path, init=False)
    """The path to the animal-specific reference .ME (motion estimator) file, used to align the Mesoscope's imaging
    field to the same view across all data acquisition sessions."""
    roi_path: Path = field(default_factory=Path, init=False)
    """The path to the animal-specific reference .ROI (Region-of-Interest) file, used to restore the same imaging
    field across all data acquisition sessions."""
    kinase_path: Path = field(default_factory=Path, init=False)
    """The path to the 'kinase.bin' file used to lock the MATLAB's runtime function (setupAcquisition.m) into the data
    acquisition mode until the kinase marker is removed by the VRPC."""
    phosphatase_path: Path = field(default_factory=Path, init=False)
    """The path to the 'phosphatase.bin' file used to gracefully terminate the MATLAB's runtimes locked into the data
    acquisition mode by the presence of the 'kinase.bin' file."""

    def __post_init__(
        self,
    ) -> None:
        """Resolves the managed directory layout, creating any missing directory components."""
        # Resolves additional paths using the input root paths
        self.motion_estimator_path = self.persistent_data_path.joinpath("MotionEstimator.me")
        self.roi_path = self.persistent_data_path.joinpath("fov.roi")
        self.session_specific_path = self.meso_data_path.joinpath(self.session)
        self.mesoscope_data_path = self.meso_data_path.joinpath("mesoscope_data")
        self.kinase_path = self.mesoscope_data_path.joinpath("kinase.bin")
        self.phosphatase_path = self.mesoscope_data_path.joinpath("phosphatase.bin")

        # Ensures that the shared data directory and the persistent data directory exist.
        ensure_directory_exists(self.mesoscope_data_path)
        ensure_directory_exists(self.persistent_data_path)


@dataclass()
class _VRPCDestinations:
    """Defines the layout of the long-term data storage infrastructure mounted to the VRPC's filesystem via the SMB
    protocol used to store the session's data after acquisition.
    """

    nas_data_path: Path
    """The path to the session's data directory on the Synology NAS."""
    server_data_path: Path
    """The path to the session's data directory on the BioHPC server."""
