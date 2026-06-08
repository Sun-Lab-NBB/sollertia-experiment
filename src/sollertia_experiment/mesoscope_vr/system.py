"""Provides the configuration, persistent state, and filesystem-layout assets that define the Mesoscope-VR data
acquisition system.
"""

from enum import IntEnum, StrEnum
from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_video_system import EncoderSpeedPresets
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from sollertia_shared_assets import (
    CONFIGURATION_DIRECTORY,
    AnimalData,
    SessionData,
    SessionTypes,
    TaskTemplate,
    AcquisitionSystems,
    get_data_root,
    get_task_templates_directory,
    create_experiment_configuration,
    populate_default_experiment_states,
)
from ataraxis_data_structures import YamlConfig

from ..vr_task import VRTaskConfiguration
from ..cross_system import (
    StorageDestination,
    StorageDestinations,
    SystemConfiguration,
    get_system_configuration_data,
    register_system_configuration,
    create_system_configuration_file as _create_system_configuration_file,
)

MESOSCOPE_VR_SESSIONS: tuple[str, str, str, str] = (
    SessionTypes.LICK_TRAINING,
    SessionTypes.RUN_TRAINING,
    SessionTypes.MESOSCOPE_EXPERIMENT,
    SessionTypes.WINDOW_CHECKING,
)
"""Defines the data acquisition session types supported by the Mesoscope-VR data acquisition system."""


class MesoscopeStorageDestination(StrEnum):
    """Defines the canonical long-term storage destinations anticipated by the Mesoscope-VR data acquisition system.

    These members seed the default storage_directories configuration and define the order in which destinations are
    preferred when a single source of truth is required, such as during animal migration. The configuration is not
    restricted to these members: a system can configure any number of destinations under arbitrary names.
    """

    NAS = "NAS"
    """The Network-Attached-Storage backup volume. Preferred when pulling data, due to a typically faster transfer
    speed."""
    SERVER = "Server"
    """The remote compute server used as the primary long-term storage and analysis destination."""


class MesoscopeAcquisitionOrder(StrEnum):
    """Defines the supported plane-acquisition orders for the Mesoscope reference and high-definition z-stacks."""

    INTERLEAVED = "interleaved"
    """Iterates over the target planes once per acquired volume, acquiring one frame at each plane before repeating
    (Z1, Z2, Z1, Z2)."""
    SMOOTH = "smooth"
    """Acquires all averaged frames at one target plane before advancing to the next plane (Z1, Z1, Z2, Z2)."""


@dataclass(frozen=True, slots=True)
class RunTrainingThresholdLimits:
    """Defines the absolute bounds applied to the run training running speed and epoch duration thresholds.

    The run training logic clamps the effective speed and duration thresholds to these bounds, and the runtime control
    GUI uses them to constrain the user-facing target threshold spin boxes. Sharing a single definition keeps the
    acquisition runtime and the control GUI in agreement on the achievable threshold range.
    """

    minimum_speed_cm_s: float = 0.1
    """The lower bound, in centimeters per second, of the running speed threshold."""
    maximum_speed_cm_s: float = 5.0
    """The upper bound, in centimeters per second, of the running speed threshold."""
    minimum_duration_s: float = 0.05
    """The lower bound, in seconds, of the running epoch duration threshold."""
    maximum_duration_s: float = 5.0
    """The upper bound, in seconds, of the running epoch duration threshold."""


RUN_TRAINING_THRESHOLD_LIMITS: RunTrainingThresholdLimits = RunTrainingThresholdLimits()
"""The active run training speed and duration threshold limits shared by the acquisition runtime and the control GUI."""


@dataclass(slots=True)
class MesoscopeFileSystem:
    """Stores the filesystem configuration of the Mesoscope-VR data acquisition system.

    Notes:
        The local data root (the directory under which all projects are stored on this machine) is no longer part
        of this configuration. It is owned by the Sollertia platform as the shared data root; resolve it with
        ``get_data_root()`` and set it with the ``slsa configure data-root`` command.
    """

    mesoscope_directory: Path = Path()
    """The absolute path to the local-filesystem-mounted directory where all Mesoscope-acquired data is aggregated
    during acquisition by the PC that manages the Mesoscope during runtime."""
    storage_directories: dict[str, Path] = field(
        default_factory=lambda: {
            MesoscopeStorageDestination.NAS.value: Path(),
            MesoscopeStorageDestination.SERVER.value: Path(),
        }
    )
    """Maps the name of each long-term storage destination to the absolute path of the local-filesystem-mounted
    directory where all projects are stored on that destination. Destinations whose path is left unset (an empty path)
    are treated as not configured and are skipped during data transfer and removal. The mapping order defines the
    preference order used when the data must be pulled back to the acquisition system for any reason."""


@dataclass(slots=True)
class MesoscopeGoogleSheets:
    """Stores the identifiers for the Google Sheets used by the Mesoscope-VR data acquisition system.

    Notes:
        Both sheet identifiers are optional. A sheet whose identifier is left unset (an empty string) is treated as not
        configured, and the data exchange that depends on it is skipped with a warning. If neither
        identifier is set, the system disables all Google Sheets integration and preprocesses sessions without
        snapshotting surgery records or updating the water restriction log.
    """

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
    face_camera_configuration_path: Path = Path()
    """The absolute path to the GenICam configuration .yaml file that records the expected node configuration of the
    face camera. An empty path means no stored configuration is associated with this camera. When set, it is used to
    verify, dump, or restore the camera's GenICam configuration."""
    body_camera_index: int = 1
    """The index of the body camera in the list of all available Harvester-managed cameras."""
    body_camera_display_frame_rate: int = 25
    """The rate, in frames per second, at which the body camera's acquired frames are displayed in the live preview
    window. This is independent of the rate at which frames are saved to disk."""
    body_camera_quantization: int = 20
    """The quantization parameter used by the body camera to encode acquired frames as video files."""
    body_camera_preset: EncoderSpeedPresets = EncoderSpeedPresets.SLOWEST
    """The encoding speed preset used by the body camera to encode acquired frames as video files."""
    body_camera_configuration_path: Path = Path()
    """The absolute path to the GenICam configuration .yaml file that records the expected node configuration of the
    body camera. An empty path means no stored configuration is associated with this camera. When set, it is used to
    verify, dump, or restore the camera's GenICam configuration."""


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
class MesoscopeAcquisition:
    """Stores the online motion-estimation and z-stack acquisition configuration of the Mesoscope-VR system.

    These parameters configure the reference motion estimator and the high-definition reference z-stack that the
    ScanImagePC generates at the start of each runtime. The VRPC delivers them to the runAcquisition MATLAB function
    over MQTT with each command that consumes them, so this configuration is the single source of truth for the
    Mesoscope acquisition geometry.
    """

    z_step_um: int = 20
    """The spacing, in micrometers, between consecutive target imaging planes in the acquired z-stack."""
    z_range_um: tuple[int, int] = (1050, 1050)
    """The [minimum, maximum] z-plane range to image, in micrometers. Equal boundaries image a single plane at that
    depth; distinct boundaries image the inclusive slice between them."""
    z_exclusion_um: tuple[int, int] = (0, 0)
    """The [minimum, maximum] boundaries, in micrometers, of the non-imaged exclusion zone used for two-plane imaging.
    Equal boundaries disable two-plane imaging. When the boundaries differ, they must fall within z_range_um."""
    acquisition_order: MesoscopeAcquisitionOrder = MesoscopeAcquisitionOrder.INTERLEAVED
    """The order in which the target planes are acquired when building the reference and high-definition z-stacks."""
    registration_channel: int = 1
    """The acquisition channel used for online motion registration and the high-definition reference z-stack."""
    field_curvature_correction: bool = False
    """Determines whether ScanImage field curvature correction is enabled during acquisition. The appropriate setting
    depends on the specific microscope."""
    frames_per_reference_plane: int = 20
    """The number of frames acquired and averaged at each reference plane. Larger values improve motion
    characterization at the cost of longer processing and higher acquisition-machine load."""
    zstack_scale_factor: float = 2.0
    """The factor by which the X and Y resolution of each ROI is scaled when acquiring the high-definition reference
    z-stack. The scaling preserves the original ROI aspect ratios."""

    def __post_init__(self) -> None:
        """Validates that the acquisition parameters are positive and the z-range and exclusion-zone boundaries are
        correctly ordered and bounded.
        """
        # The positivity guards mirror the validation the runAcquisition MATLAB arguments block enforced before these
        # parameters moved into this configuration.
        positive_parameters: tuple[tuple[str, float], ...] = (
            ("z_step_um", self.z_step_um),
            ("registration_channel", self.registration_channel),
            ("frames_per_reference_plane", self.frames_per_reference_plane),
            ("zstack_scale_factor", self.zstack_scale_factor),
        )
        for name, value in positive_parameters:
            if value <= 0:
                message = (
                    f"Unable to initialize MesoscopeAcquisition. The {name} field must be a positive value, but got "
                    f"{value}."
                )
                console.error(message=message, error=ValueError)

        if self.z_range_um[0] <= 0 or self.z_range_um[0] > self.z_range_um[1]:
            message = (
                f"Unable to initialize MesoscopeAcquisition. The z_range_um boundaries must be positive and ordered as "
                f"(minimum, maximum), but got {self.z_range_um}."
            )
            console.error(message=message, error=ValueError)

        # Equal exclusion boundaries disable two-plane imaging, so only a configured (unequal) zone is range-checked.
        minimum, maximum = self.z_exclusion_um
        if minimum > maximum:
            message = (
                f"Unable to initialize MesoscopeAcquisition. The z_exclusion_um boundaries must be ordered as "
                f"(minimum, maximum), but got {self.z_exclusion_um}."
            )
            console.error(message=message, error=ValueError)

        if minimum != maximum and not (self.z_range_um[0] <= minimum and maximum <= self.z_range_um[1]):
            message = (
                f"Unable to initialize MesoscopeAcquisition. The configured z_exclusion_um zone {self.z_exclusion_um} "
                f"must fall within the z_range_um boundaries {self.z_range_um}."
            )
            console.error(message=message, error=ValueError)


@dataclass(slots=True)
class MesoscopeVRAssets:
    """Stores the Virtual Reality task asset configuration of the Mesoscope-VR data acquisition system.

    These assets consist of the Zaber motor controllers that position the head-fixed animal and the associated hardware
    to optimize Virtual Reality task performance. They also include the runtime configuration used to communicate with
    the Unity game engine that runs the task.
    """

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
class MesoscopeSystemConfiguration(SystemConfiguration):
    """Defines the hardware and software asset configuration for the Mesoscope-VR data acquisition system."""

    name: str = "mesoscope"
    """The descriptive name of the data acquisition system."""
    filesystem: MesoscopeFileSystem = field(default_factory=MesoscopeFileSystem)
    """Stores the filesystem configuration."""
    sheets: MesoscopeGoogleSheets = field(default_factory=MesoscopeGoogleSheets)
    """Stores the identifiers for the Google Sheets."""
    cameras: MesoscopeCameras = field(default_factory=MesoscopeCameras)
    """Stores the video cameras configuration."""
    microcontrollers: MesoscopeMicroControllers = field(default_factory=MesoscopeMicroControllers)
    """Stores the microcontrollers configuration."""
    acquisition: MesoscopeAcquisition = field(default_factory=MesoscopeAcquisition)
    """Stores the Mesoscope motion-estimation and z-stack acquisition configuration."""
    assets: MesoscopeVRAssets = field(default_factory=MesoscopeVRAssets)
    """Stores the Virtual Reality task asset configuration."""

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


class MesoscopeVRStates(IntEnum):
    """Defines the set of codes used by the Mesoscope-VR data acquisition system to communicate its runtime state."""

    IDLE = 0
    """The system is currently not conducting a data acquisition session."""
    REST = 1
    """The system is conducting the 'rest' period of an experiment session."""
    RUN = 2
    """The system is conducting the 'run' period of an experiment session."""
    LICK_TRAINING = 3
    """The system is conducting the lick training session."""
    RUN_TRAINING = 4
    """The system is conducting the run training session."""

    @classmethod
    def to_dict(cls) -> dict[str, int]:
        """Converts the enumeration members to a mapping of each lowercased member name, with underscores replaced by
        spaces, to its value.
        """
        return {member.name.lower().replace("_", " "): member.value for member in cls}


@dataclass(slots=True)
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
        self.zaber_positions_path = self.persistent_data_path.joinpath("zaber_positions.yaml")
        self.mesoscope_positions_path = self.persistent_data_path.joinpath("mesoscope_positions.yaml")
        self.window_screenshot_path = self.persistent_data_path.joinpath("window_screenshot.png")

        if self.session_type == SessionTypes.LICK_TRAINING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("lick_training_descriptor.yaml")
        elif self.session_type == SessionTypes.RUN_TRAINING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("run_training_descriptor.yaml")
        elif self.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            self.session_descriptor_path = self.persistent_data_path.joinpath("mesoscope_experiment_descriptor.yaml")
        elif self.session_type == SessionTypes.WINDOW_CHECKING:
            self.session_descriptor_path = self.persistent_data_path.joinpath("window_checking_descriptor.yaml")

        else:
            message = (
                f"Unable to resolve the filesystem layout for the Mesoscope-VR data acquisition system. The session "
                f"type must be one of the supported types ({','.join(MESOSCOPE_VR_SESSIONS)}), but got "
                f"'{self.session_type}'."
            )
            console.error(message=message, error=ValueError)

        ensure_directory_exists(path=self.persistent_data_path)


@dataclass(slots=True)
class _ScanImagePCData:
    """Defines the layout of the ScanImagePC's Mesoscope data root directory (the configured mesoscope_directory) used
    to aggregate all Mesoscope-acquired data during a data acquisition session's runtime.
    """

    session: str
    """The unique identifier of the session for which this instance was initialized."""
    mesoscope_root_path: Path
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

    def __post_init__(
        self,
    ) -> None:
        """Resolves the managed directory layout, creating any missing directory components."""
        self.motion_estimator_path = self.persistent_data_path.joinpath("MotionEstimator.me")
        self.roi_path = self.persistent_data_path.joinpath("fov.roi")
        self.session_specific_path = self.mesoscope_root_path.joinpath(self.session)
        self.mesoscope_data_path = self.mesoscope_root_path.joinpath("mesoscope_data")

        ensure_directory_exists(path=self.mesoscope_data_path)
        ensure_directory_exists(path=self.persistent_data_path)


# Registers the Mesoscope-VR system configuration with the shared cross-system registry so the shared create / resolve
# / load helpers can operate on it. The shared helpers own the file lifecycle; this package only adds the registration
# and the typed get_system_configuration() wrapper below. The shared get_system_configuration_path is re-exported as-is.
register_system_configuration(system=AcquisitionSystems.MESOSCOPE_VR, configuration_class=MesoscopeSystemConfiguration)


def create_system_configuration_file() -> None:
    """Creates the .yaml configuration file for the Mesoscope-VR data acquisition system and configures the local
    machine (PC) to use this file for all future acquisition-system-related calls.

    This package only supports the Mesoscope-VR acquisition system, so this thin wrapper always creates the
    Mesoscope-VR configuration by delegating to the shared cross-system create_system_configuration_file, which owns
    the file-creation logic.
    """
    _create_system_configuration_file(system=AcquisitionSystems.MESOSCOPE_VR)


def create_experiment_configuration_file(
    project: str,
    experiment: str,
    template: str,
    state_count: int,
    reward_size: float,
    reward_tone_duration: int,
    puff_duration: int,
    occupancy_duration: int,
) -> None:
    """Creates a Mesoscope-VR experiment configuration file from a task template under the configured data root.

    Resolves the target project directory under the local data root, instantiates the named task template from the
    configured task templates directory, builds a Mesoscope-VR experiment configuration with the provided trial
    defaults, populates the requested number of default-valued runtime states, and writes the result to the project's
    configuration directory.

    Args:
        project: The name of the project under which to create the experiment configuration file.
        experiment: The name of the experiment, used as the stem of the created configuration file.
        template: The name of the task template to instantiate, given as the template filename without the .yaml
            extension.
        state_count: The number of default-valued runtime states to generate.
        reward_size: The default water reward volume, in microliters, for lick-type trials.
        reward_tone_duration: The default reward tone duration, in milliseconds, for lick-type trials.
        puff_duration: The default gas puff duration, in milliseconds, for occupancy-type trials.
        occupancy_duration: The default occupancy threshold duration, in milliseconds, for occupancy-type trials.

    Raises:
        ValueError: If the target project does not exist under the data root.
        FileNotFoundError: If the named task template does not exist in the configured task templates directory.
    """
    root_directory = get_data_root()
    project_path = root_directory.joinpath(project)
    if not project_path.exists():
        message = (
            f"Unable to generate the {experiment} experiment's configuration file: the project '{project}' does not "
            f"exist under the data root {root_directory}. Use 'slsa configure project' first."
        )
        console.error(message=message, error=ValueError)

    file_path = project_path.joinpath(CONFIGURATION_DIRECTORY, f"{experiment}.yaml")

    templates_directory = get_task_templates_directory()
    template_path = templates_directory.joinpath(f"{template}.yaml")
    if not template_path.exists():
        available_templates = sorted([template_file.stem for template_file in templates_directory.glob("*.yaml")])
        message = (
            f"Unable to generate the '{experiment}' experiment configuration. The template '{template}' was "
            f"not found in {templates_directory}. Available templates: "
            f"{', '.join(available_templates) if available_templates else 'none'}."
        )
        console.error(message=message, error=FileNotFoundError)

    task_template = TaskTemplate.from_yaml(file_path=template_path)

    experiment_configuration = create_experiment_configuration(
        template=task_template,
        system=AcquisitionSystems.MESOSCOPE_VR,
        unity_scene_name=template,
        default_reward_size_ul=reward_size,
        default_reward_tone_duration_ms=reward_tone_duration,
        default_puff_duration_ms=puff_duration,
        default_occupancy_duration_ms=occupancy_duration,
    )

    populate_default_experiment_states(
        experiment_configuration=experiment_configuration,
        state_count=state_count,
    )

    experiment_configuration.to_yaml(file_path=file_path)
    console.echo(
        message=f"{experiment} experiment's configuration file: created from template '{template}'.",
        level=LogLevel.SUCCESS,
    )


def get_system_configuration() -> MesoscopeSystemConfiguration:
    """Loads the local system configuration file and verifies that the host-machine belongs to the Mesoscope-VR data
    acquisition system.

    Returns:
        The initialized MesoscopeSystemConfiguration instance that stores the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the local working directory does not contain exactly one system configuration file.
        TypeError: If the host-machine does not belong to the Mesoscope-VR data acquisition system.
    """
    system_configuration = get_system_configuration_data()
    if not isinstance(system_configuration, MesoscopeSystemConfiguration):
        belongs_to = getattr(system_configuration, "name", "an unknown")
        message = (
            f"Unable to resolve the configuration for the Mesoscope-VR data acquisition system, as the host-machine "
            f"belongs to the {belongs_to} data acquisition system. Use the 'sle mesoscope configure system' CLI "
            f"command to reconfigure the host-machine to belong to the Mesoscope-VR data acquisition system."
        )
        console.error(message=message, error=TypeError)
        # console.error() raises but is not typed NoReturn, so mypy needs an explicit raise to narrow the return type.
        # noinspection PyUnreachableCode
        raise TypeError(message)  # pragma: no cover
    return system_configuration


class MesoscopeData:
    """Defines the Mesoscope-VR data acquisition system's filesystem layout used to acquire and preprocess the target
    session's data.

    Args:
        system_configuration: The MesoscopeSystemConfiguration instance whose filesystem settings define the resolved
            directory layout.
        session_data: The SessionData instance that defines the processed data acquisition session.

    Attributes:
        vrpc_data: Defines the layout of the session-specific VRPC's persistent data directory.
        scanimagepc_data: Defines the layout of the ScanImagePC's mesoscope data directory.
        destinations: Defines the configured long-term data storage destinations mounted to the VRPC's filesystem. Only
            destinations whose storage root is configured in the system configuration are included.
        unconfigured_destinations: Stores the names of the long-term storage destinations whose storage root is not
            configured for the host-machine, used to warn about skipped data backups during preprocessing.
    """

    def __init__(self, system_configuration: MesoscopeSystemConfiguration, session_data: SessionData) -> None:
        session = session_data.session_name

        # Anchors the animal on the platform data root, then rebinds it onto the Mesoscope acquisition mount and each
        # long-term storage destination to resolve the per-root copies of the same logical session.
        local_animal = AnimalData(
            root=get_data_root(), project_name=session_data.project_name, animal_id=session_data.animal_id
        )
        mesoscope_animal = local_animal.for_root(root=system_configuration.filesystem.mesoscope_directory)

        self.vrpc_data: _VRPCPersistentData = _VRPCPersistentData(
            session_type=session_data.session_type,
            persistent_data_path=local_animal.persistent_data_path,
        )

        self.scanimagepc_data: _ScanImagePCData = _ScanImagePCData(
            session=session,
            mesoscope_root_path=system_configuration.filesystem.mesoscope_directory,
            persistent_data_path=mesoscope_animal.persistent_data_path,
        )

        # Long-term storage destinations. Resolves a StorageDestination only for each storage root that is configured
        # (set to a non-default path) in the system configuration. This supports host machines that lack some or all
        # long-term storage destinations. Destinations whose root is left unset are recorded under
        # unconfigured_destinations so the preprocessing pipeline can warn about the skipped backups. The configuration
        # order is preserved, defining the preference order used when a single destination is required.
        resolved_destinations: list[StorageDestination] = []
        unconfigured_destinations: list[str] = []
        for destination_name, destination_root in system_configuration.filesystem.storage_directories.items():
            if destination_root == Path():
                unconfigured_destinations.append(destination_name)
                continue
            resolved_destinations.append(
                StorageDestination(
                    name=destination_name,
                    session_path=local_animal.for_root(root=destination_root).session_path(session_name=session),
                )
            )
        self.destinations: StorageDestinations = StorageDestinations(destinations=tuple(resolved_destinations))
        self.unconfigured_destinations: tuple[str, ...] = tuple(unconfigured_destinations)

    def __repr__(self) -> str:
        """Returns a string representation of the MesoscopeData instance."""
        return (
            f"MesoscopeData(destinations={self.destinations}, "
            f"unconfigured_destinations={self.unconfigured_destinations})"
        )
