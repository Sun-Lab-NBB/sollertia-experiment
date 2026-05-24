"""Provides utility assets shared by other modules of the mesoscope_vr package."""

from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from sollertia_shared_assets import SessionData, SessionTypes

from .configuration import MesoscopeSystemConfiguration, get_system_configuration_data


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
