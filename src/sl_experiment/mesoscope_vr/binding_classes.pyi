from pathlib import Path

from _typeshed import Incomplete
from .positions import ZaberPositions
from .configuration import (
    MesoscopeCameras as MesoscopeCameras,
    MesoscopeExternalAssets as MesoscopeExternalAssets,
    MesoscopeMicroControllers as MesoscopeMicroControllers,
)
from ataraxis_video_system import VideoSystem
from ataraxis_data_structures import DataLogger as DataLogger
from ataraxis_communication_interface import MicroControllerInterface

from .zaber_bindings import (
    ZaberAxis as ZaberAxis,
    ZaberConnection as ZaberConnection,
)
from ..shared_components import (
    TTLInterface as TTLInterface,
    LickInterface as LickInterface,
    BrakeInterface as BrakeInterface,
    ValveInterface as ValveInterface,
    ScreenInterface as ScreenInterface,
    TorqueInterface as TorqueInterface,
    EncoderInterface as EncoderInterface,
    GasPuffValveInterface as GasPuffValveInterface,
)

class ZaberMotors:
    _headbar: ZaberConnection
    _wheel: ZaberConnection
    _lickport: ZaberConnection
    _headbar_z: ZaberAxis
    _headbar_pitch: ZaberAxis
    _headbar_roll: ZaberAxis
    _lickport_z: ZaberAxis
    _lickport_y: ZaberAxis
    _lickport_x: ZaberAxis
    _wheel_x: ZaberAxis
    _previous_positions: ZaberPositions | None
    def __init__(
        self, zaber_positions: ZaberPositions | None, zaber_configuration: MesoscopeExternalAssets
    ) -> None: ...
    def restore_position(self) -> None: ...
    def prepare_motors(self) -> None: ...
    def park_position(self) -> None: ...
    def maintenance_position(self) -> None: ...
    def mount_position(self) -> None: ...
    def unmount_position(self) -> None: ...
    def generate_position_snapshot(self) -> ZaberPositions: ...
    def wait_until_idle(self) -> None: ...
    def disconnect(self) -> None: ...
    def park_motors(self) -> None: ...
    def unpark_motors(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...

class MicroControllerInterfaces:
    _started: bool
    _configuration: MesoscopeMicroControllers
    brake: Incomplete
    valve: Incomplete
    gas_puff_valve: Incomplete
    screens: Incomplete
    _actor: MicroControllerInterface
    mesoscope_frame: TTLInterface
    lick: LickInterface
    torque: TorqueInterface
    _sensor: MicroControllerInterface
    wheel_encoder: EncoderInterface
    _encoder: MicroControllerInterface
    def __init__(self, data_logger: DataLogger, microcontroller_configuration: MesoscopeMicroControllers) -> None: ...
    def __del__(self) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class VideoSystems:
    _face_camera_started: bool
    _body_camera_started: bool
    _face_camera: VideoSystem
    _body_camera: VideoSystem
    def __init__(
        self, data_logger: DataLogger, camera_configuration: MesoscopeCameras, output_directory: Path
    ) -> None: ...
    def __del__(self) -> None: ...
    def start_face_camera(self) -> None: ...
    def start_body_camera(self) -> None: ...
    def save_face_camera_frames(self) -> None: ...
    def save_body_camera_frames(self) -> None: ...
    def stop(self) -> None: ...
