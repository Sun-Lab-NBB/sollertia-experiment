from enum import IntEnum, StrEnum
from dataclasses import field, dataclass

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from sl_shared_assets import (
    SessionData,
    GasPuffTrial,
    ExperimentState as ExperimentState,
    WaterRewardTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration,
)

from .configuration import MesoscopeSystemConfiguration as MesoscopeSystemConfiguration
from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import MQTTCommunication

from .tools import (
    MesoscopeData as MesoscopeData,
    CachedMotifDecomposer as CachedMotifDecomposer,
    get_system_configuration as get_system_configuration,
)
from .runtime_ui import RuntimeControlUI as RuntimeControlUI
from .visualizers import (
    VisualizerMode as VisualizerMode,
    BehaviorVisualizer as BehaviorVisualizer,
)
from .maintenance_ui import MaintenanceControlUI as MaintenanceControlUI
from .binding_classes import (
    ZaberMotors as ZaberMotors,
    VideoSystems as VideoSystems,
    MicroControllerInterfaces as MicroControllerInterfaces,
)
from ..shared_components import (
    BrakeInterface as BrakeInterface,
    ValveInterface as ValveInterface,
    GasPuffValveInterface as GasPuffValveInterface,
    get_version_data as get_version_data,
    get_animal_project as get_animal_project,
    get_project_experiments as get_project_experiments,
)
from .data_preprocessing import (
    purge_session as purge_session,
    preprocess_session_data as preprocess_session_data,
    rename_mesoscope_directory as rename_mesoscope_directory,
)

_RESPONSE_DELAY: int
_RENDERING_SEPARATION_DELAY: int
_response_delay_timer: Incomplete

def _generate_mesoscope_position_snapshot(session_data: SessionData, mesoscope_data: MesoscopeData) -> None: ...
def _generate_zaber_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, zaber_motors: ZaberMotors
) -> None: ...
def _setup_zaber_motors(zaber_motors: ZaberMotors) -> None: ...
def _reset_zaber_motors(zaber_motors: ZaberMotors) -> None: ...
def _setup_mesoscope(session_data: SessionData, mesoscope_data: MesoscopeData) -> None: ...
def _verify_descriptor_update(
    descriptor: MesoscopeExperimentDescriptor
    | LickTrainingDescriptor
    | RunTrainingDescriptor
    | WindowCheckingDescriptor,
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
) -> None: ...

class _MesoscopeVRStates(IntEnum):
    IDLE = 0
    REST = 1
    RUN = 2
    LICK_TRAINING = 3
    RUN_TRAINING = 4
    @classmethod
    def to_dict(cls) -> dict[str, int]: ...

class _MesoscopeVRMQTTTopics(StrEnum):
    UNITY_TERMINATION = "Gimbl/Session/Stop"
    UNITY_STARTUP = "Gimbl/Session/Start"
    CUE_SEQUENCE = "CueSequence/"
    CUE_SEQUENCE_REQUEST = "CueSequenceTrigger/"
    DISABLE_LICK_GUIDANCE = "RequireLick/True/"
    ENABLE_LICK_GUIDANCE = "RequireLick/False/"
    DISABLE_OCCUPANCY_GUIDANCE = "RequireWait/True/"
    ENABLE_OCCUPANCY_GUIDANCE = "RequireWait/False/"
    SHOW_REWARD_ZONE_BOUNDARY = "VisibleMarker/True/"
    HIDE_REWARD_ZONE_BOUNDARY = "VisibleMarker/False/"
    UNITY_SCENE_REQUEST = "SceneNameTrigger/"
    UNITY_SCENE = "SceneName/"
    STIMULUS = "Gimbl/Stimulus/"
    TRIGGER_DELAY = "Gimbl/TriggerDelay/"
    ENCODER_DATA = "LinearTreadmill/Data"
    LICK_EVENT = "LickPort/"

class _MesoscopeVRLogMessageCodes(IntEnum):
    SYSTEM_STATE = 1
    RUNTIME_STATE = 2
    REINFORCING_GUIDANCE_STATE = 3
    AVERSIVE_GUIDANCE_STATE = 4
    DISTANCE_SNAPSHOT = 5

@dataclass
class _TrialState:
    completed: int = ...
    distances: NDArray[np.float64] = field(default_factory=Incomplete)
    reinforcing_guided_trials: int = ...
    reinforcing_failed_trials: int = ...
    reinforcing_recovery_threshold: int = ...
    reinforcing_recovery_trials: int = ...
    reinforcing_rewarded: bool = ...
    reinforcing_rewards: tuple[tuple[float, int], ...] = ...
    aversive_guided_trials: int = ...
    aversive_failed_trials: int = ...
    aversive_recovery_threshold: int = ...
    aversive_recovery_trials: int = ...
    aversive_succeeded: bool = ...
    aversive_puff_durations: tuple[int, ...] = ...
    trial_structures: dict[str, WaterRewardTrial | GasPuffTrial] = field(default_factory=dict)
    def trial_completed(self, traveled_distance: float) -> bool: ...
    def get_current_reward(self) -> tuple[float, int]: ...
    def get_current_puff_duration(self) -> int: ...
    def is_current_trial_aversive(self) -> bool: ...
    def advance_trial(self) -> int: ...

@dataclass
class _UnityState:
    position: np.float64 = field(default_factory=Incomplete)
    cue_sequence: NDArray[np.uint8] = field(default_factory=Incomplete)
    terminated: bool = ...
    reinforcing_guidance_enabled: bool = ...
    aversive_guidance_enabled: bool = ...

class _MesoscopeVRSystem:
    _mesoscope_frame_delay: int
    _speed_calculation_window: int
    _source_id: np.uint8
    _started: bool
    _terminated: bool
    _paused: bool
    _mesoscope_started: bool
    descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor
    _experiment_configuration: MesoscopeExperimentConfiguration | None
    _system_configuration: MesoscopeSystemConfiguration
    _session_data: SessionData
    _mesoscope_data: MesoscopeData
    _system_state: int
    _runtime_state: int
    _timestamp_timer: PrecisionTimer
    _distance: np.float64
    _lick_count: np.uint64
    _unconsumed_reward_count: int
    _pause_start_time: int
    paused_time: int
    _delivered_water_volume: np.float64
    _mesoscope_frame_count: np.uint64
    _mesoscope_terminated: bool
    _running_speed: np.float64
    _speed_timer: Incomplete
    _paused_water_volume: np.float64
    _unity_state: _UnityState
    _trial_state: _TrialState
    _logger: DataLogger
    _microcontrollers: MicroControllerInterfaces
    _cameras: VideoSystems
    _zaber_motors: ZaberMotors
    _unity: MQTTCommunication
    _mesoscope_timer: PrecisionTimer
    _motif_decomposer: Incomplete
    _ui: RuntimeControlUI
    _visualizer: BehaviorVisualizer
    def __init__(
        self,
        session_data: SessionData,
        session_descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor,
        experiment_configuration: MesoscopeExperimentConfiguration | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def _generate_hardware_state_snapshot(self) -> None: ...
    def _generate_session_descriptor(self) -> None: ...
    def _setup_unity(self) -> None: ...
    def _wait_for_unity_topic(self, expected_topic: str) -> bytes | bytearray: ...
    def _clear_unity_buffer(self) -> None: ...
    def _get_cue_sequence(self) -> None: ...
    def _decompose_cue_sequence_into_trials(self) -> None: ...
    @staticmethod
    def _decompose_sequence_numba_flat(
        cue_sequence: NDArray[np.uint8],
        motifs_flat: NDArray[np.uint8],
        motif_starts: NDArray[np.int32],
        motif_lengths: NDArray[np.int32],
        motif_indices: NDArray[np.int32],
        max_trials: int,
    ) -> tuple[NDArray[np.int32], int]: ...
    def _start_mesoscope(self) -> None: ...
    def _stop_mesoscope(self) -> None: ...
    def _clear_mesoscope_markers(self) -> None: ...
    def _checkpoint(self) -> None: ...
    def _toggle_reinforcing_guidance(self, *, enable_guidance: bool) -> None: ...
    def _toggle_aversive_guidance(self, *, enable_guidance: bool) -> None: ...
    def _change_system_state(self, new_state: int) -> None: ...
    def change_runtime_state(self, new_state: int) -> None: ...
    def idle(self) -> None: ...
    def rest(self) -> None: ...
    def run(self) -> None: ...
    def lick_train(self) -> None: ...
    def run_train(self) -> None: ...
    def update_visualizer_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None: ...
    def _deliver_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> None: ...
    def _simulate_reward(self, tone_duration: int = 300) -> None: ...
    def resolve_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> bool: ...
    def runtime_cycle(self) -> None: ...
    def _data_cycle(self) -> None: ...
    def _unity_cycle(self) -> None: ...
    def _ui_cycle(self) -> None: ...
    def _mesoscope_cycle(self) -> None: ...
    def _pause_runtime(self) -> None: ...
    def _resume_runtime(self) -> None: ...
    def _terminate_runtime(self) -> None: ...
    def setup_reinforcing_guidance(
        self, initial_guided_trials: int = 3, recovery_mode_threshold: int = 9, recovery_guided_trials: int = 3
    ) -> None: ...
    def setup_aversive_guidance(
        self, initial_guided_trials: int = 0, recovery_mode_threshold: int = 9, recovery_guided_trials: int = 3
    ) -> None: ...
    @property
    def terminated(self) -> bool: ...
    @property
    def running_speed(self) -> np.float64: ...
    @property
    def speed_modifier(self) -> int: ...
    @property
    def duration_modifier(self) -> int: ...
    @property
    def dispensed_water_volume(self) -> float: ...

def window_checking_logic(experimenter: str, project_name: str, animal_id: str) -> None: ...
def lick_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    reward_size: float | None = None,
    reward_tone_duration: int | None = None,
    minimum_reward_delay: int | None = None,
    maximum_reward_delay: int | None = None,
    maximum_water_volume: float | None = None,
    maximum_training_time: int | None = None,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def run_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    reward_size: float | None = None,
    reward_tone_duration: int | None = None,
    initial_speed_threshold: float | None = None,
    initial_duration_threshold: float | None = None,
    speed_increase_step: float | None = None,
    duration_increase_step: float | None = None,
    increase_threshold: float | None = None,
    maximum_water_volume: float | None = None,
    maximum_training_time: int | None = None,
    maximum_idle_time: float | None = None,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def experiment_logic(
    experimenter: str,
    project_name: str,
    experiment_name: str,
    animal_id: str,
    animal_weight: float,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def maintenance_logic() -> None: ...
