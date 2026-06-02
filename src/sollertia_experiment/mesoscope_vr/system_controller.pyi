import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from sollertia_shared_assets import (
    SessionData as SessionData,
    TaskTemplate as TaskTemplate,
    RunTrainingDescriptor,
    LickTrainingDescriptor as LickTrainingDescriptor,
    MesoscopeExperimentDescriptor as MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration as MesoscopeExperimentConfiguration,
)
from ataraxis_data_structures import DataLogger

from .system import (
    MesoscopeData as MesoscopeData,
    ZaberPositions as ZaberPositions,
    MesoscopeVRStates as MesoscopeVRStates,
    MesoscopePositions as MesoscopePositions,
    MesoscopeSystemConfiguration as MesoscopeSystemConfiguration,
    get_system_configuration as get_system_configuration,
)
from ..vr_task import (
    VRTaskDriver as VRTaskDriver,
    VRTaskEventKind as VRTaskEventKind,
    load_vr_task_template as load_vr_task_template,
)
from .runtime_ui import RuntimeControlUI as RuntimeControlUI
from .visualizer import (
    VisualizerMode as VisualizerMode,
    BehaviorVisualizer as BehaviorVisualizer,
)
from .binding_classes import (
    ZaberMotors as ZaberMotors,
    VideoSystems as VideoSystems,
    MicroControllerInterfaces as MicroControllerInterfaces,
)
from .data_preprocessing import (
    purge_session as purge_session,
    preprocess_session_data as preprocess_session_data,
    rename_mesoscope_directory as rename_mesoscope_directory,
)
from .acquisition_components import (
    _RESPONSE_DELAY as _RESPONSE_DELAY,
    _TrialState as _TrialState,
    _setup_mesoscope as _setup_mesoscope,
    _reset_zaber_motors as _reset_zaber_motors,
    _setup_zaber_motors as _setup_zaber_motors,
    _response_delay_timer as _response_delay_timer,
    _generate_zaber_snapshot as _generate_zaber_snapshot,
    _verify_descriptor_update as _verify_descriptor_update,
    _MesoscopeVRLogMessageCodes as _MesoscopeVRLogMessageCodes,
    _generate_mesoscope_position_snapshot as _generate_mesoscope_position_snapshot,
)

_MINIMUM_CPU_COUNT: int
_GUIDED_RUNTIME_STATE_CODE: int
_MESOSCOPE_START_TIMEOUT_MS: int
_EXPECTED_FRAME_PULSES: int
_MICROLITERS_PER_MILLILITER: float

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
    _trial_state: _TrialState
    _logger: DataLogger
    _microcontrollers: MicroControllerInterfaces
    _cameras: VideoSystems
    _zaber_motors: ZaberMotors
    _task_template: TaskTemplate | None
    _vr_task: VRTaskDriver | None
    _mesoscope_timer: PrecisionTimer
    _ui: RuntimeControlUI
    _visualizer: BehaviorVisualizer
    def __init__(
        self,
        session_data: SessionData,
        session_descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor,
        experiment_configuration: MesoscopeExperimentConfiguration | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def change_runtime_state(self, new_state: int) -> None: ...
    def idle(self) -> None: ...
    def rest(self) -> None: ...
    def run(self) -> None: ...
    def lick_train(self) -> None: ...
    def run_train(self) -> None: ...
    def update_visualizer_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None: ...
    def publish_runtime_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None: ...
    def resolve_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> bool: ...
    def runtime_cycle(self) -> None: ...
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
    def _generate_hardware_state_snapshot(self) -> None: ...
    def _checkpoint(self) -> None: ...
    def _start_mesoscope(self) -> None: ...
    def _clear_mesoscope_markers(self) -> None: ...
    def _stop_mesoscope(self) -> None: ...
    def _change_system_state(self, new_state: int) -> None: ...
    def _log_cue_sequence(self, cue_sequence: NDArray[np.uint8]) -> None: ...
    def _log_reinforcing_guidance_change(self, *, enabled: bool) -> None: ...
    def _log_aversive_guidance_change(self, *, enabled: bool) -> None: ...
    def _refresh_trial_state_from_vr_decomposition(self) -> None: ...
    def _build_trial_parameter_arrays(
        self, *, trial_names: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, int], ...], tuple[int, ...]]: ...
    def _deliver_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> None: ...
    def _simulate_reward(self, tone_duration: int = 300) -> None: ...
    def _data_cycle(self) -> None: ...
    def _unity_cycle(self) -> None: ...
    def _ui_cycle(self) -> None: ...
    def _mesoscope_cycle(self) -> None: ...
    def _pause_runtime(self) -> None: ...
    def _resume_runtime(self) -> None: ...
    def _terminate_runtime(self) -> None: ...
    def _generate_session_descriptor(self) -> None: ...
