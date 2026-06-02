from enum import IntEnum
from multiprocessing import Process

from _typeshed import Incomplete
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QLabel, QGroupBox, QMainWindow, QPushButton, QDoubleSpinBox
from ataraxis_data_structures import SharedMemoryArray

from .system import RUN_TRAINING_THRESHOLD_LIMITS as RUN_TRAINING_THRESHOLD_LIMITS
from .visualizer import VisualizerMode as VisualizerMode

_SPEED_THRESHOLD_SCALE: int
_DURATION_THRESHOLD_SCALE: int
_MODIFIER_STEP_CM_S: float
_UI_REFRESH_INTERVAL_MS: int
_EXIT_FEEDBACK_DELAY_MS: int

class _DataArrayIndex(IntEnum):
    TERMINATION = 0
    EXIT_SIGNAL = 1
    REWARD_SIGNAL = 2
    SPEED_MODIFIER = 3
    DURATION_MODIFIER = 4
    PAUSE_STATE = 5
    OPEN_VALVE = 6
    CLOSE_VALVE = 7
    REWARD_VOLUME = 8
    REINFORCING_GUIDANCE_ENABLED = 9
    AVERSIVE_GUIDANCE_ENABLED = 10
    GAS_VALVE_OPEN = 11
    GAS_VALVE_CLOSE = 12
    GAS_VALVE_PUFF = 13
    GAS_VALVE_PUFF_DURATION = 14
    SETUP_COMPLETE = 15
    RUNTIME_SPEED_THRESHOLD = 16
    RUNTIME_DURATION_THRESHOLD = 17

class _WaterValveTrackerIndex(IntEnum):
    OPEN_STATE = 2

class _GasPuffTrackerIndex(IntEnum):
    OPEN_STATE = 1

class RuntimeControlUI:
    _data_array: SharedMemoryArray
    _valve_tracker: SharedMemoryArray
    _gas_puff_tracker: SharedMemoryArray
    _mode: VisualizerMode
    _has_reinforcing_trials: bool
    _has_aversive_trials: bool
    _ui_process: Process | None
    _started: bool
    def __init__(self, valve_tracker: SharedMemoryArray, gas_puff_tracker: SharedMemoryArray) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    def start(
        self, mode: VisualizerMode | int = ..., *, has_reinforcing_trials: bool = True, has_aversive_trials: bool = True
    ) -> None: ...
    def shutdown(self) -> None: ...
    def set_pause_state(self, *, paused: bool) -> None: ...
    def set_reinforcing_guidance_state(self, *, enabled: bool) -> None: ...
    def set_aversive_guidance_state(self, *, enabled: bool) -> None: ...
    def set_setup_complete(self) -> None: ...
    @property
    def exit_signal(self) -> bool: ...
    @property
    def reward_signal(self) -> bool: ...
    @property
    def speed_modifier(self) -> int: ...
    @property
    def duration_modifier(self) -> int: ...
    def set_runtime_thresholds(self, speed_threshold_cm_s: float, duration_threshold_ms: float) -> None: ...
    @property
    def pause_runtime(self) -> bool: ...
    @property
    def open_valve(self) -> bool: ...
    @property
    def close_valve(self) -> bool: ...
    @property
    def reward_volume(self) -> int: ...
    @property
    def enable_reinforcing_guidance(self) -> bool: ...
    @property
    def enable_aversive_guidance(self) -> bool: ...
    @property
    def gas_valve_open_signal(self) -> bool: ...
    @property
    def gas_valve_close_signal(self) -> bool: ...
    @property
    def gas_valve_puff_signal(self) -> bool: ...
    @property
    def gas_valve_puff_duration(self) -> int: ...
    def _run_ui_process(
        self, mode: VisualizerMode, *, has_reinforcing_trials: bool, has_aversive_trials: bool
    ) -> None: ...

class _ControlUIWindow(QMainWindow):
    _data_array: SharedMemoryArray
    _valve_tracker: SharedMemoryArray
    _gas_puff_tracker: SharedMemoryArray
    _mode: VisualizerMode
    _has_reinforcing_trials: bool
    _has_aversive_trials: bool
    _is_paused: bool
    _setup_complete: bool
    _reinforcing_guidance_enabled: bool
    _aversive_guidance_enabled: bool
    _reward_in_progress: bool
    _puff_in_progress: bool
    _last_auto_speed: int
    _last_auto_duration: int
    def __init__(
        self,
        data_array: SharedMemoryArray,
        valve_tracker: SharedMemoryArray,
        gas_puff_tracker: SharedMemoryArray,
        mode: VisualizerMode | int = ...,
        *,
        has_reinforcing_trials: bool = True,
        has_aversive_trials: bool = True,
    ) -> None: ...
    def closeEvent(self, event: QCloseEvent | None) -> None: ...
    _exit_button: Incomplete
    _pause_button: Incomplete
    _reinforcing_guidance_button: QPushButton | None
    _aversive_guidance_button: QPushButton | None
    _runtime_status_label: Incomplete
    _valve_open_button: Incomplete
    _valve_close_button: Incomplete
    _reward_button: Incomplete
    _volume_spinbox: Incomplete
    _valve_status_label: Incomplete
    _gas_valve_open_button: QPushButton | None
    _gas_valve_close_button: QPushButton | None
    _gas_puff_button: QPushButton | None
    _gas_duration_spinbox: QDoubleSpinBox | None
    _gas_valve_status_label: QLabel | None
    _speed_group: QGroupBox | None
    _duration_group: QGroupBox | None
    _speed_spinbox: QDoubleSpinBox | None
    _duration_spinbox: QDoubleSpinBox | None
    def _setup_ui(self) -> None: ...
    def _apply_qt6_styles(self) -> None: ...
    _monitor_timer: Incomplete
    def _setup_monitoring(self) -> None: ...
    def _check_external_state(self) -> None: ...
    def _exit_runtime(self) -> None: ...
    def _deliver_reward(self) -> None: ...
    def _open_valve(self) -> None: ...
    def _close_valve(self) -> None: ...
    def _toggle_pause(self) -> None: ...
    def _update_reward_volume(self) -> None: ...
    def _update_speed_modifier(self) -> None: ...
    def _update_duration_modifier(self) -> None: ...
    def _sync_run_training_spinbox(
        self,
        spinbox: QDoubleSpinBox | None,
        auto_index: _DataArrayIndex,
        modifier_index: _DataArrayIndex,
        last_auto: int,
        divisor: float,
    ) -> int: ...
    @staticmethod
    def _refresh_button_style(button: QPushButton) -> None: ...
    def _update_reinforcing_guidance_ui(self) -> None: ...
    def _update_aversive_guidance_ui(self) -> None: ...
    def _toggle_reinforcing_guidance(self) -> None: ...
    def _toggle_aversive_guidance(self) -> None: ...
    def _update_pause_ui(self) -> None: ...
    def _disable_valve_open_close_buttons(self) -> None: ...
    def _update_gas_puff_duration(self) -> None: ...
    def _gas_valve_open(self) -> None: ...
    def _gas_valve_close(self) -> None: ...
    def _gas_valve_puff(self) -> None: ...
