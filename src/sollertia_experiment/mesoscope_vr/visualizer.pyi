from enum import IntEnum
from typing import Any

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import Event
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg

_FONTDICT_AXIS_LABEL: dict[str, str | int]
_FONTDICT_TITLE: dict[str, str | int]
_FONTDICT_LEGEND: dict[str, str | int]
_LINE_STYLE_DICT: dict[str, str]
_PALETTE_DICT: dict[str, tuple[float, float, float]]
_TRIAL_HISTORY_SIZE: int
_SPEED_AXIS_YLIM: tuple[float, float]
_BINARY_AXIS_YLIM: tuple[float, float]
_TRIAL_RECTANGLE_WIDTH: float
_TRIAL_RECTANGLE_OFFSET: float

class VisualizerMode(IntEnum):
    LICK_TRAINING = 0
    RUN_TRAINING = 1
    EXPERIMENT = 2

class BehaviorVisualizer:
    _event_tick_true: Incomplete
    _event_tick_false: Incomplete
    _time_window: int
    _time_step: int
    _update_timer: Incomplete
    _timestamps: NDArray[np.float32]
    _lick_data: NDArray[np.uint8]
    _valve_data: NDArray[np.uint8]
    _puff_data: NDArray[np.uint8]
    _speed_data: NDArray[np.float64]
    _valve_event: bool
    _puff_event: bool
    _lick_event: bool
    _running_speed: np.float64
    _lick_line: Line2D | None
    _valve_line: Line2D | None
    _puff_line: Line2D | None
    _speed_line: Line2D | None
    _figure: Figure | None
    _lick_axis: Axes | None
    _valve_axis: Axes | None
    _puff_axis: Axes | None
    _speed_axis: Axes | None
    _blit_manager: _BlitManager | None
    _speed_threshold_line: Line2D | None
    _duration_threshold_line: Line2D | None
    _speed_threshold_text: Text | None
    _duration_threshold_text: Text | None
    _is_open: bool
    _once: bool
    _mode: VisualizerMode | int
    _trial_types: NDArray[np.int8]
    _trial_outcomes: NDArray[np.int8]
    _total_trials: int
    _trial_axis: Axes | None
    _reinforcing_rectangles: list[Rectangle]
    _aversive_rectangles: list[Rectangle]
    _has_reinforcing_trials: bool
    _has_aversive_trials: bool
    def __init__(self) -> None: ...
    def open(
        self, mode: VisualizerMode | int = ..., *, has_reinforcing_trials: bool = True, has_aversive_trials: bool = True
    ) -> None: ...
    def __del__(self) -> None: ...
    def update(self) -> None: ...
    def update_run_training_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None: ...
    def add_lick_event(self) -> None: ...
    def add_valve_event(self) -> None: ...
    def add_puff_event(self) -> None: ...
    def update_running_speed(self, running_speed: np.float64) -> None: ...
    def add_trial_outcome(self, *, is_aversive: bool, succeeded: bool, was_guided: bool) -> None: ...
    def close(self) -> None: ...
    def _sample_data(self) -> None: ...
    def _setup_trial_axis(self) -> None: ...
    @staticmethod
    def _update_trial_rectangle(rectangles: list[Rectangle], index: int, outcome: np.int8) -> None: ...

def _plt_palette(color: str) -> tuple[float, float, float]: ...
def _plt_line_styles(line_style: str) -> str: ...

class _BlitManager:
    _canvas: Incomplete
    _figure: Incomplete
    _animated_artists: Incomplete
    _background: Any
    _connection_id: Incomplete
    def __init__(self, canvas: FigureCanvasAgg, animated_artists: list[Line2D]) -> None: ...
    def update(self) -> None: ...
    def refresh(self) -> None: ...
    def _on_draw(self, _event: Event) -> None: ...
