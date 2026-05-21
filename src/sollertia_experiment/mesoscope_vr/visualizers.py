"""Provides the Visualizer class that renders the animal's task performance data in real time during training and
experiment runtimes.
"""

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")  # Uses QT backend for performance and compatibility with Linux

from ataraxis_time import PrecisionTimer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    from matplotlib.figure import Figure

# Updates plotting dictionaries to preferentially use Arial text style and specific sizes for different text elements
# in plots. General parameters and the font size for axes' tick numbers.
plt.rcParams.update({"font.family": "Arial", "font.weight": "normal", "xtick.labelsize": 16, "ytick.labelsize": 16})
_fontdict_axis_label = {"family": "Arial", "weight": "normal", "size": 18}  # Axis label fonts.
_fontdict_title = {"family": "Arial", "weight": "normal", "size": 20}  # Title fonts.
_fontdict_legend = {"family": "Arial", "weight": "normal", "size": 14}  # Legend fonts.

# Initializes dictionaries to map colloquial names to specific linestyle and color parameters.
_line_style_dict = {"solid": "-", "dashed": "--", "dotdashed": "_.", "dotted": ":"}
_palette_dict = {
    "green": (0.000, 0.639, 0.408),
    "blue": (0.000, 0.525, 0.749),
    "red": (0.769, 0.008, 0.137),
    "yellow": (1.000, 0.827, 0.000),
    "purple": (0.549, 0.000, 0.749),
    "orange": (1.000, 0.502, 0.000),
    "pink": (0.945, 0.569, 0.608),
    "black": (0.000, 0.000, 0.000),
    "white": (1.000, 1.000, 1.000),
    "gray": (0.500, 0.500, 0.500),
}

# The number of trials to display in the trial performance panel.
_TRIAL_HISTORY_SIZE: int = 20


class VisualizerMode(IntEnum):
    """Defines the display modes for the BehaviorVisualizer."""

    LICK_TRAINING = 0
    """Displays only lick sensor and valve plots."""
    RUN_TRAINING = 1
    """Displays lick, valve, and running speed plots."""
    EXPERIMENT = 2
    """Displays all plots including the trial performance panel."""


def _plt_palette(color: str) -> tuple[float, float, float]:
    """Converts colloquial color names to pyplot RGB color codes.

    Args:
        color: The colloquial name of the color to be retrieved. Available options are: 'green', 'blue', 'red',
            'yellow', 'purple', 'orange', 'pink', 'black', 'white', 'gray'.

    Returns:
        A list of R, G, and B values for the requested color.

    Raises:
        KeyError: If the input color name is not recognized.
    """
    try:
        return _palette_dict[color]
    except KeyError:
        message = (
            f"Unexpected color name '{color}' encountered when converting the colloquial color name to RGB array. "
            f"Provide one of the supported color arguments: {', '.join(_palette_dict.keys())}."
        )
        console.error(message=message, error=KeyError)
        # Fallback to appease mypy. Should not be reachable.
        raise KeyError(message) from None  # pragma: no cover


def _plt_line_styles(line_style: str) -> str:
    """Converts colloquial line style names to pyplot's 'linestyle' string-codes.

    Args:
        line_style: The colloquial name for the line style to be used. Available options are: 'solid', 'dashed',
            'dotdashed' and 'dotted'.

    Returns:
        The string-code for the requested line style.

    Raises:
        KeyError: If the input line style is not recognized.
    """
    try:
        return str(_line_style_dict[line_style])
    except KeyError:
        message = (
            f"Unexpected line style name '{line_style}' encountered when converting the colloquial line style name to "
            f"the pyplot linestyle string. Provide one of the supported line style arguments: "
            f"{', '.join(_line_style_dict.keys())}."
        )
        console.error(message=message, error=KeyError)
        # Fallback to appease mypy. Should not be reachable.
        raise KeyError(message) from None  # pragma: no cover


class BehaviorVisualizer:
    """Visualizes lick, valve, and running speed data in real time.

    Notes:
        This class is designed to run in the main thread of the runtime control process. To update the visualized data,
        call the 'update' class method as part of the runtime cycle method.

        Calling this initializer does not open the visualizer plot window. Call the open() class method to finalize
        the visualizer initialization before starting runtime.

    Attributes:
        _event_tick_true: Stores a NumPy uint8 value of 1 to expedite visualization data processing.
        _event_tick_false: Stores a NumPy uint8 value of 0 to expedite visualization data processing.
        _time_window: Specifies the time window, in seconds, to visualize during runtime.
        _time_step: Specifies the interval, in milliseconds, at which to update the visualization plots.
        _update_timer: The PrecisionTimer instance used to enforce the visualization plot update interval set by the
            _time_step attribute.
        _timestamps: Stores the timestamps of the displayed data during visualization runtime.
        _lick_data: Stores the data used to generate the lick sensor state plot.
        _valve_data: Stores the data used to generate the solenoid valve state plot.
        _puff_data: Stores the data used to generate the air puff valve state plot.
        _speed_data: Stores the data used to generate the running speed plot.
        _lick_event: Determines whether the lick sensor has reported a new lick event since the last visualizer update.
        _valve_event: Determines whether the valve was used to deliver a new reward since the last visualizer update.
        _puff_event: Determines whether the air puff valve was activated since the last visualizer update.
        _lick_line: The line for the lick sensor data plot.
        _valve_line: The line for the solenoid valve data plot.
        _puff_line: The line for the air puff valve data plot.
        _speed_line: The line for the average running speed data plot.
        _figure: The figure that displays the plots.
        _lick_axis: The axis for the lick sensor data plot.
        _valve_axis: The axis for the solenoid valve data plot.
        _puff_axis: The axis for the air puff valve data plot (only in EXPERIMENT mode with aversive trials).
        _speed_axis: The axis for the average running speed data plot.
        _speed_threshold_line: The horizontal line that shows the running speed threshold used during training sessions.
        _duration_threshold_line: The vertical line that shows the running epoch duration used during training sessions.
        _running_speed: The current running speed of the animal, in cm / s, averaged over a time-window of 50 ms.
        _once: Limits certain visualizer setup operations to only be called once during runtime.
        _is_open: Tracks whether the visualizer plot has been created.
        _speed_threshold_text: The text object that communicates the speed threshold value to the user.
        _duration_threshold_text: The text object that communicates the running epoch duration value to the user.
        _mode: The runtime mode that determines the visualizer layout.
        _trial_axis: The axis for the trial performance panel (only in experiment mode).
        _trial_types: Stores the most recent trial types with values -1=empty, 0=reinforcing, 1=aversive.
            Newest at the rightmost index.
        _trial_outcomes: Stores the most recent trial outcomes with values -1=empty, 0=failed, 1=success, 2=guided.
        _total_trials: The total number of trials recorded, used for x-axis labeling.
        _reinforcing_rectangles: The rectangle patches for visualizing reinforcing trial outcomes.
        _aversive_rectangles: The rectangle patches for visualizing aversive trial outcomes.
        _has_reinforcing_trials: Determines whether the experiment includes reinforcing (water reward) trials.
        _has_aversive_trials: Determines whether the experiment includes aversive (gas puff) trials.
    """

    # Pre-initializes NumPy event ticks to slightly reduce cyclic visualizer update speed.
    _event_tick_true = np.uint8(1)
    _event_tick_false = np.uint8(0)

    def __init__(
        self,
    ) -> None:
        # Currently, the class is statically configured to visualize the sliding window of 10 seconds updated every
        # 25 ms.
        self._time_window: int = 10
        self._time_step: int = 25
        self._update_timer = PrecisionTimer("ms")

        # Pre-creates the structures used to store the displayed data during visualization runtime.
        self._timestamps: NDArray[np.float32] = np.arange(
            start=0 - self._time_window, stop=self._time_step / 1000, step=self._time_step / 1000, dtype=np.float32
        )
        self._lick_data: NDArray[np.uint8] = np.zeros_like(a=self._timestamps, dtype=np.uint8)
        self._valve_data: NDArray[np.uint8] = np.zeros_like(a=self._timestamps, dtype=np.uint8)
        self._puff_data: NDArray[np.uint8] = np.zeros_like(a=self._timestamps, dtype=np.uint8)
        self._speed_data: NDArray[np.float64] = np.zeros_like(a=self._timestamps, dtype=np.float64)
        self._valve_event: bool = False
        self._puff_event: bool = False
        self._lick_event: bool = False
        self._running_speed: np.float64 = np.float64(0)

        # Line objects (to be created during open())
        self._lick_line: Line2D | None = None
        self._valve_line: Line2D | None = None
        self._puff_line: Line2D | None = None
        self._speed_line: Line2D | None = None

        # Figure objects (to be created during open())
        self._figure: Figure | None = None
        self._lick_axis: Axes | None = None
        self._valve_axis: Axes | None = None
        self._puff_axis: Axes | None = None
        self._speed_axis: Axes | None = None

        # Running speed threshold and duration threshold lines.
        self._speed_threshold_line: Line2D | None = None
        self._duration_threshold_line: Line2D | None = None

        # Text annotations.
        self._speed_threshold_text: Text | None = None
        self._duration_threshold_text: Text | None = None

        self._is_open: bool = False
        self._once: bool = True

        self._mode: VisualizerMode | int = VisualizerMode.EXPERIMENT

        # Stores trial types: -1=empty, 0=reinforcing, 1=aversive. Newest trial is at the rightmost index.
        self._trial_types: NDArray[np.int8] = np.full(shape=_TRIAL_HISTORY_SIZE, fill_value=-1, dtype=np.int8)
        # Stores trial outcomes: -1=empty, 0=failed, 1=success, 2=guided. Newest trial is at the rightmost index.
        self._trial_outcomes: NDArray[np.int8] = np.full(shape=_TRIAL_HISTORY_SIZE, fill_value=-1, dtype=np.int8)
        self._total_trials: int = 0  # Tracks total trial count for x-axis labeling.

        self._trial_axis: Axes | None = None
        self._reinforcing_rectangles: list[Rectangle] = []
        self._aversive_rectangles: list[Rectangle] = []

        # Trial type flags, set during open() based on experiment configuration.
        self._has_reinforcing_trials: bool = True
        self._has_aversive_trials: bool = True

    def open(
        self,
        mode: VisualizerMode | int = VisualizerMode.EXPERIMENT,
        *,
        has_reinforcing_trials: bool = True,
        has_aversive_trials: bool = True,
    ) -> None:
        """Opens the visualization window and initializes all matplotlib components.

        Notes:
            This method must be called before any visualization updates can occur.

        Args:
            mode: The display mode that determines the subplot layout. Must be a valid VisualizerMode
                enumeration member.
            has_reinforcing_trials: Determines whether the experiment includes reinforcing (water reward) trials.
                When True, the trial panel shows a row for reinforcing trial outcomes.
            has_aversive_trials: Determines whether the experiment includes aversive (gas puff) trials.
                When True, the trial panel shows a row for aversive trial outcomes.
        """
        if self._is_open:
            return

        self._mode = mode
        self._has_reinforcing_trials = has_reinforcing_trials
        self._has_aversive_trials = has_aversive_trials

        # Creates the figure with a mode-dependent subplot layout.
        if mode == VisualizerMode.LICK_TRAINING:
            self._figure, (self._lick_axis, self._valve_axis) = plt.subplots(
                2,
                1,
                figsize=(10, 2.9),
                sharex=True,
                num="Runtime Behavior Visualizer",
                gridspec_kw={
                    "hspace": 0.5,
                    "left": 0.11,
                    "right": 0.97,
                    "top": 0.88,
                    "bottom": 0.22,
                    "height_ratios": [1, 1],
                },
            )
            self._speed_axis = None
            self._trial_axis = None
        elif mode == VisualizerMode.RUN_TRAINING:
            self._figure, (self._lick_axis, self._valve_axis, self._speed_axis) = plt.subplots(
                3,
                1,
                figsize=(10, 6),
                sharex=True,
                num="Runtime Behavior Visualizer",
                gridspec_kw={
                    "hspace": 0.4,
                    "left": 0.11,
                    "right": 0.97,
                    "top": 0.94,
                    "bottom": 0.11,
                    "height_ratios": [1, 1, 3],
                },
            )
            self._trial_axis = None
        else:  # VisualizerMode.EXPERIMENT
            # Uses GridSpec with a spacer row to have tight spacing between lick/valve/speed but larger gap before
            # trial. Figure size and trial panel height adapt based on whether one or both trial types are enabled.
            # When aversive trials are enabled, an additional air puff axis is included.
            spacer_ratio = 0.3

            if has_reinforcing_trials and has_aversive_trials:
                # Both trial types: larger figure with 2-row trial panel and puff axis.
                fig_height = 10.5
                trial_height_ratio = 2.3
                top_margin = 0.97
                bottom_margin = 0.07
                height_ratios = [1, 1, 1, 3, spacer_ratio, trial_height_ratio]
            elif has_aversive_trials:
                # Aversive only: includes puff axis with 1-row trial panel.
                fig_height = 9.5
                trial_height_ratio = 1.5
                top_margin = 0.96
                bottom_margin = 0.08
                height_ratios = [1, 1, 1, 3, spacer_ratio, trial_height_ratio]
            else:
                # Reinforcing only: no puff axis, 1-row trial panel.
                fig_height = 8.5
                trial_height_ratio = 1.5
                top_margin = 0.96
                bottom_margin = 0.08
                height_ratios = [1, 1, 3, spacer_ratio, trial_height_ratio]

            self._figure = plt.figure(num="Runtime Behavior Visualizer", figsize=(10, fig_height))
            grid_spec = GridSpec(
                nrows=len(height_ratios),
                ncols=1,
                figure=self._figure,
                height_ratios=height_ratios,
                hspace=0.4,
                left=0.16,
                right=0.97,
                top=top_margin,
                bottom=bottom_margin,
            )

            # Creates axes based on layout configuration.
            self._lick_axis = self._figure.add_subplot(grid_spec[0])
            self._valve_axis = self._figure.add_subplot(grid_spec[1])
            if has_aversive_trials:
                # Layout with puff axis: lick, valve, puff, speed, spacer, trial.
                self._puff_axis = self._figure.add_subplot(grid_spec[2])
                self._speed_axis = self._figure.add_subplot(grid_spec[3])
                self._trial_axis = self._figure.add_subplot(grid_spec[5])
            else:
                # Layout without puff axis: lick, valve, speed, spacer, trial.
                self._puff_axis = None
                self._speed_axis = self._figure.add_subplot(grid_spec[2])
                self._trial_axis = self._figure.add_subplot(grid_spec[4])

        self._lick_axis.set_title(label="Lick Sensor State", fontdict=_fontdict_title)
        self._lick_axis.set_ylim(bottom=-0.05, top=1.05)
        self._lick_axis.set_xlim(left=-self._time_window, right=0)
        self._lick_axis.set_xlabel(xlabel="")
        self._lick_axis.yaxis.set_major_locator(FixedLocator(locs=[0, 1]))
        self._lick_axis.yaxis.set_major_formatter(FixedFormatter(seq=["No Lick", "Lick"]))

        self._valve_axis.set_title(label="Reward Valve State", fontdict=_fontdict_title)
        self._valve_axis.set_ylim(bottom=-0.05, top=1.05)
        self._valve_axis.set_xlim(left=-self._time_window, right=0)
        self._valve_axis.set_xlabel(xlabel="")
        self._valve_axis.yaxis.set_major_locator(FixedLocator(locs=[0, 1]))
        self._valve_axis.yaxis.set_major_formatter(FixedFormatter(seq=["Closed", "Open"]))

        # Configures the air puff axis, which only exists in EXPERIMENT mode with aversive trials.
        if self._puff_axis is not None:
            self._puff_axis.set_title(label="Air Puff Valve State", fontdict=_fontdict_title)
            self._puff_axis.set_ylim(bottom=-0.05, top=1.05)
            self._puff_axis.set_xlim(left=-self._time_window, right=0)
            self._puff_axis.set_xlabel(xlabel="")
            self._puff_axis.yaxis.set_major_locator(FixedLocator(locs=[0, 1]))
            self._puff_axis.yaxis.set_major_formatter(FixedFormatter(seq=["Closed", "Open"]))

        # Configures the speed axis, which only exists in RUN_TRAINING and EXPERIMENT modes.
        if self._speed_axis is not None:
            self._speed_axis.set_title(label="Average Running Speed", fontdict=_fontdict_title)
            self._speed_axis.set_ylim(bottom=-2, top=42)
            self._speed_axis.set_xlim(left=-self._time_window, right=0)
            self._speed_axis.set_ylabel(ylabel="Speed (cm / s)", fontdict=_fontdict_axis_label)
            self._speed_axis.yaxis.labelpad = 10
            self._speed_axis.set_xlabel(xlabel="Time (s)", fontdict=_fontdict_axis_label)
            self._speed_axis.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False))
            self._speed_axis.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
            plt.setp(self._lick_axis.get_xticklabels(), visible=False)
            plt.setp(self._valve_axis.get_xticklabels(), visible=False)
            if self._puff_axis is not None:
                plt.setp(self._puff_axis.get_xticklabels(), visible=False)
        else:
            # In LICK_TRAINING mode, the valve axis is the bottom plot and shows the x-axis labels.
            self._valve_axis.set_xlabel(xlabel="Time (s)", fontdict=_fontdict_axis_label)
            self._valve_axis.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
            plt.setp(self._lick_axis.get_xticklabels(), visible=False)

        (self._lick_line,) = self._lick_axis.plot(
            self._timestamps,
            self._lick_data,
            drawstyle="steps-post",
            color=_plt_palette("red"),
            linewidth=2,
            alpha=1.0,
            linestyle="solid",
        )

        (self._valve_line,) = self._valve_axis.plot(
            self._timestamps,
            self._valve_data,
            drawstyle="steps-post",
            color=_plt_palette("blue"),
            linewidth=2,
            alpha=1.0,
            linestyle="solid",
        )

        # Creates the air puff plot for EXPERIMENT mode with aversive trials.
        if self._puff_axis is not None:
            (self._puff_line,) = self._puff_axis.plot(
                self._timestamps,
                self._puff_data,
                drawstyle="steps-post",
                color=_plt_palette("gray"),
                linewidth=2,
                alpha=1.0,
                linestyle="solid",
            )

        # Creates the speed plot and threshold lines for RUN_TRAINING and experiment modes.
        if self._speed_axis is not None:
            (self._speed_line,) = self._speed_axis.plot(
                self._timestamps,
                self._speed_data,
                color=_plt_palette("green"),
                linewidth=2,
                alpha=1.0,
                linestyle="solid",
            )

            self._speed_threshold_line = self._speed_axis.axhline(
                y=0.05, color=_plt_palette("black"), linestyle="dashed", linewidth=1.5, alpha=0.5, visible=False
            )
            self._duration_threshold_line = self._speed_axis.axvline(
                x=-0.05, color=_plt_palette("black"), linestyle="dashed", linewidth=1.5, alpha=0.5, visible=False
            )

            self._speed_threshold_text = self._speed_axis.text(
                -self._time_window + 0.5,  # x position: left edge and padding
                40,  # y position: near top of plot
                f"Target speed: {0:.2f} cm/s",
                fontdict=_fontdict_legend,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 1.0, "edgecolor": "none", "pad": 3},
            )

            self._duration_threshold_text = self._speed_axis.text(
                -self._time_window + 0.5,  # x position: left edge and padding
                35.5,  # y position: below speed text
                f"Target duration: {0:.2f} s",
                fontdict=_fontdict_legend,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 1.0, "edgecolor": "none", "pad": 3},
            )

        # Sets up the trial performance panel, which only exists in experiment modes.
        if self._trial_axis is not None:
            self._setup_trial_axis()

        plt.show(block=False)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()

        self._is_open = True

    def __del__(self) -> None:
        """Ensures that the visualization is terminated before the instance is garbage-collected."""
        self.close()

    def update(self) -> None:
        """Re-renders the visualization plot managed by the instance to include the data acquired since the last
        update() call.

        Notes:
            The method has an internal update frequency limiter and is designed to be called without any external
            update frequency control.
        """
        # Does not do anything until the figure is opened (created)
        if not self._is_open:
            return

        # Ensures the plot is not updated any faster than necessary to resolve the time_step used by the plot.
        if self._update_timer.elapsed < self._time_step:
            return

        self._update_timer.reset()

        # Replaces the oldest timestamp data with the current data.
        self._sample_data()

        # Updates the artists with new data.
        self._lick_line.set_data(self._timestamps, self._lick_data)  # type: ignore[union-attr]
        self._valve_line.set_data(self._timestamps, self._valve_data)  # type: ignore[union-attr]
        if self._puff_line is not None:
            self._puff_line.set_data(self._timestamps, self._puff_data)
        if self._speed_line is not None:
            self._speed_line.set_data(self._timestamps, self._speed_data)

        # Renders the changes.
        self._figure.canvas.draw()  # type: ignore[union-attr]
        self._figure.canvas.flush_events()  # type: ignore[union-attr]

    def update_run_training_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None:
        """Updates the running speed and duration threshold lines to use the input anchor values.

        Args:
            speed_threshold: The speed, in centimeter per second, the animal needs to maintain to get water rewards.
            duration_threshold: The duration, in milliseconds, the animal has to maintain the above-threshold speed to
                get water rewards.
        """
        # Does not do anything until the figure is opened (created) or if speed axis doesn't exist.
        if not self._is_open or self._speed_axis is None:
            return

        # Converts from milliseconds to seconds.
        duration_threshold /= 1000

        # Updates line positions.
        self._speed_threshold_line.set_ydata([speed_threshold, speed_threshold])  # type: ignore[union-attr]
        self._duration_threshold_line.set_xdata([-duration_threshold, -duration_threshold])  # type: ignore[union-attr]

        # Updates text annotations with current threshold values.
        self._speed_threshold_text.set_text(f"Target speed: {speed_threshold:.2f} cm/s")  # type: ignore[union-attr]
        self._duration_threshold_text.set_text(  # type: ignore[union-attr]
            f"Target duration: {duration_threshold:.2f} s"
        )

        # Ensures the visibility is only changed once during runtime.
        if self._once:
            self._speed_threshold_line.set_visible(True)  # type: ignore[union-attr]
            self._duration_threshold_line.set_visible(True)  # type: ignore[union-attr]
            self._once = False

        # Renders the changes.
        self._figure.canvas.draw()  # type: ignore[union-attr]
        self._figure.canvas.flush_events()  # type: ignore[union-attr]

    def close(self) -> None:
        """Closes the visualized figure and cleans up the resources used by the instance during runtime."""
        if self._is_open and self._figure is not None:
            plt.close(self._figure)
            self._is_open = False

    def _sample_data(self) -> None:
        """Updates the visualization data arrays with the data accumulated since the last visualization update."""
        # Rolls arrays by one position to the left, so the first element becomes the last.
        self._valve_data = np.roll(self._valve_data, shift=-1)
        self._lick_data = np.roll(self._lick_data, shift=-1)

        # Replaces the last element (previously the first or 'oldest' value) with new data.

        # If the runtime has detected at least one lick event since the last visualizer update, emits a lick tick.
        if self._lick_event:
            self._lick_data[-1] = self._event_tick_true
        else:
            self._lick_data[-1] = self._event_tick_false
        self._lick_event = False  # Resets the lick event flag.

        # If the runtime has detected at least one water reward (valve) event since the last visualizer update, emits a
        # valve activation tick.
        if self._valve_event:
            self._valve_data[-1] = self._event_tick_true
        else:
            self._valve_data[-1] = self._event_tick_false
        self._valve_event = False  # Resets the valve event flag.

        # If the runtime has detected at least one air puff event since the last visualizer update, emits a puff tick.
        # Only updates if puff axis exists (EXPERIMENT mode with aversive trials).
        if self._puff_axis is not None:
            self._puff_data = np.roll(self._puff_data, shift=-1)
            if self._puff_event:
                self._puff_data[-1] = self._event_tick_true
            else:
                self._puff_data[-1] = self._event_tick_false
            self._puff_event = False  # Resets the puff event flag.

        # The speed value is updated ~every 50 milliseconds. Until the update timeout is exhausted, at each graph
        # update cycle the last speed point is overwritten with the previous speed point. This generates a
        # sequence of at most 2 identical speed readouts and is not noticeable to the user. Only updates if speed axis
        # exists (not in LICK_TRAINING mode).
        if self._speed_axis is not None:
            self._speed_data = np.roll(self._speed_data, shift=-1)
            self._speed_data[-1] = self._running_speed

    def add_lick_event(self) -> None:
        """Instructs the visualizer to render a new lick event during the next update cycle."""
        self._lick_event = True

    def add_valve_event(self) -> None:
        """Instructs the visualizer to render a new valve activation (reward) event during the next update cycle."""
        self._valve_event = True

    def add_puff_event(self) -> None:
        """Instructs the visualizer to render a new air puff valve activation event during the next update cycle."""
        self._puff_event = True

    def update_running_speed(self, running_speed: np.float64) -> None:
        """Instructs the visualizer to render the provided running speed datapoint during the next update cycle."""
        self._running_speed = running_speed

    def _setup_trial_axis(self) -> None:
        """Initializes the trial performance panel with empty rectangle patches.

        This method creates a visualization showing the most recent trials. The layout adapts based on which trial
        types are enabled: when both reinforcing and aversive trials are enabled, trials appear in a 2-row layout
        with reinforcing trials on the bottom and aversive trials on top. When only one trial type is enabled,
        a single-row layout is used.
        """
        if self._trial_axis is None:
            return

        self._trial_axis.set_title(label="Trial Performance", fontdict=_fontdict_title)
        self._trial_axis.set_xlim(left=0.5, right=_TRIAL_HISTORY_SIZE + 0.5)
        self._trial_axis.set_xlabel(xlabel="Trial Number", fontdict=_fontdict_axis_label)
        self._trial_axis.set_xticks(ticks=range(1, _TRIAL_HISTORY_SIZE + 1))
        self._trial_axis.set_xticklabels(labels=[""] * _TRIAL_HISTORY_SIZE)

        # Configures the Y-axis layout based on which trial types are enabled.
        if self._has_reinforcing_trials and self._has_aversive_trials:
            # Two-row layout: reinforcing on bottom, aversive on top.
            self._trial_axis.set_ylim(bottom=-0.1, top=1.0)
            self._trial_axis.set_yticks(ticks=[0.25, 0.75])
            self._trial_axis.set_yticklabels(labels=["Reinforcing", "Aversive"])
            self._trial_axis.axhline(y=0.5, color=_plt_palette(color="gray"), linestyle="-", linewidth=0.5, alpha=0.5)
            reinforcing_y = 0.05
            aversive_y = 0.55
            rect_height = 0.4
        elif self._has_reinforcing_trials:
            # Single-row layout: reinforcing only.
            self._trial_axis.set_ylim(bottom=-0.1, top=1.0)
            self._trial_axis.set_yticks(ticks=[0.45])
            self._trial_axis.set_yticklabels(labels=["Reinforcing"])
            reinforcing_y = 0.15
            aversive_y = 0.0  # Not used.
            rect_height = 0.6
        else:
            # Single-row layout: aversive only.
            self._trial_axis.set_ylim(bottom=-0.1, top=1.0)
            self._trial_axis.set_yticks(ticks=[0.45])
            self._trial_axis.set_yticklabels(labels=["Aversive"])
            reinforcing_y = 0.0  # Not used.
            aversive_y = 0.15
            rect_height = 0.6

        # Adds color legend for trial outcomes.
        legend_elements = [
            Rectangle(xy=(0, 0), width=1, height=1, facecolor=_plt_palette(color="green"), label="Succeeded"),
            Rectangle(xy=(0, 0), width=1, height=1, facecolor=_plt_palette(color="red"), label="Failed"),
            Rectangle(xy=(0, 0), width=1, height=1, facecolor=_plt_palette(color="gray"), label="Guided"),
        ]
        self._trial_axis.legend(
            handles=legend_elements,
            loc="lower right",
            ncol=3,
            fontsize=10,
            framealpha=0.9,
            edgecolor="none",
            bbox_to_anchor=(1.0, -0.02),
        )

        # Creates the reinforcing trial rectangles if reinforcing trials are enabled.
        self._reinforcing_rectangles = []
        if self._has_reinforcing_trials:
            for i in range(_TRIAL_HISTORY_SIZE):
                rect = Rectangle(
                    xy=(i + 1 - 0.175, reinforcing_y),
                    width=0.35,
                    height=rect_height,
                    facecolor=_plt_palette("gray"),
                    edgecolor="none",
                    alpha=0.3,
                    visible=False,
                )
                self._trial_axis.add_patch(rect)
                self._reinforcing_rectangles.append(rect)

        # Creates the aversive trial rectangles if aversive trials are enabled.
        self._aversive_rectangles = []
        if self._has_aversive_trials:
            for i in range(_TRIAL_HISTORY_SIZE):
                rect = Rectangle(
                    xy=(i + 1 - 0.175, aversive_y),
                    width=0.35,
                    height=rect_height,
                    facecolor=_plt_palette("gray"),
                    edgecolor="none",
                    alpha=0.3,
                    visible=False,
                )
                self._trial_axis.add_patch(rect)
                self._aversive_rectangles.append(rect)

    def add_trial_outcome(self, *, is_aversive: bool, succeeded: bool, was_guided: bool) -> None:
        """Records a trial outcome and updates the trial performance visualization.

        Args:
            is_aversive: Determines whether the trial was an aversive (gas puff) trial. If False, the trial is
                treated as a reinforcing (water reward) trial.
            succeeded: Determines whether the animal succeeded in the trial. For reinforcing trials, success means
                the animal received a reward. For aversive trials, success means the animal avoided the puff.
            was_guided: Determines whether the trial was in guidance mode (automatic rewards/puffs).
        """
        if self._trial_axis is None:
            return

        # Increments total trial count for x-axis labeling.
        self._total_trials += 1

        # Maps the boolean outcome flags to integer values: 2=guided, 1=success, 0=failure.
        if was_guided:
            outcome = np.int8(2)
        elif succeeded:
            outcome = np.int8(1)
        else:
            outcome = np.int8(0)

        # Rolls arrays left by 1 position and inserts new trial at the rightmost position.
        self._trial_types = np.roll(a=self._trial_types, shift=-1)
        self._trial_outcomes = np.roll(a=self._trial_outcomes, shift=-1)
        self._trial_types[-1] = np.int8(1) if is_aversive else np.int8(0)
        self._trial_outcomes[-1] = outcome

        # Redraws all rectangles based on the arrays. Newest trial is at the rightmost index.
        for index in range(_TRIAL_HISTORY_SIZE):
            trial_type = self._trial_types[index]
            trial_outcome = self._trial_outcomes[index]
            if trial_type == -1:
                # Empty slot, hides rectangles for enabled trial types.
                if self._has_reinforcing_trials:
                    self._reinforcing_rectangles[index].set_visible(False)
                if self._has_aversive_trials:
                    self._aversive_rectangles[index].set_visible(False)
            elif trial_type == 1:
                # Aversive trial.
                if self._has_reinforcing_trials:
                    self._reinforcing_rectangles[index].set_visible(False)
                if self._has_aversive_trials:
                    self._update_trial_rectangle(
                        rectangles=self._aversive_rectangles, index=index, outcome=trial_outcome
                    )
            else:
                # Reinforcing trial.
                if self._has_aversive_trials:
                    self._aversive_rectangles[index].set_visible(False)
                if self._has_reinforcing_trials:
                    self._update_trial_rectangle(
                        rectangles=self._reinforcing_rectangles, index=index, outcome=trial_outcome
                    )

        # Updates x-axis labels. Empty positions get empty labels, filled positions get trial numbers.
        num_displayed = min(self._total_trials, _TRIAL_HISTORY_SIZE)
        start_trial_number = self._total_trials - num_displayed + 1
        labels: list[str] = []
        for index in range(_TRIAL_HISTORY_SIZE):
            if self._trial_types[index] == -1:
                labels.append("")
            else:
                labels.append(str(start_trial_number + index - (_TRIAL_HISTORY_SIZE - num_displayed)))
        self._trial_axis.set_xticklabels(labels=labels)

    @staticmethod
    def _update_trial_rectangle(rectangles: list[Rectangle], index: int, outcome: np.int8) -> None:
        """Updates a single trial rectangle based on the outcome value.

        Args:
            rectangles: The list of rectangle patches (either reinforcing or aversive).
            index: The index of the trial in the circular buffer (0-19).
            outcome: The outcome value (-1=empty, 0=failure, 1=success, 2=guided).
        """
        if index >= len(rectangles):
            return

        rect = rectangles[index]

        # Sets rectangle color based on outcome: green=success, red=failure, gray=guided.
        if outcome == 1:
            rect.set_facecolor(_plt_palette("green"))
        elif outcome == 0:
            rect.set_facecolor(_plt_palette("red"))
        else:
            rect.set_facecolor(_plt_palette("gray"))

        rect.set_alpha(1.0)
        rect.set_visible(True)
