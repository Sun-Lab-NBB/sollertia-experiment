"""Standalone simulation that pairs a Mesoscope-VR runtime control panel with the matching behavior visualizer and
drives both with randomly generated data.

This script is a manual review tool for evaluating the look and feel of an acquisition runtime without any hardware.
Set the '_MODE' configuration variable near the top of the file to choose which runtime to simulate; the control
panel and the visualizer layout are paired to the selected mode:

- LICK_TRAINING: the lick sensor and reward valve plots.
- RUN_TRAINING: adds the running speed plot and the speed and duration threshold lines driven by the control-panel
  modifier spin boxes.
- EXPERIMENT: adds the air puff plot and the two-row trial performance panel, plus the reinforcing and aversive
  guidance toggles and the gas puff valve controls. The '_HAS_REINFORCING_TRIALS' and '_HAS_AVERSIVE_TRIALS' flags
  select the EXPERIMENT sub-layout.

The control panel starts paused, mirroring a real session. Click 'Resume Runtime' to start the synthetic data stream,
exercise the available controls to see them reflected in the plots, and click 'Terminate Runtime' (or close a window,
or press Ctrl+C) to end the simulation. Run it from the repository root with 'python live.py'.
"""

import sys
from pathlib import Path

# Prepends the in-repo source tree to the import path so the simulation always exercises the local source (including
# any uncommitted changes) rather than an installed copy of the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataclasses import field, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray

from sollertia_experiment.mesoscope_vr.runtime_ui import RuntimeControlUI
from sollertia_experiment.mesoscope_vr.visualizer import VisualizerMode, BehaviorVisualizer

# Simulation configuration. Edit these values to choose which acquisition runtime to simulate. '_MODE' selects the
# paired visualizer layout and control panel; the trial-type flags only apply to the EXPERIMENT mode.
_MODE: VisualizerMode = VisualizerMode.RUN_TRAINING
"""The acquisition runtime to simulate. Set to VisualizerMode.LICK_TRAINING, RUN_TRAINING, or EXPERIMENT."""
_HAS_REINFORCING_TRIALS: bool = True
"""Determines whether the EXPERIMENT simulation includes reinforcing (water reward) trials."""
_HAS_AVERSIVE_TRIALS: bool = True
"""Determines whether the EXPERIMENT simulation includes aversive (gas puff) trials."""

# The cadence, in milliseconds, at which new running speed and lick samples are generated. Matches the ~50 ms running
# speed update interval used by the real acquisition runtime.
_DATA_INTERVAL_MS: int = 50
"""The interval, in milliseconds, between synthetic behavior samples."""
_TRIAL_INTERVAL_MS: int = 4000
"""The interval, in milliseconds, between recorded trial outcomes."""
_LOOP_DELAY_MS: int = 5
"""The minimum delay, in milliseconds, between simulation loop iterations."""
_VALVE_PULSE_MS: int = 150
"""The duration, in milliseconds, the reward valve stays open during a single reward delivery."""
_PUFF_PULSE_MS: int = 100
"""The duration, in milliseconds, the gas valve stays open during a single air puff delivery."""

_MAXIMUM_RUNNING_SPEED: float = 40.0
"""The upper bound, in centimeters per second, for the simulated running speed."""
_SPEED_STEP_SCALE: float = 3.0
"""The standard deviation, in centimeters per second, of the per-sample running speed random walk."""

_LICK_PROBABILITY: float = 0.15
"""The baseline probability of emitting a lick on any given behavior sample."""
_LICK_BOOST_PROBABILITY: float = 0.6
"""The elevated lick probability used during the brief consumption window after a reward."""
_LICK_BOOST_SAMPLES: int = 8
"""The number of behavior samples for which the elevated lick probability applies after a reward."""

_SPONTANEOUS_REWARD_PROBABILITY: float = 0.01
"""The probability of spontaneously delivering a reward on any given behavior sample."""
_SPONTANEOUS_PUFF_PROBABILITY: float = 0.008
"""The probability of spontaneously delivering an air puff on any given behavior sample."""
_TRIAL_SUCCESS_PROBABILITY: float = 0.7
"""The probability that a non-guided trial is scored as a success."""
_AVERSIVE_TRIAL_PROBABILITY: float = 0.5
"""The probability that a recorded trial is an aversive (gas puff) trial when both trial types are enabled."""

_BASE_SPEED_THRESHOLD_CM_S: float = 10.0
"""The baseline running speed threshold, in centimeters per second, used in the run-training simulation."""
_BASE_DURATION_THRESHOLD_MS: float = 2000.0
"""The baseline running epoch duration threshold, in milliseconds, used in the run-training simulation."""
_SPEED_MODIFIER_SCALE: float = 0.01
"""The running speed threshold change, in centimeters per second, applied per unit of the speed modifier control."""
_DURATION_MODIFIER_SCALE: float = 10.0
"""The duration threshold change, in milliseconds, applied per unit of the duration modifier control."""


def _new_millisecond_timer() -> PrecisionTimer:
    """Returns a millisecond-precision PrecisionTimer instance."""
    return PrecisionTimer(precision=TimerPrecisions.MILLISECOND)


@dataclass(slots=True)
class _SimulationState:
    """Holds the mutable state of the behavior simulation across update cycles.

    Args:
        rng: The random number generator used to produce the synthetic behavior stream.
        mode: The visualizer display mode being simulated.
        has_reinforcing_trials: Determines whether the EXPERIMENT simulation includes reinforcing trials.
        has_aversive_trials: Determines whether the EXPERIMENT simulation includes aversive trials.
    """

    rng: np.random.Generator
    """The random number generator used to produce the synthetic behavior stream."""
    mode: VisualizerMode
    """The visualizer display mode being simulated."""
    has_reinforcing_trials: bool = True
    """Determines whether the EXPERIMENT simulation includes reinforcing (water reward) trials."""
    has_aversive_trials: bool = True
    """Determines whether the EXPERIMENT simulation includes aversive (gas puff) trials."""
    running_speed: float = 0.0
    """The current simulated running speed, in centimeters per second."""
    valve_held_open: bool = False
    """Tracks whether the user is holding the reward valve open via the control panel."""
    gas_held_open: bool = False
    """Tracks whether the user is holding the gas valve open via the control panel."""
    reward_pulse_active: bool = False
    """Tracks whether a timed reward valve pulse is currently in progress."""
    puff_pulse_active: bool = False
    """Tracks whether a timed air puff valve pulse is currently in progress."""
    lick_boost_samples: int = 0
    """The number of remaining samples for which the elevated post-reward lick probability applies."""
    last_speed_modifier: int = 0
    """The speed modifier value last pushed to the visualizer, used to detect control-panel changes."""
    last_duration_modifier: int = 0
    """The duration modifier value last pushed to the visualizer, used to detect control-panel changes."""
    thresholds_pushed: bool = False
    """Tracks whether the run-training thresholds have been pushed to the visualizer at least once."""
    reward_pulse_timer: PrecisionTimer = field(default_factory=_new_millisecond_timer)
    """The timer that bounds the reward valve open duration."""
    puff_pulse_timer: PrecisionTimer = field(default_factory=_new_millisecond_timer)
    """The timer that bounds the gas valve open duration."""


def _create_trackers() -> tuple[SharedMemoryArray, SharedMemoryArray]:
    """Creates the reward valve and gas puff state trackers expected by the control panel.

    The trackers mirror the layouts used by the real ValveModule and GasPuffValveInterface: the valve tracker uses
    index 2 and the puff tracker uses index 1 to report the open and close state read by the GUI.

    Returns:
        A tuple storing the reward valve tracker and the gas puff tracker.
    """
    valve_tracker = SharedMemoryArray.create_array(
        name="live_simulation_valve_tracker", prototype=np.zeros(shape=3, dtype=np.float64), exists_ok=True
    )
    puff_tracker = SharedMemoryArray.create_array(
        name="live_simulation_puff_tracker", prototype=np.zeros(shape=2, dtype=np.uint32), exists_ok=True
    )
    return valve_tracker, puff_tracker


def _fire_reward(visualizer: BehaviorVisualizer, state: _SimulationState) -> None:
    """Renders a water reward delivery and opens the reward valve for a brief pulse."""
    visualizer.add_valve_event()
    state.reward_pulse_active = True
    state.reward_pulse_timer.reset()
    # Animals lick more frequently while consuming a reward, so the lick rate is briefly elevated.
    state.lick_boost_samples = _LICK_BOOST_SAMPLES


def _fire_puff(visualizer: BehaviorVisualizer, state: _SimulationState) -> None:
    """Renders an air puff delivery and opens the gas valve for a brief pulse."""
    visualizer.add_puff_event()
    state.puff_pulse_active = True
    state.puff_pulse_timer.reset()


def _advance_behavior(visualizer: BehaviorVisualizer, state: _SimulationState) -> None:
    """Advances the synthetic running speed and lick stream by one sample and emits spontaneous deliveries."""
    # Running speed is only displayed in the RUN_TRAINING and EXPERIMENT layouts.
    if state.mode in (VisualizerMode.RUN_TRAINING, VisualizerMode.EXPERIMENT):
        state.running_speed += float(state.rng.normal(loc=0.0, scale=_SPEED_STEP_SCALE))
        state.running_speed = float(np.clip(state.running_speed, 0.0, _MAXIMUM_RUNNING_SPEED))
        visualizer.update_running_speed(running_speed=np.float64(state.running_speed))

    # Licks and water rewards are present in every display mode.
    lick_probability = _LICK_PROBABILITY
    if state.lick_boost_samples > 0:
        lick_probability = _LICK_BOOST_PROBABILITY
        state.lick_boost_samples -= 1
    if state.rng.random() < lick_probability:
        visualizer.add_lick_event()
    if state.rng.random() < _SPONTANEOUS_REWARD_PROBABILITY:
        _fire_reward(visualizer=visualizer, state=state)

    # The air puff valve only exists in the EXPERIMENT layout with aversive trials enabled.
    if (
        state.mode == VisualizerMode.EXPERIMENT
        and state.has_aversive_trials
        and state.rng.random() < _SPONTANEOUS_PUFF_PROBABILITY
    ):
        _fire_puff(visualizer=visualizer, state=state)


def _record_trial(visualizer: BehaviorVisualizer, ui: RuntimeControlUI, state: _SimulationState) -> None:
    """Records a random trial outcome, honoring the enabled trial types and the guidance toggles in the GUI."""
    # Chooses the trial type from those enabled for the simulated session.
    if state.has_reinforcing_trials and state.has_aversive_trials:
        is_aversive = bool(state.rng.random() < _AVERSIVE_TRIAL_PROBABILITY)
    else:
        is_aversive = state.has_aversive_trials

    # Reflects the relevant guidance toggle so that enabling guidance in the GUI visibly changes trial outcomes.
    was_guided = ui.enable_aversive_guidance if is_aversive else ui.enable_reinforcing_guidance
    # Guided trials always succeed, mirroring the automatic reward or puff avoidance the guidance modes provide.
    succeeded = True if was_guided else bool(state.rng.random() < _TRIAL_SUCCESS_PROBABILITY)
    visualizer.add_trial_outcome(is_aversive=is_aversive, succeeded=succeeded, was_guided=was_guided)


def _update_thresholds(visualizer: BehaviorVisualizer, ui: RuntimeControlUI, state: _SimulationState) -> None:
    """Pushes the run-training speed and duration thresholds derived from the control-panel modifier spin boxes.

    The update only reaches the visualizer when a modifier value changes, so dragging the speed or duration control
    moves the corresponding threshold line without forcing a redraw on every idle cycle.
    """
    speed_modifier = ui.speed_modifier
    duration_modifier = ui.duration_modifier
    if (
        state.thresholds_pushed
        and speed_modifier == state.last_speed_modifier
        and duration_modifier == state.last_duration_modifier
    ):
        return

    state.last_speed_modifier = speed_modifier
    state.last_duration_modifier = duration_modifier
    state.thresholds_pushed = True

    speed_threshold = float(
        np.clip(_BASE_SPEED_THRESHOLD_CM_S + speed_modifier * _SPEED_MODIFIER_SCALE, 0.0, _MAXIMUM_RUNNING_SPEED)
    )
    duration_threshold_ms = max(0.0, _BASE_DURATION_THRESHOLD_MS + duration_modifier * _DURATION_MODIFIER_SCALE)
    visualizer.update_run_training_thresholds(
        speed_threshold=np.float64(speed_threshold), duration_threshold=np.float64(duration_threshold_ms)
    )


def _process_control_signals(ui: RuntimeControlUI, visualizer: BehaviorVisualizer, state: _SimulationState) -> None:
    """Reflects the latest control-panel button signals into the simulation and the visualizer."""
    if ui.reward_signal:
        _fire_reward(visualizer=visualizer, state=state)
    if ui.gas_valve_puff_signal:
        _fire_puff(visualizer=visualizer, state=state)
    if ui.open_valve:
        state.valve_held_open = True
    if ui.close_valve:
        state.valve_held_open = False
    if ui.gas_valve_open_signal:
        state.gas_held_open = True
    if ui.gas_valve_close_signal:
        state.gas_held_open = False


def _sync_valve_states(
    valve_tracker: SharedMemoryArray, puff_tracker: SharedMemoryArray, state: _SimulationState
) -> None:
    """Expires active delivery pulses and writes the resulting valve states to the shared trackers."""
    if state.reward_pulse_active and state.reward_pulse_timer.elapsed >= _VALVE_PULSE_MS:
        state.reward_pulse_active = False
    if state.puff_pulse_active and state.puff_pulse_timer.elapsed >= _PUFF_PULSE_MS:
        state.puff_pulse_active = False

    # Index 2 of the valve tracker and index 1 of the puff tracker hold the open and close state read by the GUI.
    valve_tracker[2] = 1 if (state.valve_held_open or state.reward_pulse_active) else 0
    puff_tracker[1] = 1 if (state.gas_held_open or state.puff_pulse_active) else 0


def main() -> None:
    """Runs the paired control-panel and behavior-visualizer simulation for the configured runtime mode."""
    # The EXPERIMENT layout requires at least one trial type to render a meaningful trial performance panel.
    if _MODE == VisualizerMode.EXPERIMENT and not (_HAS_REINFORCING_TRIALS or _HAS_AVERSIVE_TRIALS):
        message = (
            "Unable to simulate the experiment runtime. At least one of the reinforcing or aversive trial types must "
            "be enabled, but both are disabled in the simulation configuration."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Starting the Mesoscope-VR live simulation in {_MODE.name} mode.")

    valve_tracker, puff_tracker = _create_trackers()
    ui = RuntimeControlUI(valve_tracker=valve_tracker, gas_puff_tracker=puff_tracker)
    visualizer = BehaviorVisualizer()
    state = _SimulationState(
        rng=np.random.default_rng(),
        mode=_MODE,
        has_reinforcing_trials=_HAS_REINFORCING_TRIALS,
        has_aversive_trials=_HAS_AVERSIVE_TRIALS,
    )

    data_timer = _new_millisecond_timer()
    trial_timer = _new_millisecond_timer()
    loop_timer = _new_millisecond_timer()

    # Starts the control panel before opening the visualizer. The GUI runs in a forked child process, so it must be
    # spawned before the main process initializes the Qt-backed visualizer, mirroring the real acquisition runtime.
    ui.start(mode=_MODE, has_reinforcing_trials=_HAS_REINFORCING_TRIALS, has_aversive_trials=_HAS_AVERSIVE_TRIALS)
    visualizer.open(
        mode=_MODE, has_reinforcing_trials=_HAS_REINFORCING_TRIALS, has_aversive_trials=_HAS_AVERSIVE_TRIALS
    )

    # Run training displays speed and duration threshold lines, so the simulation seeds them before the loop starts.
    if _MODE == VisualizerMode.RUN_TRAINING:
        _update_thresholds(visualizer=visualizer, ui=ui, state=state)

    console.echo(message="Click 'Resume Runtime' in the control panel to start the synthetic data stream.")

    try:
        while True:
            _process_control_signals(ui=ui, visualizer=visualizer, state=state)
            if ui.exit_signal:
                break

            # Run-training thresholds track the GUI modifier controls even while paused, so the threshold lines move
            # as the user adjusts the speed and duration spin boxes.
            if _MODE == VisualizerMode.RUN_TRAINING:
                _update_thresholds(visualizer=visualizer, ui=ui, state=state)

            # Advances the synthetic behavior only while the runtime is unpaused, mirroring the real system.
            if not ui.pause_runtime:
                if data_timer.elapsed >= _DATA_INTERVAL_MS:
                    data_timer.reset()
                    _advance_behavior(visualizer=visualizer, state=state)
                if _MODE == VisualizerMode.EXPERIMENT and trial_timer.elapsed >= _TRIAL_INTERVAL_MS:
                    trial_timer.reset()
                    _record_trial(visualizer=visualizer, ui=ui, state=state)

            _sync_valve_states(valve_tracker=valve_tracker, puff_tracker=puff_tracker, state=state)

            try:
                visualizer.update()
            except Exception:
                # The visualizer window was closed, so the simulation ends.
                break

            loop_timer.delay(delay=_LOOP_DELAY_MS, allow_sleep=True)
    except KeyboardInterrupt:
        console.echo(message="Received a keyboard interrupt. Shutting down the simulation.")
    finally:
        visualizer.close()
        ui.shutdown()
        # The control panel does not own the trackers, so the simulation releases their shared memory buffers here.
        for tracker in (valve_tracker, puff_tracker):
            tracker.disconnect()
            tracker.destroy()
        console.echo(message="Simulation finished.")


if __name__ == "__main__":
    main()
