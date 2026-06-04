"""Provides the shared runtime-state types and the hardware setup, teardown, and snapshot helpers used by the
Mesoscope-VR data acquisition runtime.
"""

from __future__ import annotations

from enum import IntEnum
import math
import atexit
import shutil
from typing import TYPE_CHECKING
from dataclasses import field, fields, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    SessionData,
    GasPuffTrial,
    SessionTypes,
    WaterRewardTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
)

from .system import MesoscopeData, MesoscopePositions
from .runtime_ui import collect_surgery_quality, collect_experimenter_notes
from ..cross_system import request_text, wait_for_enter, request_confirmation

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .binding_classes import ZaberMotors
    from .mesoscope_driver import MesoscopeDriver


RESPONSE_DELAY: int = 2000
"""Specifies the number of milliseconds to delay showing the response prompt after showing a message that requires
user interaction."""


class _ResponseDelayTimer:
    """Owns the shared PrecisionTimer used to pace the rendering of terminal outputs during runtime.

    Notes:
        The timer is wrapped in this holder, rather than stored directly as a module constant, so that its underlying
        nanobind-bound C++ object can be released at interpreter shutdown. If that object is still referenced when the
        ataraxis_time extension is finalized, nanobind prints a spurious 'leaked instance' warning to the terminal.
        Since every runtime module shares this single holder by reference, the holder owns the only reference to the
        C++ timer, so releasing it here frees the object before the extension teardown check runs.
    """

    def __init__(self) -> None:
        self._timer: PrecisionTimer | None = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        atexit.register(self._release)

    def _release(self) -> None:
        """Drops the wrapped PrecisionTimer so its C++ object is freed before the ataraxis_time extension teardown."""
        self._timer = None

    def delay(self, delay: int, *, allow_sleep: bool = False, block: bool = False) -> None:
        """Delays for the requested number of milliseconds, forwarding to the wrapped PrecisionTimer."""
        if self._timer is None:
            return
        self._timer.delay(delay=delay, allow_sleep=allow_sleep, block=block)

    def reset(self) -> None:
        """Resets the reference point of the wrapped PrecisionTimer to the current time."""
        if self._timer is None:
            return
        self._timer.reset()

    @property
    def elapsed(self) -> int:
        """Returns the number of milliseconds elapsed since the last reset of the wrapped PrecisionTimer."""
        if self._timer is None:
            return 0
        return self._timer.elapsed


RESPONSE_DELAY_TIMER: _ResponseDelayTimer = _ResponseDelayTimer()
"""The shared timer used to pace the rendering of terminal outputs that require user interaction during runtime."""


class MesoscopeVRLogMessageCodes(IntEnum):
    """Defines the set of codes used by the Mesoscope-VR data acquisition to specify the ongoing events when logging
    the system data acquired during runtime.
    """

    SYSTEM_STATE = 1
    """The system has changed its (configuration) state."""
    RUNTIME_STATE = 2
    """The acquired session has changed its (runtime) state."""
    REINFORCING_GUIDANCE_STATE = 3
    """The system has changed the reinforcing (water reward) trial guidance state."""
    AVERSIVE_GUIDANCE_STATE = 4
    """The system has changed the aversive (gas puff) trial guidance state."""
    DISTANCE_SNAPSHOT = 5
    """The system has taken a snapshot of the total distance traveled by the animal at the time Unity signaled runtime
    termination (emergency pause)."""


@dataclass(slots=True)
class TrialState:
    """Tracks the state of the Mesoscope-VR-acquired session's task trials.

    This dataclass consolidates all trial-related state tracking attributes used during experiment runtimes to
    monitor trial progression, manage task guidance modes, and determine stimulus delivery conditions. Supports both
    reinforcing (water reward) and aversive (gas puff) trial types.
    """

    # Overall trial tracking.
    completed: int = 0
    """The total number of trials completed by the animal since the last cue sequence reset or runtime onset."""
    distances: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    """Stores the total cumulative distance, in centimeters, the animals would travel at the end of each trial."""

    # Reinforcing (water reward) trial tracking.
    reinforcing_guided_trials: int = 0
    """The remaining number of reinforcing trials for which to maintain the lick guidance mode."""
    reinforcing_failed_trials: int = 0
    """The number of consecutive reinforcing trials for which the animal did not receive a water reward."""
    reinforcing_recovery_threshold: int = 0
    """The number of consecutively failed reinforcing trials after which to engage recovery guidance mode."""
    reinforcing_recovery_trials: int = 0
    """The number of guided reinforcing trials to use when recovery mode is triggered."""
    reinforcing_rewarded: bool = False
    """Tracks whether the current reinforcing trial has been rewarded."""
    reinforcing_rewards: tuple[tuple[float, int], ...] = ((0.0, 0),)
    """Stores the reward size (volume in μL) and tone duration (ms) for each trial, with 0 for trials of the other
    type."""

    # Aversive (gas puff) trial tracking.
    aversive_guided_trials: int = 0
    """The remaining number of aversive trials for which to maintain the occupancy guidance mode."""
    aversive_failed_trials: int = 0
    """The number of consecutive aversive trials for which the animal failed to meet occupancy requirements."""
    aversive_recovery_threshold: int = 0
    """The number of consecutively failed aversive trials after which to engage recovery guidance mode."""
    aversive_recovery_trials: int = 0
    """The number of guided aversive trials to use when recovery mode is triggered."""
    aversive_succeeded: bool = False
    """Tracks whether the animal met the occupancy requirement for the current aversive trial."""
    aversive_puff_durations: tuple[int, ...] = (100,)
    """Stores the gas puff duration (ms) for each trial, with 0 for trials of the other type."""

    # Trial structure configuration.
    trial_structures: dict[str, WaterRewardTrial | GasPuffTrial] = field(default_factory=dict)
    """Maps trial structure names to their configuration objects."""

    def trial_completed(self, traveled_distance: float) -> bool:
        """Determines whether the current trial is complete based on the total distance traveled by the animal.

        Args:
            traveled_distance: The total cumulative distance, in centimeters, traveled by the animal since the last
                cue sequence reset or runtime onset.

        Returns:
            True if the animal has traveled beyond the current trial's distance threshold, False otherwise. Returns
            False if all trials have been completed.
        """
        if self.completed >= len(self.distances):
            return False
        return bool(traveled_distance > self.distances[self.completed])

    def get_current_reward(self) -> tuple[float, int]:
        """Retrieves the reward parameters for the current reinforcing trial.

        Returns:
            A tuple containing the reward size in microliters and the reward tone duration in milliseconds.
        """
        return self.reinforcing_rewards[self.completed]

    def get_current_puff_duration(self) -> int:
        """Retrieves the gas puff duration for the current aversive trial.

        Returns:
            The gas puff duration in milliseconds.
        """
        return self.aversive_puff_durations[self.completed]

    def is_current_trial_aversive(self) -> bool:
        """Determines whether the current trial is an aversive (gas puff) trial from its nonzero per-trial puff
        duration.

        Notes:
            The accessor indexes the per-trial puff duration array at the current trial position, so it is only valid
            while trial_completed() returns False.

        Returns:
            True if the current trial stores a nonzero gas puff duration, False otherwise.
        """
        return self.aversive_puff_durations[self.completed] > 0

    def advance_trial(self) -> int:
        """Advances the trial tracking state to the next trial.

        Returns:
            The updated count of consecutively failed trials for the current trial type.
        """
        # Captures trial type BEFORE incrementing to update the correct failure counters.
        is_aversive = self.is_current_trial_aversive()
        self.completed += 1

        if is_aversive:
            # Aversive trial: success = met occupancy requirement (no puff delivered).
            if not self.aversive_succeeded:
                self.aversive_failed_trials += 1
            else:
                self.aversive_failed_trials = 0
            self.aversive_succeeded = False
            return self.aversive_failed_trials
        # Reinforcing trial: success = received water reward.
        if not self.reinforcing_rewarded:
            self.reinforcing_failed_trials += 1
        else:
            self.reinforcing_failed_trials = 0
        self.reinforcing_rewarded = False
        return self.reinforcing_failed_trials


def generate_mesoscope_position_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, mesoscope_driver: MesoscopeDriver
) -> None:
    """Queries the current Mesoscope imaging position from the ScanImagePC and writes it as a mesoscope_positions.yaml
    file to the session and the animal's persistent directories.

    Notes:
        Most position fields are queried directly from the ScanImage software over MQTT, so the mesoscope control
        driver must still be connected and idle when this function runs. The red-dot alignment position is the only
        field that cannot be queried, so it is entered manually, defaulting to the previous runtime's value.

    Args:
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
        mesoscope_driver: The MesoscopeDriver instance used to query the Mesoscope state over MQTT.
    """
    # If the session was not fully initialized (nk.bin marker exists), skips the snapshot generation.
    if session_data.raw_data.nk_path.exists():
        return

    # Loads the previous runtime's red-dot alignment position, if available, to offer as the default. This is the only
    # position field that cannot be queried from the ScanImage software.
    previous_red_dot_alignment_z = 0.0
    if mesoscope_data.vrpc_data.mesoscope_positions_path.exists():
        previous_positions = MesoscopePositions.from_yaml(file_path=mesoscope_data.vrpc_data.mesoscope_positions_path)
        previous_red_dot_alignment_z = previous_positions.red_dot_alignment_z

    # Queries the live mesoscope positions from the ScanImagePC, then fills in the red-dot alignment Z position from
    # operator input, as it is the only field the ScanImage software cannot report.
    mesoscope_positions = mesoscope_driver.query_state()
    mesoscope_positions.red_dot_alignment_z = _prompt_red_dot_alignment(previous_value=previous_red_dot_alignment_z)

    # Rounds every position down to at most three decimal places, discarding the spurious sub-micrometer and
    # sub-millidegree precision reported by the ScanImage software before the snapshot is persisted.
    for position_field in fields(mesoscope_positions):
        rounded_value = _floor_to_three_decimals(value=getattr(mesoscope_positions, position_field.name))
        setattr(mesoscope_positions, position_field.name, rounded_value)

    # Writes the snapshot to the session's raw_data directory and to the animal's persistent directory, overwriting any
    # existing persistent file so it can seed the next runtime.
    mesoscope_positions.to_yaml(file_path=session_data.system_raw_data.mesoscope_positions_path)
    mesoscope_positions.to_yaml(file_path=mesoscope_data.vrpc_data.mesoscope_positions_path)

    console.echo(message="Mesoscope positions: Saved.", level=LogLevel.SUCCESS)


def generate_zaber_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, zaber_motors: ZaberMotors
) -> None:
    """Creates a snapshot of the current Zaber motor positions and saves it as a zaber_positions.yaml file.

    Args:
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
        zaber_motors: The ZaberMotors instance that manages the Zaber assets used by the session for which the
            snapshot is generated.
    """
    # If at least one of the managed motor groups is not connected, does not run the snapshot generation sequence.
    # Also, if the session failed to properly initialize, as marked by the presence of the nk.bin marker.
    if not zaber_motors.is_connected or session_data.raw_data.nk_path.exists():
        return

    zaber_positions = zaber_motors.generate_position_snapshot()

    # Saves the newly generated file both to the persistent directory and to the session directory. Note, saving to the
    # persistent data directory automatically overwrites any existing position file.
    zaber_positions.to_yaml(file_path=mesoscope_data.vrpc_data.zaber_positions_path)
    zaber_positions.to_yaml(file_path=session_data.system_raw_data.zaber_positions_path)

    message = "Zaber motor positions: Saved."
    console.echo(message=message, level=LogLevel.SUCCESS)


def setup_zaber_motors(zaber_motors: ZaberMotors) -> None:
    """If necessary, carries out the Zaber motor setup and positioning sequence.

    Args:
        zaber_motors: The ZaberMotors instance that manages the Zaber motors used during runtime.
    """
    # Determines whether to carry out the Zaber motor positioning sequence.
    message = (
        "Do you want to carry out the Zaber motor setup sequence for this runtime? Only enter 'no' if the animal is "
        "already positioned inside the Mesoscope enclosure."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

    # Blocks until the operator confirms or declines the Zaber motor setup sequence.
    if not request_confirmation(message="Carry out the Zaber motor setup sequence?", default=False):
        # Aborts method runtime, as no further Zaber setup is required.
        return

    # Since it is now possible to shut down Zaber motors without fixing HeadBarRoll position, requests the user
    # to verify this manually.
    message = (
        "Check that the HeadBarRoll motor has a positive (>0) angle. If the angle is negative (<0), the motor will "
        "collide with the stopper during homing, which will DAMAGE the motor."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is
    # safe to proceed with motor movements.
    message = (
        "Preparing to move Zaber motors into mounting position. Remove the mesoscope objective, swivel out the "
        "VR screens, and make sure the animal is NOT mounted in the Mesoscope's enclosure."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Homes all managed motors in parallel.
    zaber_motors.prepare_motors()

    # Moves all motors to the animal mounting position.
    zaber_motors.mount_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Gives the user time to mount the animal and requires confirmation before proceeding further.
    message = (
        "Preparing to move the motors into the imaging position. Mount the animal onto the VR rig. Do NOT "
        "adjust any motors manually at this time. Do NOT install the mesoscope objective."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Restores all motors to the positions used during the previous session's runtime.
    zaber_motors.restore_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def run_shutdown_step(description: str, step: Callable[[], None]) -> None:
    """Executes a single shutdown callable, isolating it so that an error or interrupt does not propagate.

    The Mesoscope-VR shutdown sequences tear down several subprocess-backed assets in turn. Allowing an exception or a
    repeated KeyboardInterrupt from one asset to propagate would skip the remaining teardown steps and leave the
    orphaned subprocesses to be collected by the garbage collector, which tears down their shared-memory managers out
    of order and cascades into multiprocessing errors. This helper contains each failure so the remaining steps still
    run, while the originally propagating exception (if any) resumes once the shutdown sequence completes.

    Args:
        description: A short gerund phrase naming the step, used to contextualize an error encountered while running it.
        step: The zero-argument callable that performs the shutdown step.
    """
    try:
        step()
    except (Exception, KeyboardInterrupt) as error:
        message = (
            f"Encountered an error while {description} during the Mesoscope-VR shutdown sequence: {error!r}. "
            f"Continuing with the remaining shutdown steps."
        )
        console.echo(message=message, level=LogLevel.ERROR)


def reset_zaber_motors(zaber_motors: ZaberMotors) -> None:
    """If necessary, carries out the Zaber motor parking and shutdown sequence.

    Args:
        zaber_motors: The ZaberMotors instance that manages the Zaber motors used during runtime.
    """
    # If at least one of the managed motor groups is not connected, does not run the reset sequence.
    if not zaber_motors.is_connected:
        return

    # Determines whether to carry out the Zaber motor shutdown sequence.
    message = (
        "Do you want to carry out Zaber motor shutdown sequence? If ending a successful runtime, enter 'yes'. If "
        "terminating a failed runtime to restart it, enter 'no'. Note! Entering 'yes' does NOT move any motors."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

    # Blocks until the operator confirms or declines the Zaber motor shutdown sequence.
    if not request_confirmation(message="Carry out the Zaber motor shutdown sequence?", default=False):
        # Disconnects from Zaber motors. This does not change motor positions but does lock (park) all motors
        # before disconnecting.
        zaber_motors.disconnect()
        return

    # Helps with removing the animal from the enclosure by retracting the LickPort in the Y-axis (moving it away
    # from the animal).
    message = "Retracting the lick-port away from the animal..."
    console.echo(message=message, level=LogLevel.INFO)
    zaber_motors.unmount_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = "Uninstall the mesoscope objective and REMOVE the animal from the Mesoscope's enclosure."
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Moves all motors to the hardcoded parking positions.
    zaber_motors.park_position()

    # Disconnects from Zaber motors. This does not change motor positions but does lock (park) all motors
    # before disconnecting.
    zaber_motors.disconnect()

    message = "Zaber motors: Reset."
    console.echo(message=message, level=LogLevel.SUCCESS)


def setup_mesoscope(
    session_data: SessionData, mesoscope_data: MesoscopeData, mesoscope_driver: MesoscopeDriver
) -> None:
    """Guides the user through the sequence of steps that prepares the Mesoscope for the data acquisition runtime.

    Notes:
        The mesoscope is controlled over MQTT. After the ScanImagePC reports that the runAcquisition function has
        connected, this function preloads the persisted reference estimator as an alignment aid, guides the user
        through mounting and alignment, and commands the reference generation once the alignment screenshot appears.

    Args:
        session_data: The SessionData instance that defines the session for which the Mesoscope is being prepared.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
        mesoscope_driver: The MesoscopeDriver instance used to command the ScanImage software over MQTT.
    """
    # Determines whether the acquired session is a Window Checking session.
    window_checking: bool = session_data.session_type == SessionTypes.WINDOW_CHECKING

    # Step 0: Clears out the mesoscope_data directory.
    # Ensures that the mesoscope_data directory is reset before running the mesoscope's preparation sequence. To
    # minimize the risk of important data loss, this procedure now requires the user to remove the files manually.
    while True:
        existing_files = list(mesoscope_data.scanimagepc_data.mesoscope_data_path.glob("*"))

        if not existing_files:
            break

        message = (
            f"Unable to prepare the Mesoscope for the data acquisition runtime. The preparation requires the shared "
            f"'mesoscope_data' ScanImagePC directory to be empty, but the directory contains the following unexpected "
            f"files: {','.join(file.name for file in existing_files)}. Clear the directory from all existing files "
            f"before proceeding."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
        wait_for_enter(message="Press Enter to continue.")

    # Waits for the ScanImage control interface to come online, then preloads the persisted reference estimator (if one
    # exists for the animal) as an alignment aid. The estimator path is local to the ScanImagePC filesystem, so the
    # VRPC sends only the project and animal identifiers and the ScanImagePC resolves the path under its own Mesoscope
    # data root. Automatic motion correction stays disabled so the user aligns the mesoscope manually during the next
    # step.
    mesoscope_driver.await_alive()
    mesoscope_driver.preload(project=session_data.project_name, animal=session_data.animal_id)

    # Step 1: Resolves the imaging plane.
    # If the previous session's mesoscope positions were saved, loads the imaging coordinates and displays them to the
    # user.
    if not window_checking and mesoscope_data.vrpc_data.mesoscope_positions_path.exists():
        previous_positions: MesoscopePositions = MesoscopePositions.from_yaml(
            file_path=mesoscope_data.vrpc_data.mesoscope_positions_path,
        )
        message = (
            f"Follow the steps of the mesoscope preparation protocol available from the sl-protocols repository. "
            f"Previous mesoscope coordinates were: x={previous_positions.mesoscope_x}, "
            f"y={previous_positions.mesoscope_y}, roll={previous_positions.mesoscope_roll}, "
            f"z={previous_positions.mesoscope_z}, fast_z={previous_positions.mesoscope_fast_z}, "
            f"tip={previous_positions.mesoscope_tip}, tilt={previous_positions.mesoscope_tilt}, "
            f"laser_power={previous_positions.laser_power_mw}, "
            f"red_dot_alignment_z={previous_positions.red_dot_alignment_z}."
        )
    elif not window_checking:
        message = (
            f"No previous mesoscope imaging position data found for the animal {session_data.animal_id}. Follow the "
            f"steps of the window checking protocol available from the sl-protocols repository to establish the "
            f"imaging plane for the animal."
        )
    else:
        message = (
            "Follow the steps of the window checking protocol available from the sl-protocols repository to establish "
            "the imaging plane for the animal."
        )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Step 2: Generates the screenshot of the red-dot alignment and the cranial window.
    message = (
        "Generate the screenshot of the red-dot alignment, the imaging plane state (cell activity), and the "
        "ScanImage acquisition parameters by pressing the 'Win + PrtSc' combination."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    # Ensures that the screenshot is created before proceeding further.
    while True:
        screenshots = list(mesoscope_data.scanimagepc_data.mesoscope_root_path.glob("*.png"))

        if len(screenshots) == 1:
            break

        message = (
            f"Unable to retrieve the screenshot from the ScanImage PC. Expected a single .png file inside the "
            f"'mesodata' ScanImagePC directory, but instead found {len(screenshots)} candidate files. Ensure that the "
            f"directory only stores the .png screenshot generated during the previous preparation step."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
        wait_for_enter(message="Press Enter to continue.")

    # Transfers the screenshot to the session's raw_data directory (window_screenshot.png).
    screenshot_path = session_data.system_raw_data.window_screenshot_path

    # Moves the screenshot from the ScanImagePC to the VRPC.
    shutil.move(src=screenshots.pop(), dst=screenshot_path)

    # Copies the screenshot to the animal's persistent data directory so that it can be reused during the next
    # runtime.
    shutil.copy2(src=screenshot_path, dst=mesoscope_data.vrpc_data.window_screenshot_path)

    # Window checking sessions require special handling.
    if window_checking:
        # Since window checking may reveal that the evaluated animal is not fit for participating in experiments,
        # optionally allows aborting the runtime early for window checking sessions.
        message = "Do you want to generate the ROI and MotionEstimator snapshots for this animal?"
        console.echo(message=message, level=LogLevel.INFO)
        RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

        # Blocks until the operator confirms or declines generating the metadata snapshots.
        if not request_confirmation(
            message="Generate the ROI and MotionEstimator snapshots for this animal?", default=False
        ):
            # Aborts the runtime if the user does not intend to generate the ROI and MotionEstimator data.
            console.echo(message="Mesoscope preparation: Complete.", level=LogLevel.SUCCESS)
            return

    # Step 3: Commands the ScanImagePC to generate the new session estimator and high-definition z-stack and arm the
    # mesoscope for acquisition. The alignment screenshot detected above gates this lengthy preparation step.

    # Verifies the ScanImage imaging parameters before the lengthy reference generation. The runAcquisition function no
    # longer blocks on this confirmation once launched, so it is surfaced here, immediately before the
    # reference-generation command is dispatched.
    message = (
        "Ensure the following ScanImage imaging parameters are applied before generating the reference: the laser is "
        "enabled and its power is set, the ROI frame rate is ~10 Hz, the scan phase is ~0.8888, and PMT AutoOn is "
        "enabled."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue.")

    mesoscope_driver.generate_reference()

    # Window checking sessions only need the generated reference files, so they release the mesoscope without acquiring
    # any session frames.
    if window_checking:
        mesoscope_driver.abort()

    # The reference generation produces 3 files: MotionEstimator.me, fov.roi, and zstack.tiff.
    target_files = (
        mesoscope_data.scanimagepc_data.mesoscope_data_path.joinpath("MotionEstimator.me"),
        mesoscope_data.scanimagepc_data.mesoscope_data_path.joinpath("fov.roi"),
        mesoscope_data.scanimagepc_data.mesoscope_data_path.joinpath("zstack.tiff"),
    )

    # Waits until the necessary files are generated on the ScanImagePC.
    while True:
        missing_files = [file for file in target_files if not file.exists()]

        if not missing_files:
            break

        missing_names = ", ".join(file.name for file in missing_files)

        message = (
            f"Unable to confirm that the ScanImagePC has generated the required acquisition data files, as the "
            f"following expected files are missing from the 'mesoscope_data' directory: {missing_names}. Ensure the "
            f"runAcquisition function is running on the ScanImagePC and retry."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
        wait_for_enter(message="Press Enter to continue.")

    console.echo(message="Mesoscope preparation: Complete.", level=LogLevel.SUCCESS)


def finalize_session_descriptor(
    descriptor: MesoscopeExperimentDescriptor
    | LickTrainingDescriptor
    | RunTrainingDescriptor
    | WindowCheckingDescriptor,
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
) -> None:
    """Collects the supervising experimenter's session notes, writes the completed descriptor to the session's
    raw_data directory, and caches a copy to the animal's persistent directory.

    The notes are entered through a blocking terminal prompt instead of by manually editing the session_descriptor.yaml
    file, removing the filesystem round-trip previously required to annotate each session. For window checking
    sessions, the experimenter is additionally prompted for the cranial window quality rating, which is otherwise left
    at its default value.

    Args:
        descriptor: The session_descriptor.yaml-convertible instance to complete and cache to the acquired session's
            data directory.
        session_data: The SessionData instance that defines the session for which the descriptor file is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # Window checking sessions additionally capture the experimenter's cranial window quality rating on a 0-3 scale.
    # The rating is propagated to the surgery log Google Sheet during preprocessing; the other session types do not
    # track a window quality rating.
    if isinstance(descriptor, WindowCheckingDescriptor):
        descriptor.surgery_quality = collect_surgery_quality(session_name=session_data.session_name)

    # Collects the experimenter notes through a blocking terminal prompt and stores them inside the descriptor. The
    # runtime control UI is already shut down at this point, so the prompt runs on the main thread without competing
    # GUIs.
    descriptor.experimenter_notes = collect_experimenter_notes(session_name=session_data.session_name)

    # Saves the completed descriptor as a .yaml file inside the session's raw_data directory.
    descriptor.to_yaml(file_path=session_data.raw_data.session_descriptor_path)
    console.echo(message="Session descriptor file: Created.", level=LogLevel.SUCCESS)

    # Copies the descriptor to the animal's persistent directory. This is primarily used during training to restore
    # the training parameters between training sessions of the same type.
    shutil.copy2(
        src=session_data.raw_data.session_descriptor_path,
        dst=mesoscope_data.vrpc_data.session_descriptor_path,
    )


def _prompt_red_dot_alignment(previous_value: float) -> float:
    """Prompts the operator for the red-dot alignment Z position, defaulting to the currently stored value.

    Notes:
        The red-dot alignment Z position is the only Mesoscope position field that cannot be queried from the
        ScanImage software, so it is entered manually. Submitting an empty response keeps the stored value.

    Args:
        previous_value: The currently stored red-dot alignment Z position, offered as the default.

    Returns:
        The red-dot alignment Z position to record, in micrometers.
    """
    message = (
        f"Enter the red-dot alignment Z position, in micrometers, used during this runtime. The currently stored "
        f"value is {previous_value}. Leave the response empty to keep the stored value."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

    response: str = request_text(
        message="Enter the red-dot alignment Z position, in micrometers:",
        validate=_validate_red_dot_response,
    )
    if not response.strip():
        return previous_value
    return float(response)


def _validate_red_dot_response(response: str) -> bool | str:
    """Validates a red-dot alignment Z position response, accepting a number or an empty response.

    Args:
        response: The raw text entered by the operator.

    Returns:
        True when the response is empty or parses as a number, or an error message describing the constraint.
    """
    if not response.strip():
        return True
    try:
        float(response)
    except ValueError:
        return "Enter a numeric value or leave the response empty to keep the stored value."
    return True


def _floor_to_three_decimals(value: float) -> float:
    """Rounds the given value down to at most three decimal places.

    Notes:
        The value is floored toward negative infinity rather than truncated toward zero, so negative inputs round
        down to the next lower three-decimal value.

    Args:
        value: The floating-point position value to round down.

    Returns:
        The value rounded down to at most three decimal places.
    """
    return math.floor(value * 1000.0) / 1000.0
