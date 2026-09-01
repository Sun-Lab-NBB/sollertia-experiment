"""Provides the shared runtime-state types and the hardware setup, teardown, and snapshot helpers used by the
Mesoscope-VR data acquisition runtime.
"""

from __future__ import annotations

import os
import sys
from enum import IntEnum
import math
import atexit
import shutil
from typing import TYPE_CHECKING
from decimal import ROUND_FLOOR, Decimal
from dataclasses import field, fields, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    SessionData,
    SessionTypes,
    MesoscopeGasPuffTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeWaterRewardTrial,
    MesoscopeExperimentDescriptor,
)

from .system import MesoscopeData, MesoscopePositions
from .runtime_ui import collect_surgery_quality, collect_experimenter_notes, collect_experimenter_given_water_volume
from ..cross_system import (
    request_text,
    wait_for_enter,
    request_confirmation,
    request_required_confirmation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .binding_classes import ZaberMotors
    from .mesoscope_driver import MesoscopeDriver


RESPONSE_DELAY: int = 500
"""Specifies the number of milliseconds to delay showing the response prompt after showing a message that requires
user interaction."""
_DEFAULT_TOTAL_WATER_VOLUME_ML: float = 1.0
"""The total session water volume, in milliliters, offered as the fallback default when the experimenter is prompted for
the amount of water the animal should receive at session teardown and no prior session recorded a water intake."""
_THREE_DECIMAL_QUANTUM: Decimal = Decimal("0.001")
"""The decimal grid step the Mesoscope position values are floored onto before they are written to the position
snapshot."""
_REFERENCE_WRITE_PERMISSION_BITS: int = 0o666
"""The permission bits requested for a staged reference file, before the process umask narrows them.

Notes:
    Matches the bits ataraxis-data-structures requests in atomic_write(), so a reference file published here carries
    the same permissions as every other file the platform writes from scratch, rather than inheriting whatever mode
    the ScanImagePC copy happened to carry.
"""
_REFERENCE_RENAME_RETRY_COUNT: int = 5
"""The maximum number of times publishing a staged reference file is attempted before the failure propagates."""
_REFERENCE_RENAME_RETRY_DELAY_MILLISECONDS: int = 500
"""The delay in milliseconds before the second attempt to publish a staged reference file.

Notes:
    The delay doubles on each further attempt, matching the back-off ataraxis-data-structures applies in
    atomic_write(). Windows refuses the rename while a scanner holds the destination open, and the holder releases it
    on its own after a duration this module cannot predict.
"""


class _ResponseDelayTimer:
    """Owns the shared PrecisionTimer used to pace the rendering of terminal outputs during runtime.

    Notes:
        The timer is wrapped in this holder, rather than stored directly as a module constant, so that its underlying
        nanobind-bound C++ object can be released at interpreter shutdown. If that object is still referenced when the
        ataraxis_time extension is finalized, nanobind prints a spurious 'leaked instance' warning to the terminal.
        Since every runtime module shares this single holder by reference, the holder owns the only reference to the
        C++ timer, so releasing it here frees the object before the extension teardown check runs.

    Attributes:
        _timer: The PrecisionTimer instance used to pace the terminal output rendering, or None once it has been
            released at interpreter shutdown.
    """

    def __init__(self) -> None:
        self._timer: PrecisionTimer | None = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        atexit.register(self._release)

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

    def _release(self) -> None:
        """Drops the wrapped PrecisionTimer so its C++ object is freed before the ataraxis_time extension teardown."""
        self._timer = None


RESPONSE_DELAY_TIMER: _ResponseDelayTimer = _ResponseDelayTimer()
"""The shared timer used to pace the rendering of terminal outputs that require user interaction during runtime."""


class MesoscopeVRLogMessageCodes(IntEnum):
    """Defines the set of codes used by the Mesoscope-VR data acquisition system to specify the ongoing events when
    logging the system data acquired during runtime.
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
    MESOSCOPE_ACQUISITION_STATE = 6
    """The system has changed whether it expects the Mesoscope to be acquiring the session's frames. The value is one
    while the ScanImagePC is expected to write session frames and zero otherwise, which brackets the periods during
    which the logged frame acquisition pulses correspond to saved frames."""


@dataclass(slots=True)
class TrialState:
    """Tracks the state of the Mesoscope-VR-acquired session's task trials.

    This dataclass consolidates all trial-related state tracking attributes used during experiment runtimes to
    monitor trial progression, manage task guidance modes, and determine stimulus delivery conditions. Supports both
    reinforcing (water reward) and aversive (gas puff) trial types.
    """

    completed: int = 0
    """The total number of trials completed by the animal since the last cue sequence reset or runtime onset."""
    distances: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    """Stores the total cumulative distance, in centimeters, the animal will have traveled at the end of each trial."""

    reinforcing_guided_trials: int = 0
    """The remaining number of reinforcing trials for which to maintain the lick guidance mode."""
    reinforcing_failed_trials: int = 0
    """The number of consecutive reinforcing trials for which the animal did not receive a water reward."""
    reinforcing_recovery_threshold: int = 0
    """The number of consecutively failed reinforcing trials after which to engage recovery guidance mode."""
    reinforcing_recovery_trials: int = 0
    """The number of guided reinforcing trials to use when recovery mode is triggered."""
    reinforcing_rewarded: bool = False
    """Determines whether the current reinforcing trial has been rewarded."""
    reinforcing_rewards: tuple[tuple[float, int], ...] = ((0.0, 0),)
    """Stores the reward size (volume in μL) and tone duration (ms) for each trial, with 0 for trials of the other
    type."""

    aversive_guided_trials: int = 0
    """The remaining number of aversive trials for which to maintain the occupancy guidance mode."""
    aversive_failed_trials: int = 0
    """The number of consecutive aversive trials for which the animal failed to meet occupancy requirements."""
    aversive_recovery_threshold: int = 0
    """The number of consecutively failed aversive trials after which to engage recovery guidance mode."""
    aversive_recovery_trials: int = 0
    """The number of guided aversive trials to use when recovery mode is triggered."""
    aversive_succeeded: bool = False
    """Determines whether the animal met the occupancy requirement for the current aversive trial."""
    aversive_puff_durations: tuple[int, ...] = (100,)
    """Stores the gas puff duration (ms) for each trial, with 0 for trials of the other type."""

    trial_structures: dict[str, MesoscopeWaterRewardTrial | MesoscopeGasPuffTrial] = field(default_factory=dict)
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

    def is_current_trial_aversive(self) -> bool:
        """Determines whether the current trial is an aversive (gas puff) trial from its nonzero per-trial puff
        duration.

        Notes:
            The accessor indexes the per-trial puff duration array at the current trial position, so it is only valid
            while that position is below the length of the array. A trial_completed() call returning True guarantees
            that bound for the trial it just resolved.

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
            # An aversive trial succeeds when the animal meets the occupancy requirement and no puff is delivered.
            if not self.aversive_succeeded:
                self.aversive_failed_trials += 1
            else:
                self.aversive_failed_trials = 0
            self.aversive_succeeded = False
            return self.aversive_failed_trials
        # A reinforcing trial succeeds when the animal receives a water reward.
        if not self.reinforcing_rewarded:
            self.reinforcing_failed_trials += 1
        else:
            self.reinforcing_failed_trials = 0
        self.reinforcing_rewarded = False
        return self.reinforcing_failed_trials


@dataclass(frozen=True, slots=True)
class _PreviousSessionWaterContext:
    """Stores the animal weight and total water intake recovered from the most recent non-window-checking session."""

    animal_weight_g: float
    """The animal's weight, in grams, recorded at the start of the most recent prior session."""
    received_water_volume_ml: float
    """The total water volume, in milliliters, the animal received during the most recent prior session. This sums the
    runtime-dispensed and experimenter-given volumes and excludes water dispensed while the system was paused."""


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
    if not zaber_motors.is_connected or session_data.raw_data.nk_path.exists():
        return

    zaber_positions = zaber_motors.generate_position_snapshot()

    # Note, saving to the persistent data directory automatically overwrites any existing position file.
    zaber_positions.to_yaml(file_path=mesoscope_data.vrpc_data.zaber_positions_path)
    zaber_positions.to_yaml(file_path=session_data.system_raw_data.zaber_positions_path)

    message = "Zaber motor positions: Saved."
    console.echo(message=message, level=LogLevel.SUCCESS)


def setup_zaber_motors(zaber_motors: ZaberMotors) -> None:
    """If necessary, carries out the Zaber motor setup and positioning sequence.

    Args:
        zaber_motors: The ZaberMotors instance that manages the Zaber motors used during runtime.
    """
    message = (
        "Do you want to carry out the Zaber motor setup sequence for this runtime? Only enter 'no' if the animal is "
        "already positioned inside the Mesoscope enclosure."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

    # Blocks until the operator confirms or declines the Zaber motor setup sequence. The prompt has no default, so an
    # accidental empty Enter cannot silently skip positioning the motors.
    if not request_required_confirmation(message="Carry out the Zaber motor setup sequence?"):
        return

    # Shutting down the Zaber motors does not fix the HeadBarRoll position, so the user verifies the angle manually
    # before homing.
    message = (
        "Check that the HeadBarRoll motor has a positive (>0) angle. If the angle is negative (<0), the motor will "
        "collide with the stopper during homing, which will DAMAGE the motor."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue")

    # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is
    # safe to proceed with motor movements.
    message = (
        "Preparing to move Zaber motors into mounting position. Remove the mesoscope objective, swivel out the "
        "VR screens, and make sure the animal is NOT mounted in the Mesoscope's enclosure."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue")

    zaber_motors.prepare_motors()

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
    wait_for_enter(message="Press Enter to continue")

    zaber_motors.restore_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def reset_zaber_motors(zaber_motors: ZaberMotors) -> None:
    """If necessary, carries out the Zaber motor parking and shutdown sequence.

    Args:
        zaber_motors: The ZaberMotors instance that manages the Zaber motors used during runtime.
    """
    if not zaber_motors.is_connected:
        return

    message = (
        "Do you want to carry out Zaber motor shutdown sequence? If ending a successful runtime, enter 'yes'. If "
        "terminating a failed runtime to restart it, enter 'no'. Note! Entering 'yes' retracts the lick-port and "
        "then moves all motors to their parking positions. Entering 'no' only locks the motors in place."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

    # Blocks until the operator confirms or declines the Zaber motor shutdown sequence. The prompt has no default, so
    # an accidental empty Enter cannot silently choose whether the motors are parked.
    if not request_required_confirmation(message="Carry out the Zaber motor shutdown sequence?"):
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
    wait_for_enter(message="Press Enter to continue")

    zaber_motors.park_position()

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

        When a persisted reference already exists for the animal, the function additionally offers to replace it with
        the snapshot generated during this session, applying the choice once the new reference files are confirmed.

    Args:
        session_data: The SessionData instance that defines the session for which the Mesoscope is being prepared.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
        mesoscope_driver: The MesoscopeDriver instance used to command the ScanImage software over MQTT.
    """
    window_checking: bool = session_data.session_type == SessionTypes.WINDOW_CHECKING

    # Captures whether the operator chooses to replace the animal's persisted reference (MotionEstimator.me and
    # fov.roi) with the snapshot generated this session. Defaults to keeping the existing reference.
    replace_reference: bool = False

    # Step 0: Clears out the mesoscope_data directory.
    # Ensures that the mesoscope_data directory is reset before running the mesoscope's preparation sequence. To
    # minimize the risk of important data loss, this procedure requires the user to remove the files manually.
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
        wait_for_enter(message="Press Enter to continue")

    # Waits for the ScanImage control interface to come online, then preloads the persisted reference estimator (if one
    # exists for the animal) as an alignment aid. The estimator path is local to the ScanImagePC filesystem, so the
    # VRPC sends only the project and animal identifiers and the ScanImagePC resolves the path under its own Mesoscope
    # data root. Automatic motion correction stays disabled so the user aligns the mesoscope manually during the next
    # step.
    mesoscope_driver.await_alive()
    mesoscope_driver.preload(project=session_data.project_name, animal=session_data.animal_id)

    # Step 1: Resolves the imaging plane.
    # If a previous session's mesoscope positions were saved, loads the imaging coordinates and displays them to the
    # user. This applies to every session type, including window checking: whenever a persisted reference estimator
    # exists to preload above, the matching position snapshot also exists, so re-checking a previously imaged animal
    # reveals the prior coordinates the same way experiment sessions do.
    if mesoscope_data.vrpc_data.mesoscope_positions_path.exists():
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
    else:
        message = (
            f"No previous mesoscope imaging position data found for the animal {session_data.animal_id}. Follow the "
            f"steps of the window checking protocol available from the sl-protocols repository to establish the "
            f"imaging plane for the animal."
        )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue")

    # Step 2: Generates the screenshot of the red-dot alignment and the cranial window.
    message = (
        "Generate the screenshot of the red-dot alignment, the imaging plane state (cell activity), and the "
        "ScanImage acquisition parameters by pressing the 'Win + PrtSc' combination."
    )
    console.echo(message=message, level=LogLevel.INFO)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue")

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
        wait_for_enter(message="Press Enter to continue")

    screenshot_path = session_data.system_raw_data.window_screenshot_path

    shutil.move(src=screenshots.pop(), dst=screenshot_path)

    # Copies the screenshot to the animal's persistent data directory so that it can be reused during the next
    # runtime.
    shutil.copy2(src=screenshot_path, dst=mesoscope_data.vrpc_data.window_screenshot_path)

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

    # Verifies the ScanImage imaging parameters before the lengthy reference generation. The runAcquisition function
    # does not block on this confirmation once launched, so it is surfaced here, immediately before the
    # reference-generation command is dispatched.
    message = (
        "Ensure the following ScanImage imaging parameters are applied before generating the reference: the laser is "
        "enabled and its power is set, the ROI frame rate is ~10 Hz, the scan phase is ~0.8888, and PMT AutoOn is "
        "enabled."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
    wait_for_enter(message="Press Enter to continue")

    # Once the operator confirms the mesoscope is configured, offers to replace the animal's persisted reference with
    # the snapshot about to be generated. The prompt only appears when a reference already exists, because the first
    # reference for an animal is persisted automatically during preprocessing. The decision is captured here, before
    # the lengthy generation, because the runtime proceeds directly into the session once the mesoscope is armed.
    if (
        mesoscope_data.scanimagepc_data.motion_estimator_path.exists()
        or mesoscope_data.scanimagepc_data.roi_path.exists()
    ):
        replace_reference = request_confirmation(
            message=(
                f"Replace the persisted reference (MotionEstimator.me and fov.roi) for animal "
                f"{session_data.animal_id} with the snapshot generated this session?"
            ),
            default=False,
        )

    # Reference generation can fail for operator-fixable reasons reported by the ScanImagePC (for example, no active ROI
    # within the scanner FOV). Surfacing the exact ScanImagePC error and holding here for a retry lets the operator
    # correct the issue on the spot, rather than propagating the error into a session teardown and data purge.
    while True:
        try:
            mesoscope_driver.generate_reference()
            break
        except RuntimeError as error:
            message = (
                f"Mesoscope reference generation failed. {error} Address the issue on the ScanImagePC (for example, "
                f"ensure at least one active ROI with a scanfield exists within the scanner FOV), then retry. Select "
                f"'no' to abort the session and shut down."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
            if not request_confirmation(message="Retry mesoscope reference generation?", default=True):
                raise

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
        wait_for_enter(message="Press Enter to continue")

    # Applies the reference-replacement decision captured before generation, copying the freshly generated files over
    # the animal's persisted reference. Both files are copied together so a previously half-written reference pair is
    # fully refreshed. The matching mesoscope_positions snapshot is updated during session finalization, so an aborted
    # session may pair this reference with the previous positions until the next completed session.
    if replace_reference:
        _publish_reference_pair(
            motion_estimator_source=target_files[0],
            roi_source=target_files[1],
            motion_estimator_destination=mesoscope_data.scanimagepc_data.motion_estimator_path,
            roi_destination=mesoscope_data.scanimagepc_data.roi_path,
        )
        console.echo(message="Mesoscope reference: Replaced.", level=LogLevel.SUCCESS)

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

    The notes are entered through a blocking terminal prompt, so annotating a session takes no filesystem round-trip.
    For window checking sessions, the experimenter is additionally prompted for the cranial window quality rating,
    which is otherwise left at its default value. For the other session types, the experimenter is instead shown the
    session water summary and prompted for the total water the animal should receive, and the additional volume to
    hand-deliver is recorded as the experimenter-given water volume. The prompt defaults to the total water the animal
    received on the previous session, falling back to a standard default when no prior session recorded a water
    intake.

    Args:
        descriptor: The session_descriptor.yaml-convertible instance to complete and cache to the acquired session's
            data directory.
        session_data: The SessionData instance that defines the session for which the descriptor file is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # Window checking sessions additionally capture the experimenter's cranial window quality rating on a 0-3 scale.
    # The rating is propagated to the surgery log Google Sheet during preprocessing, and the other session types do
    # not track a window quality rating.
    if isinstance(descriptor, WindowCheckingDescriptor):
        descriptor.surgery_quality = collect_surgery_quality(session_name=session_data.session_name)
    else:
        # Non-window-checking sessions report the water delivered during the session alongside the animal's current
        # and previous weights and the water it received on the previous session, then prompt for the total water the
        # animal should receive. The returned value is the additional volume the experimenter must hand-deliver to
        # reach that total, counting only session water.
        previous_context = _resolve_previous_session_water_context(
            persistent_data_path=mesoscope_data.vrpc_data.persistent_data_path
        )
        previous_weight_g = previous_context.animal_weight_g if previous_context is not None else None
        previous_received_water_volume_ml = (
            previous_context.received_water_volume_ml if previous_context is not None else None
        )
        # Pre-fills the prompt with the total water the animal received on the previous session, letting the
        # experimenter keep the same intake by accepting the default. Falls back to the standard default when no prior
        # session recorded a water intake.
        default_total_water_volume_ml = (
            previous_received_water_volume_ml
            if previous_received_water_volume_ml is not None
            else _DEFAULT_TOTAL_WATER_VOLUME_ML
        )
        descriptor.experimenter_given_water_volume_ml = collect_experimenter_given_water_volume(
            current_weight_g=descriptor.animal_weight_g,
            previous_weight_g=previous_weight_g,
            previous_received_water_volume_ml=previous_received_water_volume_ml,
            session_water_volume_ml=descriptor.dispensed_water_volume_ml,
            default_total_water_volume_ml=default_total_water_volume_ml,
        )

    # Collects the experimenter notes through a blocking terminal prompt and stores them inside the descriptor. The
    # runtime control UI is already shut down at this point, so the prompt runs on the main thread without competing
    # GUIs.
    descriptor.experimenter_notes = collect_experimenter_notes(session_name=session_data.session_name)

    descriptor.to_yaml(file_path=session_data.raw_data.session_descriptor_path)
    console.echo(message="Session descriptor file: Created.", level=LogLevel.SUCCESS)

    # Copies the descriptor to the animal's persistent directory. This is primarily used during training to restore
    # the training parameters between training sessions of the same type.
    shutil.copy2(
        src=session_data.raw_data.session_descriptor_path,
        dst=mesoscope_data.vrpc_data.session_descriptor_path,
    )


def _publish_reference_pair(
    motion_estimator_source: Path,
    roi_source: Path,
    motion_estimator_destination: Path,
    roi_destination: Path,
) -> None:
    """Replaces the animal's persisted reference pair with the freshly generated motion estimator and ROI files.

    Notes:
        The two files describe one imaging field and are only meaningful together, so both are staged beside their
        destinations and flushed to disk before either is published. A failure while staging removes every staged file
        and leaves the destinations untouched. A failure between the two renames restores the destination the first
        rename already replaced, from a copy staged alongside it, so the pair never survives half-replaced.

        The restore is itself a rename, so the only window this cannot close is a crash of the interpreter or the host
        between the two renames, which a filesystem transaction would be needed to close.

        Each staged file is created with the permission bits ataraxis-data-structures requests in atomic_write(),
        narrowed by the process umask, so the published pair carries the platform's usual permissions rather than the
        mode the ScanImagePC copy happened to carry.

    Args:
        motion_estimator_source: The path to the freshly generated MotionEstimator.me file.
        roi_source: The path to the freshly generated fov.roi file.
        motion_estimator_destination: The path to the animal's persisted MotionEstimator.me file.
        roi_destination: The path to the animal's persisted fov.roi file.

    Raises:
        OSError: If a staged file cannot be written.
        PermissionError: If a destination stays locked by another process for every publishing attempt.
    """
    staged_files: list[tuple[Path, Path]] = []
    replaced_files: list[tuple[Path, Path]] = []
    try:
        for source, destination in (
            (motion_estimator_source, motion_estimator_destination),
            (roi_source, roi_destination),
        ):
            staged_files.append((_stage_reference_file(source=source, destination=destination), destination))

            # Stages a copy of the destination the rename below replaces, so a failure on the second rename can put
            # the first destination back rather than leaving a mismatched pair behind.
            if destination.exists():
                replaced_files.append(
                    (
                        _stage_reference_file(source=destination, destination=destination.with_suffix(".previous")),
                        destination,
                    )
                )

        for staged_path, destination in staged_files:
            _publish_staged_file(staged_path=staged_path, destination=destination)
    except BaseException:
        for staged_path, destination in replaced_files:
            _restore_replaced_file(staged_path=staged_path, destination=destination)
        for staged_path, _ in staged_files:
            staged_path.unlink(missing_ok=True)
        raise
    else:
        for staged_path, _ in replaced_files:
            staged_path.unlink(missing_ok=True)


def _restore_replaced_file(staged_path: Path, destination: Path) -> None:
    """Puts a replaced reference file back from the copy staged before it was overwritten.

    Notes:
        The restore runs while an error is already propagating, so a failure here is reported and swallowed rather
        than allowed to replace the error that triggered the rollback.

    Args:
        staged_path: The path to the staged copy of the destination's previous contents.
        destination: The destination path to restore.
    """
    try:
        _publish_staged_file(staged_path=staged_path, destination=destination)
    except (Exception, KeyboardInterrupt) as error:
        message = (
            f"Unable to restore the previous contents of {destination} after the mesoscope reference publication "
            f"failed: {error!r}. The animal's persisted reference pair may be mismatched, so regenerate it before the "
            f"next imaging session."
        )
        console.echo(message=message, level=LogLevel.ERROR)


def _stage_reference_file(source: Path, destination: Path) -> Path:
    """Copies the source file into a temporary file beside its destination and flushes it to disk.

    Notes:
        The temporary file is named after the destination and the writing process, so two runtimes publishing the
        same destination cannot collide on it. It is created in the destination's own directory, so the rename that
        publishes it stays within one filesystem.

    Args:
        source: The path to the file whose contents are staged.
        destination: The path to which the staged file is later published.

    Returns:
        The path to the staged temporary file.

    Raises:
        OSError: If the temporary file cannot be created or written.
    """
    staged_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")

    # Opens the staged file directly rather than through mkstemp(), which hardcodes the 0o600 bits that suit a private
    # scratch file and would leave a published reference readable only by the account that wrote it.
    descriptor = os.open(
        path=staged_path, flags=os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=_REFERENCE_WRITE_PERMISSION_BITS
    )
    try:
        with os.fdopen(fd=descriptor, mode="wb") as file:
            file.write(source.read_bytes())

            # Forces the contents out of the userspace and kernel buffers, so a host losing power after the rename
            # finds a complete file rather than a partial one.
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        # The caller only learns of a staged file once this function returns it, so a failure after the file exists
        # removes it here rather than leaving it for the caller's rollback.
        staged_path.unlink(missing_ok=True)
        raise

    return staged_path


def _publish_staged_file(staged_path: Path, destination: Path) -> None:
    """Renames a staged reference file over its destination path.

    Notes:
        Windows refuses the rename while another process holds the destination open without sharing its deletion,
        which a scanner or an indexer does for as long as it takes to read the file. The attempt is repeated with a
        doubling delay there, since the holder releases the file on its own. Every other platform replaces an open
        destination without complaint and renames on the first attempt.

    Args:
        staged_path: The path to the staged temporary file.
        destination: The path to which the staged file is renamed.

    Raises:
        PermissionError: If the destination stays locked by another process for every attempt.
    """
    if sys.platform != "win32":
        staged_path.replace(target=destination)
        return

    delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    delay = _REFERENCE_RENAME_RETRY_DELAY_MILLISECONDS
    for attempt in range(_REFERENCE_RENAME_RETRY_COUNT):
        try:
            staged_path.replace(target=destination)
        except PermissionError:
            if attempt == _REFERENCE_RENAME_RETRY_COUNT - 1:
                raise
            delay_timer.delay(delay=delay, allow_sleep=True, block=False)
            delay *= 2
        else:
            return


def _resolve_previous_session_water_context(persistent_data_path: Path) -> _PreviousSessionWaterContext | None:
    """Returns the animal weight and total water intake from the most recent non-window-checking session, or None.

    Notes:
        Both values are read from the newest per-session-type descriptor that a prior session cached in the animal's
        persistent directory. Window checking descriptors are skipped because they record neither the animal's weight
        nor its water intake. The descriptor filenames mirror the persistent layout defined by the _VRPCPersistentData
        class. The total intake matches the preprocessing definition, summing the runtime-dispensed and
        experimenter-given volumes while excluding water dispensed during the paused state.

    Args:
        persistent_data_path: The path to the animal's persistent directory that stores the cached descriptors.

    Returns:
        A _PreviousSessionWaterContext carrying the most recent prior session's animal weight and total water intake,
        or None when no prior session recorded them.
    """
    descriptor_file_names = (
        "lick_training_descriptor.yaml",
        "run_training_descriptor.yaml",
        "mesoscope_experiment_descriptor.yaml",
    )
    existing_paths = [
        candidate_path
        for file_name in descriptor_file_names
        if (candidate_path := persistent_data_path / file_name).exists()
    ]
    if not existing_paths:
        return None

    newest_path = max(existing_paths, key=lambda path: path.stat().st_mtime)
    if newest_path.name == descriptor_file_names[0]:
        descriptor: LickTrainingDescriptor | RunTrainingDescriptor | MesoscopeExperimentDescriptor = (
            LickTrainingDescriptor.from_yaml(file_path=newest_path)
        )
    elif newest_path.name == descriptor_file_names[1]:
        descriptor = RunTrainingDescriptor.from_yaml(file_path=newest_path)
    else:
        descriptor = MesoscopeExperimentDescriptor.from_yaml(file_path=newest_path)

    received_water_volume_ml = round(
        descriptor.dispensed_water_volume_ml + descriptor.experimenter_given_water_volume_ml, ndigits=3
    )
    return _PreviousSessionWaterContext(
        animal_weight_g=float(descriptor.animal_weight_g),
        received_water_volume_ml=received_water_volume_ml,
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
    """Validates a red-dot alignment Z position response, accepting a finite number or an empty response.

    Args:
        response: The raw text entered by the operator.

    Returns:
        True when the response is empty or parses as a finite number, or an error message describing the constraint.
    """
    if not response.strip():
        return True
    try:
        parsed_value = float(response)
    except ValueError:
        return "Enter a numeric value or leave the response empty to keep the stored value."
    if not math.isfinite(parsed_value):
        return "Enter a finite numeric value. Infinity and not-a-number responses are not valid positions."
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
    # Floors on the decimal grid, reached through the shortest decimal representation of the input, because scaling a
    # binary float by 1000 lands an input that already carries three decimals just below its grid point and floors it
    # one step low.
    return float(Decimal(str(value)).quantize(exp=_THREE_DECIMAL_QUANTUM, rounding=ROUND_FLOOR))
