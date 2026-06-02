"""Provides the shared runtime-state types and the hardware setup, teardown, and snapshot helpers used by the
Mesoscope-VR data acquisition runtime.
"""

from __future__ import annotations

from enum import IntEnum
import shutil
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

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

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .binding_classes import ZaberMotors


_RESPONSE_DELAY: int = 2000
"""Specifies the number of milliseconds to delay showing the response prompt after showing a message that requires
user interaction."""

_response_delay_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The PrecisionTimer instance used to support the proper rendering of all terminal outputs used during runtime."""


class _MesoscopeVRLogMessageCodes(IntEnum):
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
class _TrialState:
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


def _generate_mesoscope_position_snapshot(session_data: SessionData, mesoscope_data: MesoscopeData) -> None:
    """Forces the user to update the mesoscope_positions.yaml file to reflect the current mesoscope's imaging position
    coordinates and copies the validated file into the animal's persistent directory.

    Args:
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # If the session was not fully initialized (nk.bin marker exists), skips the snapshot generation.
    if session_data.raw_data.nk_path.exists():
        return

    # Loads the previous position data into memory.
    previous_mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(
        file_path=mesoscope_data.vrpc_data.mesoscope_positions_path,
    )

    # Forces the user to update the cached mesoscope position coordinates with the current data.
    message = (
        f"Update the data inside the mesoscope_positions.yaml file stored under the {session_data.session_name} "
        f"session's 'raw_data' directory to reflect the current mesoscope objective position."
    )
    console.echo(message=message, level=LogLevel.INFO)
    # Delays for 2 seconds to ensure the user reads the message before continuing.
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Defines the error message for file formatting issues.
    io_error_message = (
        f"Unable to read the data from the {session_data.session_name} session's mesoscope_positions.yaml file. This "
        f"indicates that the file was mis-formatted during editing. Make sure the file contents follow the .YAML "
        f"format before retrying."
    )

    # Defines the validation error message for unchanged positions.
    validation_error_message = (
        f"Failed to verify that the mesoscope_positions.yaml file stored inside the {session_data.session_name} "
        f"session's raw_data directory has been updated to include the mesoscope imaging coordinates used during "
        f"runtime. Edit the mesoscope_positions.yaml file to update the position fields with coordinates "
        f"displayed in the ScanImage software or on the ThorLabs pad. Make sure to save the changes by pressing "
        f"the 'CTRL+S' combination."
    )

    # Continuously attempts to read and validate the Mesoscope positions data until successful.
    while True:
        # Attempts to read the current mesoscope positions from the session file.
        # noinspection PyBroadException
        try:
            mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(
                file_path=session_data.system_raw_data.mesoscope_positions_path,
            )
        except Exception:
            console.echo(message=io_error_message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")
            continue

        # Validates that the user has updated the position data.
        if (
            mesoscope_positions.mesoscope_x != previous_mesoscope_positions.mesoscope_x
            or mesoscope_positions.mesoscope_y != previous_mesoscope_positions.mesoscope_y
            or mesoscope_positions.mesoscope_z != previous_mesoscope_positions.mesoscope_z
            or mesoscope_positions.mesoscope_roll != previous_mesoscope_positions.mesoscope_roll
            or mesoscope_positions.mesoscope_fast_z != previous_mesoscope_positions.mesoscope_fast_z
            or mesoscope_positions.mesoscope_tip != previous_mesoscope_positions.mesoscope_tip
            or mesoscope_positions.mesoscope_tilt != previous_mesoscope_positions.mesoscope_tilt
            or mesoscope_positions.laser_power_mw != previous_mesoscope_positions.laser_power_mw
            or mesoscope_positions.red_dot_alignment_z != previous_mesoscope_positions.red_dot_alignment_z
        ):
            break

        # If positions match, requests the user to update the file.
        console.echo(message=validation_error_message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")

    # Copies the updated mesoscope positions data into the animal's persistent directory.
    shutil.copy2(
        src=session_data.system_raw_data.mesoscope_positions_path,
        dst=mesoscope_data.vrpc_data.mesoscope_positions_path,
    )


def _generate_zaber_snapshot(
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


def _setup_zaber_motors(zaber_motors: ZaberMotors) -> None:
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
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

    # Blocks until a valid answer is received from the user.
    while True:
        user_input = input("Enter 'yes' or 'no': ").strip().lower()
        answer = user_input[0] if user_input else ""

        if answer == "n":
            # Aborts method runtime, as no further Zaber setup is required.
            return

        if answer == "y":
            # Proceeds with the setup sequence.
            break

    # Since it is now possible to shut down Zaber motors without fixing HeadBarRoll position, requests the user
    # to verify this manually.
    message = (
        "Check that the HeadBarRoll motor has a positive (>0) angle. If the angle is negative (<0), the motor will "
        "collide with the stopper during homing, which will DAMAGE the motor."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Initializes the Zaber positioning sequence. This relies heavily on user feedback to confirm that it is
    # safe to proceed with motor movements.
    message = (
        "Preparing to move Zaber motors into mounting position. Remove the mesoscope objective, swivel out the "
        "VR screens, and make sure the animal is NOT mounted in the Mesoscope's enclosure."
    )
    console.echo(message=message, level=LogLevel.WARNING)
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

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
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Restores all motors to the positions used during the previous session's runtime.
    zaber_motors.restore_position()

    message = "Motor Positioning: Complete."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _reset_zaber_motors(zaber_motors: ZaberMotors) -> None:
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
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

    while True:
        user_input = input("Enter 'yes' or 'no': ").strip().lower()
        answer = user_input[0] if user_input else ""

        # Continues with the rest of the shutdown runtime.
        if answer == "y":
            break

        # Ends the runtime, as there is no need to move Zaber motors.
        if answer == "n":
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
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Moves all motors to the hardcoded parking positions.
    zaber_motors.park_position()

    # Disconnects from Zaber motors. This does not change motor positions but does lock (park) all motors
    # before disconnecting.
    zaber_motors.disconnect()

    message = "Zaber motors: Reset."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _setup_mesoscope(session_data: SessionData, mesoscope_data: MesoscopeData) -> None:
    """Guides the user through the sequence of steps that prepares the Mesoscope for the data acquisition runtime.

    Args:
        session_data: The SessionData instance that defines the session for which the Mesoscope is being prepared.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
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
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

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
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Step 2: Generates the screenshot of the red-dot alignment and the cranial window.
    message = (
        "Generate the screenshot of the red-dot alignment, the imaging plane state (cell activity), and the "
        "ScanImage acquisition parameters by pressing the 'Win + PrtSc' combination."
    )
    console.echo(message=message, level=LogLevel.INFO)
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

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
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

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
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

        # Blocks until a valid answer is received from the user.
        while True:
            user_input = input("Enter 'yes' or 'no': ").strip().lower()
            answer = user_input[0] if user_input else ""

            if answer == "n":
                # Aborts the runtime if the user does not intend to generate the ROI and MotionEstimator data.
                console.echo(message="Mesoscope preparation: Complete.", level=LogLevel.SUCCESS)
                return

            if answer == "y":
                # Proceeds with the metadata file acquisition sequence.
                break

        # Ensures that kinase is removed, while the phosphatase is present. This aborts the runtime
        # after generating the zstack.tiff and the MotionEstimator.me files.
        mesoscope_data.scanimagepc_data.kinase_path.unlink(missing_ok=True)
        mesoscope_data.scanimagepc_data.phosphatase_path.touch()

    else:
        # For all other runtimes, resets the kinase and phosphatase markers before instructing the user to start the
        # acquisition preparation function.
        mesoscope_data.scanimagepc_data.kinase_path.unlink(missing_ok=True)
        mesoscope_data.scanimagepc_data.phosphatase_path.unlink(missing_ok=True)

    # Step 3: Generates the new MotionEstimator file and arms the mesoscope for acquisition.
    message = (
        "Call the 'setupAcquisition(hSI, hSICtl)' function via MATLAB's command line interface on the ScanImagePC to "
        "prepare and arm the mesoscope to acquire the session's data."
    )
    console.echo(message=message, level=LogLevel.INFO)
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # The preparation function generates 3 files: MotionEstimator.me, fov.roi, and zstack.tiff.
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
            f"following expected files are missing from the 'mesoscope_data' directory: {missing_names}. Rerun the "
            f"setupAcquisition(hSI, hSICtl) function to generate the requested files."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

    console.echo(message="Mesoscope preparation: Complete.", level=LogLevel.SUCCESS)


def _verify_descriptor_update(
    descriptor: MesoscopeExperimentDescriptor
    | LickTrainingDescriptor
    | RunTrainingDescriptor
    | WindowCheckingDescriptor,
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
) -> None:
    """Caches the input session's descriptor to disk and forces the user supervising the session's data acquisition to
    update the data stored inside the cached descriptor file with the notes made during runtime.

    Args:
        descriptor: The session_descriptor.yaml-convertible instance to cache to the acquired session's data directory.
        session_data: The SessionData instance that defines the session for which the descriptor file is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # Saves the descriptor as a .yaml file.
    descriptor.to_yaml(file_path=session_data.raw_data.session_descriptor_path)
    console.echo(message="Session descriptor precursor file: Created.", level=LogLevel.SUCCESS)

    # Instructs the user to add user-collected data to the cached descriptor file.
    message = (
        f"Update the data inside the session_descriptor.yaml file stored under the {session_data.session_name} "
        f"session's 'raw_data' directory to include the notes and data collected by the user supervising the runtime "
        f"during the session's data acquisition."
    )

    console.echo(message=message, level=LogLevel.INFO)
    _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
    input("Enter anything to continue: ")

    # Defines error messages for file operations.
    io_error_message = (
        f"Unable to read the data from the {session_data.session_name} session's session_descriptor.yaml file. This "
        f"indicates that the file was mis-formatted during editing. Make sure the file contents follow the .YAML "
        f"format before retrying."
    )
    validation_error_message = (
        f"Failed to verify that the session_descriptor.yaml file stored inside the {session_data.session_name} "
        f"session's raw_data directory has been updated to include the supervising user's notes taken during "
        f"runtime. Manually edit the session_descriptor.yaml file and replace the default text under the "
        f"'experimenter_notes' field with the notes taken during runtime. Make sure to save the changes by pressing "
        f"the 'CTRL+S' combination."
    )

    # Continuously attempts to read and validate the session descriptor until successful.
    while True:
        # Attempts to read the session's descriptor data from the .yaml file.
        # noinspection PyBroadException
        try:
            descriptor = descriptor.from_yaml(file_path=session_data.raw_data.session_descriptor_path)
        except Exception:
            console.echo(message=io_error_message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")
            continue

        # Validates that the user has updated the experimenter notes.
        # noinspection PyUnresolvedReferences
        if "Replace this with your notes." not in descriptor.experimenter_notes:
            break

        # If validation fails, prompts the user to update the file.
        console.echo(message=validation_error_message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")

    # If the descriptor has passed the verification, copies it up to the animal's persistent directory. This is a
    # feature primarily used during training to restore the training parameters between training sessions of the
    # same type.
    shutil.copy2(
        src=session_data.raw_data.session_descriptor_path,
        dst=mesoscope_data.vrpc_data.session_descriptor_path,
    )
