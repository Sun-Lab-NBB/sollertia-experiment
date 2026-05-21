"""Provides the assets for executing data acquisition sessions and maintenance runtimes via the Mesoscope-VR data
acquisition system.
"""

import os
import copy
from enum import IntEnum, StrEnum
import json
from json import dumps
import shutil as sh
from pathlib import Path
import tempfile
from dataclasses import field, dataclass

from tqdm import tqdm
from numba import njit  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray  # noqa: TC002
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, convert_time, get_timestamp
from sollertia_shared_assets import (
    SessionData,
    GasPuffTrial,
    SessionTypes,
    ExperimentState,
    WaterRewardTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    MesoscopeHardwareState,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration,
)
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import DataLogger, LogPackage
from ataraxis_communication_interface import MQTTCommunication, MicroControllerInterface

from .tools import MesoscopeData, CachedMotifDecomposer, get_system_configuration
from .positions import MesoscopePositions, ZaberPositions
from .configuration import MesoscopeSystemConfiguration
from .runtime_ui import RuntimeControlUI
from .visualizers import VisualizerMode, BehaviorVisualizer
from .maintenance_ui import MaintenanceControlUI
from .binding_classes import ZaberMotors, VideoSystems, MicroControllerInterfaces
from ..shared_components import (
    BrakeInterface,
    ValveInterface,
    GasPuffValveInterface,
    get_version_data,
    get_animal_project,
    get_project_experiments,
)
from .data_preprocessing import purge_session, preprocess_session_data, rename_mesoscope_directory

_RESPONSE_DELAY: int = 2000
"""Specifies the number of milliseconds to delay showing the response prompt after showing a message that requires 
user interaction."""

_RENDERING_SEPARATION_DELAY = 500
"""Specifies the number of milliseconds to delay between rendering console outputs (stderr) and non-console outputs 
(stdout) to prevent the two renders from overlapping."""

_response_delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
"""The PrecisionTimer instance used to support the proper rendering of all terminal outputs used during runtime."""


# Defines shared methods to make their use consistent between window checking and other runtimes.
def _generate_mesoscope_position_snapshot(session_data: SessionData, mesoscope_data: MesoscopeData) -> None:
    """Generates a precursor mesoscope_positions.yaml file and forces the user to update it to reflect
    the current mesoscope's imaging position coordinates.

    Args:
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # If the session was not fully initialized (nk.bin marker exists), skips the snapshot generation.
    if session_data.raw_data.nk_path.exists():
        return

    # Loads the previous position data into memory.
    previous_mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(
        file_path=mesoscope_data.vrpc_data.mesoscope_positions_path
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

    # Defines the error message for file formatting issues
    io_error_message = (
        f"Unable to read the data from the {session_data.session_name} session's mesoscope_positions.yaml file. This "
        f"indicates that the file was mis-formatted during editing. Make sure the file contents follow the .YAML "
        f"format before retrying."
    )

    # Defines the validation error message for unchanged positions
    validation_error_message = (
        f"Failed to verify that the mesoscope_positions.yaml file stored inside the {session_data.session_name} "
        f"session's raw_data directory has been updated to include the mesoscope imaging coordinates used during "
        f"runtime. Edit the mesoscope_positions.yaml file to update the position fields with coordinates "
        f"displayed in the ScanImage software or on the ThorLabs pad. Make sure to save the changes by pressing "
        f"the 'CTRL+S' combination."
    )

    # Continuously attempts to read and validate the Mesoscope positions data until successful
    while True:
        # Attempts to read the current mesoscope positions from the session file
        # noinspection PyBroadException
        try:
            mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(
                file_path=session_data.raw_data.mesoscope_positions_path,
            )
        except Exception:
            console.echo(message=io_error_message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")
            continue

        # Validates that the user has updated the position data
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

        # If positions match, request the user to update the file
        console.echo(message=validation_error_message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")

    # Copies the updated mesoscope positions data into the animal's persistent directory.
    sh.copy2(
        src=session_data.raw_data.mesoscope_positions_path,
        dst=mesoscope_data.vrpc_data.mesoscope_positions_path,
    )


def _generate_zaber_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, zaber_motors: ZaberMotors
) -> None:
    """Creates a snapshot of the current Zaber motor positions and saves it as a zaber_positions.yaml file.

    Args:
        zaber_motors: The ZaberMotors instance that manages the Zaber assets used by the session for which the
            snapshot is generated.
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
        mesoscope_data: The MesoscopeData instance that defines the current Mesoscope-VR system's configuration.
    """
    # If at least one of the managed motor groups is not connected, does not run the snapshot generation sequence.
    # Also, if the session failed to properly initialize, as marked by the presence of the nk.bin marker.
    if not zaber_motors.is_connected or session_data.raw_data.nk_path.exists():
        return

    # Generates the snapshot
    zaber_positions = zaber_motors.generate_position_snapshot()

    # Saves the newly generated file both to the persistent directory and to the session directory. Note, saving to the
    # persistent data directory automatically overwrites any existing position file.
    zaber_positions.to_yaml(file_path=mesoscope_data.vrpc_data.zaber_positions_path)
    zaber_positions.to_yaml(file_path=session_data.raw_data.zaber_positions_path)

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

        # Continues with the rest of the shutdown runtime
        if answer == "y":
            break

        # Ends the runtime, as there is no need to move Zaber motors.
        if answer == "n":
            # Disconnects from Zaber motors. This does not change motor positions but does lock (park) all motors
            # before disconnecting.
            zaber_motors.disconnect()
            return

    # Helps with removing the animal from the enclosure by retracting the lick-port in the Y-axis (moving it away
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
        session_data: The SessionData instance that defines the session for which the snapshot is generated.
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
            f"files: {','.join([file.name for file in existing_files])}. Clear the directory from all existing files "
            f"before proceeding."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

    # Step 1: Resolves the imaging plane.
    # If the previous session's mesoscope positions were saved, loads the imaging coordinates and displays them to the
    # user
    if not window_checking and mesoscope_data.vrpc_data.mesoscope_positions_path.exists():
        previous_positions: MesoscopePositions = MesoscopePositions.from_yaml(
            file_path=mesoscope_data.vrpc_data.mesoscope_positions_path
        )
        message = (
            f"Follow the steps of the mesoscope preparation protocol available from the sl-protocols repository."
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
        screenshots = list(mesoscope_data.scanimagepc_data.meso_data_path.glob("*.png"))

        if screenshots:
            break

        message = (
            f"Unable to retrieve the screenshot from the ScanImage PC. Expected a single .png file inside the "
            f"'mesodata' ScanImagePC directory, but instead found {len(screenshots)} candidate files. Ensure that the "
            f"directory only stores the .png screenshot generated during the previous preparation step."
        )
        console.echo(message=message, level=LogLevel.ERROR)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

    # Transfers the screenshot to the session's mesoscope_frames directory
    screenshot_path = session_data.raw_data.window_screenshot_path
    sh.move(screenshots.pop(), screenshot_path)  # Moves the screenshot from the ScanImagePC to the VRPC

    # Copies the screenshot to the animal's persistent data directory so that it can be reused during the next
    # runtime.
    sh.copy2(screenshot_path, mesoscope_data.vrpc_data.window_screenshot_path)

    # Window checking sessions require special handling.
    if window_checking:
        # Since window checking may reveal that the evaluated animal is not fit for participating in experiments,
        # optionally allows aborting the runtime early for window checking sessions.
        message = "Do you want to generate the ROI and MotionEstimator snapshots for this animal?"
        console.echo(message=message, level=LogLevel.INFO)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

        # Blocks until a valid answer is received from the user
        while True:
            user_input = input("Enter 'yes' or 'no': ").strip().lower()
            answer = user_input[0] if user_input else ""

            if answer == "n":
                # Aborts the runtime if the user does not intend to generate the ROI and MotionEstimator data
                console.echo(message="Mesoscope preparation: Complete.", level=LogLevel.SUCCESS)
                return

            if answer == "y":
                # Proceeds with the metadata file acquisition sequence
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
        missing_files = [f for f in target_files if not f.exists()]

        if not missing_files:
            break

        missing_names = ", ".join(f.name for f in missing_files)

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

    # Defines error messages for file operations
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

    # Continuously attempts to read and validate the session descriptor until successful
    while True:
        # Attempts to read the session's descriptor data from the .yaml file.
        # noinspection PyBroadException
        try:
            descriptor = descriptor.from_yaml(file_path=session_data.raw_data.session_descriptor_path)
        except Exception:
            console.echo(message=io_error_message, level=LogLevel.ERROR)
            input("Enter anything to continue: ")
            continue

        # Validates that the user has updated the experimenter notes
        # noinspection PyUnresolvedReferences
        if "Replace this with your notes." not in descriptor.experimenter_notes:
            break

        # If validation fails, prompt the user to update the file
        console.echo(message=validation_error_message, level=LogLevel.ERROR)
        input("Enter anything to continue: ")

    # If the descriptor has passed the verification, copies it up to the animal's persistent directory. This is a
    # feature primarily used during training to restore the training parameters between training sessions of the
    # same type.
    sh.copy2(
        src=session_data.raw_data.session_descriptor_path,
        dst=mesoscope_data.vrpc_data.session_descriptor_path,
    )


class _MesoscopeVRStates(IntEnum):
    """Defines the set of codes used by the Mesoscope-VR data acquisition system to communicate its runtime state."""

    IDLE = 0
    """The system is currently not conducting a data acquisition session."""
    REST = 1
    """The system is conducting the 'rest' period of an experiment session."""
    RUN = 2
    """The system is conducting the 'run' period of an experiment session."""
    LICK_TRAINING = 3
    """The system is conducting the lick training session."""
    RUN_TRAINING = 4
    """The system is conducting the run training session."""

    @classmethod
    def to_dict(cls) -> dict[str, int]:
        """Converts the instance's data to a dictionary mapping, replacing underscores with spaces."""
        return {member.name.lower().replace("_", " "): member.value for member in cls}


class _MesoscopeVRMQTTTopics(StrEnum):
    """Defines the set of MQTT topics used by the Mesoscope-VR data acquisition system to communicate with the Unity
    game engine.

    Notes:
        The topics defined in this enumeration are used in addition to the topic defined by the hardware module
        interfaces used by the system.
    """

    UNITY_TERMINATION = "Gimbl/Session/Stop"
    """Stops the Unity game session."""
    UNITY_STARTUP = "Gimbl/Session/Start"
    """Starts the Unity game session."""
    CUE_SEQUENCE = "CueSequence/"
    """The topic to which Unity sends the sequence of VR cues used by the current game session."""
    CUE_SEQUENCE_REQUEST = "CueSequenceTrigger/"
    """Requests Unity to send the sequence of VR cues used by the current game session."""
    DISABLE_LICK_GUIDANCE = "RequireLick/True/"
    """Disables lick guidance for reinforcing trials (animal must lick to trigger reward)."""
    ENABLE_LICK_GUIDANCE = "RequireLick/False/"
    """Enables lick guidance for reinforcing trials (reward on collision without lick)."""
    DISABLE_OCCUPANCY_GUIDANCE = "RequireWait/True/"
    """Disables occupancy guidance for aversive trials (animal must meet duration requirement)."""
    ENABLE_OCCUPANCY_GUIDANCE = "RequireWait/False/"
    """Enables occupancy guidance for aversive trials (brake pulse on early exit)."""
    SHOW_REWARD_ZONE_BOUNDARY = "VisibleMarker/True/"
    """Requests Unity to show the task guidance mode collision box to the animal."""
    HIDE_REWARD_ZONE_BOUNDARY = "VisibleMarker/False/"
    """Requests Unity to hide the task guidance mode collision box from the animal."""
    UNITY_SCENE_REQUEST = "SceneNameTrigger/"
    """Requests Unity to send the name of the currently used game scene."""
    UNITY_SCENE = "SceneName/"
    """The topic to which Unity sends the name of the currently used game scene."""
    STIMULUS = "Gimbl/Stimulus/"
    """The topic used by Unity to notify the runtime when the animal triggers a stimulus (water reward or gas puff)."""
    TRIGGER_DELAY = "Gimbl/TriggerDelay/"
    """The topic to which Unity sends the occupancy delay to enforce by briefly pulsing the brake."""
    ENCODER_DATA = "LinearTreadmill/Data"
    """Sends animal motion (distance) updates to Unity."""
    LICK_EVENT = "LickPort/"
    """Sends lick event notifications to Unity."""


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
    """The system has taken a snapshot of the total distance traveled by the animal due to changing the VR wall cue 
    sequence."""


@dataclass
class _TrialState:
    """Tracks the state of the Mesoscope-VR-acquired session's task trials.

    This dataclass consolidates all trial-related state tracking attributes used during experiment runtimes to
    monitor trial progression, manage task guidance modes, and determine stimulus delivery conditions. Supports both
    reinforcing (water reward) and aversive (gas puff) trial types.
    """

    # Overall trial tracking
    completed: int = 0
    """The total number of trials completed by the animal since the last cue sequence reset or runtime onset."""
    distances: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    """Stores the total cumulative distance, in centimeters, the animals would travel at the end of each trial."""

    # Reinforcing (water reward) trial tracking
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
    """Stores the reward size (volume in μL) and tone duration (ms) for each reinforcing trial."""

    # Aversive (gas puff) trial tracking
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
    """Stores the gas puff duration (ms) for each aversive trial."""

    # Trial structure configuration
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
        return traveled_distance > self.distances[self.completed]

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
        """Checks whether the current trial is an aversive (gas puff) trial.

        Returns:
            True if the current trial is a GasPuffTrial, False otherwise.
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


@dataclass
class _UnityState:
    """Tracks the state of the Mesoscope-VR-acquired session's Virtual Reality task environment managed by the Unity
    game engine.

    This dataclass consolidates all Unity-related state tracking attributes used during experiment runtimes to
    monitor the Virtual Reality environment state, manage task guidance modes, and facilitate communication between
    the Mesoscope-VR system and the Unity game engine via MQTT.
    """

    position: np.float64 = field(default_factory=lambda: np.float64(0.0))
    """The current absolute position of the animal, in Unity units, relative to the origin of the Virtual Reality task 
    environment's track."""
    cue_sequence: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=0, dtype=np.uint8))
    """The sequence of the Virtual Reality environment wall cues used by the session's task environment. This array 
    defines the visual cues displayed to the animal as it progresses through the virtual track."""
    terminated: bool = False
    """Tracks whether the system has detected that the Unity game engine has unexpectedly terminated its runtime. When
    True, the system enters an emergency pause state to allow the user to restart Unity."""
    reinforcing_guidance_enabled: bool = False
    """Tracks the state of the reinforcing trial guidance mode."""
    aversive_guidance_enabled: bool = False
    """Tracks the state of the aversive trial guidance mode."""


class _MesoscopeVRSystem:
    """Provides methods for conducting data acquisition sessions using the Mesoscope-VR system.

    Notes:
        Calling this initializer does not instantiate all assets required for the runtime. Use the start() method
        before calling other instance methods to properly initialize all required runtime assets and remote
        processes.

        This instance statically reserves the id code '1' to label its log entries.

    Args:
        session_data: The SessionData instance that defines the session for which to acquire the data.
        session_descriptor: The partially configured SessionDescriptor instance that stores the task metadata of the
            session for which to acquire the data.
        experiment_configuration: The MesoscopeExperimentConfiguration instance that specifies the experiment
            configuration to use during the session's data acquisition or None, if the session is not a mesoscope
            experiment session.

    Attributes:
        _mesoscope_frame_delay: The maximum delay, in milliseconds, that can separate the acquisition of any two
            consecutive mesoscope frames, when the mesoscope functions as expected.
        _speed_calculation_window: Determines the window size, in milliseconds, used to calculate the recorded animal's
            running speed.
        _source_id: The unique identifier code of the instance, used to identify the instance in the generated
            data log messages.
        _started: Tracks whether the session's data acquisition has started.
        _terminated: Tracks whether the session's data acquisition has terminated.
        _paused: Tracks whether the session's data acquisition has been temporarily paused.
        _mesoscope_started: Tracks whether the system has started acquiring Mesoscope frames.
        descriptor: The SessionDescriptor instance for the session whose data is acquired by the system during
            runtime.
        _experiment_configuration: The MesoscopeExperimentConfiguration instance for the session whose data is acquired
            by the system during runtime or None, if the session is not of the 'mesoscope experiment' type.
        _system_configuration: The MesoscopeSystemConfiguration instance that defines the configuration of the data
            acquisition system.
        _session_data: The SessionData instance that defines the session whose data is acquired by the system during
            runtime.
        _mesoscope_data: The MesoscopeData instance that defines the filesystem layout of the data acquisition system.
        _system_state: The code that communicates the current Mesoscope-VR system's state.
        _runtime_state: The code that communicates the current data acquisition session's task state (stage).
        _timestamp_timer: The PrecisionTimer instance that timestamps log entries generated by the instance.
        _distance: The total cumulative distance, in centimeters, traveled by the animal since runtime onset.
        _lick_count: The total number of licks performed by the animal since runtime onset.
        _unconsumed_reward_count: The number of rewards delivered to the animal that has not yet been consumed
            by the animal.
        _pause_start_time: The absolute time, in microseconds elapsed since the UTC epoch onset, of the last
            runtime pause onset.
        paused_time: The total time, in seconds, the session's data acquisition runtime spent in the paused
            (idle) state.
        _delivered_water_volume: The total volume of water dispensed by the water delivery valve during the
            active data acquisition state.
        _mesoscope_frame_count: Tracks the number of frames acquired by the Mesoscope since the last mesoscope frame
            acquisition onset.
        _mesoscope_terminated: Tracks whether the system has detected that the Mesoscope has unexpectedly
            terminated its runtime.
        _running_speed: The animal's running speed, in centimeters per second, computed over the last 50 milliseconds.
        _speed_timer: The PrecisionTimer instance used to compute the animal's running speed in 50-millisecond
            intervals.
        _paused_water_volume: Tracks the total volume of water, in milliliters, dispensed by the water delivery valve
            when the session's data acquisition was paused.
        _logger: The DataLogger instance that logs the data from all sources managed by the Mesoscope-VR instance.
        _microcontrollers: The MicroControllerInterfaces instance that interfaces with the Actor, Sensor, and Encoder
            microcontrollers used during runtime.
        _cameras: The VideoSystems instance that interfaces with the face and body cameras used during runtime.
        _zaber_motors: The ZaberMotors instance that interfaces with the HeadBar, LickPort, and Wheel motor groups.
        _unity: The MQTTCommunication instance that bidirectionally transfers data between this instance and the Unity
            game engine managing the session's Virtual Reality task environment.
        _ui: The RuntimeControlUI instance that maintains a Graphical User Interface that allows the user to
            control the session's runtime.
        _visualizer: The BehaviorVisualizer instance used during runtime to visualize the animal's behavior or
            None, if the managed runtime does not require behavior visualization.
        _mesoscope_timer: The PrecisionTimer instance used to track the delay between receiving consecutive
            mesoscope frame acquisition pulses.
        _motif_decomposer: The MotifDecomposer instance used during runtime to decompose long VR cue sequences
            into the sequence of trials and corresponding cumulative traveled distances associated with each trial.
        _unity_state: The _UnityState instance that tracks the state of the Virtual Reality task environment managed
            by the Unity game engine.
        _trial_state: The _TrialState instance that tracks the progression of trials during experiment runtimes.

    Raises:
        RuntimeError: If the host-machine does not have enough logical CPU cores to support the runtime.
    """

    # Statically assigns mesoscope frame checking window and speed calculation window, in milliseconds.
    _mesoscope_frame_delay: int = 300
    _speed_calculation_window: int = 50

    # Reserves logging source ID code 1 for this class
    _source_id: np.uint8 = np.uint8(1)

    def __init__(
        self,
        session_data: SessionData,
        session_descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor,
        experiment_configuration: MesoscopeExperimentConfiguration | None = None,
    ) -> None:
        # Creates runtime state tracking flags
        self._started: bool = False
        self._terminated: bool = False
        self._paused: bool = False
        self._mesoscope_started: bool = False

        # Pre-runtime check to ensure that the host-machine has enough cores to facilitate the data acquisition.
        # 3 cores for microcontrollers, 1 core for the data logger, 4 cores for the video systems
        # (2 producers, 2 consumers), 1 core for the central process calling this method, 1 core for
        # the main GUI: 10 cores total.
        _minimum_cpu_count = 10
        cpu_count = os.cpu_count()
        if cpu_count is None or not cpu_count >= _minimum_cpu_count:
            message = (
                f"Unable to initialize the Mesoscope-VR system runtime control class. The host PC must have at least "
                f"10 logical CPU cores available for this runtime to work as expected, but only {cpu_count} cores are "
                f"available."
            )
            console.error(message=message, error=RuntimeError)

        # Caches SessionDescriptor and MesoscopeExperimentConfiguration instances to class attributes.
        self.descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor = (
            session_descriptor
        )
        self._experiment_configuration: MesoscopeExperimentConfiguration | None = experiment_configuration

        # Caches the descriptor to disk. Primarily, this is required for preprocessing the data if the session's runtime
        # terminates unexpectedly.
        self.descriptor.to_yaml(file_path=session_data.raw_data.session_descriptor_path)

        # Resolves and caches the Mesoscope-VR and the processed session's configuration parameters.
        self._system_configuration: MesoscopeSystemConfiguration = get_system_configuration()
        self._session_data: SessionData = session_data
        self._mesoscope_data: MesoscopeData = MesoscopeData(
            session_data=session_data, system_configuration=self._system_configuration
        )

        # Generates a precursor MesoscopePositions file and dumps it to the session's raw_data directory.
        # If a previous set of mesoscope position coordinates is available, overwrites the 'default' mesoscope
        # coordinates with the positions loaded from the snapshot stored inside the persistent_data directory of the
        # animal.
        if self._mesoscope_data.vrpc_data.mesoscope_positions_path.exists():
            # Loading and re-dumping the data updates the contents of the position's file to dynamically integrate any
            # upstream changes in the sollertia-shared-assets into the file structure.
            previous_mesoscope_positions: MesoscopePositions = MesoscopePositions.from_yaml(
                file_path=self._mesoscope_data.vrpc_data.mesoscope_positions_path
            )
            previous_mesoscope_positions.to_yaml(file_path=session_data.raw_data.mesoscope_positions_path)

        # If previous position data is not available, creates a new MesoscopePositions instance with default position
        # values.
        else:
            # Caches the precursor file to the raw_data session directory and to the persistent data directory.
            precursor = MesoscopePositions()
            precursor.to_yaml(file_path=session_data.raw_data.mesoscope_positions_path)
            precursor.to_yaml(file_path=self._mesoscope_data.vrpc_data.mesoscope_positions_path)

        # Defines the asset used to set and maintain combinations of system and runtime (task) states.
        self._system_state: int = 0
        self._runtime_state: int = 0
        self._timestamp_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MICROSECOND)

        # Initializes the tracker attributes used to cyclically handle data updates during runtime.
        self._distance: np.float64 = np.float64(0.0)
        self._lick_count: np.uint64 = np.uint64(0)
        self._unconsumed_reward_count: int = 0
        self._pause_start_time: int = 0
        self.paused_time: int = 0
        self._delivered_water_volume: np.float64 = np.float64(0.0)
        self._mesoscope_frame_count: np.uint64 = np.uint64(0)
        self._mesoscope_terminated: bool = False
        self._running_speed: np.float64 = np.float64(0.0)
        self._speed_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        self._paused_water_volume: np.float64 = np.float64(0.0)

        # Initializes Unity and trial state tracking dataclasses.
        self._unity_state: _UnityState = _UnityState()
        self._trial_state: _TrialState = _TrialState()

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data.raw_data_path,
            instance_name="behavior",
            thread_count=10,
        )

        # Initializes the binding class for all MicroController Interfaces.
        self._microcontrollers: MicroControllerInterfaces = MicroControllerInterfaces(
            data_logger=self._logger, microcontroller_configuration=self._system_configuration.microcontrollers
        )

        # Initializes the binding class for all VideoSystems.
        self._cameras: VideoSystems = VideoSystems(
            data_logger=self._logger,
            output_directory=self._session_data.raw_data.camera_data_path,
            camera_configuration=self._system_configuration.cameras,
        )

        # The ZaberLauncher UI cannot connect to the ports managed by Python bindings, so it must be initialized before
        # connecting to motor groups from Python.
        message = (
            "Preparing to connect to all managed Zaber motors. Make sure that the ZaberLauncher app is running before "
            "proceeding further. If the ZaberLauncher is not running, it will be IMPOSSIBLE to manually control the "
            "Zaber motors."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

        # If the system has a snapshot of the Zaber positions used during a previous runtime, loads it into memory and
        # restores all Zaber motors to that snapshot. Otherwise, uses predefined default positions and expects the
        # user to fine-tune them as necessary.
        if self._mesoscope_data.vrpc_data.zaber_positions_path.exists():
            zaber_positions = ZaberPositions.from_yaml(file_path=self._mesoscope_data.vrpc_data.zaber_positions_path)
        else:
            zaber_positions = None

        # Initializes the binding class for all Zaber motors.
        self._zaber_motors: ZaberMotors = ZaberMotors(
            zaber_positions=zaber_positions, zaber_configuration=self._system_configuration.assets
        )

        # Defines optional assets used by some, but not all runtimes.
        monitored_topics = (
            _MesoscopeVRMQTTTopics.CUE_SEQUENCE,
            _MesoscopeVRMQTTTopics.UNITY_TERMINATION,
            _MesoscopeVRMQTTTopics.UNITY_STARTUP,
            _MesoscopeVRMQTTTopics.UNITY_SCENE,
            _MesoscopeVRMQTTTopics.STIMULUS,
            _MesoscopeVRMQTTTopics.TRIGGER_DELAY,
        )  # The list of topics monitored for the incoming data sent from the Unity game engine.
        self._unity: MQTTCommunication = MQTTCommunication(
            ip=self._system_configuration.assets.unity_ip,
            port=self._system_configuration.assets.unity_port,
            monitored_topics=monitored_topics,
        )
        self._mesoscope_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        self._motif_decomposer = CachedMotifDecomposer()

        # Initializes but does not start the assets used by all runtimes. These assets need to be started in a
        # specific order, which is handled by the start() method.
        # noinspection PyProtectedMember
        self._ui: RuntimeControlUI = RuntimeControlUI(
            valve_tracker=self._microcontrollers.valve._valve_tracker,  # noqa: SLF001
            gas_puff_tracker=self._microcontrollers.gas_puff_valve._puff_tracker,  # noqa: SLF001
        )
        self._visualizer: BehaviorVisualizer = BehaviorVisualizer()

    def start(self) -> None:
        """Guides the user through a semi-interactive sequence of steps that prepares the assets used to acquire the
        session's data.

        Notes:
            This method executes a complex initialization sequence that initializes and configures all assets, internal
            (managed by the VRPC) and external (managed by other PCs and / or software) and often takes a significant
            amount of time.

            As part of its runtime, the method gradually reserves an expanding pool of host-machine's resources (CPUs,
            GPUs, memory, etc.) to support the runtime of the initialized assets.
        """
        # If the assets are already initialized, aborts the runtime early.
        if self._started:
            return

        message = "Initializing Mesoscope-VR system assets..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts the data logger
        self._logger.start()

        # Generates and logs the onset timestamp for the Mesoscope-VR system.
        onset: NDArray[np.uint8] = get_timestamp(output_format=TimestampFormats.BYTES)  # type: ignore[assignment]
        self._timestamp_timer.reset()  # Immediately resets the timer to align it with the onset timestamp.
        self._logger.input_queue.put(
            LogPackage(source_id=self._source_id, acquisition_time=np.uint64(0), serialized_data=onset)
        )  # Logs the onset timestamp

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts all microcontroller interfaces
        self._microcontrollers.start()

        # Sets the runtime into the Idle state before instructing the user to finalize runtime preparations.
        self.idle()

        # Generates a snapshot of the runtime hardware configuration. In turn, this data is used to parse the .npz log
        # files during processing.
        self._generate_hardware_state_snapshot()

        # If the session uses Virtual Reality, initializes the MQTT communication with the Unity game engine.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            self._unity.connect()  # Establishes communication with the MQTT broker.

            # Guides the user through the Unity setup sequence.
            self._setup_unity()

        # Begins acquiring and displaying frames with the all available cameras.
        self._cameras.start_face_camera()
        self._cameras.start_body_camera()

        # If necessary, carries out the Zaber motor setup and animal mounting sequence and generates a snapshot of all
        # zaber motor positions. This serves as an early checkpoint in case the runtime has to be aborted in a
        # non-graceful way (without running the stop() sequence). This way, the next runtime restarts with the
        # calibrated zaber positions. The snapshot includes any adjustment to the HeadBar positions performed during
        # the red-dot alignment.
        _setup_zaber_motors(zaber_motors=self._zaber_motors)
        _generate_zaber_snapshot(
            session_data=self._session_data, mesoscope_data=self._mesoscope_data, zaber_motors=self._zaber_motors
        )

        # If the session is a mesoscope experiment, initializes the mesoscope.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            # Instructs the user to prepare the mesoscope for data acquisition.
            _setup_mesoscope(session_data=self._session_data, mesoscope_data=self._mesoscope_data)

        # Determines the visualizer mode based on session type. This mode is used by both the runtime control UI and
        # the behavior visualizer to conditionally enable/disable UI elements.
        if self._session_data.session_type == SessionTypes.LICK_TRAINING:
            visualizer_mode = VisualizerMode.LICK_TRAINING
        elif self._session_data.session_type == SessionTypes.RUN_TRAINING:
            visualizer_mode = VisualizerMode.RUN_TRAINING
        else:
            visualizer_mode = VisualizerMode.EXPERIMENT

        # Determines which trial types are used based on the experiment configuration. This affects both the runtime
        # control UI and the visualizer layouts.
        has_reinforcing_trials = True
        has_aversive_trials = True
        if visualizer_mode == VisualizerMode.EXPERIMENT and self._experiment_configuration:
            trial_structures = self._experiment_configuration.trial_structures.values()
            has_reinforcing_trials = any(isinstance(t, WaterRewardTrial) for t in trial_structures)
            has_aversive_trials = any(isinstance(t, GasPuffTrial) for t in trial_structures)

        # Initializes the runtime control GUI with the appropriate mode and trial type configuration.
        self._ui.start(
            mode=visualizer_mode,
            has_reinforcing_trials=has_reinforcing_trials,
            has_aversive_trials=has_aversive_trials,
        )

        # Synchronizes the Unity game engine's state with the initial state of the runtime control's UI before
        # entering the checkpoint loop.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            self._unity_state.reinforcing_guidance_enabled = self._ui.enable_reinforcing_guidance
            self._toggle_reinforcing_guidance(enable_guidance=self._unity_state.reinforcing_guidance_enabled)
            self._unity_state.aversive_guidance_enabled = self._ui.enable_aversive_guidance
            self._toggle_aversive_guidance(enable_guidance=self._unity_state.aversive_guidance_enabled)

        # Initializes the runtime visualizer. This HAS to be initialized after cameras and the UI to prevent collisions
        # in the QT backend, which is used by all three assets.
        self._visualizer.open(
            mode=visualizer_mode,
            has_reinforcing_trials=has_reinforcing_trials,
            has_aversive_trials=has_aversive_trials,
        )

        # Enters the manual checkpoint loop. This loop holds the runtime and allows using the GUI to test all runtime
        # components before starting the data acquisition.
        self._checkpoint()

        # If the user chooses to abort (terminate) the runtime during checkpoint, aborts the method runtime early.
        if self._terminated:
            # Sets the flag to True to support the proper stop() method runtime.
            self._started = True
            return

        message = "Initiating data acquisition..."
        console.echo(message=message, level=LogLevel.INFO)

        # Starts saving frames from al cameras
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Starts mesoscope frame acquisition if the runtime is a mesoscope experiment.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            # Enables mesoscope frame monitoring
            self._microcontrollers.mesoscope_frame.set_monitoring_state(state=True)

            # Ensures that the frame monitoring starts before acquisition.
            _response_delay_timer.delay(delay=1000, block=False)  # Uses the global response delay timer.

            # Starts acquiring mesoscope frames.
            self._start_mesoscope()

        # The setup procedure is complete.
        self._started = True

        message = "Mesoscope-VR system: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def stop(self) -> None:
        """Stops all Mesoscope-VR system components, external assets, and ends the session's data acquisition."""
        # If all assets are already stopped, aborts the runtime early.
        if not self._started:
            return

        # Resets the _started tracker before attempting the shutdown sequence
        self._started = False

        message = "Terminating Mesoscope-VR system runtime..."
        console.echo(message=message, level=LogLevel.INFO)

        # Switches the system into the IDLE state. Since IDLE state has most modules set to stop-friendly states,
        # this is used as a shortcut to prepare the VR system for shutdown. Also, this clearly marks the end of the
        # main runtime period.
        self.idle()

        # Shuts down the UI and the visualizer.
        self._ui.shutdown()
        self._visualizer.close()

        # Disconnects from the MQTT broker that facilitates communication with Unity.
        self._unity.disconnect()

        # Stops all cameras.
        self._cameras.stop()

        # Stops mesoscope frame acquisition and monitoring if the runtime uses Mesoscope.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT and self._mesoscope_started:
            self._stop_mesoscope()
            self._microcontrollers.mesoscope_frame.set_monitoring_state(state=False)

            # Renames the mesoscope data directory to include the session name. This both clears the shared directory
            # for the next acquisition and ensures that the mesoscope data collected during runtime will be preserved
            # unless it is preprocessed or the user removes it manually.
            rename_mesoscope_directory(mesoscope_data=self._mesoscope_data)

        # Generates the snapshot of the current Zaber motor positions and saves them as a .yaml file. This has
        # to be done before Zaber motors are potentially reset back to parking position.
        _generate_zaber_snapshot(
            session_data=self._session_data, mesoscope_data=self._mesoscope_data, zaber_motors=self._zaber_motors
        )

        # Updates the internally stored SessionDescriptor instance with runtime data, saves it to disk, and instructs
        # the user to add experimenter notes and other user-defined information to the descriptor file.
        self._generate_session_descriptor()

        # Generates the snapshot of the positions used by all mesoscope's imaging axes.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            _generate_mesoscope_position_snapshot(session_data=self._session_data, mesoscope_data=self._mesoscope_data)

        # Optionally resets Zaber motors by moving them to the dedicated parking position before shutting down Zaber
        # connection. Regardless of whether the motors are moved, disconnects from the motors at the end of the method's
        # runtime.
        _reset_zaber_motors(zaber_motors=self._zaber_motors)

        # Stops all microcontroller interfaces
        self._microcontrollers.stop()

        # Stops the data logger instance
        self._logger.stop()

        message = "Data Logger: Stopped."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Cleans up all SharedMemoryArray objects and leftover references before entering data processing mode to
        # support parallel runtime preparations.
        del self._microcontrollers
        del self._zaber_motors
        del self._cameras
        del self._logger

        # Notifies the user that the acquisition is complete.
        console.echo(message="Data acquisition: Complete.", level=LogLevel.SUCCESS)

        # If the session was not fully initialized, skips the preprocessing.
        if self._session_data.raw_data.nk_path.exists():
            return

        # Determines whether to carry out data preprocessing or purging.
        message = (
            "Do you want to preprocess or purge the the acquired session's data? CRITICAL! Only enter 'purge session' "
            "if you want to permanently DELETE the session's data. All valid data REQUIRES preprocessing to ensure "
            "safe storage."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        while True:
            answer = input("Enter 'yes', 'no' or 'purge session': ")

            # Default case: preprocesses the data.
            if answer.lower() == "yes":
                preprocess_session_data(session_data=self._session_data)
                break

            # Does not carry out data preprocessing or purging. In certain scenarios, it may be necessary to skip data
            # preprocessing in favor of faster animal turnover.
            if answer.lower() == "no":
                break

            # Exclusively for failed runtimes: removes all session data from all destinations.
            if answer.lower() == "purge session":
                purge_session(session_data=self._session_data)
                break

        message = "Mesoscope-VR system runtime: Terminated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _generate_hardware_state_snapshot(self) -> None:
        """Resolves and caches the snapshot of the system's hardware configuration parameters to the acquired session's
        raw_data directory as a hardware_state.yaml file.
        """
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT and self._experiment_configuration:
            hardware_state = MesoscopeHardwareState(
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                maximum_brake_strength=float(self._microcontrollers.brake.maximum_brake_strength),
                minimum_brake_strength=float(self._microcontrollers.brake.minimum_brake_strength),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                screens_initially_on=self._microcontrollers.screens.state,
                recorded_mesoscope_ttl=True,
                delivered_gas_puffs=any(
                    isinstance(t, GasPuffTrial) for t in self._experiment_configuration.trial_structures.values()
                ),
                system_state_codes=_MesoscopeVRStates.to_dict(),
            )
        # Note, lick and run training runtimes only use a subset of all hardware modules.
        elif self._session_data.session_type == SessionTypes.LICK_TRAINING:
            hardware_state = MesoscopeHardwareState(
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                delivered_gas_puffs=False,
                system_state_codes=_MesoscopeVRStates.to_dict(),
            )
        elif self._session_data.session_type == SessionTypes.RUN_TRAINING:
            hardware_state = MesoscopeHardwareState(
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                delivered_gas_puffs=False,
                system_state_codes=_MesoscopeVRStates.to_dict(),
            )
        else:
            # It should be impossible to satisfy this error clause, but is kept for safety reasons
            message = (
                f"Unsupported session type {self._session_data.session_type} encountered when generating "
                f"the snapshot of the Mesoscope-VR system's hardware configuration."
            )
            console.error(message=message, error=ValueError)
            # A fall-back to appease mypy, should not be reachable
            raise ValueError(message)  # pragma: no cover

        # Caches the resolved hardware state to disk
        hardware_state.to_yaml(self._session_data.raw_data.hardware_state_path)
        message = "Mesoscope-VR hardware configuration snapshot: Generated."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _generate_session_descriptor(self) -> None:
        """Updates the contents of the acquired session's descriptor file with data collected during runtime and caches
        it to the session's raw_data directory.
        """
        # The presence of the 'nk.bin' marker indicates that the session has not been properly initialized. Since
        # this method can be called as part of the emergency shutdown process for a session that encountered an
        # initialization error, if the marker exists, ends the runtime early.
        if self._session_data.raw_data.nk_path.exists():
            return

        # Updates the contents of the pregenerated descriptor file and dumps it as a .yaml into the root raw_data
        # session directory.

        # Runtime water volume. This should accurately reflect the volume of water consumed by the animal during
        # runtime.
        delivered_water = self._microcontrollers.valve.delivered_volume - self._paused_water_volume
        # Converts from uL to ml
        self.descriptor.dispensed_water_volume_ml = float(round(delivered_water / 1000, ndigits=3))

        # Same as above, but tracks the total volume of water dispensed during pauses. While the animal might
        # have consumed some of that water, it is equally plausible that all water was wasted or not dispensed at all.
        self.descriptor.pause_dispensed_water_volume_ml = float(round(self._paused_water_volume / 1000, ndigits=3))

        self.descriptor.incomplete = False  # If the runtime reaches this point, the session is likely complete.

        # Precalculates the volume of water that the experimenter needs to deliver to the animal if the combined
        # volume delivered during runtime and paused state is less than 1 ml. This is used to pre-fill the
        # experimenter-delivered volume field as a convenience feature for experimenters.
        total_delivered_volume = (
            self.descriptor.dispensed_water_volume_ml + self.descriptor.pause_dispensed_water_volume_ml
        )
        if total_delivered_volume < 1:
            self.descriptor.experimenter_given_water_volume_ml = float(round(1 - total_delivered_volume, ndigits=3))

        # Ensures that the user updates the descriptor file.
        _verify_descriptor_update(
            descriptor=self.descriptor, session_data=self._session_data, mesoscope_data=self._mesoscope_data
        )

    def _setup_unity(self) -> None:
        """Guides the user through the setup sequence for the Unity game engine and the session's Virtual Reality
        task environment.
        """
        # Stage 1: Verifies that the Unity scene matches the expected task configuration.
        while True:
            # Clears the Unity communication buffer and instructs the user to arm Unity.
            self._clear_unity_buffer()
            message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
            console.echo(message=message, level=LogLevel.INFO)

            # Waits for Unity to send the startup confirmation message.
            self._wait_for_unity_topic(expected_topic=_MesoscopeVRMQTTTopics.UNITY_STARTUP)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

            # Verifies that Unity is configured to display the correct scene.
            message = "Verifying that the Unity game engine is configured to display the correct scene..."
            console.echo(message=message, level=LogLevel.INFO)

            # Sends a request for the scene (task) name to Unity.
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.UNITY_SCENE_REQUEST)

            # Blocks until Unity sends the active task scene name.
            payload = self._wait_for_unity_topic(expected_topic=_MesoscopeVRMQTTTopics.UNITY_SCENE)

            # Extracts the name of the scene running in Unity.
            scene_name: str = json.loads(payload.decode("utf-8"))["name"]
            expected_scene_name: str = self._experiment_configuration.unity_scene_name  # type: ignore[union-attr]

            if scene_name == expected_scene_name:
                # If the scene name matches the expected name, advances to the next stage.
                message = "Unity scene configuration: Confirmed."
                console.echo(message=message, level=LogLevel.SUCCESS)
                break

            # Otherwise, displays an error message and prompts the user to fix the Unity configuration.
            message = (
                f"The name of the Virtual Reality scene (task) running in Unity ({scene_name}) does not match the "
                f"scene name expected based on the session's experiment configuration ({expected_scene_name}). "
                f"Reconfigure Unity to run the correct VR task and try again."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

        # Stage 2: Verifies that the Unity task displays correctly on the VR screens.
        # Activates the VR screens so that the user can check whether the Unity task displays as expected.
        self._microcontrollers.screens.set_state(state=True)

        # Delays the runtime for 2 seconds to ensure that the VR screen controllers receive the activation pulse.
        _response_delay_timer.delay(delay=2000, block=False)

        message = (
            "Verify that the Virtual Reality scene displays on the VR screens as intended. Disable (end) Unity "
            "runtime when ready to advance to the next preparation step."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Continuously loops the display verification process until the user confirms success.
        while True:
            # Sends continuous position updates to Unity to animate the VR environment for visual verification.
            while True:
                # Prevents the motion from being too fast.
                _response_delay_timer.delay(delay=100, block=False)

                # Advances the Unity scene forward by 0.1 Unity unit (~ 10 mm).
                json_string = dumps(obj={"movement": 0.1})
                byte_array = json_string.encode("utf-8")
                self._unity.send_data(topic=_MesoscopeVRMQTTTopics.ENCODER_DATA, payload=byte_array)

                # Parses incoming data from Unity to detect termination.
                data = self._unity.get_data()
                if data is None:
                    continue

                # Breaks the animation loop when Unity sends a termination message.
                if data[0] == _MesoscopeVRMQTTTopics.UNITY_TERMINATION:
                    message = "Unity termination: Detected."
                    console.echo(message=message, level=LogLevel.INFO)
                    break

            # Prompts the user to confirm whether the display verification was successful.
            message = "Did the Virtual Reality display render correctly on the VR screens?"
            console.echo(message=message, level=LogLevel.INFO)

            # Requests the user to provide a valid answer.
            answer = ""
            while answer not in {"y", "n"}:
                user_input = input("Enter 'yes' or 'no': ").strip().lower()
                answer = user_input[0] if user_input else ""

            # Breaks the verification loop if the user confirms the displays are working correctly.
            if answer == "y":
                break

            # Otherwise, notifies the user that the verification will restart.
            message = (
                "Restarting VR display verification. Ensure Unity is properly configured and arm the task "
                "to begin the verification."
            )
            console.echo(message=message, level=LogLevel.WARNING)

            # Clears the buffer and waits for Unity to be re-armed before retrying.
            self._clear_unity_buffer()
            message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
            console.echo(message=message, level=LogLevel.INFO)
            self._wait_for_unity_topic(expected_topic=_MesoscopeVRMQTTTopics.UNITY_STARTUP)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

        # Re-arms Unity after successful display verification.
        self._clear_unity_buffer()
        message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
        console.echo(message=message, level=LogLevel.INFO)
        self._wait_for_unity_topic(expected_topic=_MesoscopeVRMQTTTopics.UNITY_STARTUP)
        message = "Unity state transition: Confirmed. Unity is now armed."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Disables the VR screens now that verification is complete.
        self._microcontrollers.screens.set_state(state=False)

        # Requests and resolves the Virtual Reality cue sequence for the task.
        self._get_cue_sequence()

        message = "Unity setup: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _wait_for_unity_topic(self, expected_topic: str) -> bytes | bytearray:
        """Waits for Unity to send a message on the specified topic.

        Args:
            expected_topic: The Unity communication topic to be monitored for incoming messages.

        Returns:
            The payload received with the message sent to the monitored topic.
        """
        while True:
            # Adds a brief delay to prevent CPU spinning.
            _response_delay_timer.delay(delay=10, block=False)

            # Retrieves the next available message from the Unity communication buffer.
            data = self._unity.get_data()

            # If the unity sent a message, checks whether the message was sent to the expected topic.
            if data is not None:
                topic, payload = data
                if topic == expected_topic:
                    return payload

    def _clear_unity_buffer(self) -> None:
        """Clears all pending messages from the MQTT communication buffer used to communicate with the Unity game
        engine.
        """
        while self._unity.has_data:
            _ = self._unity.get_data()

    def _get_cue_sequence(self) -> None:
        """Requests and resolves the Virtual Reality task environment's wall cue sequence for the acquired session."""
        # Clears the Unity communication buffer before requesting the cue sequence.
        self._clear_unity_buffer()

        message = (
            "Requesting Virtual Reality wall cue sequence from Unity. Ensure Unity is armed and the task is running."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Continuously retries until the cue sequence is successfully received.
        while True:
            # Sends the cue sequence request to Unity.
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.CUE_SEQUENCE_REQUEST)
            _response_delay_timer.reset()

            # Waits up to 5 seconds for Unity to respond with the cue sequence.
            _maximum_wait_time = 5000
            while _response_delay_timer.elapsed < _maximum_wait_time:
                # Adds small delay to prevent CPU spinning.
                _response_delay_timer.delay(delay=10, block=False)

                # Checks for incoming messages from Unity.
                data = self._unity.get_data()
                if data is None:
                    continue

                # Discards messages not related to the cue sequence.
                if data[0] != _MesoscopeVRMQTTTopics.CUE_SEQUENCE:
                    continue

                # Successfully received the cue sequence - extracts and processes it.
                self._unity_state.cue_sequence = np.array(
                    json.loads(data[1].decode("utf-8"))["cue_sequence"], dtype=np.uint8
                )

                # Logs the received sequence.
                self._logger.input_queue.put(
                    LogPackage(
                        source_id=self._source_id,
                        acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                        serialized_data=self._unity_state.cue_sequence,
                    )
                )

                # Decomposes the received cue sequence into a sequence of trials.
                self._decompose_cue_sequence_into_trials()

                # Resets the traveled distance tracker array and internal class attributes.
                self._microcontrollers.wheel_encoder.reset_distance_tracker()
                self._unity_state.position = np.float64(0.0)
                self._distance = np.float64(0.0)
                self._trial_state.completed = 0

                message = "VR cue sequence: Received."
                console.echo(message=message, level=LogLevel.SUCCESS)
                return

            # Timeout occurred - prompts the user to verify Unity is running and retry.
            message = (
                f"The Mesoscope-VR system sent a cue sequence request to Unity via the "
                f"'{_MesoscopeVRMQTTTopics.CUE_SEQUENCE_REQUEST}' topic but received no response within 5 seconds. "
                f"Ensure Unity is armed and the task is running."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

    def _decompose_cue_sequence_into_trials(self) -> None:
        """Decomposes the Virtual Reality environment's cue sequence of the acquired session into a sequence of trials.

        Notes:
            Uses a greedy longest-match approach to identify trial motifs in the processed cue sequence.

        Raises:
            RuntimeError: If the method is not able to fully decompose the Virtual Reality environment cue sequence into
                a sequence of trials.
        """
        # Extracts all trial data in a single pass to avoid redundant iterations. These arrays are indexed by trial
        # type, not by trial position in the sequence.
        trial_structures = self._experiment_configuration.trial_structures  # type: ignore[union-attr]
        trials = list(trial_structures.values())
        trial_motifs = []
        trial_distances = []
        reinforcing_rewards_by_type: list[tuple[float, int]] = []
        aversive_puff_durations_by_type: list[int] = []

        for trial in trials:
            trial_motifs.append(np.array(trial.cue_sequence, dtype=np.uint8))
            trial_distances.append(float(trial.trial_length_cm))
            if isinstance(trial, WaterRewardTrial):
                reinforcing_rewards_by_type.append((float(trial.reward_size_ul), int(trial.reward_tone_duration_ms)))
                aversive_puff_durations_by_type.append(0)  # Safe placeholder for reinforcing trials
            else:  # GasPuffTrial
                reinforcing_rewards_by_type.append((0.0, 0))  # Safe placeholder for aversive trials
                aversive_puff_durations_by_type.append(int(trial.puff_duration_ms))

        self._trial_state.trial_structures = trial_structures

        # Prepares the flattened motif data using the MotifDecomposer class.
        motifs_flat, motif_starts, motif_lengths, motif_indices, distances_array = (
            self._motif_decomposer.prepare_motif_data(trial_motifs, trial_distances)
        )

        # Estimates the maximum number of trials and calls Numba-accelerated decomposition.
        max_trials = len(self._unity_state.cue_sequence) // min(len(motif) for motif in trial_motifs) + 1
        trial_indices_array, trial_count = self._decompose_sequence_numba_flat(
            self._unity_state.cue_sequence, motifs_flat, motif_starts, motif_lengths, motif_indices, max_trials
        )

        # Checks for decomposition errors and raises RuntimeError with diagnostic information.
        if trial_count == -1:
            # Reconstructs the position where decomposition failed for error reporting.
            failed_position = sum(len(trial_motifs[index]) for index in trial_indices_array[:max_trials] if index != 0)
            remaining_cues = self._unity_state.cue_sequence[failed_position : failed_position + 20]

            message = (
                f"Unable to decompose the acquired session's Virtual Reality environment's cue sequence into a "
                f"sequence of trials. No trial motif matched the processed sequence at position {failed_position}. "
                f"The next 20 unmatched cues: {remaining_cues.tolist()}."
            )
            console.error(message=message, error=RuntimeError)

        # Constructs the cumulative distance array directly from decomposed trial indices.
        self._trial_state.distances = np.cumsum(distances_array[trial_indices_array[:trial_count]].astype(np.float64))

        # Builds per-trial reward and puff duration arrays from the decomposed sequence. Each entry corresponds to
        # a trial in the actual sequence, not a trial type.
        sequence_indices = trial_indices_array[:trial_count]
        self._trial_state.reinforcing_rewards = tuple(reinforcing_rewards_by_type[i] for i in sequence_indices)
        self._trial_state.aversive_puff_durations = tuple(aversive_puff_durations_by_type[i] for i in sequence_indices)

    @staticmethod
    @njit(cache=True)
    def _decompose_sequence_numba_flat(
        cue_sequence: NDArray[np.uint8],
        motifs_flat: NDArray[np.uint8],
        motif_starts: NDArray[np.int32],
        motif_lengths: NDArray[np.int32],
        motif_indices: NDArray[np.int32],
        max_trials: int,
    ) -> tuple[NDArray[np.int32], int]:
        """Decomposes a long sequence of Virtual Reality (VR) wall cues into individual trial motifs.

        Notes:
            This worker function is used to speed up decomposition via numba-acceleration.

        Args:
            cue_sequence: The full Virtual Reality environment cue sequence to decompose.
            motifs_flat: All trial type motifs supported by the acquired session, concatenated into a single 1D array.
            motif_starts: The starting index of each unique motif in the motifs_flat array.
            motif_lengths: The length of each unique motif in the motifs_flat array.
            motif_indices: Stores the original trial type motif indices before they are sorted to optimize the lookup
                speed.
            max_trials: The maximum number of trials that can make up the entire cue sequence.

        Returns:
            A tuple of two elements. The first element is the array of trials (trial-type indices) decoded from the
            cue sequence. The second element is the total number of trials extracted from the cue sequence.
        """
        # Prepares runtime trackers.
        trial_indices = np.zeros(max_trials, dtype=np.int32)
        trial_count = 0
        sequence_pos = 0
        sequence_length = len(cue_sequence)
        num_motifs = len(motif_lengths)

        # Decomposes the sequence into trial motifs using greedy matching. Longer motifs are matched over shorter ones.
        while sequence_pos < sequence_length and trial_count < max_trials:
            motif_found = False

            for i in range(num_motifs):
                motif_length = motif_lengths[i]

                # Checks if the current position allows for a complete motif match.
                if sequence_pos + motif_length <= sequence_length:
                    motif_start = motif_starts[i]

                    # Checks if the motif matches the current sequence position.
                    match = True
                    for j in range(motif_length):
                        if cue_sequence[sequence_pos + j] != motifs_flat[motif_start + j]:
                            match = False
                            break

                    # Records the match and advances to the next sequence position.
                    if match:
                        trial_indices[trial_count] = motif_indices[i]
                        trial_count += 1
                        sequence_pos += motif_length
                        motif_found = True
                        break

            # Returns error code if no motif matches the current position.
            if not motif_found:
                return trial_indices, -1

        return trial_indices[:trial_count], trial_count

    def _start_mesoscope(self) -> None:
        """Generates the mesoscope acquisition start marker file on the ScanImagePC and waits for the frame acquisition
        to begin.
        """
        # Clears the mesoscope marker files before attempting to start acquisition.
        self._clear_mesoscope_markers()

        # Continuously retries starting the mesoscope acquisition until successful.
        while True:
            # Resets the frame counter.
            self._microcontrollers.mesoscope_frame.reset_pulse_count()

            # Verifies that the mesoscope is not already acquiring frames.
            _response_delay_timer.delay(delay=1000, block=False)
            if self._microcontrollers.mesoscope_frame.pulse_count > 0:
                message = (
                    "Unable to trigger mesoscope frame acquisition, as the mesoscope is already acquiring frames. "
                    "This indicates that the setupAcquisition() MATLAB function did not run as expected. Re-run the "
                    "setupAcquisition function and try again."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                input("Enter anything to retry: ")
                continue

            # Clears any unexpected TIFF files the first time the method is called for a session. This ensures that the
            # number of mesoscope frame acquisition pulses always matches the number of frames recorded for the
            # session.
            if not self._mesoscope_started:
                for pattern in ["*.tif", "*.tiff"]:
                    for file in self._mesoscope_data.scanimagepc_data.mesoscope_data_path.glob(pattern):
                        # Excludes zstack files generated during the imaging field setup from cleanup.
                        if "zstack" not in file.name:
                            file.unlink(missing_ok=True)

            # Sends the acquisition trigger by creating the kinase marker file.
            self._mesoscope_data.scanimagepc_data.kinase_path.touch()

            message = "Mesoscope acquisition trigger: Sent. Waiting for the mesoscope frame acquisition to start..."
            console.echo(message=message, level=LogLevel.INFO)

            # Waits for mesoscope to start acquiring at least 10 frames.
            _response_delay_timer.reset()
            _maximum_wait_time = 15000  # 15 seconds
            _expected_pulses = 10
            while _response_delay_timer.elapsed < _maximum_wait_time:
                # Adds delay to prevent CPU spinning.
                _response_delay_timer.delay(delay=10, block=False)

                if self._microcontrollers.mesoscope_frame.pulse_count > _expected_pulses:
                    message = "Mesoscope frame acquisition: Started."
                    console.echo(message=message, level=LogLevel.SUCCESS)

                    # Sets up continuous mesoscope frame acquisition monitoring.
                    self._mesoscope_frame_count = self._microcontrollers.mesoscope_frame.pulse_count
                    self._mesoscope_timer.reset()
                    self._mesoscope_started = True
                    return

            # If the timeout window expires without receiving any mesoscope frames, clears the markers and prompts the
            # user to reconfigure the mesoscope.
            self._clear_mesoscope_markers()
            message = (
                "The Mesoscope-VR system has requested the mesoscope to start acquiring frames and failed to "
                "receive 10 frame acquisition triggers over 15 seconds. It is likely that the mesoscope has not "
                "been armed for externally-triggered frame acquisition or that the mesoscope frame monitoring "
                "module is not functioning. Make sure the Mesoscope is configured for data acquisition and try again."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

    def _stop_mesoscope(self) -> None:
        """Sends the frame acquisition stop TTL pulse to the mesoscope and waits for the frame acquisition to stop.

        This method is used internally to stop the mesoscope frame acquisition as part of the stop() method runtime.

        Notes:
            This method contains an infinite loop that waits for the mesoscope to stop generating frame acquisition
            triggers.
        """
        # Clears the mesoscope marker files to trigger acquisition shutdown.
        self._clear_mesoscope_markers()

        # Creates the phosphatase marker as a fallback termination mechanism.
        self._mesoscope_data.scanimagepc_data.phosphatase_path.touch()

        message = "Waiting for the Mesoscope to stop acquiring frames..."
        console.echo(message=message, level=LogLevel.INFO)

        # Monitors for mesoscope frame acquisition to stop.
        self._microcontrollers.mesoscope_frame.reset_pulse_count()

        while True:
            # Waits 2 seconds between checks (mesoscope runs at ~10 Hz, so 2s = ~20 frames if still running).
            _response_delay_timer.delay(delay=2000, block=False)

            # If no frames received during the 2-second delay, mesoscope has stopped.
            if self._microcontrollers.mesoscope_frame.pulse_count == 0:
                break

            # Resets counter and continues monitoring.
            self._microcontrollers.mesoscope_frame.reset_pulse_count()

        # Cleans up the phosphatase marker file.
        self._mesoscope_data.scanimagepc_data.phosphatase_path.unlink(missing_ok=True)

    def _clear_mesoscope_markers(self) -> None:
        """Clears all mesoscope acquisition marker files from the ScanImagePC's shared mesoscope data directory.

        This utility method removes both the kinase (start) and phosphatase (stop) marker files, ensuring
        a clean directory state before sending new acquisition commands to the mesoscope.
        """
        self._mesoscope_data.scanimagepc_data.kinase_path.unlink(missing_ok=True)
        self._mesoscope_data.scanimagepc_data.phosphatase_path.unlink(missing_ok=True)

    def _checkpoint(self) -> None:
        """Instructs the user to verify the functioning of all GUI-addressable Mesoscope-VR components before starting
        the session's data acquisition.
        """
        message = (
            "Runtime preparation: Complete. Carry out all final checks and adjustments, such as priming the water "
            "delivery valve. When you are ready to start the runtime, use the UI to 'resume' it."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = (
            "Note: All sensors, including the lick sensor, are DISABLED at this time. If you are running a training "
            "session, apply the electroconductive gel to the headbar to ensure the lick sensor works as expected "
            "during the runtime."
        )
        console.echo(message=message, level=LogLevel.WARNING)

        while self._ui.pause_runtime:
            self._visualizer.update()

            if self._ui.reward_signal:
                self._deliver_reward(reward_size=self._ui.reward_volume)

            if self._ui.open_valve:
                self._microcontrollers.valve.set_state(state=True)

            if self._ui.close_valve:
                self._microcontrollers.valve.set_state(state=False)

            if self._ui.gas_valve_open_signal:
                self._microcontrollers.gas_puff_valve.set_state(state=True)

            if self._ui.gas_valve_close_signal:
                self._microcontrollers.gas_puff_valve.set_state(state=False)

            if self._ui.gas_valve_puff_signal:
                self._microcontrollers.gas_puff_valve.deliver_puff(duration_ms=self._ui.gas_valve_puff_duration)

            if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
                if self._ui.enable_reinforcing_guidance != self._unity_state.reinforcing_guidance_enabled:
                    self._toggle_reinforcing_guidance(enable_guidance=self._ui.enable_reinforcing_guidance)
                if self._ui.enable_aversive_guidance != self._unity_state.aversive_guidance_enabled:
                    self._toggle_aversive_guidance(enable_guidance=self._ui.enable_aversive_guidance)

            if self._ui.exit_signal:
                self._terminate_runtime()
                if self._terminated:
                    break

        self._microcontrollers.valve.set_state(state=False)
        self._paused_water_volume += self._microcontrollers.valve.delivered_volume
        self._unconsumed_reward_count = 0

        # Signals the UI that setup is complete - this permanently disables valve open/close buttons.
        self._ui.set_setup_complete()

    def _toggle_reinforcing_guidance(self, *, enable_guidance: bool) -> None:
        """Sets the reinforcing trial guidance mode.

        Args:
            enable_guidance: Determines whether to enable or disable reinforcing guidance.
        """
        if not enable_guidance:
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.DISABLE_LICK_GUIDANCE)
        else:
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.ENABLE_LICK_GUIDANCE)

        # Logs the reinforcing guidance state change.
        log_package = LogPackage(
            source_id=self._source_id,
            acquisition_time=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array(
                [_MesoscopeVRLogMessageCodes.REINFORCING_GUIDANCE_STATE, enable_guidance], dtype=np.uint8
            ),
        )
        self._logger.input_queue.put(log_package)

        # Updates the unity state tracker.
        self._unity_state.reinforcing_guidance_enabled = enable_guidance

    def _toggle_aversive_guidance(self, *, enable_guidance: bool) -> None:
        """Sets the aversive trial guided mode.

        Args:
            enable_guidance: Determines whether to enable or disable aversive guidance.
        """
        if not enable_guidance:
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.DISABLE_OCCUPANCY_GUIDANCE)
        else:
            self._unity.send_data(topic=_MesoscopeVRMQTTTopics.ENABLE_OCCUPANCY_GUIDANCE)

        # Logs the aversive guidance state change.
        log_package = LogPackage(
            source_id=self._source_id,
            acquisition_time=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array(
                [_MesoscopeVRLogMessageCodes.AVERSIVE_GUIDANCE_STATE, enable_guidance], dtype=np.uint8
            ),
        )
        self._logger.input_queue.put(log_package)

        # Updates the unity state tracker.
        self._unity_state.aversive_guidance_enabled = enable_guidance

    def _change_system_state(self, new_state: int) -> None:
        """Updates and logs the new Mesoscope-VR system state.

        Args:
            new_state: The unique code for the newly activated Mesoscope-VR system state.
        """
        # Ensures that the _system_state attribute is set to a non-zero value after runtime initialization. This is
        # used to restore the runtime back to the pre-pause state if the runtime enters the paused state (idle), but the
        # user then chooses to resume the runtime.
        if new_state != _MesoscopeVRStates.IDLE:
            self._system_state = new_state  # Updates the Mesoscope-VR system state

        # Logs the system state update. Uses header-code 1 to indicate that the logged value is the system state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            acquisition_time=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([_MesoscopeVRLogMessageCodes.SYSTEM_STATE, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def change_runtime_state(self, new_state: int) -> None:
        """Updates and logs the new acquired session's runtime state (stage).

        Args:
            new_state: The unique code for the new session's runtime state.
        """
        # Ensures that the _runtime_state attribute is set to a non-zero value after runtime initialization. This is
        # used to restore the runtime back to the pre-pause state if the runtime enters the paused state (idle), but the
        # user then chooses to resume the runtime.
        if new_state != _MesoscopeVRStates.IDLE:
            self._runtime_state = new_state

        # Logs the runtime state update. Uses header-code 2 to indicate that the logged value is the runtime state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            acquisition_time=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([_MesoscopeVRLogMessageCodes.RUNTIME_STATE, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def idle(self) -> None:
        """Switches the Mesoscope-VR system to the idle state.

        Notes:
            This state is designed to be used exclusively during periods where the runtime pauses and does not generate
            any valid data.

            In the idle state, the brake is engaged and the screens are turned Off. All sensors other than the mesoscope
            frame acquisition TTL sensor are disabled.

            Setting the system to 'idle' also automatically changes the runtime state to 0 (idle).
        """
        # Switches runtime state to 0
        self.change_runtime_state(new_state=_MesoscopeVRStates.IDLE)

        # Blackens the VR screens
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake
        self._microcontrollers.brake.set_state(state=True)

        # Disables all sensor monitoring
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)
        self._microcontrollers.torque.set_monitoring_state(state=False)
        self._microcontrollers.lick.set_monitoring_state(state=False)

        # Sets system state to 0
        self._change_system_state(_MesoscopeVRStates.IDLE)

    def rest(self) -> None:
        """Switches the Mesoscope-VR system to the rest state.

        Notes:
            In the rest state, the brake is engaged and the screens are turned off. The encoder sensor is
            disabled, and the torque sensor is enabled.
        """
        # Enables lick monitoring
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Blackens the VR screens
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake
        self._microcontrollers.brake.set_state(state=True)

        # Suspends encoder monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)

        # Enables torque monitoring.
        self._microcontrollers.torque.set_monitoring_state(state=True)

        # Sets system state to 1
        self._change_system_state(_MesoscopeVRStates.REST)

    def run(self) -> None:
        """Switches the Mesoscope-VR system to the run state.

        Notes:
            In the rest state, the brake is disengaged and the screens are turned off. The encoder sensor is
            enabled, and the torque sensor is disabled.
        """
        # Enables lick monitoring
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Initializes encoder monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=True)

        # Disables torque monitoring.
        self._microcontrollers.torque.set_monitoring_state(state=False)

        # Activates VR screens.
        self._microcontrollers.screens.set_state(state=True)

        # Disengages the brake
        self._microcontrollers.brake.set_state(state=False)

        # Sets system state to 2
        self._change_system_state(_MesoscopeVRStates.RUN)

    def lick_train(self) -> None:
        """Switches the Mesoscope-VR system to the lick training state.

        Notes:
            In this state, the brake is engaged and the screens are turned off. The encoder sensor is
            disabled, and the torque sensor is enabled.

            Calling this method automatically switches the runtime state to 255 (active training).
        """
        # Switches runtime state to 255 (active)
        self.change_runtime_state(new_state=255)

        # Blackens the VR screens
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake
        self._microcontrollers.brake.set_state(state=True)

        # Disables encoder monitoring
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)

        # Initiates torque monitoring
        self._microcontrollers.torque.set_monitoring_state(state=True)

        # Initiates lick monitoring
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Sets system state to 3
        self._change_system_state(_MesoscopeVRStates.LICK_TRAINING)

    def run_train(self) -> None:
        """Switches the Mesoscope-VR system to the run training state.

        Notes:
            In this state, the brake is disengaged and the screens are turned off. The encoder sensor is
            enabled, and the torque sensor is disabled.

            Calling this method automatically switches the runtime state to 255 (active training).
        """
        # Switches runtime state to 255 (active)
        self.change_runtime_state(new_state=255)

        # Blackens the VR screens
        self._microcontrollers.screens.set_state(state=False)

        # Disengages the brake.
        self._microcontrollers.brake.set_state(state=False)

        # Ensures that encoder monitoring is enabled
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=True)

        # Ensures torque monitoring is disabled
        self._microcontrollers.torque.set_monitoring_state(state=False)

        # Initiates lick monitoring
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Sets system state to 4
        self._change_system_state(_MesoscopeVRStates.RUN_TRAINING)

    def update_visualizer_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None:
        """Instructs the data visualizer to update the displayed running speed and running epoch duration thresholds
        using the input data.

        Args:
            speed_threshold: The running speed threshold, in centimeters per second, which specifies how fast the
                animal should be running to satisfy the current task conditions.
            duration_threshold: The running epoch duration threshold, in milliseconds, which specifies how long the
                animal must maintain the above-threshold speed to satisfy the current task conditions.
        """
        # Each time visualizer thresholds are updated, also updates the descriptor. For this, converts NumPy scalars to
        # Python float objects (a requirement to make them YAML-compatible).
        if isinstance(self.descriptor, RunTrainingDescriptor):
            self.descriptor.final_run_speed_threshold_cm_s = round(float(speed_threshold), 2)
            # Converts time from milliseconds to seconds
            self.descriptor.final_run_duration_threshold_s = round(float(duration_threshold) / 1000, 2)

        self._visualizer.update_run_training_thresholds(
            speed_threshold=speed_threshold, duration_threshold=duration_threshold
        )

    def _deliver_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        Args:
            reward_size: The volume of water to deliver, in microliters. If this argument is set to None, the method
                will use the same volume as used during the previous reward delivery or as set via the GUI.
            tone_duration: The time, in milliseconds, for which to sound the auditory tone while delivering the reward.
        """
        self._unconsumed_reward_count += 1  # Increments the unconsumed reward count each time reward is delivered.
        self._microcontrollers.valve.deliver_reward(volume=reward_size, tone_duration=tone_duration)

        # Configures the visualizer to display the valve activation event during the next update cycle.
        self._visualizer.add_valve_event()

    def _simulate_reward(self, tone_duration: int = 300) -> None:
        """Uses the buzzer controlled by the valve module to deliver an audible tone without delivering any water
        reward.

        Args:
            tone_duration: The time, in milliseconds, for which to sound the auditory tone.
        """
        self._microcontrollers.valve.simulate_reward(tone_duration=tone_duration)

    def resolve_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> bool:
        """Depending on the current number of unconsumed rewards and runtime configuration, either delivers or simulates
        the requested volume of water reward.

        Args:
            reward_size: The volume of water to deliver, in microliters.
            tone_duration: The time, in milliseconds, for which to sound the auditory tone while delivering the reward.

        Returns:
            True if the method delivers the water reward, False if it simulates it.
        """
        # Only delivers water rewards if the current unconsumed count value is below the user-defined threshold.
        if self._unconsumed_reward_count < self.descriptor.maximum_unconsumed_rewards:
            self._deliver_reward(reward_size=reward_size, tone_duration=tone_duration)
            return True

        # Otherwise, simulates water reward by sounding the buzzer without delivering any water
        self._simulate_reward(tone_duration=tone_duration)
        return False

    def runtime_cycle(self) -> None:
        """Sequentially carries out all cyclic Mesoscope-VR runtime tasks.

        Notes:
            This method must be called as part of the runtime cycle loop of the runtime management function that
            interfaces with the Mesoscope-VR system to acquire the managed session's data.
        """
        # This loop is used to keep the runtime in the runtime cycle if runtime is paused. This effectively suspends
        # external runtime logic.
        while True:
            # Handles animal behavior data updates.
            self._data_cycle()

            # Continuously updates the visualizer
            self._visualizer.update()

            # Synchronizes the runtime state with the state of the user-facing GUI
            self._ui_cycle()

            # If the GUI was used to terminate the runtime, aborts the cycle early
            if self.terminated:
                return

            # For experiment runtime, also executes the dedicated Unity and Mesoscope cycles.
            if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
                self._unity_cycle()
                self._mesoscope_cycle()

            # As long as the runtime is not paused, returns after running the cycle once. Otherwise, continuously loops
            # the cycle until the user uses the UI to resume the runtime or terminate it.
            if not self._paused:
                return

    def _data_cycle(self) -> None:
        """Queries and synchronizes changes to animal runtime behavior metrics with Unity and the visualizer class.

        This method reads the data sent by low-level data acquisition modules and updates class attributes used to
        support runtime logic, data visualization, and Unity VR task. If necessary, it directly communicates the updates
        to Unity via MQTT and to the visualizer through appropriate methods.
        """
        # Reads the total distance traveled by the animal and the current position of the animal in Unity units.
        traveled_distance = self._microcontrollers.wheel_encoder.traveled_distance
        current_position = self._microcontrollers.wheel_encoder.absolute_position

        # Updates running speed over ~50 millisecond windows.
        if self._speed_timer.elapsed >= self._speed_calculation_window:
            self._speed_timer.reset()
            running_speed = np.float64(((traveled_distance - self._distance) / 100) * 1000)
            self._distance = traveled_distance
            self._running_speed = running_speed
            self._visualizer.update_running_speed(running_speed)

        # Handles Unity-based virtual reality task execution for experiment sessions.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            # Computes the change in the animal's position and sends updates to Unity.
            position_delta = current_position - self._unity_state.position

            if position_delta != 0:
                self._unity_state.position = current_position
                json_string = dumps(obj={"movement": position_delta})
                byte_array = json_string.encode("utf-8")
                self._unity.send_data(topic=_MesoscopeVRMQTTTopics.ENCODER_DATA, payload=byte_array)

            # Checks if the animal has completed the current trial.
            if self._trial_state.trial_completed(traveled_distance):
                # Capture outcome BEFORE advance_trial() resets the flags
                is_aversive = self._trial_state.is_current_trial_aversive()
                if is_aversive:
                    succeeded = self._trial_state.aversive_succeeded
                    was_guided = self._trial_state.aversive_guided_trials > 0
                else:
                    succeeded = self._trial_state.reinforcing_rewarded
                    was_guided = self._trial_state.reinforcing_guided_trials > 0

                failed_count = self._trial_state.advance_trial()

                # Report outcome to visualizer
                self._visualizer.add_trial_outcome(is_aversive=is_aversive, succeeded=succeeded, was_guided=was_guided)

                # Handles recovery mode activation based on trial type.
                if is_aversive:
                    threshold = self._trial_state.aversive_recovery_threshold
                    recovery_trials = self._trial_state.aversive_recovery_trials
                    if failed_count >= threshold and recovery_trials > 0:
                        self._trial_state.aversive_failed_trials = 0
                        self._trial_state.aversive_guided_trials = recovery_trials
                        self._ui.set_aversive_guidance_state(enabled=True)
                else:
                    threshold = self._trial_state.reinforcing_recovery_threshold
                    recovery_trials = self._trial_state.reinforcing_recovery_trials
                    if failed_count >= threshold and recovery_trials > 0:
                        self._trial_state.reinforcing_failed_trials = 0
                        self._trial_state.reinforcing_guided_trials = recovery_trials
                        self._ui.set_reinforcing_guidance_state(enabled=True)

        # Handles incoming lick data
        lick_count = self._microcontrollers.lick.lick_count
        if lick_count > self._lick_count:
            self._lick_count = lick_count
            self._unconsumed_reward_count = 0
            self._visualizer.add_lick_event()

            if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
                self._unity.send_data(topic=_MesoscopeVRMQTTTopics.LICK_EVENT, payload=None)

        # Handles water delivery tracking
        dispensed_water = self._microcontrollers.valve.delivered_volume - (
            self._paused_water_volume + self._delivered_water_volume
        )
        if dispensed_water > 0:
            if self._paused:
                self._paused_water_volume += dispensed_water
            else:
                self._delivered_water_volume += dispensed_water

    def _unity_cycle(self) -> None:
        """Synchronizes the state of the acquired session's Virtual Reality task environment manged by the Unity game
         engine with the current state of the Mesoscope-VR system.

        Notes:
            During each runtime cycle, the method receives and parses exactly one message stored in the
            MQTTCommunication class buffer.
        """
        # If there is no Unity data to receive, aborts the runtime early.
        data = self._unity.get_data()
        if data is None:
            return

        # Handles stimulus delivery commands from Unity (both water reward and gas puff).
        if data[0] == _MesoscopeVRMQTTTopics.STIMULUS:
            if self._trial_state.is_current_trial_aversive():
                # Aversive trial: deliver gas puff
                puff_duration = self._trial_state.get_current_puff_duration()
                self._microcontrollers.gas_puff_valve.deliver_puff(duration_ms=puff_duration)

                # Decrements the guided trial counter for aversive trials.
                if self._trial_state.aversive_guided_trials > 0:
                    self._trial_state.aversive_guided_trials -= 1
                    if self._trial_state.aversive_guided_trials == 0:
                        self._ui.set_aversive_guidance_state(enabled=False)

                # Marks the aversive trial as failed (puff was delivered).
                self._trial_state.aversive_succeeded = False
            else:
                # Reinforcing trial: deliver water reward
                reward_size, tone_duration = self._trial_state.get_current_reward()
                self.resolve_reward(reward_size=reward_size, tone_duration=tone_duration)

                # Decrements the guided trial counter for reinforcing trials.
                if self._trial_state.reinforcing_guided_trials > 0:
                    self._trial_state.reinforcing_guided_trials -= 1
                    if self._trial_state.reinforcing_guided_trials == 0:
                        self._ui.set_reinforcing_guidance_state(enabled=False)

                # Marks the reinforcing trial as rewarded.
                self._trial_state.reinforcing_rewarded = True

        # Handles occupancy guidance delay messages for brake pulsing.
        if data[0] == _MesoscopeVRMQTTTopics.TRIGGER_DELAY:
            payload = json.loads(data[1].decode("utf-8"))
            delay_ms = payload.get("delay_ms", 0)
            if delay_ms > 0:
                self._microcontrollers.brake.send_pulse(duration_ms=delay_ms)

        # Handles Unity termination messages.
        if data[0] == _MesoscopeVRMQTTTopics.UNITY_TERMINATION:
            self._unity_state.terminated = True
            self._pause_runtime()
            message = "Emergency pause: Engaged. Reason: Unity sent a runtime termination message."
            console.echo(message=message, level=LogLevel.ERROR)

            # Logs the distance snapshot.
            traveled_distance = float(self._microcontrollers.wheel_encoder.traveled_distance)
            distance_bytes = np.array([traveled_distance], dtype="<f8").view(np.uint8)

            log_package = LogPackage(
                source_id=self._source_id,
                acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                serialized_data=np.concatenate(
                    [np.array([_MesoscopeVRLogMessageCodes.DISTANCE_SNAPSHOT], dtype=np.uint8), distance_bytes]
                ),
            )
            self._logger.input_queue.put(log_package)

            message = (
                "Address the issue that prevents Unity game engine from running and resume the runtime. Re-arm the "
                "Unity scene (hit the 'play' button) before resuming the runtime. Alternatively, terminate the runtime "
                "to attempt graceful shutdown."
            )
            console.echo(message=message, level=LogLevel.INFO)
            return

    def _ui_cycle(self) -> None:
        """Queries the state of various GUI components and adjusts the runtime behavior accordingly."""
        if self._ui.pause_runtime and not self._paused:
            self._pause_runtime()
        elif not self._ui.pause_runtime and self._paused:
            self._resume_runtime()

        if self._ui.exit_signal:
            self._terminate_runtime()
            if self.terminated:
                return

        if self._ui.reward_signal:
            self._deliver_reward(reward_size=self._ui.reward_volume)
            if self._paused:
                self._unconsumed_reward_count = 0

        # Handles gas valve puff signal. Note: open/close signals are only processed during initial setup.
        if self._ui.gas_valve_puff_signal:
            self._microcontrollers.gas_puff_valve.deliver_puff(duration_ms=self._ui.gas_valve_puff_duration)

        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            # Synchronizes guidance state with UI.
            if self._ui.enable_reinforcing_guidance != self._unity_state.reinforcing_guidance_enabled:
                self._toggle_reinforcing_guidance(enable_guidance=self._ui.enable_reinforcing_guidance)
            if self._ui.enable_aversive_guidance != self._unity_state.aversive_guidance_enabled:
                self._toggle_aversive_guidance(enable_guidance=self._ui.enable_aversive_guidance)

    def _mesoscope_cycle(self) -> None:
        """Checks whether mesoscope frame acquisition is active and, if not, emergency pauses the runtime."""
        # Aborts early if the cycle is called too early or if it is no longer necessary.
        if self._mesoscope_timer.elapsed < self._mesoscope_frame_delay or self._mesoscope_terminated:
            return

        # Updates frame count and resets the timer if frames are being received normally.
        current_pulse_count = self._microcontrollers.mesoscope_frame.pulse_count
        if self._mesoscope_frame_count < current_pulse_count:
            self._mesoscope_frame_count = current_pulse_count
            self._mesoscope_timer.reset()
            return

        # Frame acquisition has stopped - enters emergency pause state.
        self._mesoscope_terminated = True
        self._pause_runtime()

        message = "Emergency pause: Engaged. Reason: Mesoscope stopped sending frame acquisition triggers."
        console.echo(message=message, level=LogLevel.ERROR)

        # Cleans up acquisition markers to facilitate restart.
        self._stop_mesoscope()

        message = (
            "Address the issue that prevents the Mesoscope from acquiring frames and resume the runtime. Follow "
            "additional instructions displayed after resuming the runtime to re-arm the mesoscope to continue "
            "acquiring frames for the current runtime. Alternatively, terminate the runtime to attempt graceful "
            "shutdown."
        )
        console.echo(message=message, level=LogLevel.INFO)

    def _pause_runtime(self) -> None:
        """Pauses the session's data acquisition.

        Notes:
            When the runtime is paused, the Mesoscope-VR system locks into its internal cycle loop and does not release
            control to the main runtime logic loop. Additionally, it switches the system into the 'idle' state,
            effectively interrupting any ongoing task. The GUI and all external assets (Unity, Mesoscope) continue
            to function as normal unless manually terminated by the user.

            Any water dispensed through the valve during the paused state does not count against the water reward limit
            of the executed task.
        """
        # Ensures that the GUI reflects that the runtime is paused. While most paused states originate from the GUI,
        # certain events may cause the main runtime cycle to activate the paused state bypassing the GUI.
        if not self._ui.pause_runtime:
            self._ui.set_pause_state(paused=True)

        # Records pause onset time
        self._pause_start_time = self._timestamp_timer.elapsed

        # Switches the Mesoscope-VR system into the idle state.
        self.idle()

        # Notifies the user that the runtime has been paused
        message = "Mesoscope-VR runtime: Paused."
        console.echo(message=message, level=LogLevel.WARNING)

        # Sets the paused flag
        self._paused = True

    def _resume_runtime(self) -> None:
        """Resumes the session's data acquisition."""
        message = "Mesoscope-VR runtime: Resumed."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # If Unity or mesoscope terminated during runtime, attempts to re-initialize Unity and restart the Mesoscope.
        if self._unity_state.terminated:
            # When the Unity game cycles, it resets the sequence of VR wall cues. This re-queries the new wall cue
            # sequence to enable accurate tracking of the animal's position in VR after reset.
            self._get_cue_sequence()

            # Resets the termination tracker if cue_sequence retrieval succeeds, indicating that the Unity has
            # been restarted.
            self._unity_state.terminated = False

        if self._mesoscope_terminated:
            # Restarting the Mesoscope is slightly different from starting it, as the user needs to call the
            # setupAcquisition() function with a special argument. Instructs the user to call the function and then
            # enters the
            # Mesoscope start sequence.
            message = (
                "If necessary call the setupAcquisition(hSI, hSICtl, recovery=true) command in the MATLAB command line "
                "interface before proceeding to resume an interrupted acquisition."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            input("Enter anything to continue: ")

            self._start_mesoscope()

            # Resets the termination tacker if Mesoscope acquisition restarts successfully.
            self._mesoscope_terminated = False

        # Updates the 'paused_time' value to reflect the time spent inside the 'paused' state. Most runtimes use this
        # public attribute to adjust the execution time of certain runtime stages or the runtime altogether.
        self.paused_time += round(
            convert_time(time=self._timestamp_timer.elapsed - self._pause_start_time, from_units="us", to_units="s")
        )

        # Restores the runtime state back to the value active before the pause.
        self.change_runtime_state(new_state=self._runtime_state)

        # Restores the system state to pre-pause condition.
        if self._system_state == _MesoscopeVRStates.IDLE:
            # This is a rare case where the pause was triggered before a valid non-idle state was activated by the
            # runtime logic function. While rare, it is not technically impossible, so it is supported here
            self.idle()
        elif self._system_state == _MesoscopeVRStates.REST:
            self.rest()
        elif self._system_state == _MesoscopeVRStates.RUN:
            self.run()
        elif self._system_state == _MesoscopeVRStates.LICK_TRAINING:
            self.lick_train()
        elif self._system_state == _MesoscopeVRStates.RUN_TRAINING:
            self.run_train()

        # Resets the paused flag
        self._paused = False

    def _terminate_runtime(self) -> None:
        """Verifies that the user intends to abort the runtime via terminal prompt and, if so, sets the runtime into
        the termination mode.
        """
        # Verifies that the user intends to abort the runtime to avoid 'misclick' terminations.
        message = "Runtime abort signal: Received. Are you sure you want to abort the runtime?"
        console.echo(message=message, level=LogLevel.WARNING)
        while True:
            user_input = input("Enter 'yes' or 'no': ").strip().lower()
            answer = user_input[0] if user_input else ""

            # Sets the runtime into the termination state, which aborts all instance cycles and the outer logic function
            # cycle.
            if answer == "y":
                self._terminated = True
                return

            # Returns without terminating the runtime
            if answer == "n":
                return

    def setup_reinforcing_guidance(
        self, initial_guided_trials: int = 3, recovery_mode_threshold: int = 9, recovery_guided_trials: int = 3
    ) -> None:
        """Configures the guidance mode for reinforcing (water reward) trials.

        Notes:
            Once this method configures the guidance handling logic, the system maintains that logic internally until
            the session's data acquisition ends or this method is called again to reconfigure the guidance parameters.

        Args:
            initial_guided_trials: The number of reinforcing trials for which to initially enable guidance mode.
            recovery_mode_threshold: The number of consecutively failed reinforcing trials after which the system
                must engage the guidance recovery mode.
            recovery_guided_trials: The number of guided reinforcing trials to use when recovery mode is triggered.
        """
        self._trial_state.reinforcing_guided_trials = initial_guided_trials
        self._trial_state.reinforcing_failed_trials = 0
        self._trial_state.reinforcing_recovery_threshold = recovery_mode_threshold
        self._trial_state.reinforcing_recovery_trials = recovery_guided_trials

        # Enables reinforcing guidance via direct GUI manipulation.
        if initial_guided_trials > 0:
            self._ui.set_reinforcing_guidance_state(enabled=True)

    def setup_aversive_guidance(
        self, initial_guided_trials: int = 0, recovery_mode_threshold: int = 9, recovery_guided_trials: int = 3
    ) -> None:
        """Configures the guidance mode for aversive (gas puff) trials.

        Notes:
            Once this method configures the guidance handling logic, the system maintains that logic internally until
            the session's data acquisition ends or this method is called again to reconfigure the guidance parameters.

        Args:
            initial_guided_trials: The number of aversive trials for which to initially enable guidance mode.
            recovery_mode_threshold: The number of consecutively failed aversive trials after which the system
                must engage the guidance recovery mode.
            recovery_guided_trials: The number of guided aversive trials to use when recovery mode is triggered.
        """
        self._trial_state.aversive_guided_trials = initial_guided_trials
        self._trial_state.aversive_failed_trials = 0
        self._trial_state.aversive_recovery_threshold = recovery_mode_threshold
        self._trial_state.aversive_recovery_trials = recovery_guided_trials

        # Enables aversive guidance via direct GUI manipulation.
        if initial_guided_trials > 0:
            self._ui.set_aversive_guidance_state(enabled=True)

    @property
    def terminated(self) -> bool:
        """Returns True if the system has entered the termination state."""
        return self._terminated

    @property
    def running_speed(self) -> np.float64:
        """Returns the current running speed of the animal in centimeters per second."""
        return self._running_speed

    @property
    def speed_modifier(self) -> int:
        """Returns the current modifier applied to the running speed threshold during run training."""
        return self._ui.speed_modifier

    @property
    def duration_modifier(self) -> int:
        """Returns the current modifier applied to the duration threshold during run training."""
        return self._ui.duration_modifier

    @property
    def dispensed_water_volume(self) -> float:
        """Returns the total volume of water, in microliters, dispensed by the valve during the current runtime."""
        return float(self._delivered_water_volume)


def window_checking_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
) -> None:
    """Guides the user though verifying the quality of the implanted cranial window and generating the initial
    Mesoscope-VR system configuration for the target animal.

    Args:
        experimenter: The unique identifier of the experimenter conducting the window checking session.
        project_name: The name of the project in which the evaluated animal participates.
        animal_id: The unique identifier of the animal being evaluated.
    """
    message = "Initializing the window checking session..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
    system_configuration = get_system_configuration()

    # Verifies that the specified project has been configured.
    project_directory = system_configuration.filesystem.root_directory.joinpath(project_name)
    if not project_directory.exists():
        message = (
            f"Unable to execute the window checking session for the animal {animal_id} participating in the project "
            f"{project_name}. The {system_configuration.name} data acquisition system is not configured to acquire "
            f"data for this project. Use the 'sl-configure project' command to configure the project before running "
            f"data acquisition sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Verifies that the animal participates exclusively in the specified project.
    animal_projects = get_animal_project(animal_id=animal_id)
    if len(animal_projects) > 1:  # Rare case, often indicative of old migration pipeline use
        message = (
            f"Unable to execute the window checking session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is associated with multiple projects managed by the "
            f"{system_configuration.name} data acquisition system, which is not allowed. The animal is associated with "
            f"the following projects: {', '.join(animal_projects)}."
        )
        console.error(message=message, error=ValueError)
    elif len(animal_projects) == 1 and animal_projects[0] != project_name:
        message = (
            f"Unable to execute the window checking session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is already associated with a different project '{animal_projects[0]}' managed "
            f"by the {system_configuration.name} data acquisition system. If necessary, use the 'sl-manage migrate' "
            f"CLI command to transfer the animal to the desired project."
        )
        console.error(message=message, error=ValueError)

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=SessionTypes.WINDOW_CHECKING,
        python_version=python_version,
        sl_experiment_version=library_version,
        acquisition_system=system_configuration,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Generates the precursor session descriptor instance and caches it to disk.
    descriptor = WindowCheckingDescriptor(
        experimenter=experimenter,
        incomplete=True,
    )
    descriptor.to_yaml(file_path=session_data.raw_data.session_descriptor_path)

    # Generates and caches the MesoscopePositions precursor file to the persistent and raw_data directories.
    precursor = MesoscopePositions()
    precursor.to_yaml(file_path=session_data.raw_data.mesoscope_positions_path)
    precursor.to_yaml(file_path=mesoscope_data.vrpc_data.mesoscope_positions_path)

    zaber_motors: ZaberMotors | None = None
    try:
        # If the animal has a snapshot of Zaber motor positions used during a previous runtime, loads and uses these
        # positions. Otherwise, uses the default positions hardcoded in the Zaber controller's non-volatile memory.
        zaber_positions = (
            ZaberPositions.from_yaml(mesoscope_data.vrpc_data.zaber_positions_path)
            if mesoscope_data.vrpc_data.zaber_positions_path.exists()
            else None
        )

        # Initializes the data logger. This initialization follows the same procedure as the _MesoscopeVRSystem class
        logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data.raw_data_path,
            instance_name="behavior",  # Creates behavior_log subdirectory under raw_data
            thread_count=10,
        )
        logger.start()

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Initializes the face camera. The body camera is not used during window checking.
        cameras = VideoSystems(
            data_logger=logger,
            output_directory=session_data.raw_data.camera_data_path,
            camera_configuration=system_configuration.cameras,
        )
        cameras.start_face_camera()
        message = "Face camera acquisition: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # The ZaberLauncher UI cannot connect to the ports managed by Python bindings, so it must be initialized before
        # connecting to motor groups from Python.
        message = (
            "Preparing to connect to all managed Zaber motors. Make sure that the ZaberLauncher app is running before "
            "proceeding further. If the ZaberLauncher is not running, it will be IMPOSSIBLE to manually control the "
            "Zaber motors."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

        # Establishes communication with Zaber motors
        zaber_motors = ZaberMotors(zaber_positions=zaber_positions, zaber_configuration=system_configuration.assets)

        # Removes the nk.bin marker to avoid automatic session cleanup during post-processing.
        session_data.runtime_initialized()

        # Prepares Zaber motors for data acquisition.
        _setup_zaber_motors(zaber_motors=zaber_motors)

        # Runs the user through the process of preparing the mesoscope and assessing the quality of the animal's cranial
        # window.
        _setup_mesoscope(session_data=session_data, mesoscope_data=mesoscope_data)

        # Retrieves current motor positions and packages them into a ZaberPositions object.
        _generate_zaber_snapshot(session_data=session_data, mesoscope_data=mesoscope_data, zaber_motors=zaber_motors)

        # Instructs the user to update the session descriptor file
        _verify_descriptor_update(descriptor=descriptor, session_data=session_data, mesoscope_data=mesoscope_data)

        # Generates the snapshot of the Mesoscope imaging position used to generate the data during window checking.
        _generate_mesoscope_position_snapshot(session_data=session_data, mesoscope_data=mesoscope_data)

        # Resets Zaber motors to their original positions.
        _reset_zaber_motors(zaber_motors=zaber_motors)

        # Terminates the face camera
        cameras.stop()

        # Stops the data logger
        logger.stop()

        # Triggers preprocessing pipeline. In this case, since there is no data to preprocess, the pipeline primarily
        # just copies the session raw_data directory to the NAS and BioHPC server.
        preprocess_session_data(session_data=session_data)

    finally:
        # If the session runtime terminates before the session was initialized, removes session data from all sources
        # before shutting down.
        if session_data.raw_data.nk_path.exists():
            message = (
                f"The runtime was unexpectedly terminated before it was able to initialize all required Mesoscope-VR "
                f"assets. Removing all leftover data from the uninitialized session from all destinations accessible "
                f"to the {system_configuration.name} data acquisition system..."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            purge_session(session_data)

        # If Zaber motors were connected, attempts to gracefully shut down the motors.
        if zaber_motors is not None:
            _reset_zaber_motors(zaber_motors=zaber_motors)

        # Ends the runtime
        message = "Window checking session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


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
) -> None:
    """Trains the animal to operate the lickport used by the Mesoscope-VR data acquisition system.

    Notes:
        The training consists of delivering water rewards via the lickport at pseudorandom intervals to teach the
        animal that rewards come out of the lick port. The training continues either until the valve
        delivers the 'maximum_water_volume' in milliliters or until the 'maximum_training_time' in minutes is reached,
        whichever comes first.

        Most arguments to this function are optional overrides. If an argument is not provided, the system loads the
        argument's value used during a previous runtime (if available) or uses a system-defined default value.

    Args:
        experimenter: The unique identifier of the experimenter conducting the training session.
        project_name: The name of the project in which the trained animal participates.
        animal_id: The unique identifier of the animal being trained.
        animal_weight: The weight of the animal, in grams, at the beginning of the session.
        reward_size: The volume of water, in microliters, to use when delivering water rewards to the animal.
        reward_tone_duration: The duration, in milliseconds, of the auditory tone played to the animal when it
            receives water rewards.
        minimum_reward_delay: The minimum time, in seconds, that has to pass between delivering two consecutive rewards.
        maximum_reward_delay: The maximum time, in seconds, that can pass between delivering two consecutive rewards.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered to the animal during
            the session.
        maximum_training_time: The maximum training time, in minutes.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before the system suspends delivering water rewards until the animal consumes all available rewards.
            Setting this argument to 0 disables forcing reward consumption.
    """
    message = "Initializing the lick training session..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
    system_configuration = get_system_configuration()

    # Verifies that the specified project has been configured.
    project_directory = system_configuration.filesystem.root_directory.joinpath(project_name)
    if not project_directory.exists():
        message = (
            f"Unable to execute the lick training session for the animal {animal_id} participating in the project "
            f"{project_name}. The {system_configuration.name} data acquisition system is not configured to acquire "
            f"data for this project. Use the 'sl-configure project' command to configure the project before running "
            f"data acquisition sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Verifies that the animal participates exclusively in the specified project.
    animal_projects = get_animal_project(animal_id=animal_id)
    if len(animal_projects) > 1:  # Rare case, often indicative of old migration pipeline use
        message = (
            f"Unable to execute the lick training session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is associated with multiple projects managed by the "
            f"{system_configuration.name} data acquisition system, which is not allowed. The animal is associated with "
            f"the following projects: {', '.join(animal_projects)}."
        )
        console.error(message=message, error=ValueError)
    elif len(animal_projects) == 1 and animal_projects[0] != project_name:
        message = (
            f"Unable to execute the lick training session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is already associated with a different project '{animal_projects[0]}' managed "
            f"by the {system_configuration.name} data acquisition system. If necessary, use the 'sl-manage migrate' "
            f"CLI command to transfer the animal to the desired project."
        )
        console.error(message=message, error=ValueError)

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=SessionTypes.LICK_TRAINING,
        python_version=python_version,
        sl_experiment_version=library_version,
        acquisition_system=system_configuration,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # If the trained animal has previously participated in this type of sessions, loads the previous session's runtime
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: LickTrainingDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory
        previous_descriptor = LickTrainingDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight
    descriptor = LickTrainingDescriptor(
        experimenter=experimenter,
        animal_weight_g=animal_weight,
    )

    # Configures the session to use either the previous session's parameters (if available) or the default parameters.
    if previous_descriptor is not None:
        # Overrides the default configuration parameters with the parameters used during the previous runtime.
        descriptor.maximum_reward_delay_s = previous_descriptor.maximum_reward_delay_s
        descriptor.minimum_reward_delay_s = previous_descriptor.minimum_reward_delay_s
        descriptor.water_reward_size_ul = previous_descriptor.water_reward_size_ul
        descriptor.reward_tone_duration_ms = previous_descriptor.reward_tone_duration_ms
        descriptor.maximum_water_volume_ml = previous_descriptor.maximum_water_volume_ml
        descriptor.maximum_training_time_min = previous_descriptor.maximum_training_time_min
        descriptor.maximum_unconsumed_rewards = previous_descriptor.maximum_unconsumed_rewards

    # If necessary, updates the descriptor with the argument override values provided by the user.
    if maximum_reward_delay is not None:
        descriptor.maximum_reward_delay_s = maximum_reward_delay
    if minimum_reward_delay is not None:
        descriptor.minimum_reward_delay_s = minimum_reward_delay
    if reward_size is not None:
        descriptor.water_reward_size_ul = reward_size
    if reward_tone_duration is not None:
        descriptor.reward_tone_duration_ms = reward_tone_duration
    if maximum_water_volume is not None:
        descriptor.maximum_water_volume_ml = maximum_water_volume
    if maximum_training_time is not None:
        descriptor.maximum_training_time_min = maximum_training_time
    if maximum_unconsumed_rewards is not None:
        descriptor.maximum_unconsumed_rewards = maximum_unconsumed_rewards

    # Validates the maximum unconsumed rewards parameter. If the maximum unconsumed reward count is below 1, disables
    # the feature by deferring the assignment until after the total number of rewards is calculated. This ensures that
    # the feature can be properly disabled by setting the limit equal to the total reward count.
    _disable_unconsumed_limit = descriptor.maximum_unconsumed_rewards < 1

    # Initializes the timer used to enforce reward delays
    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    message = "Generating the pseudorandom reward delay sequence..."
    console.echo(message=message, level=LogLevel.INFO)

    # Converts maximum volume to uL and divides it by the reward size to get the number of delays to sample from
    # the delay distribution
    num_samples = np.floor((descriptor.maximum_water_volume_ml * 1000) / descriptor.water_reward_size_ul).astype(
        np.uint64
    )

    # Generates samples from a uniform distribution within delay bounds
    rng = np.random.default_rng()
    samples = rng.uniform(descriptor.minimum_reward_delay_s, descriptor.maximum_reward_delay_s, num_samples)

    # Calculates cumulative training time for each sampled delay. This communicates the total time passed when each
    # reward is delivered to the animal
    cumulative_time = np.cumsum(samples)

    # Finds the maximum number of samples that fits within the maximum training time. This handles the (expected) cases
    # where the total training time is insufficient to deliver the maximum allowed volume of water, so the reward
    # sequence needs to be clipped.
    max_samples_idx = np.searchsorted(cumulative_time, descriptor.maximum_training_time_min * 60, side="right")

    # Slices the samples array to make the total training time be roughly the maximum requested duration.
    reward_delays: NDArray[np.float64] = samples[:max_samples_idx]

    # Aborts if no rewards fit in the requested training time. Raises an error before system initialization to allow
    # automatic session data purge.
    if max_samples_idx == 0:
        message = (
            f"Unable to generate the lick training reward sequence. The requested maximum training time "
            f"({descriptor.maximum_training_time_min} minutes) is shorter than the minimum reward delay "
            f"({descriptor.minimum_reward_delay_s} seconds). Increase the maximum training time or decrease the "
            f"minimum reward delay."
        )
        console.error(message=message, error=ValueError)

    message = (
        f"Generated a sequence of {len(reward_delays)} rewards with the total cumulative runtime of "
        f"{np.round(cumulative_time[max_samples_idx - 1] / 60, decimals=3)} minutes."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # If session runtime is limited by the total volume of delivered water, rather than the maximum runtime, clips the
    # total training time at the point where the maximum allowed water volume is delivered.
    if len(reward_delays) == len(cumulative_time):
        # Actual session time is the accumulated delay converted from seconds to minutes at the last index.
        descriptor.maximum_training_time_min = int(np.ceil(cumulative_time[-1] / 60))

    # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
    # number of rewards to be delivered.
    if _disable_unconsumed_limit:
        descriptor.maximum_unconsumed_rewards = len(reward_delays)

    system: _MesoscopeVRSystem | None = None
    try:
        # Initializes the system class
        system = _MesoscopeVRSystem(session_data=session_data, session_descriptor=descriptor)

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of errors should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exceptions as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.runtime_initialized()

        # Switches the system into lick-training mode
        system.lick_train()

        message = "Lick training: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Loops over all delays and delivers reward via the lick tube as soon as the delay expires.
        delay_timer.reset()
        for delay in tqdm(
            reward_delays,
            desc="Delivered water rewards",
            unit="reward",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} rewards [{elapsed}]",
        ):
            # This loop is executed while the code is waiting for the delay to pass. Anything that needs to be done
            # during the delay has to go here. If the session is paused during the delay cycle, the time spent in the
            # pause is used to discount the delay. This is in contrast to other sessions, where pause time actually
            # INCREASES the overall session duration.
            while delay_timer.elapsed < (delay - system.paused_time):
                system.runtime_cycle()  # Repeatedly calls the runtime cycle during the delay period

            # If the user sent the abort command, terminates the training early.
            if system.terminated:
                message = (
                    "Lick training abort signal detected. Aborting the lick training with a graceful shutdown "
                    "procedure..."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break  # Breaks the for loop

            # Resets the delay timer immediately after exiting the delay loop
            delay_timer.reset()

            # Clears the paused time at the end of each delay cycle. This has to be done to prevent future delay
            # loops from ending earlier than expected unless the session is paused again as part of that loop.
            system.paused_time = 0

            # Delivers the water reward to the animal or simulates the reward if the animal is not licking
            system.resolve_reward(
                reward_size=descriptor.water_reward_size_ul, tone_duration=descriptor.reward_tone_duration_ms
            )

        # Ensures the animal has time to consume the last reward before the LickPort is moved out of its range. Uses
        # the maximum possible time interval as the delay interval.
        delay_timer.delay(delay=descriptor.maximum_reward_delay_s, block=False)

    # RecursionErrors should not be raised by any session component except in the case that the user wants to terminate
    # the session as part of the startup checkpoint. Therefore, silences the error.
    except RecursionError:
        pass

    # Ensures that the function always attempts the graceful shutdown procedure, even if it encounters session errors.
    finally:
        # If the system was initialized, attempts to gracefully terminate system assets
        if system is not None:
            system.stop()

        # If the session terminates before the session was initialized, removes session data from all
        # sources before shutting down.
        if session_data.raw_data.nk_path.exists():
            message = (
                "The lick training session was unexpectedly terminated before it was able to initialize and start all "
                "assets. Removing all leftover data from the uninitialized session from all destinations..."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            purge_session(session_data)

        message = "Lick training session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


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
) -> None:
    """Trains the animal to run on the wheel treadmill while being head-fixed.

    Notes:
        The run training consists of making the animal run on the wheel with a desired speed, in centimeters per
        second, maintained for the desired duration of time, in seconds. Each time the animal satisfies the speed
        and duration thresholds, it receives a water reward, and the speed and duration trackers reset for the
        next training 'epoch'. Each time the animal receives 'increase_threshold' of water, the speed and duration
        thresholds increase to make the task progressively more challenging. The training continues either until the
        training time exceeds the 'maximum_training_time', or the animal receives the 'maximum_water_volume' of water,
        whichever happens earlier.

        Most arguments to this function are optional overrides. If an argument is not provided, the system loads the
        argument's value used during a previous session (if available) or uses a system-defined default value.

    Args:
        experimenter: The unique identifier of the experimenter conducting the training session.
        project_name: The name of the project in which the trained animal participates.
        animal_id: The unique identifier of the animal being trained.
        animal_weight: The weight of the animal, in grams, at the beginning of the session.
        reward_size: The volume of water, in microliters, to use when delivering water rewards to the animal.
        reward_tone_duration: The duration, in milliseconds, of the auditory tone played to the animal when it
            receives water rewards.
        initial_speed_threshold: The initial running speed threshold, in centimeters per second, that the animal must
            maintain to receive water rewards.
        initial_duration_threshold: The initial duration threshold, in seconds, that the animal must maintain
            above-threshold running speed to receive water rewards.
        speed_increase_step: The step size, in centimeters per second, by which to increase the speed threshold each
            time the animal receives 'increase_threshold' milliliters of water.
        duration_increase_step: The step size, in seconds, by which to increase the duration threshold each time the
            animal receives 'increase_threshold' milliliters of water.
        increase_threshold: The volume of water received by the animal, in milliliters, after which the speed and
            duration thresholds are increased by one step.
        maximum_water_volume: The maximum volume of water, in milliliters, that can be delivered to the animal during
            the session.
        maximum_training_time: The maximum training time, in minutes.
        maximum_idle_time: The maximum time, in seconds, the animal's speed can be below the speed threshold to
            still receive water rewards. This parameter is designed to help animals with a distinct 'step' pattern to
            not lose water rewards due to taking many large steps, rather than continuously running at a stable speed.
            Setting this argument to 0 disables this functionality.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before the system suspends delivering water rewards until the animal consumes all available rewards.
            Setting this argument to 0 disables forcing reward consumption.
    """
    message = "Initializing the run training session..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
    system_configuration = get_system_configuration()

    # Verifies that the specified project has been configured.
    project_directory = system_configuration.filesystem.root_directory.joinpath(project_name)
    if not project_directory.exists():
        message = (
            f"Unable to execute the run training session for the animal {animal_id} participating in the project "
            f"{project_name}. The {system_configuration.name} data acquisition system is not configured to acquire "
            f"data for this project. Use the 'sl-configure project' command to configure the project before running "
            f"data acquisition sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Verifies that the animal participates exclusively in the specified project.
    animal_projects = get_animal_project(animal_id=animal_id)
    if len(animal_projects) > 1:  # Rare case, often indicative of old migration pipeline use
        message = (
            f"Unable to execute the run training session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is associated with multiple projects managed by the "
            f"{system_configuration.name} data acquisition system, which is not allowed. The animal is associated with "
            f"the following projects: {', '.join(animal_projects)}."
        )
        console.error(message=message, error=ValueError)
    elif len(animal_projects) == 1 and animal_projects[0] != project_name:
        message = (
            f"Unable to execute the run training session for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is already associated with a different project '{animal_projects[0]}' managed "
            f"by the {system_configuration.name} data acquisition system. If necessary, use the 'sl-manage migrate' "
            f"CLI command to transfer the animal to the desired project."
        )
        console.error(message=message, error=ValueError)

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=SessionTypes.RUN_TRAINING,
        python_version=python_version,
        sl_experiment_version=library_version,
        acquisition_system=system_configuration,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # If the trained animal has previously participated in this type of sessions, loads the previous session's
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: RunTrainingDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory
        previous_descriptor = RunTrainingDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight
    descriptor = RunTrainingDescriptor(
        experimenter=experimenter,
        animal_weight_g=animal_weight,
    )

    # Configures the session to use either the previous session's parameters (if available) or the default parameters.
    if previous_descriptor is not None:
        # Overrides the default configuration parameters with the parameters used during the previous session.
        # For run training, initial thresholds are set to the FINAL thresholds from the previous session, so each
        # consecutive run training session begins where the previous one has ended.
        descriptor.initial_run_speed_threshold_cm_s = previous_descriptor.final_run_speed_threshold_cm_s
        descriptor.initial_run_duration_threshold_s = previous_descriptor.final_run_duration_threshold_s
        descriptor.run_speed_increase_step_cm_s = previous_descriptor.run_speed_increase_step_cm_s
        descriptor.run_duration_increase_step_s = previous_descriptor.run_duration_increase_step_s
        descriptor.increase_threshold_ml = previous_descriptor.increase_threshold_ml
        descriptor.maximum_water_volume_ml = previous_descriptor.maximum_water_volume_ml
        descriptor.maximum_training_time_min = previous_descriptor.maximum_training_time_min
        descriptor.maximum_idle_time_s = previous_descriptor.maximum_idle_time_s
        descriptor.maximum_unconsumed_rewards = previous_descriptor.maximum_unconsumed_rewards
        descriptor.water_reward_size_ul = previous_descriptor.water_reward_size_ul
        descriptor.reward_tone_duration_ms = previous_descriptor.reward_tone_duration_ms

    # If necessary, updates the descriptor with the argument override values provided by the user.
    if reward_size is not None:
        descriptor.water_reward_size_ul = reward_size
    if reward_tone_duration is not None:
        descriptor.reward_tone_duration_ms = reward_tone_duration
    if initial_speed_threshold is not None:
        descriptor.initial_run_speed_threshold_cm_s = initial_speed_threshold
    if initial_duration_threshold is not None:
        descriptor.initial_run_duration_threshold_s = initial_duration_threshold
    if speed_increase_step is not None:
        descriptor.run_speed_increase_step_cm_s = speed_increase_step
    if duration_increase_step is not None:
        descriptor.run_duration_increase_step_s = duration_increase_step
    if increase_threshold is not None:
        descriptor.increase_threshold_ml = increase_threshold
    if maximum_water_volume is not None:
        descriptor.maximum_water_volume_ml = maximum_water_volume
    if maximum_training_time is not None:
        descriptor.maximum_training_time_min = maximum_training_time
    if maximum_idle_time is not None:
        descriptor.maximum_idle_time_s = maximum_idle_time
    if maximum_unconsumed_rewards is not None:
        descriptor.maximum_unconsumed_rewards = maximum_unconsumed_rewards

    # Validates the maximum unconsumed rewards parameter. If the maximum unconsumed reward count is below 1, disables
    # the feature by deferring the assignment until after the maximum number of deliverable rewards is calculated. This
    # ensures that the feature can be properly disabled by setting the limit equal to the total reward count.
    _disable_unconsumed_limit = descriptor.maximum_unconsumed_rewards < 1

    # Validates the increase threshold parameter. The way 'increase_threshold' is used requires it to be greater than
    # 0. So if a threshold of 0 is passed, the system sets it to a very small number instead, which functions similar
    # to it being 0, but does not produce an error. Specifically, this prevents the 'division by zero' error.
    if descriptor.increase_threshold_ml <= 0:
        descriptor.increase_threshold_ml = 0.000000000001

    # Initializes the timers used during the session
    runtime_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    running_duration_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    epoch_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    # Initializes the assets used to guard against interrupting run epochs for animals that take many large steps. For
    # animals with a distinct walking pattern of many very large steps, the speed transiently dips below the threshold
    # for a
    # very brief moment of time, flagging the epoch as unrewarded. To avoid this issue, instead of interrupting the
    # epoch outright, the system now allows the speed to be below the threshold for a short period of time. These
    # assets help with that task pattern.
    epoch_timer_engaged: bool = False
    maximum_idle_time_ms = max(0.0, descriptor.maximum_idle_time_s) * 1000  # Ensures positive values and converts to ms

    # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
    # maximum number of rewards that can be delivered during the session.
    if _disable_unconsumed_limit:
        descriptor.maximum_unconsumed_rewards = int(
            np.ceil(descriptor.maximum_water_volume_ml / (descriptor.water_reward_size_ul / 1000))
        )

    # Converts all arguments used to determine the speed and duration threshold over time into numpy variables to
    # optimize the main session's runtime loop:
    initial_speed = np.float64(descriptor.initial_run_speed_threshold_cm_s)  # In centimeters per second
    maximum_speed = np.float64(5)  # In centimeters per second
    speed_step = np.float64(descriptor.run_speed_increase_step_cm_s)  # In centimeters per second

    initial_duration = np.float64(descriptor.initial_run_duration_threshold_s * 1000)  # In milliseconds
    maximum_duration = np.float64(5000)  # In milliseconds
    duration_step = np.float64(descriptor.run_duration_increase_step_s * 1000)  # In milliseconds

    water_threshold = np.float64(descriptor.increase_threshold_ml * 1000)  # In microliters
    maximum_volume = np.float64(descriptor.maximum_water_volume_ml * 1000)  # In microliters

    # Converts the training time from minutes to seconds to make it compatible with the timer precision.
    training_time = descriptor.maximum_training_time_min * 60

    # Initializes internal tracker variables:
    # Tracks the data necessary to update the training progress bar
    previous_time = 0

    # Tracks when speed and / or duration thresholds are updated. This is necessary to redraw the threshold lines in
    # the visualizer plot
    previous_speed_threshold = copy.copy(initial_speed)
    previous_duration_threshold = copy.copy(initial_duration)

    # This one-time tracker is used to initialize the speed and duration threshold visualization.
    once = True

    # Updates the descriptor with the final threshold values saved at the end of the session. These are
    # initialized to the initial thresholds and are updated during the session if the animal progresses.
    descriptor.final_run_speed_threshold_cm_s = descriptor.initial_run_speed_threshold_cm_s
    descriptor.final_run_duration_threshold_s = descriptor.initial_run_duration_threshold_s

    system: _MesoscopeVRSystem | None = None
    try:
        # Initializes the system class
        system = _MesoscopeVRSystem(session_data=session_data, session_descriptor=descriptor)

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of errors should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exceptions as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.runtime_initialized()

        # Switches the system into the run-training mode
        system.run_train()

        message = "Run training: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Creates a tqdm progress bar that tracks the overall training progress by communicating the total volume of
        # water delivered to the animal
        progress_bar = tqdm(
            total=round(descriptor.maximum_water_volume_ml, ndigits=3),
            desc="Delivered water volume",
            unit="ml",
            bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f} {postfix}",
        )

        runtime_timer.reset()
        running_duration_timer.reset()  # It is critical to reset both timers at the same time.

        # Pre-initializes the threshold trackers to avoid MyPy errors.
        speed_threshold: np.float64 = np.float64(0.0)
        duration_threshold: np.float64 = np.float64(0.0)

        # This is the main session loop of the run training mode.
        while runtime_timer.elapsed < (training_time + system.paused_time):
            system.runtime_cycle()  # Repeatedly calls the runtime cycle during training

            # If the user sent the abort command, terminates the training early.
            if system.terminated:
                message = (
                    "Run training abort signal detected. Aborting the run training with a graceful shutdown "
                    "procedure..."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break  # Breaks the while loop

            # Determines how many times the speed and duration thresholds have been increased based on the difference
            # between the total delivered water volume and the increase threshold. This dynamically adjusts the running
            # speed and duration thresholds with delivered water volume, ensuring the animal has to try progressively
            # harder to keep receiving water.
            increase_steps: np.float64 = np.floor(system.dispensed_water_volume / water_threshold)

            # Determines the speed and duration thresholds for each cycle. This factors in the user input via the
            # session control GUI. Note, user input has a static resolution of 0.01 cm/s and 0.01 s (10 ms) per step.
            speed_threshold = np.clip(
                a=initial_speed + (increase_steps * speed_step) + (system.speed_modifier * 0.01),
                a_min=0.1,  # Minimum value
                a_max=maximum_speed,  # Maximum value
            )
            duration_threshold = np.clip(
                a=initial_duration + (increase_steps * duration_step) + (system.duration_modifier * 10),
                a_min=50,  # Minimum value (0.05 seconds == 50 milliseconds)
                a_max=maximum_duration,  # Maximum value
            )

            # If any of the threshold changed relative to the previous loop iteration, updates the visualizer and
            # previous threshold trackers with new data. The update is forced at the beginning of the session to make
            # the visualizer render the threshold lines and values.
            if once or (
                duration_threshold != previous_duration_threshold or previous_speed_threshold != speed_threshold
            ):
                system.update_visualizer_thresholds(speed_threshold, duration_threshold)
                previous_speed_threshold = speed_threshold
                previous_duration_threshold = duration_threshold

                # Inactivates the 'once' tracker after the first update.
                if once:
                    once = False

            # If the speed is above the speed threshold, and the animal has been maintaining the above-threshold speed
            # for the required duration, delivers a water reward. If the speed is above the threshold, but the animal
            # has not yet maintained the required duration, the loop keeps cycling and accumulating the timer count.
            # This is done until the animal either reaches the required duration or drops below the speed threshold.
            if system.running_speed >= speed_threshold and running_duration_timer.elapsed >= duration_threshold:
                # Delivers water reward or simulates reward delivery. The method returns True if the reward was
                # delivered and False otherwise.
                if system.resolve_reward(
                    reward_size=descriptor.water_reward_size_ul, tone_duration=descriptor.reward_tone_duration_ms
                ):
                    # Updates the progress bar whenever the animal receives automated water rewards. The progress bar
                    # purposefully does not track 'manual' water rewards.
                    progress_bar.update(descriptor.water_reward_size_ul / 1000)  # Converts uL to ml

                # Also resets the timer. While animals typically stop consuming water rewards, which would reset the
                # timer, this guards against animals that carry on running without consuming water rewards.
                running_duration_timer.reset()

                # If the epoch timer was active for the current epoch, resets the timer
                epoch_timer_engaged = False

            # If the current speed is below the speed threshold, acts depending on whether the session is configured to
            # allow dipping below the threshold
            elif system.running_speed < speed_threshold:
                # If the user did not allow dipping below the speed threshold, resets the run duration timer.
                if maximum_idle_time_ms == 0:
                    running_duration_timer.reset()

                # If the user has enabled brief dips below the speed threshold, starts the epoch timer to ensure the
                # animal recovers the speed in the allotted time.
                elif not epoch_timer_engaged:
                    epoch_timer.reset()
                    epoch_timer_engaged = True

                # If epoch timer is enabled, checks whether the animal has failed to recover its running speed in time.
                # If so, resets the run duration timer.
                elif epoch_timer.elapsed >= maximum_idle_time_ms:
                    running_duration_timer.reset()
                    epoch_timer_engaged = False

            # If the animal is maintaining the required speed and the epoch timer was activated by the animal dipping
            # below the speed threshold, deactivates the timer. This is essential for ensuring the 'step discount'
            # time is applied to each case of speed dipping below the speed threshold, rather than the entire run epoch.
            elif (
                epoch_timer_engaged
                and system.running_speed >= speed_threshold
                and running_duration_timer.elapsed < duration_threshold
            ):
                epoch_timer_engaged = False

            # Updates the time display when each second passes. This updates the 'suffix' of the progress bar to keep
            # track of elapsed training time. Accounts for any additional time spent in the 'paused' state.
            elapsed_time = runtime_timer.elapsed - system.paused_time
            if elapsed_time > previous_time:
                previous_time = elapsed_time  # Updates previous time

                # Updates the time display without advancing the progress bar
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                progress_bar.set_postfix_str(
                    f"Time: {elapsed_minutes:02d}:{elapsed_seconds:02d}/{descriptor.maximum_training_time_min:02d}:00"
                )

                # Refreshes the display to show updated time without changing progress
                progress_bar.refresh()

            # If the total volume of water dispensed during the session exceeds the maximum allowed volume, aborts the
            # training early with a success message.
            if system.dispensed_water_volume >= maximum_volume:
                message = (
                    f"Run training has delivered the maximum allowed volume of water ({maximum_volume} uL). Aborting "
                    f"the training process..."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)
                break

        # Closes the progress bar if the session ends as expected
        progress_bar.close()

        # Updates the descriptor with the final thresholds reached during the session. These will be used as the
        # starting thresholds for the next session.
        descriptor.final_run_speed_threshold_cm_s = float(speed_threshold)
        descriptor.final_run_duration_threshold_s = float(duration_threshold / 1000)  # Converts back to seconds

    # RecursionErrors should not be raised by any session component except in the case that the user wants to terminate
    # the session as part of the startup checkpoint. Therefore, silences the error.
    except RecursionError:
        pass

    # Ensures that the function always attempts the graceful shutdown procedure, even if it encounters session errors.
    finally:
        # If the system was initialized, attempts to gracefully terminate system assets
        if system is not None:
            system.stop()

        # If the session terminates before the session was initialized, removes session data from all
        # sources before shutting down.
        if session_data.raw_data.nk_path.exists():
            message = (
                "The run training session was unexpectedly terminated before it was able to initialize and start all "
                "assets. Removing all leftover data from the uninitialized session from all destinations..."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            purge_session(session_data)

        message = "Run training session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


def experiment_logic(
    experimenter: str,
    project_name: str,
    experiment_name: str,
    animal_id: str,
    animal_weight: float,
    maximum_unconsumed_rewards: int | None = None,
) -> None:
    """Runs experiments using the Virtual Reality task environments and collects the brain activity data via the
    mesoscope.

    Notes:
        Each experiment is conceptualized as a sequence of experiment states (phases), which define the task and the
        types of data being collected while the system maintains the state. During the session, the system executes the
        predefined sequence of states defines in the experiment's configuration file. Once all states are executed, the
        experiment session ends.

        During the session's runtime, the task logic and the Virtual Reality world are resolved by the Unity game
        engine. This function handles the data collection and the overall runtime management.

        The maximum_unconsumed_rewards argument is an optional override. If not provided, the system loads the
        argument's value used during a previous session (if available) or uses a system-defined default value.

    Args:
        experimenter: The unique identifier of the experimenter conducting the experiment session.
        project_name: The name of the project in which the experimental animal participates.
        experiment_name: The name of the experiment to be conducted.
        animal_id: The unique identifier of the animal participating in the experiment.
        animal_weight: The weight of the animal, in grams, at the beginning of the session.
        maximum_unconsumed_rewards: The maximum number of rewards that can be delivered without the animal consuming
            them, before the system suspends delivering water rewards until the animal consumes all available rewards.
            Setting this argument to 0 disables forcing reward consumption.
    """
    message = f"Initializing the {experiment_name} experiment session..."
    console.echo(message=message, level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
    system_configuration = get_system_configuration()

    # Verifies that the specified project has been configured.
    project_directory = system_configuration.filesystem.root_directory.joinpath(project_name)
    if not project_directory.exists():
        message = (
            f"Unable to execute the {experiment_name} experiment session for the animal {animal_id} participating in "
            f"the project {project_name}. The {system_configuration.name} data acquisition system is not configured to "
            f"acquire data for this project. Use the 'sl-configure project' command to configure the project before "
            f"running data acquisition sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Prevents the user from executing the session if the project is not configured to run the requested experiment
    project_experiments = get_project_experiments(
        project=project_name, filesystem_configuration=system_configuration.filesystem
    )
    if experiment_name not in project_experiments:
        message = (
            f"Unable to execute the {experiment_name} experiment session for the animal {animal_id} participating in "
            f"the project {project_name}. The target project does not have an experiment configuration file named "
            f"after the target experiment. Use the 'sl-configure experiment' command to configure the experiment "
            f"before running experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Verifies that the animal participates exclusively in the specified project.
    animal_projects = get_animal_project(animal_id=animal_id)
    if len(animal_projects) > 1:  # Rare case, often indicative of old migration pipeline use
        message = (
            f"Unable to execute the {experiment_name} experiment session for the animal {animal_id} participating in "
            f"the project {project_name}. The animal is associated with multiple projects managed by the "
            f"{system_configuration.name} data acquisition system, which is not allowed. The animal is associated with "
            f"the following projects: {', '.join(animal_projects)}."
        )
        console.error(message=message, error=ValueError)
    elif len(animal_projects) == 1 and animal_projects[0] != project_name:
        message = (
            f"Unable to execute the {experiment_name} experiment session for the animal {animal_id} participating in "
            f"the project {project_name}. The animal is already associated with a different project "
            f"'{animal_projects[0]}' managed by the {system_configuration.name} data acquisition system. If necessary, "
            f"use the 'sl-manage migrate' CLI command to transfer the animal to the desired project."
        )
        console.error(message=message, error=ValueError)

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        project_name=project_name,
        animal_id=animal_id,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        experiment_name=experiment_name,
        python_version=python_version,
        sl_experiment_version=library_version,
        acquisition_system=system_configuration,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Uses initialized SessionData instance to load the experiment configuration data
    experiment_config: MesoscopeExperimentConfiguration = MesoscopeExperimentConfiguration.from_yaml(
        file_path=session_data.raw_data.experiment_configuration_path
    )

    # Verifies that all Mesoscope-VR states used during experiments are valid
    valid_states = {1, 2}
    state: ExperimentState
    for state in experiment_config.experiment_states.values():
        if state.system_state_code not in valid_states:
            message = (
                f"Invalid Mesoscope-VR system state code {state.system_state_code} encountered when verifying "
                f"{experiment_name} experiment configuration. Currently, only codes 1 (rest) and 2 (run) are supported "
                f"for the Mesoscope-VR system."
            )
            console.error(message=message, error=ValueError)

    # If the experimental animal has previously participated in this type of sessions, loads the previous session's
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: MesoscopeExperimentDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory
        previous_descriptor = MesoscopeExperimentDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight
    descriptor = MesoscopeExperimentDescriptor(
        experimenter=experimenter,
        animal_weight_g=animal_weight,
    )

    # Configures the session to use either the previous session's parameters (if available) or the default parameters.
    if previous_descriptor is not None:
        # Overrides the default configuration parameters with the parameters used during the previous session.
        descriptor.maximum_unconsumed_rewards = previous_descriptor.maximum_unconsumed_rewards

    # If necessary, updates the descriptor with the argument override values provided by the user.
    if maximum_unconsumed_rewards is not None:
        descriptor.maximum_unconsumed_rewards = maximum_unconsumed_rewards

    # Initializes the timer to enforce experiment state durations
    runtime_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    system: _MesoscopeVRSystem | None = None
    try:
        # Initializes the system class
        system = _MesoscopeVRSystem(
            session_data=session_data, session_descriptor=descriptor, experiment_configuration=experiment_config
        )

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of errors should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exceptions as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.runtime_initialized()

        # Main session loop. It loops over all submitted experiment states and ends the session after executing the
        # last state
        for state in experiment_config.experiment_states.values():
            runtime_timer.reset()  # Resets the timer

            # Sets the Experiment state
            system.change_runtime_state(state.experiment_state_code)

            # Resets the tracker used to update the progress bar every second
            previous_seconds = 0

            # Resolves and sets the Mesoscope-VR system state
            if state.system_state_code == _MesoscopeVRStates.REST:
                system.rest()
            elif state.system_state_code == _MesoscopeVRStates.RUN:
                system.run()
            else:
                message = (
                    f"Unsupported Mesoscope-VR system state code {state.system_state_code} encountered when executing "
                    f"the {state.experiment_state_code} state. Currently, only the following system state codes are "
                    f"supported {','.join(tuple(_MesoscopeVRStates))}."
                )
                console.error(message=message, error=ValueError)

            # Configures the reinforcing guidance parameters for the executed experiment state (stage).
            system.setup_reinforcing_guidance(
                initial_guided_trials=state.reinforcing_initial_guided_trials,
                recovery_mode_threshold=state.reinforcing_recovery_failed_threshold,
                recovery_guided_trials=state.reinforcing_recovery_guided_trials,
            )

            # Configures the aversive guidance parameters for the executed experiment state (stage).
            system.setup_aversive_guidance(
                initial_guided_trials=state.aversive_initial_guided_trials,
                recovery_mode_threshold=state.aversive_recovery_failed_threshold,
                recovery_guided_trials=state.aversive_recovery_guided_trials,
            )

            # Creates a tqdm progress bar for the current experiment state
            with tqdm(
                total=state.state_duration_s,
                desc=f"Executing experiment state {state.experiment_state_code}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s",
            ) as pbar:
                # Cycles until the state duration of seconds passes
                while runtime_timer.elapsed < (state.state_duration_s + system.paused_time):
                    # Since experiment logic is resolved by the Unity game engine, the session logic function only
                    # needs to call the runtime cycle and handle runtime termination cases.
                    system.runtime_cycle()  # Repeatedly calls the runtime cycle as part of the experiment state cycle

                    # If the user has terminated the session, breaks the while loop. The termination is also handled at
                    # the level of the 'for' loop. The error message is generated at that level, rather than here.
                    if system.terminated:
                        break

                    # Updates the progress bar every second. Note: this calculation statically discounts the time spent
                    # in the paused state.
                    delta_seconds = runtime_timer.elapsed - (previous_seconds + system.paused_time)
                    if delta_seconds > 0:
                        # While it is unlikely that delta ever exceeds 1, supports this rare case
                        pbar.update(delta_seconds)
                        previous_seconds = runtime_timer.elapsed - system.paused_time

                system.paused_time = 0  # Resets the paused time before entering the next experiment state's cycle

                # If the user sent the abort command, terminates the experiment early.
                if system.terminated:
                    message = (
                        "Experiment session abort signal detected. Aborting the experiment with a graceful shutdown "
                        "procedure..."
                    )
                    console.echo(message=message, level=LogLevel.ERROR)
                    break  # Breaks the for loop

    # RecursionErrors should not be raised by any session component except in the case that the user wants to terminate
    # the session as part of the startup checkpoint. Therefore, silences the error.
    except RecursionError:
        pass

    # Ensures that the function always attempts the graceful shutdown procedure, even if it encounters session errors.
    finally:
        # If the system was initialized, attempts to gracefully terminate system assets
        if system is not None:
            system.stop()

        # If the session terminates before the session was initialized, removes session data from all
        # sources before shutting down.
        if session_data.raw_data.nk_path.exists():
            message = (
                "The experiment session was unexpectedly terminated before it was able to initialize and start all "
                "assets. Removing all leftover data from the uninitialized session from all destinations..."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            purge_session(session_data)

        message = "Experiment session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


def maintenance_logic() -> None:
    """Encapsulates the logic used to maintain a subset of the Mesoscope-VR system's hardware components."""
    console.echo(message="Initializing Mesoscope-VR system maintenance runtime...", level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters
    system_configuration = get_system_configuration()

    # Initializes a timer used to optimize the main runtime cycling.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    # Determines whether to move all Zaber motors to the predefined maintenance positions.
    console.echo(
        message="Do you want to position the managed Zaber motors for valve calibration or referencing procedure?",
        level=LogLevel.INFO,
    )
    move_zaber_motors = ""
    while move_zaber_motors not in ["y", "n"]:
        user_input = input("Enter 'yes' or 'no': ").strip().lower()
        move_zaber_motors = user_input[0] if user_input else ""

    # All calibration procedures are executed in a temporary directory deleted after runtime
    with tempfile.TemporaryDirectory(prefix="sl_maintenance_") as output_dir:
        try:
            console.echo(message="Initializing the maintenance assets...", level=LogLevel.INFO)

            # Initializes the data logger. All log entries recorded by the logger during runtime are discarded at the
            # end of runtime, hence the name 'temporary'.
            logger = DataLogger(
                output_directory=Path(output_dir),
                instance_name="temporary",
                thread_count=10,
            )
            logger.start()

            # Initializes the interface for the Actor MicroController.
            valve: ValveInterface = ValveInterface(
                valve_calibration_data=(
                    system_configuration.microcontrollers.valve_calibration_data  # type: ignore[arg-type]
                ),
            )
            gas_puff_valve: GasPuffValveInterface = GasPuffValveInterface()
            wheel: BrakeInterface = BrakeInterface(
                minimum_brake_strength=system_configuration.microcontrollers.minimum_brake_strength_g_cm,
                maximum_brake_strength=system_configuration.microcontrollers.maximum_brake_strength_g_cm,
            )
            controller: MicroControllerInterface = MicroControllerInterface(
                controller_id=np.uint8(101),
                buffer_size=8192,
                port=system_configuration.microcontrollers.actor_port,
                name="actor",
                data_logger=logger,
                module_interfaces=(valve, gas_puff_valve, wheel),
            )
            controller.start()

            message = "Actor MicroController interface: Initialized."
            console.echo(message=message, level=LogLevel.SUCCESS)

            # Avoids the visual clash with the Zaber positioning dialog.
            _response_delay_timer.delay(delay=_RENDERING_SEPARATION_DELAY, block=False)

            # If Zaber motors are being used, initializes and moves them to the maintenance positions.
            if move_zaber_motors == "y":
                message = "Initializing Zaber motors..."
                console.echo(message=message, level=LogLevel.INFO)
                zaber_motors: ZaberMotors = ZaberMotors(
                    zaber_positions=None, zaber_configuration=system_configuration.assets
                )
                message = (
                    "Preparing to move Zaber motors to their maintenance positions. Remove the mesoscope objective, "
                    "swivel out the VR screens, and make sure the animal is NOT mounted on the rig. Failure to fulfill "
                    "these steps may DAMAGE the mesoscope and / or HARM the animal."
                )
                console.echo(message=message, level=LogLevel.WARNING)

                # Delays to ensure the user reads the message before continuing.
                _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

                input("Press Enter to continue: ")
                zaber_motors.prepare_motors()
                zaber_motors.maintenance_position()

                message = "Zaber motors: Positioned for Mesoscope-VR system maintenance."
                console.echo(message=message, level=LogLevel.SUCCESS)

            # Initializes the maintenance GUI
            # noinspection PyProtectedMember
            ui = MaintenanceControlUI(
                valve_tracker=valve._valve_tracker,  # noqa: SLF001
                gas_puff_tracker=gas_puff_valve._puff_tracker,  # noqa: SLF001
            )
            ui.start()

            # Notifies the user that the runtime is initialized.
            console.echo(
                message="Maintenance runtime: Initialized. Use the GUI to control the valve and brake.",
                level=LogLevel.SUCCESS,
            )

            # Enters the main control loop, relinquishing control to the maintenance GUI.
            while not ui.exit_signal:
                # Opens the valve
                if ui.valve_open_signal:
                    valve.set_state(state=True)

                # Closes the valve
                if ui.valve_close_signal:
                    valve.set_state(state=False)

                # Uses the valve to deliver a water reward
                if ui.valve_reward_signal:
                    valve.deliver_reward(volume=float(ui.reward_volume))

                # References the valve
                if ui.valve_reference_signal:
                    valve.reference_valve()

                # Performs the valve calibration procedure
                if ui.valve_calibrate_signal:
                    valve.calibrate_valve(pulse_duration=ui.calibration_pulse_duration)

                # Locks the wheel brake
                if ui.brake_lock_signal:
                    wheel.set_state(state=True)

                # Unlocks the wheel brake
                if ui.brake_unlock_signal:
                    wheel.set_state(state=False)

                # Opens the gas puff valve
                if ui.gas_valve_open_signal:
                    gas_puff_valve.set_state(state=True)

                # Closes the gas puff valve
                if ui.gas_valve_close_signal:
                    gas_puff_valve.set_state(state=False)

                # Delivers a gas puff
                if ui.gas_valve_pulse_signal:
                    gas_puff_valve.deliver_puff(duration_ms=ui.gas_valve_pulse_duration)

                # Delays for 5 milliseconds to avoid busy-waiting
                delay_timer.delay(delay=5, block=False)

        # Ensures that the runtime always attempts to terminate all assets gracefully
        finally:
            message = "Terminating Mesoscope-VR maintenance runtime..."
            console.echo(message=message, level=LogLevel.INFO)

            # If Zaber motors were used and are still connected, moves them to the park position.
            if move_zaber_motors == "y" and zaber_motors.is_connected:
                message = (
                    "Preparing to reset all Zaber motors. Remove all objects used during Mesoscope-VR maintenance, "
                    "such as water collection flasks, from the Mesoscope-VR cage."
                )
                console.echo(message=message, level=LogLevel.WARNING)

                # Delays for 2 seconds to ensure the user reads the message before continuing.
                _response_delay_timer.delay(delay=_RESPONSE_DELAY, block=False)

                input("Press Enter to continue: ")
                zaber_motors.park_position()
                zaber_motors.disconnect()

            # Shuts down the actor microcontroller interface.
            controller.stop()

            message = "Actor MicroController interface: Terminated."
            console.echo(message=message, level=LogLevel.SUCCESS)

            # Stops the data logger
            logger.stop()

            # Shuts down the UI
            ui.shutdown()

            message = "Mesoscope-VR system maintenance runtime: Terminated."
            console.echo(message=message, level=LogLevel.SUCCESS)
