"""Provides the assets for executing data acquisition sessions and maintenance runtimes via the Mesoscope-VR data
acquisition system.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from pathlib import Path
import tempfile

from tqdm import tqdm
import numpy as np
from ataraxis_time import TimeUnits, PrecisionTimer, TimerPrecisions, convert_time
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    ProjectData,
    SessionData,
    SessionTypes,
    ExperimentState,
    AcquisitionSystems,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration,
    get_data_root,
    get_projects_for_animal,
)
from ataraxis_data_structures import DataLogger
from ataraxis_communication_interface import MicroControllerInterface

from .system import (
    RUN_TRAINING_THRESHOLD_LIMITS,
    MesoscopeData,
    ZaberPositions,
    MesoscopeVRStates,
    MesoscopePositions,
    get_system_configuration,
)
from ..cross_system import (
    BrakeInterface,
    WaterValveInterface,
    GasPuffValveInterface,
    get_version_data,
    get_project_experiments,
)
from .maintenance_ui import MaintenanceControlUI
from .binding_classes import ZaberMotors, VideoSystems
from .mesoscope_driver import MesoscopeDriver
from .system_controller import MesoscopeVRSystem
from .data_preprocessing import purge_session, preprocess_session_data
from .acquisition_components import (
    RESPONSE_DELAY,
    RESPONSE_DELAY_TIMER,
    setup_mesoscope,
    reset_zaber_motors,
    setup_zaber_motors,
    generate_zaber_snapshot,
    finalize_session_descriptor,
    generate_mesoscope_position_snapshot,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .system import MesoscopeSystemConfiguration

_RENDERING_SEPARATION_DELAY: int = 500
"""Specifies the number of milliseconds to delay between rendering console outputs (stderr) and non-console outputs
(stdout) to prevent the two renders from overlapping."""

_MICROLITERS_PER_MILLILITER: float = 1000.0
"""Specifies the number of microliters in one milliliter, used to convert between the two water-volume units."""


# PyCharm does not narrow the Optional `zaber_motors` after assignment (the Optional is required for the finally
# guard) and resolves the descriptor union to its first member; both are false positives that mypy does not report.
# noinspection PyTypeChecker
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

    # Verifies that the specified project is configured and that the animal participates exclusively in it.
    _verify_project_configured(
        session_description="window checking session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )
    _verify_animal_project_membership(
        session_description="window checking session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        animal=ProjectData(root=get_data_root(), project_name=project_name).animal(animal_id),
        session_type=SessionTypes.WINDOW_CHECKING,
        python_version=python_version,
        sollertia_experiment_version=library_version,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
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
    precursor.to_yaml(file_path=session_data.system_raw_data.mesoscope_positions_path)
    precursor.to_yaml(file_path=mesoscope_data.vrpc_data.mesoscope_positions_path)

    zaber_motors: ZaberMotors | None = None

    # Builds the mesoscope control driver used to command the ScanImage software over the shared Virtual Reality MQTT
    # broker. The driver is connected just before the mesoscope preparation sequence and disconnected during cleanup.
    mesoscope_driver = MesoscopeDriver(
        configuration=system_configuration.assets.vr_task,
        acquisition=system_configuration.acquisition,
    )
    try:
        # If the animal has a snapshot of Zaber motor positions used during a previous runtime, loads and uses these
        # positions. Otherwise, uses the default positions hardcoded in the Zaber controller's non-volatile memory.
        zaber_positions = (
            ZaberPositions.from_yaml(file_path=mesoscope_data.vrpc_data.zaber_positions_path)
            if mesoscope_data.vrpc_data.zaber_positions_path.exists()
            else None
        )

        # Initializes the data logger. This initialization follows the same procedure as the MesoscopeVRSystem class.
        logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
            # Creates the behavior_log subdirectory under raw_data.
            instance_name="behavior",
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
        RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)
        input("Enter anything to continue: ")

        # Establishes communication with Zaber motors.
        zaber_motors = ZaberMotors(zaber_positions=zaber_positions, zaber_configuration=system_configuration.assets)

        # Removes the nk.bin marker to avoid automatic session cleanup during post-processing.
        session_data.mark_runtime_initialized()

        # Prepares Zaber motors for data acquisition.
        setup_zaber_motors(zaber_motors=zaber_motors)

        # Connects to the ScanImagePC over MQTT, then runs the user through the process of preparing the mesoscope and
        # assessing the quality of the animal's cranial window.
        mesoscope_driver.connect()
        setup_mesoscope(session_data=session_data, mesoscope_data=mesoscope_data, mesoscope_driver=mesoscope_driver)

        # Retrieves current motor positions and packages them into a ZaberPositions object.
        generate_zaber_snapshot(session_data=session_data, mesoscope_data=mesoscope_data, zaber_motors=zaber_motors)

        # Collects the experimenter notes through a GUI window and writes the completed session descriptor.
        finalize_session_descriptor(descriptor=descriptor, session_data=session_data, mesoscope_data=mesoscope_data)

        # Generates the snapshot of the Mesoscope imaging position used to generate the data during window checking by
        # querying the still-connected ScanImagePC.
        generate_mesoscope_position_snapshot(
            session_data=session_data, mesoscope_data=mesoscope_data, mesoscope_driver=mesoscope_driver
        )

        # Resets Zaber motors to their original positions.
        reset_zaber_motors(zaber_motors=zaber_motors)

        # Terminates the face camera.
        cameras.stop()

        # Stops the data logger.
        logger.stop()

        # Triggers preprocessing pipeline. In this case, since there is no data to preprocess, the pipeline primarily
        # just copies the session raw_data directory to all configured long-term storage destinations.
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
            purge_session(session_data=session_data)

        # If Zaber motors were connected, attempts to gracefully shut down the motors.
        if zaber_motors is not None:
            reset_zaber_motors(zaber_motors=zaber_motors)

        # Disconnects from the ScanImagePC. The disconnect is a no-op if the driver never connected.
        mesoscope_driver.disconnect()

        # Ends the runtime.
        message = "Window checking session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


# PyCharm does not narrow the Optional `system` after assignment (the Optional is required for the finally guard) and
# mis-infers some descriptor fields and numpy scalars; these are false positives that mypy does not report.
# noinspection PyTypeChecker,PyUnresolvedReferences
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

    # Verifies that the specified project is configured and that the animal participates exclusively in it.
    _verify_project_configured(
        session_description="lick training session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )
    _verify_animal_project_membership(
        session_description="lick training session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        animal=ProjectData(root=get_data_root(), project_name=project_name).animal(animal_id),
        session_type=SessionTypes.LICK_TRAINING,
        python_version=python_version,
        sollertia_experiment_version=library_version,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # If the trained animal has previously participated in this type of sessions, loads the previous session's runtime
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: LickTrainingDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory.
        previous_descriptor = LickTrainingDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight.
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
    disable_unconsumed_limit = descriptor.maximum_unconsumed_rewards < 1

    # Initializes the timer used to enforce reward delays.
    delay_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    message = "Generating the pseudorandom reward delay sequence..."
    console.echo(message=message, level=LogLevel.INFO)

    # Converts maximum volume to uL and divides it by the reward size to get the number of delays to sample from
    # the delay distribution.
    number_of_samples = np.floor(
        (descriptor.maximum_water_volume_ml * _MICROLITERS_PER_MILLILITER) / descriptor.water_reward_size_ul
    ).astype(np.uint64)

    # Generates samples from a uniform distribution within delay bounds.
    random_generator = np.random.default_rng()
    samples = random_generator.uniform(
        low=descriptor.minimum_reward_delay_s,
        high=descriptor.maximum_reward_delay_s,
        size=number_of_samples,
    )

    # Calculates cumulative training time for each sampled delay. This communicates the total time passed when each
    # reward is delivered to the animal.
    cumulative_time = np.cumsum(samples)

    # Finds the maximum number of samples that fits within the maximum training time. This handles the (expected) cases
    # where the total training time is insufficient to deliver the maximum allowed volume of water, so the reward
    # sequence needs to be clipped.
    maximum_sample_index = np.searchsorted(
        a=cumulative_time,
        v=convert_time(
            time=descriptor.maximum_training_time_min, from_units=TimeUnits.MINUTE, to_units=TimeUnits.SECOND
        ),
        side="right",
    )

    # Slices the samples array to make the total training time be roughly the maximum requested duration.
    reward_delays: NDArray[np.float64] = samples[:maximum_sample_index]

    # Aborts if no rewards fit in the requested training time. Raises an error before system initialization to allow
    # automatic session data purge.
    if maximum_sample_index == 0:
        message = (
            f"Unable to generate the lick training reward sequence. The requested maximum training time "
            f"({descriptor.maximum_training_time_min} minutes) is shorter than the minimum reward delay "
            f"({descriptor.minimum_reward_delay_s} seconds). Increase the maximum training time or decrease the "
            f"minimum reward delay."
        )
        console.error(message=message, error=ValueError)

    cumulative_runtime_min = float(
        np.round(
            convert_time(
                time=cumulative_time[maximum_sample_index - 1], from_units=TimeUnits.SECOND, to_units=TimeUnits.MINUTE
            ),
            decimals=3,
        )
    )
    message = (
        f"Generated a sequence of {len(reward_delays)} rewards with the total cumulative runtime of "
        f"{cumulative_runtime_min} minutes."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # If session runtime is limited by the total volume of delivered water, rather than the maximum runtime, clips the
    # total training time at the point where the maximum allowed water volume is delivered.
    if len(reward_delays) == len(cumulative_time):
        # Actual session time is the accumulated delay converted from seconds to minutes at the last index.
        descriptor.maximum_training_time_min = int(
            np.ceil(convert_time(time=cumulative_time[-1], from_units=TimeUnits.SECOND, to_units=TimeUnits.MINUTE))
        )

    # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
    # number of rewards to be delivered.
    if disable_unconsumed_limit:
        descriptor.maximum_unconsumed_rewards = len(reward_delays)

    system: MesoscopeVRSystem | None = None
    try:
        system = MesoscopeVRSystem(session_data=session_data, session_descriptor=descriptor)

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of error should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exception as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.mark_runtime_initialized()

        # Switches the system into lick-training mode.
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
                # Repeatedly calls the runtime cycle during the delay period.
                system.runtime_cycle()

            # If the user sent the abort command, terminates the training early.
            if system.terminated:
                message = (
                    "Lick training abort signal detected. Aborting the lick training with a graceful shutdown "
                    "procedure..."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break

            # Resets the delay timer immediately after exiting the delay loop.
            delay_timer.reset()

            # Clears the paused time at the end of each delay cycle. This has to be done to prevent future delay
            # loops from ending earlier than expected unless the session is paused again as part of that loop.
            system.paused_time = 0

            # Delivers the water reward to the animal or simulates the reward if the animal is not licking.
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
        # If the system was initialized, attempts to gracefully terminate system assets.
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
            purge_session(session_data=session_data)

        message = "Lick training session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


# PyCharm does not narrow the Optional `system` after assignment (the Optional is required for the finally guard) and
# mis-infers some descriptor fields; these are false positives that mypy does not report.
# noinspection PyTypeChecker,PyUnresolvedReferences
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

    # Verifies that the specified project is configured and that the animal participates exclusively in it.
    _verify_project_configured(
        session_description="run training session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )
    _verify_animal_project_membership(
        session_description="run training session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        animal=ProjectData(root=get_data_root(), project_name=project_name).animal(animal_id),
        session_type=SessionTypes.RUN_TRAINING,
        python_version=python_version,
        sollertia_experiment_version=library_version,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # If the trained animal has previously participated in this type of sessions, loads the previous session's
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: RunTrainingDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory.
        previous_descriptor = RunTrainingDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight.
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
    disable_unconsumed_limit = descriptor.maximum_unconsumed_rewards < 1

    # Determines whether the volume-driven threshold increase is disabled. A non-positive increase threshold holds the
    # speed and duration thresholds at their initial values for the entire session, while a positive threshold enables
    # the dynamic increase computed during the runtime loop.
    increase_disabled = descriptor.increase_threshold_ml <= 0

    # Initializes the timers used during the session.
    runtime_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    running_duration_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    epoch_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    # Initializes the assets used to guard against interrupting run epochs for animals that take many large steps. For
    # animals with a distinct walking pattern of many very large steps, the speed transiently dips below the threshold
    # for a very brief moment of time, flagging the epoch as unrewarded. To avoid this issue, instead of interrupting
    # the epoch outright, the system allows the speed to be below the threshold for a short period of time. These
    # assets help with that task pattern.
    epoch_timer_engaged: bool = False
    # Ensures a positive value and converts the maximum idle time from seconds to milliseconds.
    maximum_idle_time_ms = max(0.0, descriptor.maximum_idle_time_s) * 1000

    # If the maximum unconsumed reward count is below 1, disables the feature by setting the number to match the
    # maximum number of rewards that can be delivered during the session.
    if disable_unconsumed_limit:
        descriptor.maximum_unconsumed_rewards = int(
            np.ceil(descriptor.maximum_water_volume_ml / (descriptor.water_reward_size_ul / 1000))
        )

    # Converts all arguments used to determine the speed and duration threshold over time into numpy variables to
    # optimize the main session's runtime loop:
    initial_speed = np.float64(descriptor.initial_run_speed_threshold_cm_s)  # In centimeters per second
    maximum_speed = np.float64(RUN_TRAINING_THRESHOLD_LIMITS.maximum_speed_cm_s)  # In centimeters per second
    speed_step = np.float64(descriptor.run_speed_increase_step_cm_s)  # In centimeters per second

    initial_duration = np.float64(descriptor.initial_run_duration_threshold_s * 1000)  # In milliseconds
    maximum_duration = np.float64(RUN_TRAINING_THRESHOLD_LIMITS.maximum_duration_s * 1000)  # In milliseconds
    duration_step = np.float64(descriptor.run_duration_increase_step_s * 1000)  # In milliseconds

    water_threshold = np.float64(descriptor.increase_threshold_ml * 1000)  # In microliters
    maximum_volume = np.float64(descriptor.maximum_water_volume_ml * 1000)  # In microliters

    # Converts the training time from minutes to seconds to make it compatible with the timer precision.
    training_time = convert_time(
        time=descriptor.maximum_training_time_min, from_units=TimeUnits.MINUTE, to_units=TimeUnits.SECOND
    )

    # Initializes internal tracker variables:
    # Tracks the data necessary to update the training progress bar.
    previous_time = 0

    # Tracks when speed and / or duration thresholds are updated. This is necessary to redraw the threshold lines in
    # the visualizer plot.
    previous_speed_threshold = copy.copy(initial_speed)
    previous_duration_threshold = copy.copy(initial_duration)

    # This one-time tracker is used to initialize the speed and duration threshold visualization.
    once = True

    # Updates the descriptor with the final threshold values saved at the end of the session. These are
    # initialized to the initial thresholds and are updated during the session if the animal progresses.
    descriptor.final_run_speed_threshold_cm_s = descriptor.initial_run_speed_threshold_cm_s
    descriptor.final_run_duration_threshold_s = descriptor.initial_run_duration_threshold_s

    system: MesoscopeVRSystem | None = None
    try:
        system = MesoscopeVRSystem(session_data=session_data, session_descriptor=descriptor)

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of error should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exception as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.mark_runtime_initialized()

        # Switches the system into the run-training mode.
        system.run_train()

        message = "Run training: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Creates a tqdm progress bar that tracks the overall training progress by communicating the total volume of
        # water delivered to the animal. Uses tqdm directly (instead of console.progress) because the bar relies on
        # set_postfix_str() and a custom bar_format to display the running training time alongside the volume readout,
        # neither of which the ataraxis-base-utilities ProgressBar wrapper currently exposes.
        progress_bar = tqdm(
            total=round(descriptor.maximum_water_volume_ml, ndigits=3),
            desc="Delivered water volume",
            unit="ml",
            bar_format="{l_bar}{bar}| {n:.3f}/{total:.3f} {postfix}",
        )

        runtime_timer.reset()
        # It is critical to reset both timers at the same time.
        running_duration_timer.reset()

        # Pre-initializes the threshold trackers to avoid MyPy errors.
        speed_threshold: np.float64 = np.float64(0.0)
        duration_threshold: np.float64 = np.float64(0.0)

        # This is the main session loop of the run training mode.
        while runtime_timer.elapsed < (training_time + system.paused_time):
            # Repeatedly calls the runtime cycle during training.
            system.runtime_cycle()

            # If the user sent the abort command, terminates the training early.
            if system.terminated:
                message = (
                    "Run training abort signal detected. Aborting the run training with a graceful shutdown "
                    "procedure..."
                )
                console.echo(message=message, level=LogLevel.ERROR)
                break

            # Determines how many times the speed and duration thresholds have been increased based on the difference
            # between the total delivered water volume and the increase threshold. This dynamically adjusts the running
            # speed and duration thresholds with delivered water volume, ensuring the animal has to try progressively
            # harder to keep receiving water.
            increase_steps: np.float64 = (
                np.float64(0) if increase_disabled else np.floor(system.dispensed_water_volume / water_threshold)
            )

            # Computes the automatic (runtime-driven) component of each threshold, which combines the initial threshold
            # with the volume-driven increments but excludes the user-defined GUI modifier. Publishing these values to
            # the control GUI lets it display the current effective threshold and back-compute the modifier needed to
            # match user-requested absolute values.
            auto_speed: np.float64 = initial_speed + (increase_steps * speed_step)
            auto_duration: np.float64 = initial_duration + (increase_steps * duration_step)
            system.publish_runtime_thresholds(speed_threshold=auto_speed, duration_threshold=auto_duration)

            # Determines the effective speed and duration thresholds for each cycle by adding the user input from the
            # session control GUI on top of the automatic component. User input has a static resolution of 0.01 cm/s
            # and 0.01 s (10 ms) per step.
            speed_threshold = np.clip(
                a=auto_speed + (system.speed_modifier * 0.01),
                a_min=RUN_TRAINING_THRESHOLD_LIMITS.minimum_speed_cm_s,  # Minimum value
                a_max=maximum_speed,  # Maximum value
            )
            duration_threshold = np.clip(
                a=auto_duration + (system.duration_modifier * 10),
                a_min=RUN_TRAINING_THRESHOLD_LIMITS.minimum_duration_s * 1000,  # Minimum value, converted to ms
                a_max=maximum_duration,  # Maximum value
            )

            # If any of the threshold changed relative to the previous loop iteration, updates the visualizer and
            # previous threshold trackers with new data. The update is forced at the beginning of the session to make
            # the visualizer render the threshold lines and values.
            if once or (
                duration_threshold != previous_duration_threshold or previous_speed_threshold != speed_threshold
            ):
                system.update_visualizer_thresholds(
                    speed_threshold=speed_threshold, duration_threshold=duration_threshold
                )
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

                # If the epoch timer was active for the current epoch, resets the timer.
                epoch_timer_engaged = False

            # If the current speed is below the speed threshold, acts depending on whether the session is configured to
            # allow dipping below the threshold.
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
                previous_time = elapsed_time

                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                progress_bar.set_postfix_str(
                    f"Time: {elapsed_minutes:02d}:{elapsed_seconds:02d}/{descriptor.maximum_training_time_min:02d}:00"
                )

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

        # Closes the progress bar if the session ends as expected.
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
        # If the system was initialized, attempts to gracefully terminate system assets.
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
            purge_session(session_data=session_data)

        message = "Run training session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


# PyCharm does not narrow the Optional `system` after assignment (the Optional is required for the finally guard) and
# mis-infers a descriptor field; these are false positives that mypy does not report.
# noinspection PyTypeChecker,PyUnresolvedReferences
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
        predefined sequence of states defined in the experiment's configuration file. Once all states are executed, the
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

    # Verifies that the specified project is configured.
    project_directory = _verify_project_configured(
        session_description=f"{experiment_name} experiment session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )

    # Prevents the user from executing the session if the project is not configured to run the requested experiment.
    project_experiments = get_project_experiments(project_directory=project_directory)
    if experiment_name not in project_experiments:
        message = (
            f"Unable to execute the {experiment_name} experiment session for the animal {animal_id} participating in "
            f"the project {project_name}. The target project does not have an experiment configuration file named "
            f"after the target experiment. Use the 'slsa configure experiment' command to configure the experiment "
            f"before running experiment sessions."
        )
        console.error(message=message, error=FileNotFoundError)

    # Verifies that the animal participates exclusively in the specified project.
    _verify_animal_project_membership(
        session_description=f"{experiment_name} experiment session",
        system_configuration=system_configuration,
        project_name=project_name,
        animal_id=animal_id,
    )

    # Queries the current Python and library version information. This is then used to initialize the SessionData
    # instance.
    python_version, library_version = get_version_data()

    # Initializes the acquired session's data hierarchy and resolves the Mesoscope-VR's filesystem configuration.
    session_data = SessionData.create(
        animal=ProjectData(root=get_data_root(), project_name=project_name).animal(animal_id),
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        experiment_name=experiment_name,
        python_version=python_version,
        sollertia_experiment_version=library_version,
        acquisition_system=AcquisitionSystems.MESOSCOPE_VR,
    )
    mesoscope_data = MesoscopeData(session_data=session_data, system_configuration=system_configuration)

    # Uses initialized SessionData instance to load the experiment configuration data.
    experiment_config: MesoscopeExperimentConfiguration = MesoscopeExperimentConfiguration.from_yaml(
        file_path=session_data.raw_data.experiment_configuration_path
    )

    # Verifies that all Mesoscope-VR states used during experiments are valid.
    valid_states = (MesoscopeVRStates.REST, MesoscopeVRStates.RUN)
    supported_state_codes = ", ".join(f"{state.value} ({state.name.lower()})" for state in valid_states)
    state: ExperimentState
    for state in experiment_config.experiment_states.values():
        if state.system_state_code not in valid_states:
            message = (
                f"Invalid Mesoscope-VR system state code {state.system_state_code} encountered when verifying "
                f"{experiment_name} experiment configuration. Currently, only codes {supported_state_codes} are "
                f"supported for the Mesoscope-VR system."
            )
            console.error(message=message, error=ValueError)

    # If the experimental animal has previously participated in this type of sessions, loads the previous session's
    # parameters and uses them to override the default configuration parameters in the pregenerated descriptor instance.
    previous_descriptor_path = mesoscope_data.vrpc_data.session_descriptor_path
    previous_descriptor: MesoscopeExperimentDescriptor | None = None
    if previous_descriptor_path.exists():
        # Loads the previous descriptor's data from memory.
        previous_descriptor = MesoscopeExperimentDescriptor.from_yaml(file_path=previous_descriptor_path)

        message = "Previous session's configuration parameters: Applied."
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            "Previous session's configuration parameters: Not found. Using the default configuration parameters..."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Initializes the descriptor with the current session's experimenter and animal weight.
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

    # Initializes the timer to enforce experiment state durations.
    runtime_timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    system: MesoscopeVRSystem | None = None
    try:
        system = MesoscopeVRSystem(
            session_data=session_data, session_descriptor=descriptor, experiment_configuration=experiment_config
        )

        # Initializes all system assets and guides the user through hardware-specific session preparation steps.
        system.start()

        # If the user chose to terminate the session during initialization checkpoint, raises an error to jump to the
        # shutdown sequence, bypassing all other session preparation steps.
        if system.terminated:
            # Note, this specific type of error should not be raised by any other session component. Therefore, it is
            # possible to handle this type of exception as a unique marker for early user-requested session
            # termination.
            message = "The session was terminated early due to user request."
            console.echo(message=message, level=LogLevel.SUCCESS)
            raise RecursionError  # noqa: TRY301

        # Marks the session as fully initialized. This prevents session data from being automatically removed by
        # 'purge' runtimes.
        session_data.mark_runtime_initialized()

        # Main session loop. It loops over all submitted experiment states and ends the session after executing the
        # last state.
        for state in experiment_config.experiment_states.values():
            runtime_timer.reset()

            # Sets the Experiment state.
            system.change_runtime_state(new_state=state.experiment_state_code)

            # Resets the tracker used to update the progress bar every second.
            previous_seconds = 0

            # Resolves and sets the Mesoscope-VR system state.
            if state.system_state_code == MesoscopeVRStates.REST:
                system.rest()
            elif state.system_state_code == MesoscopeVRStates.RUN:
                system.run()
            else:
                message = (
                    f"Unsupported Mesoscope-VR system state code {state.system_state_code} encountered when executing "
                    f"the {state.experiment_state_code} state. Currently, only the following system state codes are "
                    f"supported {supported_state_codes}."
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

            # Creates a tqdm progress bar for the current experiment state. Uses tqdm directly because the bar relies on
            # a custom bar_format to display the percentage and elapsed seconds, which the ataraxis-base-utilities
            # ProgressBar wrapper currently does not expose.
            with tqdm(
                total=state.state_duration_s,
                desc=f"Executing experiment state {state.experiment_state_code}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s",
            ) as progress_bar:
                # Cycles until the state duration of seconds passes.
                while runtime_timer.elapsed < (state.state_duration_s + system.paused_time):
                    # Since experiment logic is resolved by the Unity game engine, the session logic function only
                    # needs to call the runtime cycle and handle runtime termination cases.
                    # Repeatedly calls the runtime cycle as part of the experiment state cycle.
                    system.runtime_cycle()

                    # If the user has terminated the session, breaks the while loop. The termination is also handled at
                    # the level of the 'for' loop. The error message is generated at that level, rather than here.
                    if system.terminated:
                        break

                    # Updates the progress bar every second. Note: this calculation statically discounts the time spent
                    # in the paused state.
                    delta_seconds = runtime_timer.elapsed - (previous_seconds + system.paused_time)
                    if delta_seconds > 0:
                        # While it is unlikely that delta ever exceeds 1, supports this rare case.
                        progress_bar.update(delta_seconds)
                        previous_seconds = runtime_timer.elapsed - system.paused_time

                # Resets the paused time before entering the next experiment state's cycle.
                system.paused_time = 0

                # If the user sent the abort command, terminates the experiment early.
                if system.terminated:
                    message = (
                        "Experiment session abort signal detected. Aborting the experiment with a graceful shutdown "
                        "procedure..."
                    )
                    console.echo(message=message, level=LogLevel.ERROR)
                    break

    # RecursionErrors should not be raised by any session component except in the case that the user wants to terminate
    # the session as part of the startup checkpoint. Therefore, silences the error.
    except RecursionError:
        pass

    # Ensures that the function always attempts the graceful shutdown procedure, even if it encounters session errors.
    finally:
        # If the system was initialized, attempts to gracefully terminate system assets.
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
            purge_session(session_data=session_data)

        message = "Experiment session: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)


def maintenance_logic() -> None:
    """Encapsulates the logic used to maintain a subset of the Mesoscope-VR system's hardware components."""
    console.echo(message="Initializing Mesoscope-VR system maintenance runtime...", level=LogLevel.INFO)

    # Queries the data acquisition system runtime parameters.
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

    # All calibration procedures are executed in a temporary directory deleted after runtime.
    with tempfile.TemporaryDirectory(prefix="sl_maintenance_") as output_directory:
        try:
            console.echo(message="Initializing the maintenance assets...", level=LogLevel.INFO)

            # Initializes the data logger. All log entries recorded by the logger during runtime are discarded at the
            # end of runtime, hence the name 'temporary'.
            logger = DataLogger(
                output_directory=Path(output_directory),
                instance_name="temporary",
                thread_count=10,
            )
            logger.start()

            # Initializes the interface for the Actor MicroController. The calibration data union is narrower than the
            # interface's declared parameter type; the mismatch is already acknowledged with the type: ignore below.
            # noinspection PyTypeChecker
            valve: WaterValveInterface = WaterValveInterface(
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
            RESPONSE_DELAY_TIMER.delay(delay=_RENDERING_SEPARATION_DELAY, block=False)

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
                RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

                input("Press Enter to continue: ")
                zaber_motors.prepare_motors()
                zaber_motors.maintenance_position()

                message = "Zaber motors: Positioned for Mesoscope-VR system maintenance."
                console.echo(message=message, level=LogLevel.SUCCESS)

            # Initializes the maintenance GUI.
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
                # Opens the valve.
                if ui.valve_open_signal:
                    valve.set_state(state=True)

                # Closes the valve.
                if ui.valve_close_signal:
                    valve.set_state(state=False)

                if ui.valve_reward_signal:
                    valve.deliver_reward(volume=float(ui.reward_volume))

                if ui.valve_reference_signal:
                    valve.reference_valve()

                if ui.valve_calibrate_signal:
                    valve.calibrate_valve(pulse_duration=ui.calibration_pulse_duration)

                # Locks the wheel brake.
                if ui.brake_lock_signal:
                    wheel.set_state(state=True)

                # Unlocks the wheel brake.
                if ui.brake_unlock_signal:
                    wheel.set_state(state=False)

                # Opens the gas puff valve.
                if ui.gas_valve_open_signal:
                    gas_puff_valve.set_state(state=True)

                # Closes the gas puff valve.
                if ui.gas_valve_close_signal:
                    gas_puff_valve.set_state(state=False)

                if ui.gas_valve_pulse_signal:
                    gas_puff_valve.deliver_puff(duration_ms=ui.gas_valve_pulse_duration)

                # Delays for 5 milliseconds to avoid busy-waiting.
                delay_timer.delay(delay=5, block=False)

        # Ensures that the runtime always attempts to terminate all assets gracefully.
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
                RESPONSE_DELAY_TIMER.delay(delay=RESPONSE_DELAY, block=False)

                input("Press Enter to continue: ")
                zaber_motors.park_position()
                zaber_motors.disconnect()

            # Shuts down the actor microcontroller interface.
            controller.stop()

            message = "Actor MicroController interface: Terminated."
            console.echo(message=message, level=LogLevel.SUCCESS)

            # Stops the data logger.
            logger.stop()

            # Shuts down the UI.
            ui.shutdown()

            message = "Mesoscope-VR system maintenance runtime: Terminated."
            console.echo(message=message, level=LogLevel.SUCCESS)


def _verify_project_configured(
    session_description: str,
    system_configuration: MesoscopeSystemConfiguration,
    project_name: str,
    animal_id: str,
) -> Path:
    """Verifies that the target project is configured on the local data acquisition system.

    Args:
        session_description: A short phrase naming the session being prepared, embedded into the error message
            (for example, "window checking session").
        system_configuration: The resolved Mesoscope-VR system configuration instance.
        project_name: The name of the project for which the session is prepared.
        animal_id: The unique identifier of the animal for which the session is prepared.

    Returns:
        The path to the configured project's root directory.
    """
    project = ProjectData(root=get_data_root(), project_name=project_name)
    if not project.exists():
        message = (
            f"Unable to execute the {session_description} for the animal {animal_id} participating in the project "
            f"{project_name}. The {system_configuration.name} data acquisition system is not configured to acquire "
            f"data for this project. Use the 'slsa configure project' command to configure the project before running "
            f"data acquisition sessions."
        )
        console.error(message=message, error=FileNotFoundError)
    return project.path


def _verify_animal_project_membership(
    session_description: str,
    system_configuration: MesoscopeSystemConfiguration,
    project_name: str,
    animal_id: str,
) -> None:
    """Verifies that the target animal participates exclusively in the specified project.

    Args:
        session_description: A short phrase naming the session being prepared, embedded into error messages
            (for example, "window checking session").
        system_configuration: The resolved Mesoscope-VR system configuration instance.
        project_name: The name of the project for which the session is prepared.
        animal_id: The unique identifier of the animal for which the session is prepared.
    """
    animal_projects = get_projects_for_animal(root_path=get_data_root(), animal_id=animal_id)
    # Rare case, often indicative of old migration pipeline use.
    if len(animal_projects) > 1:
        message = (
            f"Unable to execute the {session_description} for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is associated with multiple projects managed by the "
            f"{system_configuration.name} data acquisition system, which is not allowed. The animal is associated with "
            f"the following projects: {', '.join(animal_projects)}."
        )
        console.error(message=message, error=ValueError)
    elif len(animal_projects) == 1 and animal_projects[0] != project_name:
        message = (
            f"Unable to execute the {session_description} for the animal {animal_id} participating in the project "
            f"{project_name}. The animal is already associated with a different project '{animal_projects[0]}' managed "
            f"by the {system_configuration.name} data acquisition system. If necessary, use the 'sle mesoscope "
            f"migrate' CLI command to transfer the animal to the desired project."
        )
        console.error(message=message, error=ValueError)
