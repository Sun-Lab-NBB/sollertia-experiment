"""Provides the controller class that orchestrates the runtime of a single Mesoscope-VR data acquisition session."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, convert_time, get_timestamp
from ataraxis_base_utilities import LogLevel, console, convert_scalar_to_bytes
from sollertia_shared_assets import (
    SessionData,
    GasPuffTrial,
    SessionTypes,
    TaskTemplate,
    WaterRewardTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    MesoscopeHardwareState,
    MesoscopeExperimentDescriptor,
    MesoscopeExperimentConfiguration,
)
from ataraxis_data_structures import DataLogger, LogPackage

from .system import MesoscopeData, ZaberPositions, MesoscopePositions, get_system_configuration
from ..vr_task import (
    VRTaskDriver,
    VRTaskEventKind,
    load_vr_task_template,
)
from .runtime_ui import RuntimeControlUI
from .visualizer import VisualizerMode, BehaviorVisualizer
from .binding_classes import ZaberMotors, VideoSystems, MicroControllerInterfaces
from .data_preprocessing import purge_session, preprocess_session_data, rename_mesoscope_directory

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .system import MesoscopeSystemConfiguration

from .system import MesoscopeVRStates
from .acquisition_components import (
    _RESPONSE_DELAY,
    _TrialState,
    _setup_mesoscope,
    _reset_zaber_motors,
    _setup_zaber_motors,
    _response_delay_timer,
    _generate_zaber_snapshot,
    _verify_descriptor_update,
    _MesoscopeVRLogMessageCodes,
    _generate_mesoscope_position_snapshot,
)

_MINIMUM_CPU_COUNT: int = 10
"""The minimum number of logical CPU cores the host machine must provide to run a data acquisition session. The
runtime reserves three cores for microcontrollers, one for the data logger, four for the video systems, one for the
central process, and one for the main GUI."""

_GUIDED_RUNTIME_STATE_CODE: int = 255
"""The runtime (task) state code used to mark active training stages. This is an arbitrary uint8 experiment code
distinct from the MesoscopeVRStates system-state enum, which only spans codes 0 through 4."""

_MESOSCOPE_START_TIMEOUT_MS: int = 15000
"""The maximum time, in milliseconds, to wait for the mesoscope to begin acquiring frames after the acquisition
trigger is sent before treating the start attempt as failed."""

_EXPECTED_FRAME_PULSES: int = 10
"""The number of mesoscope frame acquisition pulses that must be received within the start timeout to confirm that
frame acquisition has begun."""

_MICROLITERS_PER_MILLILITER: float = 1000.0
"""The number of microliters in a single milliliter, used to convert dispensed water volumes from microliters to
milliliters."""


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
        _unconsumed_reward_count: The number of rewards delivered to the animal that have not yet been consumed
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
        _paused_water_volume: Tracks the total volume of water, in microliters, dispensed by the water delivery valve
            when the session's data acquisition was paused.
        _logger: The DataLogger instance that logs the data from all sources managed by the Mesoscope-VR instance.
        _microcontrollers: The MicroControllerInterfaces instance that interfaces with the Actor, Sensor, and Encoder
            microcontrollers used during runtime.
        _cameras: The VideoSystems instance that interfaces with the face and body cameras used during runtime.
        _zaber_motors: The ZaberMotors instance that interfaces with the HeadBar, LickPort, and Wheel motor groups.
        _task_template: The VR TaskTemplate loaded from the task templates directory for the experiment's Unity
            scene, or None if the managed runtime is not a Mesoscope experiment session.
        _vr_task: The VRTaskDriver instance that drives the Unity game engine that runs the Virtual Reality task or
            None, if the managed runtime is not a Mesoscope experiment session.
        _ui: The RuntimeControlUI instance that maintains a Graphical User Interface that allows the user to
            control the session's runtime.
        _visualizer: The BehaviorVisualizer instance used during runtime to visualize the animal's behavior or
            None, if the managed runtime does not require behavior visualization.
        _mesoscope_timer: The PrecisionTimer instance used to track the delay between receiving consecutive
            mesoscope frame acquisition pulses.
        _trial_state: The _TrialState instance that tracks the progression of trials during experiment runtimes.

    Raises:
        RuntimeError: If the host-machine does not have enough logical CPU cores to support the runtime.
    """

    # Statically assigns mesoscope frame checking window and speed calculation window, in milliseconds.
    _mesoscope_frame_delay: int = 300
    _speed_calculation_window: int = 50

    # Reserves logging source ID code 1 for this class.
    _source_id: np.uint8 = np.uint8(1)

    def __init__(
        self,
        session_data: SessionData,
        session_descriptor: MesoscopeExperimentDescriptor | LickTrainingDescriptor | RunTrainingDescriptor,
        experiment_configuration: MesoscopeExperimentConfiguration | None = None,
    ) -> None:
        # Creates runtime state tracking flags.
        self._started: bool = False
        self._terminated: bool = False
        self._paused: bool = False
        self._mesoscope_started: bool = False

        # Pre-runtime check to ensure that the host-machine has enough cores to facilitate the data acquisition.
        cpu_count = os.cpu_count()
        if cpu_count is None or not cpu_count >= _MINIMUM_CPU_COUNT:
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
            previous_mesoscope_positions.to_yaml(file_path=session_data.system_raw_data.mesoscope_positions_path)

        # If previous position data is not available, creates a new MesoscopePositions instance with default position
        # values.
        else:
            # Caches the precursor file to the raw_data session directory and to the persistent data directory.
            precursor = MesoscopePositions()
            precursor.to_yaml(file_path=session_data.system_raw_data.mesoscope_positions_path)
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

        # Initializes the trial state tracking dataclass.
        self._trial_state: _TrialState = _TrialState()

        # Initializes the DataLogger instance used to log data from all microcontrollers, camera frame savers, and this
        # class instance.
        self._logger: DataLogger = DataLogger(
            output_directory=session_data.raw_data_path,
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

        # Loads the VR task template and builds the Virtual Reality task driver only for Mesoscope experiment
        # sessions. Other session types do not drive Unity.
        self._task_template: TaskTemplate | None = None
        self._vr_task: VRTaskDriver | None = None
        if (
            self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT
            and self._experiment_configuration is not None
        ):
            task_template = load_vr_task_template(unity_scene_name=self._experiment_configuration.unity_scene_name)
            self._task_template = task_template
            self._vr_task = VRTaskDriver(
                configuration=self._system_configuration.assets.vr_task,
                task_template=task_template,
                expected_scene_name=self._experiment_configuration.unity_scene_name,
            )
        self._mesoscope_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # Initializes but does not start the assets used by all runtimes. These assets need to be started in a
        # specific order, which is handled by the start() method.
        # noinspection PyProtectedMember
        self._ui: RuntimeControlUI = RuntimeControlUI(
            valve_tracker=self._microcontrollers.valve._valve_tracker,  # noqa: SLF001
            gas_puff_tracker=self._microcontrollers.gas_puff_valve._puff_tracker,  # noqa: SLF001
        )
        self._visualizer: BehaviorVisualizer = BehaviorVisualizer()

    def __repr__(self) -> str:
        """Returns a string representation of the _MesoscopeVRSystem instance."""
        return (
            f"_MesoscopeVRSystem(session_type={self._session_data.session_type}, started={self._started}, "
            f"paused={self._paused})"
        )

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

        # Starts the data logger.
        self._logger.start()

        # Generates and logs the onset timestamp for the Mesoscope-VR system.
        onset: NDArray[np.uint8] = get_timestamp(output_format=TimestampFormats.BYTES)  # type: ignore[assignment]
        # Immediately resets the timer to align it with the onset timestamp.
        self._timestamp_timer.reset()
        self._logger.input_queue.put(
            LogPackage(source_id=self._source_id, acquisition_time=np.uint64(0), serialized_data=onset)
        )  # Logs the onset timestamp

        message = "DataLogger: Started."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Starts all microcontroller interfaces.
        self._microcontrollers.start()

        # Sets the runtime into the Idle state before instructing the user to finalize runtime preparations.
        self.idle()

        # Generates a snapshot of the runtime hardware configuration. In turn, this data is used to parse the .npz log
        # files during processing.
        self._generate_hardware_state_snapshot()

        # If the session uses Virtual Reality, applies the Unity scale to the encoder using the value loaded with
        # the task template, opens the MQTT connection, and runs the interactive Unity setup sequence.
        if self._vr_task is not None and self._task_template is not None and self._experiment_configuration is not None:
            self._microcontrollers.wheel_encoder.set_unity_scale(
                cm_per_unity_unit=self._task_template.vr_environment.cm_per_unity_unit
            )
            self._vr_task.connect()
            # Enables the VR screens for the duration of the interactive setup sequence so the user can verify the
            # display, then blackens them once setup completes.
            self._microcontrollers.screens.set_state(state=True)
            self._vr_task.setup()
            self._microcontrollers.screens.set_state(state=False)
            self._trial_state.trial_structures = self._experiment_configuration.trial_structures
            self._refresh_trial_state_from_vr_decomposition()
            # Resets the encoder and runtime distance trackers so that subsequent encoder messages produce position
            # deltas relative to the start of the experiment.
            self._microcontrollers.wheel_encoder.reset_distance_tracker()
            self._distance = np.float64(0.0)
            self._trial_state.completed = 0

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
        if visualizer_mode == VisualizerMode.EXPERIMENT and self._experiment_configuration is not None:
            trial_structures = self._experiment_configuration.trial_structures.values()
            has_reinforcing_trials = any(isinstance(trial, WaterRewardTrial) for trial in trial_structures)
            has_aversive_trials = any(isinstance(trial, GasPuffTrial) for trial in trial_structures)

        # Initializes the runtime control GUI with the appropriate mode and trial type configuration.
        self._ui.start(
            mode=visualizer_mode,
            has_reinforcing_trials=has_reinforcing_trials,
            has_aversive_trials=has_aversive_trials,
        )

        # Synchronizes the Unity game engine's state with the initial state of the runtime control's UI before
        # entering the checkpoint loop.
        if self._vr_task is not None:
            self._vr_task.set_reinforcing_guidance(enabled=self._ui.enable_reinforcing_guidance)
            self._log_reinforcing_guidance_change(enabled=self._ui.enable_reinforcing_guidance)
            self._vr_task.set_aversive_guidance(enabled=self._ui.enable_aversive_guidance)
            self._log_aversive_guidance_change(enabled=self._ui.enable_aversive_guidance)

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

        # Starts saving frames from all cameras.
        self._cameras.save_face_camera_frames()
        self._cameras.save_body_camera_frames()

        # Starts mesoscope frame acquisition if the runtime is a mesoscope experiment.
        if self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            # Enables mesoscope frame monitoring.
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

        # Resets the _started tracker before attempting the shutdown sequence.
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
        if self._vr_task is not None:
            self._vr_task.disconnect()

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

        # Stops all microcontroller interfaces.
        self._microcontrollers.stop()

        # Stops the data logger instance.
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
            "Do you want to preprocess or purge the acquired session's data? CRITICAL! Only enter 'purge session' "
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

    def change_runtime_state(self, new_state: int) -> None:
        """Updates and logs the new acquired session's runtime state (stage).

        Args:
            new_state: The unique code for the new session's runtime state.
        """
        # Ensures that the _runtime_state attribute is set to a non-zero value after runtime initialization. This is
        # used to restore the runtime back to the pre-pause state if the runtime enters the paused state (idle), but the
        # user then chooses to resume the runtime.
        if new_state != MesoscopeVRStates.IDLE:
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
        self.change_runtime_state(new_state=MesoscopeVRStates.IDLE)

        # Blackens the VR screens.
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake.
        self._microcontrollers.brake.set_state(state=True)

        # Disables all sensor monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)
        self._microcontrollers.torque.set_monitoring_state(state=False)
        self._microcontrollers.lick.set_monitoring_state(state=False)

        self._change_system_state(new_state=MesoscopeVRStates.IDLE)

    def rest(self) -> None:
        """Switches the Mesoscope-VR system to the rest state.

        Notes:
            In the rest state, the brake is engaged and the screens are turned off. The encoder sensor is
            disabled, the torque sensor is enabled, and the lick sensor is enabled.
        """
        # Enables lick monitoring.
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Blackens the VR screens.
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake.
        self._microcontrollers.brake.set_state(state=True)

        # Suspends encoder monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)

        # Enables torque monitoring.
        self._microcontrollers.torque.set_monitoring_state(state=True)

        self._change_system_state(new_state=MesoscopeVRStates.REST)

    def run(self) -> None:
        """Switches the Mesoscope-VR system to the run state.

        Notes:
            In the run state, the brake is disengaged and the screens are turned on. The encoder sensor is
            enabled, the torque sensor is disabled, and the lick sensor is enabled.
        """
        # Enables lick monitoring.
        self._microcontrollers.lick.set_monitoring_state(state=True)

        # Initializes encoder monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=True)

        # Disables torque monitoring.
        self._microcontrollers.torque.set_monitoring_state(state=False)

        # Activates VR screens.
        self._microcontrollers.screens.set_state(state=True)

        # Disengages the brake.
        self._microcontrollers.brake.set_state(state=False)

        self._change_system_state(new_state=MesoscopeVRStates.RUN)

    def lick_train(self) -> None:
        """Switches the Mesoscope-VR system to the lick training state.

        Notes:
            In this state, the brake is engaged and the screens are turned off. The encoder sensor is
            disabled, and the torque sensor is enabled.

            Calling this method automatically switches the runtime state to 255 (active training).
        """
        self.change_runtime_state(new_state=_GUIDED_RUNTIME_STATE_CODE)

        # Blackens the VR screens.
        self._microcontrollers.screens.set_state(state=False)

        # Engages the brake.
        self._microcontrollers.brake.set_state(state=True)

        # Disables encoder monitoring.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=False)

        # Initiates torque monitoring.
        self._microcontrollers.torque.set_monitoring_state(state=True)

        # Initiates lick monitoring.
        self._microcontrollers.lick.set_monitoring_state(state=True)

        self._change_system_state(new_state=MesoscopeVRStates.LICK_TRAINING)

    def run_train(self) -> None:
        """Switches the Mesoscope-VR system to the run training state.

        Notes:
            In this state, the brake is disengaged and the screens are turned off. The encoder sensor is
            enabled, and the torque sensor is disabled.

            Calling this method automatically switches the runtime state to 255 (active training).
        """
        self.change_runtime_state(new_state=_GUIDED_RUNTIME_STATE_CODE)

        # Blackens the VR screens.
        self._microcontrollers.screens.set_state(state=False)

        # Disengages the brake.
        self._microcontrollers.brake.set_state(state=False)

        # Ensures that encoder monitoring is enabled.
        self._microcontrollers.wheel_encoder.set_monitoring_state(state=True)

        # Ensures torque monitoring is disabled.
        self._microcontrollers.torque.set_monitoring_state(state=False)

        # Initiates lick monitoring.
        self._microcontrollers.lick.set_monitoring_state(state=True)

        self._change_system_state(new_state=MesoscopeVRStates.RUN_TRAINING)

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
            # Converts time from milliseconds to seconds.
            self.descriptor.final_run_duration_threshold_s = round(float(duration_threshold) / 1000, 2)

        self._visualizer.update_run_training_thresholds(
            speed_threshold=speed_threshold, duration_threshold=duration_threshold
        )

    def publish_runtime_thresholds(self, speed_threshold: np.float64, duration_threshold: np.float64) -> None:
        """Publishes the runtime-driven running speed and duration thresholds to the runtime control GUI.

        Unlike update_visualizer_thresholds, the values passed to this method exclude the user-defined GUI modifier.
        The GUI uses them to display the current effective thresholds and to convert user-requested absolute threshold
        values into the modifier offset consumed by the run training loop.

        Args:
            speed_threshold: The runtime-driven running speed threshold, in centimeters per second, before the user
                modifier is applied.
            duration_threshold: The runtime-driven running epoch duration threshold, in milliseconds, before the user
                modifier is applied.
        """
        self._ui.set_runtime_thresholds(
            speed_threshold_cm_s=float(speed_threshold), duration_threshold_ms=float(duration_threshold)
        )

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

        # Otherwise, simulates water reward by sounding the buzzer without delivering any water.
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

            # Continuously updates the visualizer.
            self._visualizer.update()

            # Synchronizes the runtime state with the state of the user-facing GUI.
            self._ui_cycle()

            # If the GUI was used to terminate the runtime, aborts the cycle early.
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

    def _generate_hardware_state_snapshot(self) -> None:
        """Resolves and caches the snapshot of the system's hardware configuration parameters to the acquired session's
        raw_data directory as a hardware_state.yaml file.
        """
        if (
            self._session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT
            and self._experiment_configuration is not None
        ):
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
                    isinstance(trial, GasPuffTrial)
                    for trial in self._experiment_configuration.trial_structures.values()
                ),
                system_state_codes=MesoscopeVRStates.to_dict(),
            )
        # Note, lick and run training runtimes only use a subset of all hardware modules.
        elif self._session_data.session_type == SessionTypes.LICK_TRAINING:
            hardware_state = MesoscopeHardwareState(
                torque_per_adc_unit=float(self._microcontrollers.torque.torque_per_adc_unit),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                delivered_gas_puffs=False,
                system_state_codes=MesoscopeVRStates.to_dict(),
            )
        elif self._session_data.session_type == SessionTypes.RUN_TRAINING:
            hardware_state = MesoscopeHardwareState(
                cm_per_pulse=float(self._microcontrollers.wheel_encoder.cm_per_pulse),
                lick_threshold=int(self._microcontrollers.lick.lick_threshold),
                valve_scale_coefficient=float(self._microcontrollers.valve.scale_coefficient),
                valve_nonlinearity_exponent=float(self._microcontrollers.valve.nonlinearity_exponent),
                delivered_gas_puffs=False,
                system_state_codes=MesoscopeVRStates.to_dict(),
            )
        else:
            # It should be impossible to satisfy this error clause, but is kept for safety reasons.
            message = (
                f"Unsupported session type {self._session_data.session_type} encountered when generating "
                f"the snapshot of the Mesoscope-VR system's hardware configuration."
            )
            console.error(message=message, error=ValueError)

        # Caches the resolved hardware state to disk.
        hardware_state.to_yaml(file_path=self._session_data.raw_data.hardware_state_path)
        message = "Mesoscope-VR hardware configuration snapshot: Generated."
        console.echo(message=message, level=LogLevel.SUCCESS)

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

            if self._vr_task is not None:
                if self._ui.enable_reinforcing_guidance != self._vr_task.state.reinforcing_guidance_enabled:
                    self._vr_task.set_reinforcing_guidance(enabled=self._ui.enable_reinforcing_guidance)
                    self._log_reinforcing_guidance_change(enabled=self._ui.enable_reinforcing_guidance)
                if self._ui.enable_aversive_guidance != self._vr_task.state.aversive_guidance_enabled:
                    self._vr_task.set_aversive_guidance(enabled=self._ui.enable_aversive_guidance)
                    self._log_aversive_guidance_change(enabled=self._ui.enable_aversive_guidance)

            if self._ui.exit_signal:
                self._terminate_runtime()
                if self._terminated:
                    break

        self._microcontrollers.valve.set_state(state=False)
        self._paused_water_volume += self._microcontrollers.valve.delivered_volume
        self._unconsumed_reward_count = 0

        # Signals the UI that setup is complete - this permanently disables valve open/close buttons.
        self._ui.set_setup_complete()

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

            # Waits for the mesoscope to start acquiring the expected number of frames.
            _response_delay_timer.reset()
            while _response_delay_timer.elapsed < _MESOSCOPE_START_TIMEOUT_MS:
                # Adds delay to prevent CPU spinning.
                _response_delay_timer.delay(delay=10, block=False)

                if self._microcontrollers.mesoscope_frame.pulse_count >= _EXPECTED_FRAME_PULSES:
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

    def _clear_mesoscope_markers(self) -> None:
        """Clears all mesoscope acquisition marker files from the ScanImagePC's shared mesoscope data directory.

        This utility method removes both the kinase (start) and phosphatase (stop) marker files, ensuring
        a clean directory state before sending new acquisition commands to the mesoscope.
        """
        self._mesoscope_data.scanimagepc_data.kinase_path.unlink(missing_ok=True)
        self._mesoscope_data.scanimagepc_data.phosphatase_path.unlink(missing_ok=True)

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

    def _change_system_state(self, new_state: int) -> None:
        """Updates and logs the new Mesoscope-VR system state.

        Args:
            new_state: The unique code for the newly activated Mesoscope-VR system state.
        """
        # Ensures that the _system_state attribute is set to a non-zero value after runtime initialization. This is
        # used to restore the runtime back to the pre-pause state if the runtime enters the paused state (idle), but the
        # user then chooses to resume the runtime.
        if new_state != MesoscopeVRStates.IDLE:
            self._system_state = new_state

        # Logs the system state update. Uses header-code 1 to indicate that the logged value is the system state-code.
        log_package = LogPackage(
            source_id=self._source_id,
            acquisition_time=np.uint64(self._timestamp_timer.elapsed),
            serialized_data=np.array([_MesoscopeVRLogMessageCodes.SYSTEM_STATE, new_state], dtype=np.uint8),
        )
        self._logger.input_queue.put(log_package)

    def _log_cue_sequence(self, cue_sequence: NDArray[np.uint8]) -> None:
        """Logs the Virtual Reality wall cue sequence most recently received from Unity."""
        self._logger.input_queue.put(
            LogPackage(
                source_id=self._source_id,
                acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                serialized_data=cue_sequence,
            )
        )

    def _log_reinforcing_guidance_change(self, *, enabled: bool) -> None:
        """Logs the change of the reinforcing trial guidance mode."""
        self._logger.input_queue.put(
            LogPackage(
                source_id=self._source_id,
                acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                serialized_data=np.array(
                    [_MesoscopeVRLogMessageCodes.REINFORCING_GUIDANCE_STATE, enabled], dtype=np.uint8
                ),
            )
        )

    def _log_aversive_guidance_change(self, *, enabled: bool) -> None:
        """Logs the change of the aversive trial guidance mode."""
        self._logger.input_queue.put(
            LogPackage(
                source_id=self._source_id,
                acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                serialized_data=np.array(
                    [_MesoscopeVRLogMessageCodes.AVERSIVE_GUIDANCE_STATE, enabled], dtype=np.uint8
                ),
            )
        )

    def _refresh_trial_state_from_vr_decomposition(self) -> None:
        """Logs the active Unity cue sequence and refreshes the cue-sequence-derived trial-state arrays.

        Notes:
            Bundles the work that must follow every cue-sequence retrieval from Unity (cold-start and resume after
            Unity restart). Writes the cue-sequence log packet, then copies the VR driver's freshly-decomposed
            per-trial distances and joins the new trial-name sequence with mesoscope-vr-specific reward and puff
            parameters.

            Does not touch the session-level trial counters (reinforcing_failed_trials, aversive_failed_trials,
            reinforcing_guided_trials, aversive_guided_trials) or the running trial position (_trial_state.completed).
            Callers handle sequence-local resets (encoder distance tracker, _distance, _trial_state.completed)
            independently around this refresh.
        """
        assert self._vr_task is not None  # noqa: S101  # guarded by caller
        self._log_cue_sequence(cue_sequence=self._vr_task.state.cue_sequence)
        self._trial_state.distances = self._vr_task.cue_sequence_distances
        self._trial_state.reinforcing_rewards, self._trial_state.aversive_puff_durations = (
            self._build_trial_parameter_arrays(trial_names=self._vr_task.trial_names)
        )

    def _build_trial_parameter_arrays(
        self, *, trial_names: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, int], ...], tuple[int, ...]]:
        """Joins the VR-decomposed trial sequence with mesoscope-vr-specific reward and puff parameters.

        Notes:
            The VRTaskDriver returns the ordered sequence of trial names produced by decomposing the active Unity
            cue sequence. This method looks each name up in the experiment configuration's trial-structures mapping
            and builds two parallel tuples: per-trial reward parameters (zero placeholders on aversive trials) and
            per-trial puff durations (zero placeholders on reinforcing trials). It also validates that every trial
            name produced by the decomposer has a matching entry in the experiment configuration.

        Args:
            trial_names: The ordered sequence of trial names produced by the VRTaskDriver's cue-sequence decomposition.

        Returns:
            A tuple of two parallel tuples: per-trial reinforcing reward parameters (volume, tone_duration) and
            per-trial aversive puff durations in milliseconds.

        Raises:
            ValueError: If a trial name produced by the decomposer has no matching entry in the experiment
                configuration's trial structures.
        """
        assert self._experiment_configuration is not None  # noqa: S101  # guarded by caller
        trial_structures = self._experiment_configuration.trial_structures

        reinforcing_rewards: list[tuple[float, int]] = []
        aversive_puff_durations: list[int] = []
        for trial_name in trial_names:
            if trial_name not in trial_structures:
                message = (
                    f"Unable to build the trial parameter arrays for the Mesoscope-VR system. The VR cue sequence "
                    f"decomposer produced trial '{trial_name}', but the experiment configuration has no matching "
                    f"entry. Trial names must align between the VR TaskTemplate and the experiment configuration."
                )
                console.error(message=message, error=ValueError)

            trial = trial_structures[trial_name]
            if isinstance(trial, WaterRewardTrial):
                reinforcing_rewards.append((float(trial.reward_size_ul), int(trial.reward_tone_duration_ms)))
                aversive_puff_durations.append(0)
            else:
                reinforcing_rewards.append((0.0, 0))
                aversive_puff_durations.append(int(trial.puff_duration_ms))

        return tuple(reinforcing_rewards), tuple(aversive_puff_durations)

    def _deliver_reward(self, reward_size: float = 5.0, tone_duration: int = 300) -> None:
        """Uses the solenoid valve to deliver the requested volume of water in microliters.

        Args:
            reward_size: The volume of water to deliver, in microliters.
            tone_duration: The time, in milliseconds, for which to sound the auditory tone while delivering the reward.
        """
        self._unconsumed_reward_count += 1
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
            self._visualizer.update_running_speed(running_speed=running_speed)

        # Handles Unity-based virtual reality task execution for experiment sessions.
        if self._vr_task is not None:
            # Forwards the latest animal position to Unity. The driver internally tracks the previous position and
            # only sends an MQTT message when the position has changed.
            self._vr_task.push_position(absolute_position=current_position)

            # Checks if the animal has completed the current trial.
            if self._trial_state.trial_completed(traveled_distance=traveled_distance):
                # Captures the trial outcome before advance_trial() resets the flags.
                is_aversive = self._trial_state.is_current_trial_aversive()
                if is_aversive:
                    succeeded = self._trial_state.aversive_succeeded
                    was_guided = self._trial_state.aversive_guided_trials > 0
                else:
                    succeeded = self._trial_state.reinforcing_rewarded
                    was_guided = self._trial_state.reinforcing_guided_trials > 0

                failed_count = self._trial_state.advance_trial()

                # Reports the trial outcome to the visualizer.
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

        # Handles incoming lick data.
        lick_count = self._microcontrollers.lick.lick_count
        if lick_count > self._lick_count:
            self._lick_count = lick_count
            self._unconsumed_reward_count = 0
            self._visualizer.add_lick_event()

            if self._vr_task is not None:
                self._vr_task.push_lick_event()

        # Handles water delivery tracking.
        dispensed_water = self._microcontrollers.valve.delivered_volume - (
            self._paused_water_volume + self._delivered_water_volume
        )
        if dispensed_water > 0:
            if self._paused:
                self._paused_water_volume += dispensed_water
            else:
                self._delivered_water_volume += dispensed_water

    def _unity_cycle(self) -> None:
        """Dispatches the next Unity event from the Virtual Reality task driver to the Mesoscope-VR hardware.

        Notes:
            Each call consumes at most one Unity event. The VRTaskDriver returns a typed event; this method maps the
            event back into mesoscope-specific hardware actions and runtime state updates.
        """
        if self._vr_task is None:
            return

        event = self._vr_task.cycle()

        if event.kind == VRTaskEventKind.STIMULUS_TRIGGERED:
            if self._trial_state.is_current_trial_aversive():
                # Aversive trial: delivers the gas puff.
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
                # Reinforcing trial: delivers the water reward.
                reward_size, tone_duration = self._trial_state.get_current_reward()
                self.resolve_reward(reward_size=reward_size, tone_duration=tone_duration)

                # Decrements the guided trial counter for reinforcing trials.
                if self._trial_state.reinforcing_guided_trials > 0:
                    self._trial_state.reinforcing_guided_trials -= 1
                    if self._trial_state.reinforcing_guided_trials == 0:
                        self._ui.set_reinforcing_guidance_state(enabled=False)

                # Marks the reinforcing trial as rewarded.
                self._trial_state.reinforcing_rewarded = True

        elif event.kind == VRTaskEventKind.TRIGGER_DELAY_REQUESTED:
            if event.delay_ms > 0:
                self._microcontrollers.brake.send_pulse(duration_ms=event.delay_ms)

        elif event.kind == VRTaskEventKind.UNITY_TERMINATED:
            self._pause_runtime()
            message = "Emergency pause: Engaged. Reason: Unity sent a runtime termination message."
            console.echo(message=message, level=LogLevel.ERROR)

            # Logs the distance snapshot.
            traveled_distance = float(self._microcontrollers.wheel_encoder.traveled_distance)
            distance_bytes = convert_scalar_to_bytes(value=traveled_distance, dtype=np.dtype("<f8"))

            log_package = LogPackage(
                source_id=self._source_id,
                acquisition_time=np.uint64(self._timestamp_timer.elapsed),
                serialized_data=np.concatenate(
                    [np.array([_MesoscopeVRLogMessageCodes.DISTANCE_SNAPSHOT], dtype=np.uint8), distance_bytes]
                ),
            )
            self._logger.input_queue.put(log_package)

            message = (
                "Address the issue that prevents the Unity game engine from running, then resume the runtime. The "
                "system automatically re-arms the Unity scene through the editor bridge on resume. Alternatively, "
                "terminate the runtime to attempt graceful shutdown."
            )
            console.echo(message=message, level=LogLevel.INFO)

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

        if self._vr_task is not None:
            # Synchronizes guidance state with UI.
            if self._ui.enable_reinforcing_guidance != self._vr_task.state.reinforcing_guidance_enabled:
                self._vr_task.set_reinforcing_guidance(enabled=self._ui.enable_reinforcing_guidance)
                self._log_reinforcing_guidance_change(enabled=self._ui.enable_reinforcing_guidance)
            if self._ui.enable_aversive_guidance != self._vr_task.state.aversive_guidance_enabled:
                self._vr_task.set_aversive_guidance(enabled=self._ui.enable_aversive_guidance)
                self._log_aversive_guidance_change(enabled=self._ui.enable_aversive_guidance)

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

        # Records pause onset time.
        self._pause_start_time = self._timestamp_timer.elapsed

        # Switches the Mesoscope-VR system into the idle state.
        self.idle()

        # Notifies the user that the runtime has been paused.
        message = "Mesoscope-VR runtime: Paused."
        console.echo(message=message, level=LogLevel.WARNING)

        # Sets the paused flag.
        self._paused = True

    def _resume_runtime(self) -> None:
        """Resumes the session's data acquisition."""
        message = "Mesoscope-VR runtime: Resumed."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # If Unity or mesoscope terminated during runtime, attempts to re-initialize Unity and restart the Mesoscope.
        if self._vr_task is not None and self._vr_task.state.terminated:
            # When the Unity game cycles, it resets the sequence of VR wall cues. This re-queries the new wall cue
            # sequence to enable accurate tracking of the animal's position in VR after reset, and resets the
            # termination flag once the new sequence has been received.
            self._vr_task.resume_after_unity_restart()
            self._refresh_trial_state_from_vr_decomposition()
            # Resets the runtime distance trackers to align with the fresh Unity position origin.
            self._microcontrollers.wheel_encoder.reset_distance_tracker()
            self._distance = np.float64(0.0)
            self._trial_state.completed = 0

        if self._mesoscope_terminated:
            # Restarting the Mesoscope is slightly different from starting it, as the user needs to call the
            # setupAcquisition() function with a special argument. Instructs the user to call the function and then
            # enters the Mesoscope start sequence.
            message = (
                "If necessary call the setupAcquisition(hSI, hSICtl, recovery=true) command in the MATLAB command line "
                "interface before proceeding to resume an interrupted acquisition."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            input("Enter anything to continue: ")

            self._start_mesoscope()

            # Resets the termination tracker if Mesoscope acquisition restarts successfully.
            self._mesoscope_terminated = False

        # Updates the 'paused_time' value to reflect the time spent inside the 'paused' state. Most runtimes use this
        # public attribute to adjust the execution time of certain runtime stages or the runtime altogether.
        self.paused_time += round(
            convert_time(time=self._timestamp_timer.elapsed - self._pause_start_time, from_units="us", to_units="s")
        )

        # Restores the runtime state back to the value active before the pause.
        self.change_runtime_state(new_state=self._runtime_state)

        # Restores the system state to pre-pause condition.
        if self._system_state == MesoscopeVRStates.IDLE:
            # This is a rare case where the pause was triggered before a valid non-idle state was activated by the
            # runtime logic function. While rare, it is not technically impossible, so it is supported here.
            self.idle()
        elif self._system_state == MesoscopeVRStates.REST:
            self.rest()
        elif self._system_state == MesoscopeVRStates.RUN:
            self.run()
        elif self._system_state == MesoscopeVRStates.LICK_TRAINING:
            self.lick_train()
        elif self._system_state == MesoscopeVRStates.RUN_TRAINING:
            self.run_train()

        # Resets the paused flag.
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

            # Returns without terminating the runtime.
            if answer == "n":
                return

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
        paused_water_volume = float(self._paused_water_volume)
        delivered_water = float(self._microcontrollers.valve.delivered_volume) - paused_water_volume
        self.descriptor.dispensed_water_volume_ml = float(
            round(delivered_water / _MICROLITERS_PER_MILLILITER, ndigits=3)
        )

        # Same as above, but tracks the total volume of water dispensed during pauses. While the animal might
        # have consumed some of that water, it is equally plausible that all water was wasted or not dispensed at all.
        self.descriptor.pause_dispensed_water_volume_ml = float(
            round(paused_water_volume / _MICROLITERS_PER_MILLILITER, ndigits=3)
        )

        # If the runtime reaches this point, the session is likely complete.
        self.descriptor.incomplete = False

        # Precalculates the volume of water that the experimenter needs to deliver to the animal if the combined
        # volume delivered during runtime and paused state is less than 1 ml. This is used to pre-fill the
        # experimenter-delivered volume field as a convenience feature for experimenters.
        total_delivered_volume = (
            self.descriptor.dispensed_water_volume_ml + self.descriptor.pause_dispensed_water_volume_ml
        )
        if total_delivered_volume < 1:
            self.descriptor.experimenter_given_water_volume_ml = float(round(1 - total_delivered_volume, ndigits=3))

        # Ensures that the user updates the descriptor file. The descriptor union accepted by the runtime is a strict
        # subset of the union accepted by _verify_descriptor_update, so the assignment is type-safe.
        # noinspection PyTypeChecker
        _verify_descriptor_update(
            descriptor=self.descriptor, session_data=self._session_data, mesoscope_data=self._mesoscope_data
        )
