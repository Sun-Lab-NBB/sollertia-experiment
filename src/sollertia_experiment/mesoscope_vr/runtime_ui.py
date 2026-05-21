"""Provides the graphical user interface used by the Mesoscope-VR data acquisition system to facilitate data
acquisition runtimes by allowing direct control over a subset of the system's runtime parameters and hardware.
"""

import sys
from enum import IntEnum
from functools import partial
import contextlib
from multiprocessing import Process

import numpy as np
from PySide6.QtGui import QFont, QCloseEvent
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QApplication,
    QDoubleSpinBox,
)
from ataraxis_base_utilities import console
from ataraxis_data_structures import SharedMemoryArray

from .visualizers import VisualizerMode


class _DataArrayIndex(IntEnum):
    """Defines the shared memory array indices for each runtime parameter and hardware component addressable from the
    user-facing GUI.
    """

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


class RuntimeControlUI:
    """Provides the Graphical User Interface (GUI) that allows modifying certain Mesoscope-VR runtime parameters in real
    time.

    Notes:
        The UI runs in a parallel process and requires a single CPU core to support its runtime.

        Initializing the class does not start the UI process. Call the start() method before calling any other instance
        methods to start the UI process.

    Args:
        valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes.
        gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            count to other processes.

    Attributes:
        _data_array: The SharedMemoryArray instance used to bidirectionally transfer the data between the UI process
            and other runtime processes.
        _valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes.
        _gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            count to other processes.
        _mode: The VisualizerMode that determines which UI elements are enabled.
        _ui_process: The Process instance running the GUI cycle.
        _started: Tracks whether the UI process is running.
    """

    def __init__(self, valve_tracker: SharedMemoryArray, gas_puff_tracker: SharedMemoryArray) -> None:
        # Defines the prototype array for the SharedMemoryArray initialization and sets the array elements to the
        # desired default state
        prototype = np.zeros(shape=16, dtype=np.int32)
        prototype[_DataArrayIndex.PAUSE_STATE] = 1  # Ensures all runtimes start in a paused state
        prototype[_DataArrayIndex.REINFORCING_GUIDANCE_ENABLED] = 0  # Initially disables reinforcing guidance
        prototype[_DataArrayIndex.AVERSIVE_GUIDANCE_ENABLED] = 0  # Initially disables aversive guidance
        prototype[_DataArrayIndex.REWARD_VOLUME] = 5  # Preconfigures reward delivery to use 5 uL rewards
        prototype[_DataArrayIndex.GAS_VALVE_PUFF_DURATION] = 100  # Default gas puff duration: 100 ms

        # Initializes the SharedMemoryArray instance
        self._data_array = SharedMemoryArray.create_array(
            name="runtime_control_ui", prototype=prototype, exists_ok=True
        )

        # Caches tracker references to class attributes
        self._valve_tracker = valve_tracker
        self._gas_puff_tracker = gas_puff_tracker

        # Initializes the mode to EXPERIMENT by default. The mode is set when start() is called.
        self._mode: VisualizerMode = VisualizerMode.EXPERIMENT

        # Trial type flags, set when start() is called based on experiment configuration.
        self._has_reinforcing_trials: bool = True
        self._has_aversive_trials: bool = True

        # Defines but does not automatically start the UI process. The process target is set in start() to pass
        # the mode.
        self._ui_process: Process | None = None
        self._started = False

    def __del__(self) -> None:
        """Terminates the UI process and releases the instance's shared memory buffers when garbage-collected."""
        self.shutdown()
        # Note: Does not disconnect or destroy the trackers as they're owned by their respective interfaces

    def start(
        self,
        mode: VisualizerMode | int = VisualizerMode.EXPERIMENT,
        *,
        has_reinforcing_trials: bool = True,
        has_aversive_trials: bool = True,
    ) -> None:
        """Starts the remote UI process.

        Args:
            mode: The VisualizerMode that determines which UI elements are enabled. Speed and duration threshold
                controls are only enabled for RUN_TRAINING mode. Must be a valid VisualizerMode enumeration member.
            has_reinforcing_trials: Determines whether the experiment includes reinforcing (water reward) trials.
                When True, the UI shows the reinforcing guidance toggle button.
            has_aversive_trials: Determines whether the experiment includes aversive (gas puff) trials. When True,
                the UI shows the aversive guidance toggle button and the gas puff valve control group.
        """
        # If the instance is already started, aborts early
        if self._started:
            return

        # Stores the mode and trial type flags.
        self._mode = VisualizerMode(mode)
        self._has_reinforcing_trials = has_reinforcing_trials
        self._has_aversive_trials = has_aversive_trials

        # Creates the UI process with the mode and trial type flags as arguments. Uses partial to bind keyword
        # arguments, allowing the method signature to use keyword-only boolean parameters.
        target = partial(
            self._run_ui_process,
            mode=self._mode,
            has_reinforcing_trials=self._has_reinforcing_trials,
            has_aversive_trials=self._has_aversive_trials,
        )
        self._ui_process = Process(target=target, daemon=True)

        # Starts the remote UI process.
        self._ui_process.start()

        # Connects to the shared memory array from the central runtime process and configures it to destroy the
        # shared memory buffer in case of an emergency (error) shutdown.
        self._data_array.connect()
        self._data_array.enable_buffer_destruction()

        # Connects to trackers to monitor valve and gas puff states
        self._valve_tracker.connect()
        self._gas_puff_tracker.connect()

        # Marks the instance as started
        self._started = True

    def shutdown(self) -> None:
        """Shuts down the remote UI process and releases the instance's shared memory buffer."""
        # If the instance is already shut down, aborts early.
        if not self._started:
            return

        # Shuts down the remote UI process.
        if self._ui_process is not None and self._ui_process.is_alive():
            self._data_array[_DataArrayIndex.TERMINATION] = 1  # Sends the termination signal to the remote process
            self._ui_process.terminate()
            self._ui_process.join(timeout=2.0)

        # Terminates the shared memory array buffer.
        self._data_array.disconnect()
        self._data_array.destroy()

        # Note: Does not disconnect trackers here - they're owned by their respective interfaces and disconnecting
        # them would break access to delivered_volume when generating the session descriptor during shutdown.

        # Marks the instance as shut down
        self._started = False

    def _run_ui_process(
        self,
        mode: VisualizerMode,
        *,
        has_reinforcing_trials: bool,
        has_aversive_trials: bool,
    ) -> None:
        """Runs UI management cycle in a parallel process.

        Args:
            mode: The VisualizerMode that determines which UI elements are enabled.
            has_reinforcing_trials: Determines whether the experiment includes reinforcing (water reward) trials.
            has_aversive_trials: Determines whether the experiment includes aversive (gas puff) trials.
        """
        self._data_array.connect()
        self._valve_tracker.connect()
        self._gas_puff_tracker.connect()

        try:
            app = QApplication(sys.argv)
            app.setApplicationName("Mesoscope-VR Control Panel")
            app.setOrganizationName("Sollertia")
            app.setStyle("Fusion")

            window = _ControlUIWindow(
                self._data_array,
                self._valve_tracker,
                self._gas_puff_tracker,
                mode=mode,
                has_reinforcing_trials=has_reinforcing_trials,
                has_aversive_trials=has_aversive_trials,
            )
            window.show()

            app.exec()
        except Exception as e:
            message = (
                f"Unable to initialize the GUI application for the main runtime user interface. "
                f"Encountered the following error {e}."
            )
            console.error(message=message, error=RuntimeError)
        finally:
            self._data_array.disconnect()
            self._valve_tracker.disconnect()
            self._gas_puff_tracker.disconnect()

    def set_pause_state(self, *, paused: bool) -> None:
        """Configures the GUI to reflect the current data acquisition session's runtime state.

        Args:
            paused: Determines whether the session is paused or running.
        """
        self._data_array[_DataArrayIndex.PAUSE_STATE] = 1 if paused else 0

    def set_reinforcing_guidance_state(self, *, enabled: bool) -> None:
        """Configures the GUI to reflect the data acquisition session's reinforcing trial guidance state.

        Args:
            enabled: Determines whether the reinforcing guidance mode is currently enabled.
        """
        self._data_array[_DataArrayIndex.REINFORCING_GUIDANCE_ENABLED] = 1 if enabled else 0

    def set_aversive_guidance_state(self, *, enabled: bool) -> None:
        """Configures the GUI to reflect the data acquisition session's aversive trial guidance state.

        Args:
            enabled: Determines whether the aversive guidance mode is currently enabled.
        """
        self._data_array[_DataArrayIndex.AVERSIVE_GUIDANCE_ENABLED] = 1 if enabled else 0

    def set_setup_complete(self) -> None:
        """Signals the GUI that the initial setup phase is complete and the runtime has started.

        Notes:
            Once setup is complete, the valve open/close buttons are permanently disabled for the remainder of the
            runtime. This method should be called after the initial checkpoint loop exits.
        """
        self._data_array[_DataArrayIndex.SETUP_COMPLETE] = 1

    @property
    def exit_signal(self) -> bool:
        """Returns True if the user has requested the system to abort the data acquisition session's runtime."""
        exit_flag = bool(self._data_array[_DataArrayIndex.EXIT_SIGNAL])
        self._data_array[_DataArrayIndex.EXIT_SIGNAL] = 0
        return exit_flag

    @property
    def reward_signal(self) -> bool:
        """Returns True if the user has requested the system to deliver a water reward."""
        reward_flag = bool(self._data_array[_DataArrayIndex.REWARD_SIGNAL])
        self._data_array[_DataArrayIndex.REWARD_SIGNAL] = 0
        return reward_flag

    @property
    def speed_modifier(self) -> int:
        """Returns the current user-defined running speed threshold modifier."""
        return int(self._data_array[_DataArrayIndex.SPEED_MODIFIER])

    @property
    def duration_modifier(self) -> int:
        """Returns the current user-defined running epoch duration threshold modifier."""
        return int(self._data_array[_DataArrayIndex.DURATION_MODIFIER])

    @property
    def pause_runtime(self) -> bool:
        """Returns True if the user has requested the system to pause the data acquisition session's runtime."""
        return bool(self._data_array[_DataArrayIndex.PAUSE_STATE])

    @property
    def open_valve(self) -> bool:
        """Returns True if the user has requested the system to open the water delivery valve."""
        open_flag = bool(self._data_array[_DataArrayIndex.OPEN_VALVE])
        self._data_array[_DataArrayIndex.OPEN_VALVE] = 0
        return open_flag

    @property
    def close_valve(self) -> bool:
        """Returns True if the user has requested the system to close the water delivery valve."""
        close_flag = bool(self._data_array[_DataArrayIndex.CLOSE_VALVE])
        self._data_array[_DataArrayIndex.CLOSE_VALVE] = 0
        return close_flag

    @property
    def reward_volume(self) -> int:
        """Returns the current user-defined volume of water dispensed by the valve when delivering water rewards."""
        return int(self._data_array[_DataArrayIndex.REWARD_VOLUME])

    @property
    def enable_reinforcing_guidance(self) -> bool:
        """Returns True if the user has enabled the reinforcing trial guidance mode."""
        return bool(self._data_array[_DataArrayIndex.REINFORCING_GUIDANCE_ENABLED])

    @property
    def enable_aversive_guidance(self) -> bool:
        """Returns True if the user has enabled the aversive trial guidance mode."""
        return bool(self._data_array[_DataArrayIndex.AVERSIVE_GUIDANCE_ENABLED])

    @property
    def gas_valve_open_signal(self) -> bool:
        """Returns True if the user has requested to open the gas puff valve."""
        signal = bool(self._data_array[_DataArrayIndex.GAS_VALVE_OPEN])
        self._data_array[_DataArrayIndex.GAS_VALVE_OPEN] = 0
        return signal

    @property
    def gas_valve_close_signal(self) -> bool:
        """Returns True if the user has requested to close the gas puff valve."""
        signal = bool(self._data_array[_DataArrayIndex.GAS_VALVE_CLOSE])
        self._data_array[_DataArrayIndex.GAS_VALVE_CLOSE] = 0
        return signal

    @property
    def gas_valve_puff_signal(self) -> bool:
        """Returns True if the user has requested to deliver a gas puff."""
        signal = bool(self._data_array[_DataArrayIndex.GAS_VALVE_PUFF])
        self._data_array[_DataArrayIndex.GAS_VALVE_PUFF] = 0
        return signal

    @property
    def gas_valve_puff_duration(self) -> int:
        """Returns the current user-defined gas puff duration in milliseconds."""
        return int(self._data_array[_DataArrayIndex.GAS_VALVE_PUFF_DURATION])


class _ControlUIWindow(QMainWindow):
    """Generates, renders, and maintains the Mesoscope-VR acquisition system's runtime GUI application window.

    Attributes:
        _data_array: The SharedMemoryArray instance used to bidirectionally transfer the data between the UI process
            and other runtime processes.
        _valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes during runtime.
        _gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            data to other processes during runtime.
        _mode: The VisualizerMode that determines which UI elements are enabled.
        _has_reinforcing_trials: Determines whether the experiment includes reinforcing (water reward) trials.
        _has_aversive_trials: Determines whether the experiment includes aversive (gas puff) trials.
        _is_paused: Tracks whether the runtime is paused.
        _setup_complete: Tracks whether the initial setup phase is complete. Once True, valve open/close buttons
            are permanently disabled.
        _reinforcing_guidance_enabled: Tracks whether reinforcing trial guidance is enabled.
        _aversive_guidance_enabled: Tracks whether aversive trial guidance is enabled.
        _reward_in_progress: Tracks whether a reward delivery is in progress.
        _puff_in_progress: Tracks whether a gas puff delivery is in progress.
    """

    def __init__(
        self,
        data_array: SharedMemoryArray,
        valve_tracker: SharedMemoryArray,
        gas_puff_tracker: SharedMemoryArray,
        mode: VisualizerMode | int = VisualizerMode.EXPERIMENT,
        *,
        has_reinforcing_trials: bool = True,
        has_aversive_trials: bool = True,
    ) -> None:
        super().__init__()

        self._data_array: SharedMemoryArray = data_array
        self._valve_tracker: SharedMemoryArray = valve_tracker
        self._gas_puff_tracker: SharedMemoryArray = gas_puff_tracker
        self._mode: VisualizerMode = VisualizerMode(mode)
        self._has_reinforcing_trials: bool = has_reinforcing_trials
        self._has_aversive_trials: bool = has_aversive_trials

        self._is_paused: bool = True
        self._setup_complete: bool = False
        self._reinforcing_guidance_enabled: bool = False
        self._aversive_guidance_enabled: bool = False

        # Tracks whether a reward delivery is in progress.
        self._reward_in_progress: bool = False
        # Tracks whether a gas puff delivery is in progress.
        self._puff_in_progress: bool = False

        # Configures the window title
        self.setWindowTitle("Mesoscope-VR Control Panel")

        # Calculates window height based on visible elements.
        # Base height includes: runtime control (without guidance buttons) and valve control.
        base_height = 380
        if self._mode == VisualizerMode.RUN_TRAINING:
            base_height += 100  # Speed and duration threshold controls.
        elif self._mode == VisualizerMode.EXPERIMENT:
            if has_reinforcing_trials:
                base_height += 45  # Reinforcing guidance button.
            if has_aversive_trials:
                base_height += 45  # Aversive guidance button.
                base_height += 130  # Gas puff valve control group.
        self.setFixedSize(450, base_height)

        # Sets up the interactive UI
        self._setup_ui()
        self._setup_monitoring()

        # Applies Qt6-optimized styling and scaling parameters
        self._apply_qt6_styles()

    def _setup_ui(self) -> None:
        """Creates and arranges all UI elements."""
        # Initializes the main widget container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Generates the central bounding box (the bounding box around all UI elements)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Runtime Control Group
        runtime_control_group = QGroupBox("Runtime Control")
        runtime_control_layout = QVBoxLayout(runtime_control_group)
        runtime_control_layout.setSpacing(6)

        # Runtime termination (exit) button
        self.exit_btn = QPushButton("✖ Terminate Runtime")
        self.exit_btn.setToolTip("Gracefully ends the runtime and initiates the shutdown procedure.")
        # noinspection PyUnresolvedReferences
        self.exit_btn.clicked.connect(self._exit_runtime)
        self.exit_btn.setObjectName("exitButton")

        # Runtime Pause / Unpause (resume) button
        self.pause_btn = QPushButton("▶️ Resume Runtime")
        self.pause_btn.setToolTip("Pauses or resumes the runtime.")
        # noinspection PyUnresolvedReferences
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setObjectName("resumeButton")

        # Configures the main control buttons
        for button in [self.exit_btn, self.pause_btn]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(button)

        # Reinforcing Guidance button (only shown in EXPERIMENT mode with reinforcing trials).
        self.reinforcing_guidance_btn: QPushButton | None = None
        if self._mode == VisualizerMode.EXPERIMENT and self._has_reinforcing_trials:
            self.reinforcing_guidance_btn = QPushButton("🎯 Enable Reinforcing Guidance")
            self.reinforcing_guidance_btn.setToolTip("Toggles reinforcing trial guidance mode on or off.")
            # noinspection PyUnresolvedReferences
            self.reinforcing_guidance_btn.clicked.connect(self._toggle_reinforcing_guidance)
            self.reinforcing_guidance_btn.setObjectName("reinforcingGuidanceButton")
            self.reinforcing_guidance_btn.setMinimumHeight(35)
            self.reinforcing_guidance_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(self.reinforcing_guidance_btn)

        # Aversive Guidance button (only shown in EXPERIMENT mode with aversive trials).
        self.aversive_guidance_btn: QPushButton | None = None
        if self._mode == VisualizerMode.EXPERIMENT and self._has_aversive_trials:
            self.aversive_guidance_btn = QPushButton("🎯 Enable Aversive Guidance")
            self.aversive_guidance_btn.setToolTip("Toggles aversive trial guidance mode on or off.")
            # noinspection PyUnresolvedReferences
            self.aversive_guidance_btn.clicked.connect(self._toggle_aversive_guidance)
            self.aversive_guidance_btn.setObjectName("aversiveGuidanceButton")
            self.aversive_guidance_btn.setMinimumHeight(35)
            self.aversive_guidance_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(self.aversive_guidance_btn)

        # Adds runtime status tracker to the same box
        self.runtime_status_label = QLabel("Runtime Status: ⏸️ Paused")
        self.runtime_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        runtime_status_font = QFont()
        runtime_status_font.setPointSize(35)
        runtime_status_font.setBold(True)
        self.runtime_status_label.setFont(runtime_status_font)
        self.runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        runtime_control_layout.addWidget(self.runtime_status_label)

        # Adds the runtime control box to the UI widget
        main_layout.addWidget(runtime_control_group)

        # Reward Valve Control Group
        valve_group = QGroupBox("Reward Valve Control")
        valve_layout = QVBoxLayout(valve_group)
        valve_layout.setSpacing(6)

        # Arranges valve control buttons in a horizontal layout
        valve_buttons_layout = QHBoxLayout()

        # Valve open
        self.valve_open_btn = QPushButton("🔓 Open")
        self.valve_open_btn.setToolTip("Opens the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.valve_open_btn.clicked.connect(self._open_valve)
        self.valve_open_btn.setObjectName("valveOpenButton")

        # Valve close
        self.valve_close_btn = QPushButton("🔒 Close")
        self.valve_close_btn.setToolTip("Closes the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.valve_close_btn.clicked.connect(self._close_valve)
        self.valve_close_btn.setObjectName("valveCloseButton")

        # Reward button
        self.reward_btn = QPushButton("● Reward")
        self.reward_btn.setToolTip("Delivers 5 uL of water through the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self.reward_btn.clicked.connect(self._deliver_reward)
        self.reward_btn.setObjectName("rewardButton")

        # Configures the buttons to expand when the UI is resized, but use a fixed height of 35 points
        for button in [self.valve_open_btn, self.valve_close_btn, self.reward_btn]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            valve_buttons_layout.addWidget(button)

        valve_layout.addLayout(valve_buttons_layout)

        # Valve status and volume control section - horizontal layout
        valve_status_layout = QHBoxLayout()
        valve_status_layout.setSpacing(6)

        # Volume control on the left
        volume_label = QLabel("Reward volume:")
        volume_label.setObjectName("volumeLabel")

        self.volume_spinbox = QDoubleSpinBox()
        self.volume_spinbox.setRange(1, 20)  # Ranges from 1 to 20
        self.volume_spinbox.setValue(5)  # Default value
        self.volume_spinbox.setDecimals(0)  # Integer precision
        self.volume_spinbox.setSuffix(" μL")  # Adds units suffix
        self.volume_spinbox.setToolTip("Sets water reward volume. Accepts values between 1 and 20 μL.")
        self.volume_spinbox.setMinimumHeight(30)
        # noinspection PyUnresolvedReferences
        self.volume_spinbox.valueChanged.connect(self._update_reward_volume)

        # Adds volume controls to the left side
        valve_status_layout.addWidget(volume_label)
        valve_status_layout.addWidget(self.volume_spinbox)

        # Adds the valve status tracker on the right
        self.valve_status_label = QLabel("Valve: 🔒 Closed")
        self.valve_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        valve_status_font = QFont()
        valve_status_font.setPointSize(35)
        valve_status_font.setBold(True)
        self.valve_status_label.setFont(valve_status_font)
        self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
        valve_status_layout.addWidget(self.valve_status_label)

        # Add the horizontal status layout to the main valve layout
        valve_layout.addLayout(valve_status_layout)

        # Adds the valve control box to the UI widget
        main_layout.addWidget(valve_group)

        # Gas Puff Valve Control Group (only shown in EXPERIMENT mode with aversive trials).
        self.gas_valve_open_btn: QPushButton | None = None
        self.gas_valve_close_btn: QPushButton | None = None
        self.gas_puff_btn: QPushButton | None = None
        self.gas_duration_spinbox: QDoubleSpinBox | None = None
        self.gas_valve_status_label: QLabel | None = None

        if self._mode == VisualizerMode.EXPERIMENT and self._has_aversive_trials:
            gas_valve_group = QGroupBox("Gas Puff Valve Control")
            gas_valve_layout = QVBoxLayout(gas_valve_group)
            gas_valve_layout.setSpacing(6)

            # Arranges gas valve control buttons in a horizontal layout
            gas_valve_buttons_layout = QHBoxLayout()

            # Gas valve open
            self.gas_valve_open_btn = QPushButton("🔓 Open")
            self.gas_valve_open_btn.setToolTip("Opens the gas puff valve.")
            # noinspection PyUnresolvedReferences
            self.gas_valve_open_btn.clicked.connect(self._gas_valve_open)
            self.gas_valve_open_btn.setObjectName("gasValveOpenButton")

            # Gas valve close
            self.gas_valve_close_btn = QPushButton("🔒 Close")
            self.gas_valve_close_btn.setToolTip("Closes the gas puff valve.")
            # noinspection PyUnresolvedReferences
            self.gas_valve_close_btn.clicked.connect(self._gas_valve_close)
            self.gas_valve_close_btn.setObjectName("gasValveCloseButton")

            # Gas puff button
            self.gas_puff_btn = QPushButton("💨 Puff")
            self.gas_puff_btn.setToolTip("Delivers a gas puff with the specified duration.")
            # noinspection PyUnresolvedReferences
            self.gas_puff_btn.clicked.connect(self._gas_valve_puff)
            self.gas_puff_btn.setObjectName("gasPuffButton")

            # Configures the buttons to expand when the UI is resized, but use a fixed height of 35 points
            for button in [self.gas_valve_open_btn, self.gas_valve_close_btn, self.gas_puff_btn]:
                button.setMinimumHeight(35)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                gas_valve_buttons_layout.addWidget(button)

            gas_valve_layout.addLayout(gas_valve_buttons_layout)

            # Gas valve status and duration control section - horizontal layout
            gas_valve_status_layout = QHBoxLayout()
            gas_valve_status_layout.setSpacing(6)

            # Duration control on the left
            gas_duration_label = QLabel("Puff duration:")
            gas_duration_label.setObjectName("volumeLabel")

            self.gas_duration_spinbox = QDoubleSpinBox()
            self.gas_duration_spinbox.setRange(10, 350)
            self.gas_duration_spinbox.setValue(100)
            self.gas_duration_spinbox.setDecimals(0)
            self.gas_duration_spinbox.setSuffix(" ms")
            self.gas_duration_spinbox.setToolTip("Sets gas puff duration. Accepts values between 10 and 350 ms.")
            self.gas_duration_spinbox.setMinimumHeight(30)
            # noinspection PyUnresolvedReferences
            self.gas_duration_spinbox.valueChanged.connect(self._update_gas_puff_duration)

            # Adds duration controls to the left side
            gas_valve_status_layout.addWidget(gas_duration_label)
            gas_valve_status_layout.addWidget(self.gas_duration_spinbox)

            # Adds the gas valve status tracker on the right
            self.gas_valve_status_label = QLabel("Valve: 🔒 Closed")
            self.gas_valve_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            gas_valve_status_font = QFont()
            gas_valve_status_font.setPointSize(35)
            gas_valve_status_font.setBold(True)
            self.gas_valve_status_label.setFont(gas_valve_status_font)
            self.gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
            gas_valve_status_layout.addWidget(self.gas_valve_status_label)

            # Adds the horizontal status layout to the main gas valve layout
            gas_valve_layout.addLayout(gas_valve_status_layout)

            # Adds the gas valve control box to the UI widget
            main_layout.addWidget(gas_valve_group)

        # Adds Run Training controls in a horizontal layout (only shown in RUN_TRAINING mode).
        self._speed_group: QGroupBox | None = None
        self._duration_group: QGroupBox | None = None
        self.speed_spinbox: QDoubleSpinBox | None = None
        self.duration_spinbox: QDoubleSpinBox | None = None

        if self._mode == VisualizerMode.RUN_TRAINING:
            controls_layout = QHBoxLayout()
            controls_layout.setSpacing(6)

            # Running Speed Threshold Control Group
            self._speed_group = QGroupBox("Speed Threshold")
            speed_layout = QVBoxLayout(self._speed_group)

            # Speed Modifier
            speed_status_label = QLabel("Current Modifier:")
            speed_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            speed_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
            speed_layout.addWidget(speed_status_label)
            self.speed_spinbox = QDoubleSpinBox()
            self.speed_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 cm/s
            self.speed_spinbox.setValue(0)  # Default value
            self.speed_spinbox.setDecimals(0)  # Integer precision
            self.speed_spinbox.setToolTip("Sets the running speed threshold modifier value.")
            self.speed_spinbox.setMinimumHeight(30)
            # noinspection PyUnresolvedReferences
            self.speed_spinbox.valueChanged.connect(self._update_speed_modifier)
            speed_layout.addWidget(self.speed_spinbox)

            # Running Duration Threshold Control Group
            self._duration_group = QGroupBox("Duration Threshold")
            duration_layout = QVBoxLayout(self._duration_group)

            # Duration modifier
            duration_status_label = QLabel("Current Modifier:")
            duration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            duration_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
            duration_layout.addWidget(duration_status_label)
            self.duration_spinbox = QDoubleSpinBox()
            self.duration_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 s
            self.duration_spinbox.setValue(0)  # Default value
            self.duration_spinbox.setDecimals(0)  # Integer precision
            self.duration_spinbox.setToolTip("Sets the running duration threshold modifier value.")
            # noinspection PyUnresolvedReferences
            self.duration_spinbox.valueChanged.connect(self._update_duration_modifier)
            duration_layout.addWidget(self.duration_spinbox)

            # Adds speed and duration threshold modifiers to the main UI widget
            controls_layout.addWidget(self._speed_group)
            controls_layout.addWidget(self._duration_group)
            main_layout.addLayout(controls_layout)

    def _apply_qt6_styles(self) -> None:
        """Applies optimized styling to all UI elements managed by this instance."""
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: #ecf0f1;
                    }

                    QGroupBox {
                        font-weight: bold;
                        font-size: 14pt;
                        border: 2px solid #bdc3c7;
                        border-radius: 8px;
                        margin: 25px 6px 6px 6px;
                        padding-top: 10px;
                        background-color: #ffffff;
                    }

                    QGroupBox::title {
                        subcontrol-origin: margin;
                        subcontrol-position: top center;
                        left: 0px;
                        right: 0px;
                        padding: 0 8px 0 8px;
                        color: #2c3e50;
                        background-color: transparent;
                        border: none;
                    }

                    QPushButton {
                        background-color: #ffffff;
                        border: 2px solid #bdc3c7;
                        border-radius: 6px;
                        padding: 6px 8px;
                        font-size: 12pt;
                        font-weight: 500;
                        color: #2c3e50;
                        min-height: 20px;
                    }

                    QPushButton:hover {
                        background-color: #f8f9fa;
                        border-color: #3498db;
                        color: #2980b9;
                    }

                    QPushButton:pressed {
                        background-color: #e9ecef;
                        border-color: #2980b9;
                    }

                    QPushButton#exitButton {
                        background-color: #e74c3c;
                        color: white;
                        border-color: #c0392b;
                        font-weight: bold;
                    }

                    QPushButton#exitButton:hover {
                        background-color: #c0392b;
                        border-color: #a93226;
                    }

                    QPushButton#pauseButton {
                        background-color: #f39c12;
                        color: white;
                        border-color: #e67e22;
                        font-weight: bold;
                    }

                    QPushButton#pauseButton:hover {
                        background-color: #e67e22;
                        border-color: #d35400;
                    }

                    QPushButton#resumeButton {
                        background-color: #27ae60;
                        color: white;
                        border-color: #229954;
                        font-weight: bold;
                    }

                    QPushButton#resumeButton:hover {
                        background-color: #229954;
                        border-color: #1e8449;
                    }

                    QPushButton#valveOpenButton {
                        background-color: #27ae60;
                        color: white;
                        border-color: #229954;
                        font-weight: bold;
                    }

                    QPushButton#valveOpenButton:hover {
                        background-color: #229954;
                        border-color: #1e8449;
                    }

                    QPushButton#valveOpenButton:disabled {
                        background-color: #ecf0f1;
                        color: #95a5a6;
                        border-color: #bdc3c7;
                    }

                    QPushButton#valveCloseButton {
                        background-color: #e67e22;
                        color: white;
                        border-color: #d35400;
                        font-weight: bold;
                    }

                    QPushButton#valveCloseButton:hover {
                        background-color: #d35400;
                        border-color: #ba4a00;
                    }

                    QPushButton#valveCloseButton:disabled {
                        background-color: #ecf0f1;
                        color: #95a5a6;
                        border-color: #bdc3c7;
                    }

                    QPushButton#rewardButton {
                        background-color: #3498db;
                        color: white;
                        border-color: #2980b9;
                        font-weight: bold;
                    }

                    QPushButton#rewardButton:hover {
                        background-color: #2980b9;
                        border-color: #21618c;
                    }

                    QLabel {
                        color: #2c3e50;
                        font-size: 12pt;
                    }

                    QLabel#volumeLabel {
                        color: #2c3e50;
                        font-size: 12pt;
                        font-weight: bold;
                    }

                    QDoubleSpinBox {
                        border: 2px solid #bdc3c7;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-weight: bold;
                        font-size: 12pt;
                        background-color: white;
                        color: #2c3e50;
                        min-height: 20px;
                    }

                    QDoubleSpinBox:focus {
                        border-color: #3498db;
                    }

                    QDoubleSpinBox::up-button {
                        subcontrol-origin: border;
                        subcontrol-position: top right;
                        width: 20px;
                        background-color: #f8f9fa;
                        border: 1px solid #bdc3c7;
                        border-top-right-radius: 4px;
                        border-bottom: none;
                    }

                    QDoubleSpinBox::up-button:hover {
                        background-color: #e9ecef;
                        border-color: #3498db;
                    }

                    QDoubleSpinBox::up-button:pressed {
                        background-color: #dee2e6;
                    }

                    QDoubleSpinBox::up-arrow {
                        image: none;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-bottom: 6px solid #2c3e50;
                        width: 0px;
                        height: 0px;
                    }

                    QDoubleSpinBox::down-button {
                        subcontrol-origin: border;
                        subcontrol-position: bottom right;
                        width: 20px;
                        background-color: #f8f9fa;
                        border: 1px solid #bdc3c7;
                        border-bottom-right-radius: 4px;
                        border-top: none;
                    }

                    QDoubleSpinBox::down-button:hover {
                        background-color: #e9ecef;
                        border-color: #3498db;
                    }

                    QDoubleSpinBox::down-button:pressed {
                        background-color: #dee2e6;
                    }

                    QDoubleSpinBox::down-arrow {
                        image: none;
                        border-left: 4px solid transparent;
                        border-right: 4px solid transparent;
                        border-top: 6px solid #2c3e50;
                        width: 0px;
                        height: 0px;
                    }

                    QSlider::groove:horizontal {
                        border: 1px solid #bdc3c7;
                        height: 8px;
                        background: #ecf0f1;
                        margin: 2px 0;
                        border-radius: 4px;
                    }

                    QSlider::handle:horizontal {
                        background: #3498db;
                        border: 2px solid #2980b9;
                        width: 20px;
                        margin: -6px 0;
                        border-radius: 10px;
                    }

                    QSlider::handle:horizontal:hover {
                        background: #2980b9;
                        border-color: #21618c;
                    }

                    QSlider::handle:horizontal:pressed {
                        background: #21618c;
                    }

                    QSlider::sub-page:horizontal {
                        background: #3498db;
                        border: 1px solid #2980b9;
                        height: 8px;
                        border-radius: 4px;
                    }

                    QSlider::add-page:horizontal {
                        background: #ecf0f1;
                        border: 1px solid #bdc3c7;
                        height: 8px;
                        border-radius: 4px;
                    }

                    QSlider::groove:vertical {
                        border: 1px solid #bdc3c7;
                        width: 8px;
                        background: #ecf0f1;
                        margin: 0 2px;
                        border-radius: 4px;
                    }

                    QSlider::handle:vertical {
                        background: #3498db;
                        border: 2px solid #2980b9;
                        height: 20px;
                        margin: 0 -6px;
                        border-radius: 10px;
                    }

                    QSlider::handle:vertical:hover {
                        background: #2980b9;
                        border-color: #21618c;
                    }

                    QSlider::handle:vertical:pressed {
                        background: #21618c;
                    }

                    QSlider::sub-page:vertical {
                        background: #ecf0f1;
                        border: 1px solid #bdc3c7;
                        width: 8px;
                        border-radius: 4px;
                    }

                    QSlider::add-page:vertical {
                        background: #3498db;
                        border: 1px solid #2980b9;
                        width: 8px;
                        border-radius: 4px;
                    }

                    QPushButton#reinforcingGuidanceButton {
                        background-color: #3498db;
                        color: white;
                        border-color: #2980b9;
                        font-weight: bold;
                    }

                    QPushButton#reinforcingGuidanceButton:hover {
                        background-color: #2980b9;
                        border-color: #1f6dad;
                    }

                    QPushButton#reinforcingGuidanceDisableButton {
                        background-color: #95a5a6;
                        color: white;
                        border-color: #7f8c8d;
                        font-weight: bold;
                    }

                    QPushButton#reinforcingGuidanceDisableButton:hover {
                        background-color: #7f8c8d;
                        border-color: #6c7b7d;
                    }

                    QPushButton#aversiveGuidanceButton {
                        background-color: #9b59b6;
                        color: white;
                        border-color: #8e44ad;
                        font-weight: bold;
                    }

                    QPushButton#aversiveGuidanceButton:hover {
                        background-color: #8e44ad;
                        border-color: #7d3c98;
                    }

                    QPushButton#aversiveGuidanceDisableButton {
                        background-color: #95a5a6;
                        color: white;
                        border-color: #7f8c8d;
                        font-weight: bold;
                    }

                    QPushButton#aversiveGuidanceDisableButton:hover {
                        background-color: #7f8c8d;
                        border-color: #6c7b7d;
                    }

                    QPushButton#gasValveOpenButton {
                        background-color: #27ae60;
                        color: white;
                        border-color: #229954;
                        font-weight: bold;
                    }

                    QPushButton#gasValveOpenButton:hover {
                        background-color: #229954;
                        border-color: #1e8449;
                    }

                    QPushButton#gasValveOpenButton:disabled {
                        background-color: #ecf0f1;
                        color: #95a5a6;
                        border-color: #bdc3c7;
                    }

                    QPushButton#gasValveCloseButton {
                        background-color: #e67e22;
                        color: white;
                        border-color: #d35400;
                        font-weight: bold;
                    }

                    QPushButton#gasValveCloseButton:hover {
                        background-color: #d35400;
                        border-color: #ba4a00;
                    }

                    QPushButton#gasValveCloseButton:disabled {
                        background-color: #ecf0f1;
                        color: #95a5a6;
                        border-color: #bdc3c7;
                    }

                    QPushButton#gasPuffButton {
                        background-color: #3498db;
                        color: white;
                        border-color: #2980b9;
                        font-weight: bold;
                    }

                    QPushButton#gasPuffButton:hover {
                        background-color: #2980b9;
                        border-color: #21618c;
                    }
                """)

    def _setup_monitoring(self) -> None:
        """Sets up a QTimer to monitor the runtime termination status."""
        self.monitor_timer = QTimer(self)
        # noinspection PyUnresolvedReferences
        self.monitor_timer.timeout.connect(self._check_external_state)
        self.monitor_timer.start(100)  # Checks every 100 ms

    def _check_external_state(self) -> None:
        """Checks the state of externally addressable UI elements and updates the managed GUI to reflect the
        externally driven changes.
        """
        # noinspection PyBroadException
        try:
            # If the termination flag has been set to 1, terminates the GUI process
            if self._data_array[_DataArrayIndex.TERMINATION] == 1:
                self.close()

            # Checks for external pause state changes and, if necessary, updates the GUI to reflect the current
            # runtime state (running or paused).
            external_pause_state = bool(self._data_array[_DataArrayIndex.PAUSE_STATE])
            if external_pause_state != self._is_paused:
                # External pause state changed, update UI accordingly
                self._is_paused = external_pause_state
                self._update_pause_ui()

            # Checks for external reinforcing guidance state changes and, if necessary, updates the GUI.
            external_reinforcing_guidance = bool(self._data_array[_DataArrayIndex.REINFORCING_GUIDANCE_ENABLED])
            if external_reinforcing_guidance != self._reinforcing_guidance_enabled:
                self._reinforcing_guidance_enabled = external_reinforcing_guidance
                self._update_reinforcing_guidance_ui()

            # Checks for external aversive guidance state changes and, if necessary, updates the GUI.
            external_aversive_guidance = bool(self._data_array[_DataArrayIndex.AVERSIVE_GUIDANCE_ENABLED])
            if external_aversive_guidance != self._aversive_guidance_enabled:
                self._aversive_guidance_enabled = external_aversive_guidance
                self._update_aversive_guidance_ui()

            # Checks for setup complete state change. Once setup is complete, valve open/close buttons are
            # permanently disabled.
            external_setup_complete = bool(self._data_array[_DataArrayIndex.SETUP_COMPLETE])
            if external_setup_complete and not self._setup_complete:
                self._setup_complete = True
                self._disable_valve_open_close_buttons()

            # Reads valve tracker state (index 2 contains open/close state: 0=closed, 1=open).
            water_valve_state = int(self._valve_tracker[2])

            # Reads gas puff tracker state (index 1 contains open/close state: 0=closed, 1=open).
            gas_valve_state = int(self._gas_puff_tracker[1])

            # Detects when water valve closes (state transitions to closed while reward was in progress).
            if self._reward_in_progress and water_valve_state == 0:
                self._reward_in_progress = False
                self.valve_status_label.setText("Valve: 🔒 Closed")
                self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

            # Detects when gas puff delivery completes (state transitions to closed while puff was in progress).
            # Only updates if aversive trials are enabled (gas_valve_status_label exists).
            if self._puff_in_progress and gas_valve_state == 0 and self.gas_valve_status_label is not None:
                self._puff_in_progress = False
                self.gas_valve_status_label.setText("Valve: 🔒 Closed")
                self.gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

        except Exception:
            self.close()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        """Handles GUI window close events.

        Args:
            event: The Qt-generated window shutdown event instance.
        """
        # Sends a runtime termination signal via the SharedMemoryArray before accepting the close event.
        # noinspection PyBroadException
        with contextlib.suppress(Exception):
            self._data_array[_DataArrayIndex.TERMINATION] = 1
        if event is not None:
            event.accept()

    def _exit_runtime(self) -> None:
        """Instructs the system to terminate the runtime."""
        previous_status = self.runtime_status_label.text()
        style = self.runtime_status_label.styleSheet()
        self._data_array[_DataArrayIndex.EXIT_SIGNAL] = 1
        self.runtime_status_label.setText("✖ Exit signal sent")
        self.runtime_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        self.exit_btn.setText("✖ Exit Requested")
        self.exit_btn.setEnabled(False)

        # Resets the button after 2 seconds
        QTimer.singleShot(2000, lambda: self.exit_btn.setText("✖ Terminate Runtime"))
        QTimer.singleShot(2000, lambda: self.exit_btn.setStyleSheet("QLabel { color: #c0392b; font-weight: bold; }"))
        QTimer.singleShot(2000, lambda: self.exit_btn.setEnabled(True))

        # Restores the status back to the previous state
        QTimer.singleShot(2000, lambda: self.runtime_status_label.setText(previous_status))
        QTimer.singleShot(2000, lambda: self.runtime_status_label.setStyleSheet(style))

    def _deliver_reward(self) -> None:
        """Instructs the system to deliver a water reward to the animal."""
        self._data_array[_DataArrayIndex.REWARD_SIGNAL] = 1
        self._reward_in_progress = True
        self.valve_status_label.setText("Valve: 💧 Delivering")
        self.valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")

    def _open_valve(self) -> None:
        """Instructs the system to open the water delivery valve."""
        self._data_array[_DataArrayIndex.OPEN_VALVE] = 1
        self.valve_status_label.setText("Valve: 🔓 Opened")
        self.valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _close_valve(self) -> None:
        """Instructs the system to close the water delivery valve."""
        self._data_array[_DataArrayIndex.CLOSE_VALVE] = 1
        self.valve_status_label.setText("Valve: 🔒 Closed")
        self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _toggle_pause(self) -> None:
        """Instructs the system to pause or resume the data acquisition session's runtime."""
        self._is_paused = not self._is_paused
        self._data_array[_DataArrayIndex.PAUSE_STATE] = 1 if self._is_paused else 0
        self._update_pause_ui()

    def _update_reward_volume(self) -> None:
        """Updates the volume used by the system when delivering water rewards to match the current GUI
        configuration.
        """
        self._data_array[_DataArrayIndex.REWARD_VOLUME] = int(self.volume_spinbox.value())

    def _update_speed_modifier(self) -> None:
        """Updates the running speed threshold modifier to match the current GUI configuration."""
        if self.speed_spinbox is not None:
            self._data_array[_DataArrayIndex.SPEED_MODIFIER] = int(self.speed_spinbox.value())

    def _update_duration_modifier(self) -> None:
        """Updates the running epoch duration modifier to match the current GUI configuration."""
        if self.duration_spinbox is not None:
            self._data_array[_DataArrayIndex.DURATION_MODIFIER] = int(self.duration_spinbox.value())

    @staticmethod
    def _refresh_button_style(button: QPushButton) -> None:
        """Refreshes button styles after object name change."""
        button.style().unpolish(button)  # type: ignore[union-attr]
        button.style().polish(button)  # type: ignore[union-attr]
        button.update()

    def _update_reinforcing_guidance_ui(self) -> None:
        """Updates the GUI to reflect the current reinforcing trial guidance state."""
        if self.reinforcing_guidance_btn is None:
            return

        if self._reinforcing_guidance_enabled:
            self.reinforcing_guidance_btn.setText("🚫 Disable Reinforcing Guidance")
            self.reinforcing_guidance_btn.setObjectName("reinforcingGuidanceDisableButton")
        else:
            self.reinforcing_guidance_btn.setText("🎯 Enable Reinforcing Guidance")
            self.reinforcing_guidance_btn.setObjectName("reinforcingGuidanceButton")

        # Refreshes styles after object name change
        self._refresh_button_style(button=self.reinforcing_guidance_btn)

    def _update_aversive_guidance_ui(self) -> None:
        """Updates the GUI to reflect the current aversive trial guidance state."""
        if self.aversive_guidance_btn is None:
            return

        if self._aversive_guidance_enabled:
            self.aversive_guidance_btn.setText("🚫 Disable Aversive Guidance")
            self.aversive_guidance_btn.setObjectName("aversiveGuidanceDisableButton")
        else:
            self.aversive_guidance_btn.setText("🎯 Enable Aversive Guidance")
            self.aversive_guidance_btn.setObjectName("aversiveGuidanceButton")

        # Refreshes styles after object name change
        self._refresh_button_style(button=self.aversive_guidance_btn)

    def _toggle_reinforcing_guidance(self) -> None:
        """Instructs the system to enable or disable the reinforcing trial guidance mode."""
        self._reinforcing_guidance_enabled = not self._reinforcing_guidance_enabled
        self._data_array[_DataArrayIndex.REINFORCING_GUIDANCE_ENABLED] = 1 if self._reinforcing_guidance_enabled else 0
        self._update_reinforcing_guidance_ui()

    def _toggle_aversive_guidance(self) -> None:
        """Instructs the system to enable or disable the aversive trial guidance mode."""
        self._aversive_guidance_enabled = not self._aversive_guidance_enabled
        self._data_array[_DataArrayIndex.AVERSIVE_GUIDANCE_ENABLED] = 1 if self._aversive_guidance_enabled else 0
        self._update_aversive_guidance_ui()

    def _update_pause_ui(self) -> None:
        """Updates the GUI to reflect the current data acquisition runtime pause state."""
        if self._is_paused:
            self.pause_btn.setText("▶️ Resume Runtime")
            self.pause_btn.setObjectName("resumeButton")
            self.runtime_status_label.setText("Runtime Status: ⏸️ Paused")
            self.runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        else:
            self.pause_btn.setText("⏸️ Pause Runtime")
            self.pause_btn.setObjectName("pauseButton")
            self.runtime_status_label.setText("Runtime Status: 🟢 Running")
            self.runtime_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

        # Refresh styles after object name change
        self._refresh_button_style(button=self.pause_btn)

    def _disable_valve_open_close_buttons(self) -> None:
        """Permanently disables valve open/close buttons after setup is complete."""
        self.valve_open_btn.setEnabled(False)
        self.valve_close_btn.setEnabled(False)
        if self.gas_valve_open_btn is not None:
            self.gas_valve_open_btn.setEnabled(False)
        if self.gas_valve_close_btn is not None:
            self.gas_valve_close_btn.setEnabled(False)

    def _update_gas_puff_duration(self) -> None:
        """Updates the gas puff duration to match the current GUI configuration."""
        if self.gas_duration_spinbox is not None:
            self._data_array[_DataArrayIndex.GAS_VALVE_PUFF_DURATION] = int(self.gas_duration_spinbox.value())

    def _gas_valve_open(self) -> None:
        """Instructs the system to open the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_OPEN] = 1
        if self.gas_valve_status_label is not None:
            self.gas_valve_status_label.setText("Valve: 🔓 Opened")
            self.gas_valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _gas_valve_close(self) -> None:
        """Instructs the system to close the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_CLOSE] = 1
        if self.gas_valve_status_label is not None:
            self.gas_valve_status_label.setText("Valve: 🔒 Closed")
            self.gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _gas_valve_puff(self) -> None:
        """Instructs the system to deliver a gas puff."""
        self._data_array[_DataArrayIndex.GAS_VALVE_PUFF] = 1
        self._puff_in_progress = True
        if self.gas_valve_status_label is not None:
            self.gas_valve_status_label.setText("Valve: 💨 Puffing")
            self.gas_valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")
