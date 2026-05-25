"""Provides the graphical user interface used by the Mesoscope-VR data acquisition system to facilitate data
acquisition runtimes by allowing direct control over a subset of the system's runtime parameters and hardware.
"""

from __future__ import annotations

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

from .visualizer import VisualizerMode


class _DataArrayIndex(IntEnum):
    """Defines the shared memory array indices for each runtime parameter and hardware component addressable from the
    user-facing GUI.
    """

    TERMINATION = 0
    """Signals the UI process to terminate and shut down the GUI window."""
    EXIT_SIGNAL = 1
    """Tracks the user's request to abort the data acquisition session's runtime."""
    REWARD_SIGNAL = 2
    """Tracks the user's request to deliver a water reward."""
    SPEED_MODIFIER = 3
    """Stores the user-defined running speed threshold modifier."""
    DURATION_MODIFIER = 4
    """Stores the user-defined running epoch duration threshold modifier."""
    PAUSE_STATE = 5
    """Tracks whether the runtime is paused (1) or running (0)."""
    OPEN_VALVE = 6
    """Tracks the user's request to open the water delivery valve."""
    CLOSE_VALVE = 7
    """Tracks the user's request to close the water delivery valve."""
    REWARD_VOLUME = 8
    """Stores the user-defined water reward volume in microliters."""
    REINFORCING_GUIDANCE_ENABLED = 9
    """Tracks whether the reinforcing trial guidance mode is enabled."""
    AVERSIVE_GUIDANCE_ENABLED = 10
    """Tracks whether the aversive trial guidance mode is enabled."""
    GAS_VALVE_OPEN = 11
    """Tracks the user's request to open the gas puff valve."""
    GAS_VALVE_CLOSE = 12
    """Tracks the user's request to close the gas puff valve."""
    GAS_VALVE_PUFF = 13
    """Tracks the user's request to deliver a gas puff."""
    GAS_VALVE_PUFF_DURATION = 14
    """Stores the user-defined gas puff duration in milliseconds."""
    SETUP_COMPLETE = 15
    """Tracks whether the initial setup phase is complete."""


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

        self._data_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name="runtime_control_ui", prototype=prototype, exists_ok=True
        )

        self._valve_tracker: SharedMemoryArray = valve_tracker
        self._gas_puff_tracker: SharedMemoryArray = gas_puff_tracker

        # Initializes the mode to EXPERIMENT by default. The mode is set when start() is called.
        self._mode: VisualizerMode = VisualizerMode.EXPERIMENT

        # Trial type flags, set when start() is called based on experiment configuration.
        self._has_reinforcing_trials: bool = True
        self._has_aversive_trials: bool = True

        # Defines but does not automatically start the UI process. The process target is set in start() to pass
        # the mode.
        self._ui_process: Process | None = None
        self._started: bool = False

    def __del__(self) -> None:
        """Terminates the UI process and releases the instance's shared memory buffers when garbage-collected."""
        self.shutdown()
        # Note: Does not disconnect or destroy the trackers as they're owned by their respective interfaces

    def __repr__(self) -> str:
        """Returns a string representation of the RuntimeControlUI instance."""
        return f"RuntimeControlUI(mode={self._mode}, started={self._started})"

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
        if self._started:
            return

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
        ui_process = Process(target=target, daemon=True)
        self._ui_process = ui_process

        ui_process.start()

        # Connects to the shared memory array from the central runtime process and configures it to destroy the
        # shared memory buffer in case of an emergency (error) shutdown.
        self._data_array.connect()
        self._data_array.enable_buffer_destruction()

        # Connects to trackers to monitor valve and gas puff states
        self._valve_tracker.connect()
        self._gas_puff_tracker.connect()

        self._started = True

    def shutdown(self) -> None:
        """Shuts down the remote UI process and releases the instance's shared memory buffer."""
        if not self._started:
            return

        if self._ui_process is not None and self._ui_process.is_alive():
            self._data_array[_DataArrayIndex.TERMINATION] = 1  # Sends the termination signal to the remote process
            self._ui_process.terminate()
            self._ui_process.join(timeout=2.0)

        self._data_array.disconnect()
        self._data_array.destroy()

        # Note: Does not disconnect trackers here - they're owned by their respective interfaces and disconnecting
        # them would break access to delivered_volume when generating the session descriptor during shutdown.

        self._started = False

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

    def _run_ui_process(
        self,
        mode: VisualizerMode,
        *,
        has_reinforcing_trials: bool,
        has_aversive_trials: bool,
    ) -> None:
        """Runs the UI management cycle in a parallel process.

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

        self._reward_in_progress: bool = False
        self._puff_in_progress: bool = False

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

        self._setup_ui()
        self._setup_monitoring()

        self._apply_qt6_styles()

    def _setup_ui(self) -> None:
        """Creates and arranges all UI elements."""
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

        self._exit_button = QPushButton("✖ Terminate Runtime")
        self._exit_button.setToolTip("Gracefully ends the runtime and initiates the shutdown procedure.")
        # noinspection PyUnresolvedReferences
        self._exit_button.clicked.connect(self._exit_runtime)
        self._exit_button.setObjectName("exitButton")

        self._pause_button = QPushButton("▶️ Resume Runtime")
        self._pause_button.setToolTip("Pauses or resumes the runtime.")
        # noinspection PyUnresolvedReferences
        self._pause_button.clicked.connect(self._toggle_pause)
        self._pause_button.setObjectName("resumeButton")

        # Configures the main control buttons
        for button in [self._exit_button, self._pause_button]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(button)

        # Reinforcing Guidance button (only shown in EXPERIMENT mode with reinforcing trials).
        self._reinforcing_guidance_button: QPushButton | None = None
        if self._mode == VisualizerMode.EXPERIMENT and self._has_reinforcing_trials:
            reinforcing_guidance_button = QPushButton("🎯 Enable Reinforcing Guidance")
            reinforcing_guidance_button.setToolTip("Toggles reinforcing trial guidance mode on or off.")
            # noinspection PyUnresolvedReferences
            reinforcing_guidance_button.clicked.connect(self._toggle_reinforcing_guidance)
            reinforcing_guidance_button.setObjectName("reinforcingGuidanceButton")
            reinforcing_guidance_button.setMinimumHeight(35)
            reinforcing_guidance_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(reinforcing_guidance_button)
            self._reinforcing_guidance_button = reinforcing_guidance_button

        # Aversive Guidance button (only shown in EXPERIMENT mode with aversive trials).
        self._aversive_guidance_button: QPushButton | None = None
        if self._mode == VisualizerMode.EXPERIMENT and self._has_aversive_trials:
            aversive_guidance_button = QPushButton("🎯 Enable Aversive Guidance")
            aversive_guidance_button.setToolTip("Toggles aversive trial guidance mode on or off.")
            # noinspection PyUnresolvedReferences
            aversive_guidance_button.clicked.connect(self._toggle_aversive_guidance)
            aversive_guidance_button.setObjectName("aversiveGuidanceButton")
            aversive_guidance_button.setMinimumHeight(35)
            aversive_guidance_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            runtime_control_layout.addWidget(aversive_guidance_button)
            self._aversive_guidance_button = aversive_guidance_button

        # Adds runtime status tracker to the same box
        self._runtime_status_label = QLabel("Runtime Status: ⏸️ Paused")
        self._runtime_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        runtime_status_font = QFont()
        runtime_status_font.setPointSize(35)
        runtime_status_font.setBold(True)
        self._runtime_status_label.setFont(runtime_status_font)
        self._runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        runtime_control_layout.addWidget(self._runtime_status_label)

        # Adds the runtime control box to the UI widget
        main_layout.addWidget(runtime_control_group)

        # Reward Valve Control Group
        valve_group = QGroupBox("Reward Valve Control")
        valve_layout = QVBoxLayout(valve_group)
        valve_layout.setSpacing(6)

        # Arranges valve control buttons in a horizontal layout
        valve_buttons_layout = QHBoxLayout()

        self._valve_open_button = QPushButton("🔓 Open")
        self._valve_open_button.setToolTip("Opens the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self._valve_open_button.clicked.connect(self._open_valve)
        self._valve_open_button.setObjectName("valveOpenButton")

        self._valve_close_button = QPushButton("🔒 Close")
        self._valve_close_button.setToolTip("Closes the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self._valve_close_button.clicked.connect(self._close_valve)
        self._valve_close_button.setObjectName("valveCloseButton")

        self._reward_button = QPushButton("● Reward")
        self._reward_button.setToolTip("Delivers 5 uL of water through the solenoid valve.")
        # noinspection PyUnresolvedReferences
        self._reward_button.clicked.connect(self._deliver_reward)
        self._reward_button.setObjectName("rewardButton")

        # Configures the buttons to expand when the UI is resized, but use a fixed height of 35 points
        for button in [self._valve_open_button, self._valve_close_button, self._reward_button]:
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

        self._volume_spinbox = QDoubleSpinBox()
        self._volume_spinbox.setRange(1, 20)
        self._volume_spinbox.setValue(5)
        self._volume_spinbox.setDecimals(0)
        self._volume_spinbox.setSuffix(" μL")
        self._volume_spinbox.setToolTip("Sets water reward volume. Accepts values between 1 and 20 μL.")
        self._volume_spinbox.setMinimumHeight(30)
        # noinspection PyUnresolvedReferences
        self._volume_spinbox.valueChanged.connect(self._update_reward_volume)

        # Adds volume controls to the left side
        valve_status_layout.addWidget(volume_label)
        valve_status_layout.addWidget(self._volume_spinbox)

        # Adds the valve status tracker on the right
        self._valve_status_label = QLabel("Valve: 🔒 Closed")
        self._valve_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        valve_status_font = QFont()
        valve_status_font.setPointSize(35)
        valve_status_font.setBold(True)
        self._valve_status_label.setFont(valve_status_font)
        self._valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
        valve_status_layout.addWidget(self._valve_status_label)

        # Add the horizontal status layout to the main valve layout
        valve_layout.addLayout(valve_status_layout)

        # Adds the valve control box to the UI widget
        main_layout.addWidget(valve_group)

        # Gas Puff Valve Control Group (only shown in EXPERIMENT mode with aversive trials).
        self._gas_valve_open_button: QPushButton | None = None
        self._gas_valve_close_button: QPushButton | None = None
        self._gas_puff_button: QPushButton | None = None
        self._gas_duration_spinbox: QDoubleSpinBox | None = None
        self._gas_valve_status_label: QLabel | None = None

        if self._mode == VisualizerMode.EXPERIMENT and self._has_aversive_trials:
            gas_valve_group = QGroupBox("Gas Puff Valve Control")
            gas_valve_layout = QVBoxLayout(gas_valve_group)
            gas_valve_layout.setSpacing(6)

            # Arranges gas valve control buttons in a horizontal layout
            gas_valve_buttons_layout = QHBoxLayout()

            gas_valve_open_button = QPushButton("🔓 Open")
            gas_valve_open_button.setToolTip("Opens the gas puff valve.")
            # noinspection PyUnresolvedReferences
            gas_valve_open_button.clicked.connect(self._gas_valve_open)
            gas_valve_open_button.setObjectName("gasValveOpenButton")

            gas_valve_close_button = QPushButton("🔒 Close")
            gas_valve_close_button.setToolTip("Closes the gas puff valve.")
            # noinspection PyUnresolvedReferences
            gas_valve_close_button.clicked.connect(self._gas_valve_close)
            gas_valve_close_button.setObjectName("gasValveCloseButton")

            gas_puff_button = QPushButton("💨 Puff")
            gas_puff_button.setToolTip("Delivers a gas puff with the specified duration.")
            # noinspection PyUnresolvedReferences
            gas_puff_button.clicked.connect(self._gas_valve_puff)
            gas_puff_button.setObjectName("gasPuffButton")

            # Configures the buttons to expand when the UI is resized, but use a fixed height of 35 points
            for button in [gas_valve_open_button, gas_valve_close_button, gas_puff_button]:
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

            gas_duration_spinbox = QDoubleSpinBox()
            gas_duration_spinbox.setRange(10, 350)
            gas_duration_spinbox.setValue(100)
            gas_duration_spinbox.setDecimals(0)
            gas_duration_spinbox.setSuffix(" ms")
            gas_duration_spinbox.setToolTip("Sets gas puff duration. Accepts values between 10 and 350 ms.")
            gas_duration_spinbox.setMinimumHeight(30)
            # noinspection PyUnresolvedReferences
            gas_duration_spinbox.valueChanged.connect(self._update_gas_puff_duration)

            # Adds duration controls to the left side
            gas_valve_status_layout.addWidget(gas_duration_label)
            gas_valve_status_layout.addWidget(gas_duration_spinbox)

            # Adds the gas valve status tracker on the right
            gas_valve_status_label = QLabel("Valve: 🔒 Closed")
            gas_valve_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            gas_valve_status_font = QFont()
            gas_valve_status_font.setPointSize(35)
            gas_valve_status_font.setBold(True)
            gas_valve_status_label.setFont(gas_valve_status_font)
            gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")
            gas_valve_status_layout.addWidget(gas_valve_status_label)

            # Adds the horizontal status layout to the main gas valve layout
            gas_valve_layout.addLayout(gas_valve_status_layout)

            # Adds the gas valve control box to the UI widget
            main_layout.addWidget(gas_valve_group)

            # Caches the widget references accessed by the monitoring and signal-handler methods.
            self._gas_valve_open_button = gas_valve_open_button
            self._gas_valve_close_button = gas_valve_close_button
            self._gas_puff_button = gas_puff_button
            self._gas_duration_spinbox = gas_duration_spinbox
            self._gas_valve_status_label = gas_valve_status_label

        # Adds Run Training controls in a horizontal layout (only shown in RUN_TRAINING mode).
        self._speed_group: QGroupBox | None = None
        self._duration_group: QGroupBox | None = None
        self._speed_spinbox: QDoubleSpinBox | None = None
        self._duration_spinbox: QDoubleSpinBox | None = None

        if self._mode == VisualizerMode.RUN_TRAINING:
            controls_layout = QHBoxLayout()
            controls_layout.setSpacing(6)

            # Running Speed Threshold Control Group
            speed_group = QGroupBox("Speed Threshold")
            speed_layout = QVBoxLayout(speed_group)

            # Speed Modifier
            speed_status_label = QLabel("Current Modifier:")
            speed_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            speed_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
            speed_layout.addWidget(speed_status_label)
            speed_spinbox = QDoubleSpinBox()
            speed_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 cm/s
            speed_spinbox.setValue(0)
            speed_spinbox.setDecimals(0)
            speed_spinbox.setToolTip("Sets the running speed threshold modifier value.")
            speed_spinbox.setMinimumHeight(30)
            # noinspection PyUnresolvedReferences
            speed_spinbox.valueChanged.connect(self._update_speed_modifier)
            speed_layout.addWidget(speed_spinbox)

            # Running Duration Threshold Control Group
            duration_group = QGroupBox("Duration Threshold")
            duration_layout = QVBoxLayout(duration_group)

            # Duration modifier
            duration_status_label = QLabel("Current Modifier:")
            duration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            duration_status_label.setStyleSheet("QLabel { font-weight: bold; color: #34495e; }")
            duration_layout.addWidget(duration_status_label)
            duration_spinbox = QDoubleSpinBox()
            duration_spinbox.setRange(-1000, 1000)  # Factoring in the step of 0.01, this allows -20 to +20 s
            duration_spinbox.setValue(0)
            duration_spinbox.setDecimals(0)
            duration_spinbox.setToolTip("Sets the running duration threshold modifier value.")
            # noinspection PyUnresolvedReferences
            duration_spinbox.valueChanged.connect(self._update_duration_modifier)
            duration_layout.addWidget(duration_spinbox)

            # Adds speed and duration threshold modifiers to the main UI widget
            controls_layout.addWidget(speed_group)
            controls_layout.addWidget(duration_group)
            main_layout.addLayout(controls_layout)

            # Caches the widget references accessed by the modifier-update handler methods.
            self._speed_group = speed_group
            self._duration_group = duration_group
            self._speed_spinbox = speed_spinbox
            self._duration_spinbox = duration_spinbox

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
        self._monitor_timer = QTimer(self)
        # noinspection PyUnresolvedReferences
        self._monitor_timer.timeout.connect(self._check_external_state)
        self._monitor_timer.start(100)  # Checks every 100 ms

    def _check_external_state(self) -> None:
        """Checks the state of externally addressable UI elements and updates the managed GUI to reflect the
        externally driven changes.
        """
        # noinspection PyBroadException
        try:
            # If the termination flag has been set, terminates the GUI process
            if bool(self._data_array[_DataArrayIndex.TERMINATION]):
                self.close()

            # Checks for external pause state changes and, if necessary, updates the GUI to reflect the current
            # runtime state (running or paused).
            external_pause_state = bool(self._data_array[_DataArrayIndex.PAUSE_STATE])
            if external_pause_state != self._is_paused:
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
                self._valve_status_label.setText("Valve: 🔒 Closed")
                self._valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

            # Detects when gas puff delivery completes (state transitions to closed while puff was in progress).
            # Only updates if aversive trials are enabled (gas_valve_status_label exists).
            if self._puff_in_progress and gas_valve_state == 0 and self._gas_valve_status_label is not None:
                self._puff_in_progress = False
                self._gas_valve_status_label.setText("Valve: 🔒 Closed")
                self._gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

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
        previous_status = self._runtime_status_label.text()
        style = self._runtime_status_label.styleSheet()
        self._data_array[_DataArrayIndex.EXIT_SIGNAL] = 1
        self._runtime_status_label.setText("✖ Exit signal sent")
        self._runtime_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        self._exit_button.setText("✖ Exit Requested")
        self._exit_button.setEnabled(False)

        # Resets the button after 2 seconds
        exit_button_style = "QLabel { color: #c0392b; font-weight: bold; }"
        QTimer.singleShot(2000, lambda: self._exit_button.setText("✖ Terminate Runtime"))
        QTimer.singleShot(2000, lambda: self._exit_button.setStyleSheet(exit_button_style))
        QTimer.singleShot(2000, lambda: self._exit_button.setEnabled(True))

        # Restores the status back to the previous state
        QTimer.singleShot(2000, lambda: self._runtime_status_label.setText(previous_status))
        QTimer.singleShot(2000, lambda: self._runtime_status_label.setStyleSheet(style))

    def _deliver_reward(self) -> None:
        """Instructs the system to deliver a water reward to the animal."""
        self._data_array[_DataArrayIndex.REWARD_SIGNAL] = 1
        self._reward_in_progress = True
        self._valve_status_label.setText("Valve: 💧 Delivering")
        self._valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")

    def _open_valve(self) -> None:
        """Instructs the system to open the water delivery valve."""
        self._data_array[_DataArrayIndex.OPEN_VALVE] = 1
        self._valve_status_label.setText("Valve: 🔓 Opened")
        self._valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _close_valve(self) -> None:
        """Instructs the system to close the water delivery valve."""
        self._data_array[_DataArrayIndex.CLOSE_VALVE] = 1
        self._valve_status_label.setText("Valve: 🔒 Closed")
        self._valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _toggle_pause(self) -> None:
        """Instructs the system to pause or resume the data acquisition session's runtime."""
        self._is_paused = not self._is_paused
        self._data_array[_DataArrayIndex.PAUSE_STATE] = 1 if self._is_paused else 0
        self._update_pause_ui()

    def _update_reward_volume(self) -> None:
        """Updates the volume used by the system when delivering water rewards to match the current GUI
        configuration.
        """
        self._data_array[_DataArrayIndex.REWARD_VOLUME] = int(self._volume_spinbox.value())

    def _update_speed_modifier(self) -> None:
        """Updates the running speed threshold modifier to match the current GUI configuration."""
        if self._speed_spinbox is not None:
            self._data_array[_DataArrayIndex.SPEED_MODIFIER] = int(self._speed_spinbox.value())

    def _update_duration_modifier(self) -> None:
        """Updates the running epoch duration modifier to match the current GUI configuration."""
        if self._duration_spinbox is not None:
            self._data_array[_DataArrayIndex.DURATION_MODIFIER] = int(self._duration_spinbox.value())

    @staticmethod
    def _refresh_button_style(button: QPushButton) -> None:
        """Refreshes button styles after object name change."""
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _update_reinforcing_guidance_ui(self) -> None:
        """Updates the GUI to reflect the current reinforcing trial guidance state."""
        if self._reinforcing_guidance_button is None:
            return

        if self._reinforcing_guidance_enabled:
            self._reinforcing_guidance_button.setText("🚫 Disable Reinforcing Guidance")
            self._reinforcing_guidance_button.setObjectName("reinforcingGuidanceDisableButton")
        else:
            self._reinforcing_guidance_button.setText("🎯 Enable Reinforcing Guidance")
            self._reinforcing_guidance_button.setObjectName("reinforcingGuidanceButton")

        # Refreshes styles after object name change
        self._refresh_button_style(button=self._reinforcing_guidance_button)

    def _update_aversive_guidance_ui(self) -> None:
        """Updates the GUI to reflect the current aversive trial guidance state."""
        if self._aversive_guidance_button is None:
            return

        if self._aversive_guidance_enabled:
            self._aversive_guidance_button.setText("🚫 Disable Aversive Guidance")
            self._aversive_guidance_button.setObjectName("aversiveGuidanceDisableButton")
        else:
            self._aversive_guidance_button.setText("🎯 Enable Aversive Guidance")
            self._aversive_guidance_button.setObjectName("aversiveGuidanceButton")

        # Refreshes styles after object name change
        self._refresh_button_style(button=self._aversive_guidance_button)

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
            self._pause_button.setText("▶️ Resume Runtime")
            self._pause_button.setObjectName("resumeButton")
            self._runtime_status_label.setText("Runtime Status: ⏸️ Paused")
            self._runtime_status_label.setStyleSheet("QLabel { color: #f39c12; font-weight: bold; }")
        else:
            self._pause_button.setText("⏸️ Pause Runtime")
            self._pause_button.setObjectName("pauseButton")
            self._runtime_status_label.setText("Runtime Status: 🟢 Running")
            self._runtime_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

        # Refresh styles after object name change
        self._refresh_button_style(button=self._pause_button)

    def _disable_valve_open_close_buttons(self) -> None:
        """Permanently disables valve open/close buttons after setup is complete."""
        self._valve_open_button.setEnabled(False)
        self._valve_close_button.setEnabled(False)
        if self._gas_valve_open_button is not None:
            self._gas_valve_open_button.setEnabled(False)
        if self._gas_valve_close_button is not None:
            self._gas_valve_close_button.setEnabled(False)

    def _update_gas_puff_duration(self) -> None:
        """Updates the gas puff duration to match the current GUI configuration."""
        if self._gas_duration_spinbox is not None:
            self._data_array[_DataArrayIndex.GAS_VALVE_PUFF_DURATION] = int(self._gas_duration_spinbox.value())

    def _gas_valve_open(self) -> None:
        """Instructs the system to open the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_OPEN] = 1
        if self._gas_valve_status_label is not None:
            self._gas_valve_status_label.setText("Valve: 🔓 Opened")
            self._gas_valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _gas_valve_close(self) -> None:
        """Instructs the system to close the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_CLOSE] = 1
        if self._gas_valve_status_label is not None:
            self._gas_valve_status_label.setText("Valve: 🔒 Closed")
            self._gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _gas_valve_puff(self) -> None:
        """Instructs the system to deliver a gas puff."""
        self._data_array[_DataArrayIndex.GAS_VALVE_PUFF] = 1
        self._puff_in_progress = True
        if self._gas_valve_status_label is not None:
            self._gas_valve_status_label.setText("Valve: 💨 Puffing")
            self._gas_valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")
