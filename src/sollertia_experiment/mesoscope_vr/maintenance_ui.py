"""Provides a graphical user interface for Mesoscope-VR system maintenance operations."""

import sys
from enum import IntEnum
import contextlib
from multiprocessing import Process

import numpy as np
from PyQt6.QtGui import QFont, QCloseEvent
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
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


class _DataArrayIndex(IntEnum):
    """Defines the shared memory array indices for each runtime parameter and hardware component addressable from the
    user-facing GUI.
    """

    TERMINATION = 0
    VALVE_OPEN = 1
    VALVE_CLOSE = 2
    VALVE_REWARD = 3
    VALVE_REFERENCE = 4
    VALVE_CALIBRATE = 5
    BRAKE_LOCK = 6
    BRAKE_UNLOCK = 7
    REWARD_VOLUME = 8
    CALIBRATION_PULSE_DURATION = 9
    GAS_VALVE_OPEN = 10
    GAS_VALVE_CLOSE = 11
    GAS_VALVE_PULSE = 12
    GAS_VALVE_PULSE_DURATION = 13


class MaintenanceControlUI:
    """Provides the Graphical User Interface (GUI) that allows controlling the Mesoscope-VR hardware during
    maintenance runtimes.

    Notes:
        The UI runs in a parallel process and requires a single CPU core to support its runtime.

        Initializing the class does not start the UI process. Call the start() method before calling any other
        instance methods to start the UI process.

    Args:
        valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes.
        gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            count to other processes.

    Attributes:
        _data_array: The SharedMemoryArray instance used to bidirectionally transfer data between the UI process
            and the maintenance runtime process.
        _valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes.
        _gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            count to other processes.
        _ui_process: The Process instance running the GUI cycle.
        _started: Tracks whether the UI process is running.
    """

    def __init__(self, valve_tracker: SharedMemoryArray, gas_puff_tracker: SharedMemoryArray) -> None:
        # Defines the prototype array for SharedMemoryArray initialization
        prototype = np.zeros(shape=14, dtype=np.uint32)
        prototype[_DataArrayIndex.TERMINATION] = 0
        prototype[_DataArrayIndex.GAS_VALVE_PULSE_DURATION] = 100  # Default gas puff duration: 100 ms
        prototype[_DataArrayIndex.REWARD_VOLUME] = 5  # Default 5 uL
        prototype[_DataArrayIndex.CALIBRATION_PULSE_DURATION] = 30  # Default 30 ms

        # Initializes the SharedMemoryArray instance
        self._data_array = SharedMemoryArray.create_array(
            name="maintenance_control_ui", prototype=prototype, exists_ok=True
        )

        # Caches tracker references to class attributes
        self._valve_tracker = valve_tracker
        self._gas_puff_tracker = gas_puff_tracker

        # Defines but does not automatically start the UI process
        self._ui_process = Process(target=self._run_ui_process, daemon=True)
        self._started = False

    def __del__(self) -> None:
        """Terminates the UI process and releases the instance's shared memory buffers when garbage-collected."""
        self.shutdown()
        # Note: Does not disconnect or destroy the trackers as they're owned by their respective interfaces

    def start(self) -> None:
        """Starts the remote UI process."""
        if self._started:
            return

        self._ui_process.start()
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

        if self._ui_process.is_alive():
            self._data_array[_DataArrayIndex.TERMINATION] = 1
            self._ui_process.terminate()
            self._ui_process.join(timeout=2.0)

        self._data_array.disconnect()
        self._data_array.destroy()

        # Note: Does not disconnect trackers here - they're owned by their respective interfaces and disconnecting
        # them would break access to delivered_volume when generating the session descriptor during shutdown.

        self._started = False

    def _run_ui_process(self) -> None:
        """Runs UI management cycle in a parallel process."""
        self._data_array.connect()
        self._valve_tracker.connect()
        self._gas_puff_tracker.connect()

        try:
            app = QApplication(sys.argv)
            app.setApplicationName("Mesoscope-VR Maintenance Panel")
            app.setOrganizationName("Sollertia")
            app.setStyle("Fusion")

            window = _MaintenanceUIWindow(self._data_array, self._valve_tracker, self._gas_puff_tracker)
            window.show()

            app.exec()
        except Exception as e:
            message = (
                f"Unable to initialize the GUI application for the maintenance user interface. "
                f"Encountered the following error {e}."
            )
            console.error(message=message, error=RuntimeError)
        finally:
            self._data_array.disconnect()
            self._valve_tracker.disconnect()
            self._gas_puff_tracker.disconnect()

    @property
    def exit_signal(self) -> bool:
        """Returns True if the user has requested to terminate the maintenance runtime."""
        return bool(self._data_array[_DataArrayIndex.TERMINATION])

    @property
    def valve_open_signal(self) -> bool:
        """Returns True if the user has requested to open the valve."""
        signal = bool(self._data_array[_DataArrayIndex.VALVE_OPEN])
        self._data_array[_DataArrayIndex.VALVE_OPEN] = 0
        return signal

    @property
    def valve_close_signal(self) -> bool:
        """Returns True if the user has requested to close the valve."""
        signal = bool(self._data_array[_DataArrayIndex.VALVE_CLOSE])
        self._data_array[_DataArrayIndex.VALVE_CLOSE] = 0
        return signal

    @property
    def valve_reward_signal(self) -> bool:
        """Returns True if the user has requested to deliver a reward."""
        signal = bool(self._data_array[_DataArrayIndex.VALVE_REWARD])
        self._data_array[_DataArrayIndex.VALVE_REWARD] = 0
        return signal

    @property
    def valve_reference_signal(self) -> bool:
        """Returns True if the user has requested to run valve reference calibration."""
        signal = bool(self._data_array[_DataArrayIndex.VALVE_REFERENCE])
        self._data_array[_DataArrayIndex.VALVE_REFERENCE] = 0
        return signal

    @property
    def valve_calibrate_signal(self) -> bool:
        """Returns True if the user has requested valve calibration."""
        signal = bool(self._data_array[_DataArrayIndex.VALVE_CALIBRATE])
        self._data_array[_DataArrayIndex.VALVE_CALIBRATE] = 0
        return signal

    @property
    def brake_lock_signal(self) -> bool:
        """Returns True if the user has requested to lock the brake."""
        signal = bool(self._data_array[_DataArrayIndex.BRAKE_LOCK])
        self._data_array[_DataArrayIndex.BRAKE_LOCK] = 0
        return signal

    @property
    def brake_unlock_signal(self) -> bool:
        """Returns True if the user has requested to unlock the brake."""
        signal = bool(self._data_array[_DataArrayIndex.BRAKE_UNLOCK])
        self._data_array[_DataArrayIndex.BRAKE_UNLOCK] = 0
        return signal

    @property
    def reward_volume(self) -> int:
        """Returns the current user-defined volume of water dispensed when delivering water rewards."""
        return int(self._data_array[_DataArrayIndex.REWARD_VOLUME])

    @property
    def calibration_pulse_duration(self) -> int:
        """Returns the current user-defined calibration pulse duration in milliseconds."""
        return int(self._data_array[_DataArrayIndex.CALIBRATION_PULSE_DURATION])

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
    def gas_valve_pulse_signal(self) -> bool:
        """Returns True if the user has requested to pulse the gas puff valve."""
        signal = bool(self._data_array[_DataArrayIndex.GAS_VALVE_PULSE])
        self._data_array[_DataArrayIndex.GAS_VALVE_PULSE] = 0
        return signal

    @property
    def gas_valve_pulse_duration(self) -> int:
        """Returns the current user-defined gas puff pulse duration in milliseconds."""
        return int(self._data_array[_DataArrayIndex.GAS_VALVE_PULSE_DURATION])


class _MaintenanceUIWindow(QMainWindow):
    """Generates, renders, and maintains the Mesoscope-VR acquisition system's maintenance GUI application window.

    Attributes:
        _data_array: The SharedMemoryArray instance used to bidirectionally transfer the data between the UI process
            and other runtime processes.
        _valve_tracker: The SharedMemoryArray instance used by the ValveModule to export the valve's state to other
            processes during runtime.
        _gas_puff_tracker: The SharedMemoryArray instance used by the GasPuffValveInterface to export the gas puff
            data to other processes during runtime.
        _reward_in_progress: Tracks whether a reward delivery is in progress.
        _calibration_in_progress: Tracks whether a calibration procedure is in progress.
        _referencing_in_progress: Tracks whether a referencing procedure is in progress.
        _puff_in_progress: Tracks whether a gas puff delivery is in progress.
    """

    def __init__(
        self, data_array: SharedMemoryArray, valve_tracker: SharedMemoryArray, gas_puff_tracker: SharedMemoryArray
    ) -> None:
        super().__init__()

        self._data_array: SharedMemoryArray = data_array
        self._valve_tracker: SharedMemoryArray = valve_tracker
        self._gas_puff_tracker: SharedMemoryArray = gas_puff_tracker

        # Tracks whether a reward delivery is in progress.
        self._reward_in_progress: bool = False
        # Tracks whether a calibration procedure is in progress.
        self._calibration_in_progress: bool = False
        # Tracks whether a referencing procedure is in progress.
        self._referencing_in_progress: bool = False
        # Tracks whether a gas puff delivery is in progress.
        self._puff_in_progress: bool = False

        self.setWindowTitle("Mesoscope-VR Maintenance Panel")
        self.setFixedSize(550, 750)

        self._setup_ui()
        self._setup_monitoring()
        self._apply_qt6_styles()

    def _setup_ui(self) -> None:
        """Creates and arranges all UI elements."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Reward Valve Control Group
        valve_group = QGroupBox("Reward Valve Control")
        valve_layout = QVBoxLayout(valve_group)
        valve_layout.setSpacing(6)

        # Basic valve controls
        basic_valve_layout = QHBoxLayout()

        self.valve_open_btn = QPushButton("🔓 Open")
        self.valve_open_btn.setToolTip("Open the valve")
        # noinspection PyUnresolvedReferences
        self.valve_open_btn.clicked.connect(self._valve_open)
        self.valve_open_btn.setObjectName("valveOpenButton")

        self.valve_close_btn = QPushButton("🔒 Close")
        self.valve_close_btn.setToolTip("Close the valve")
        # noinspection PyUnresolvedReferences
        self.valve_close_btn.clicked.connect(self._valve_close)
        self.valve_close_btn.setObjectName("valveCloseButton")

        for button in [self.valve_open_btn, self.valve_close_btn]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            basic_valve_layout.addWidget(button)

        valve_layout.addLayout(basic_valve_layout)

        # Volume control and reward section
        volume_reward_layout = QHBoxLayout()
        volume_reward_layout.setSpacing(6)

        # Volume control on the left (same width as Open button)
        volume_sub_layout = QHBoxLayout()
        volume_sub_layout.setSpacing(6)
        volume_label = QLabel("Reward volume:")
        volume_label.setObjectName("volumeLabel")

        self.volume_spinbox = QDoubleSpinBox()
        self.volume_spinbox.setRange(1, 20)
        self.volume_spinbox.setValue(5)
        self.volume_spinbox.setDecimals(0)
        self.volume_spinbox.setSuffix(" μL")
        self.volume_spinbox.setToolTip("Sets water reward volume. Accepts values between 1 and 20 μL.")
        self.volume_spinbox.setMinimumHeight(35)
        # noinspection PyUnresolvedReferences
        self.volume_spinbox.valueChanged.connect(self._update_reward_volume)

        volume_sub_layout.addWidget(volume_label)
        volume_sub_layout.addWidget(self.volume_spinbox)

        # Reward button on the right (same width as Close button)
        self.valve_reward_btn = QPushButton("💧 Reward")
        self.valve_reward_btn.setToolTip("Deliver water reward with specified volume")
        # noinspection PyUnresolvedReferences
        self.valve_reward_btn.clicked.connect(self._valve_reward)
        self.valve_reward_btn.setObjectName("rewardButton")
        self.valve_reward_btn.setMinimumHeight(35)

        volume_reward_layout.addLayout(volume_sub_layout, stretch=1)
        volume_reward_layout.addWidget(self.valve_reward_btn, stretch=1)

        valve_layout.addLayout(volume_reward_layout)

        # Valve status
        self.valve_status_label = QLabel("Valve: Closed")
        self.valve_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(12)
        status_font.setBold(True)
        self.valve_status_label.setFont(status_font)
        self.valve_status_label.setStyleSheet("QLabel { color: #7f8c8d; font-weight: bold; }")
        valve_layout.addWidget(self.valve_status_label)

        main_layout.addWidget(valve_group)

        # Calibration Group
        calibration_group = QGroupBox("Valve Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        calibration_layout.setSpacing(6)

        # Reference button
        self.valve_reference_btn = QPushButton("🔄 Reference (200 x 5 μL)")
        self.valve_reference_btn.setToolTip("Run reference valve calibration (200 pulses x 5 μL)")
        # noinspection PyUnresolvedReferences
        self.valve_reference_btn.clicked.connect(self._valve_reference)
        self.valve_reference_btn.setObjectName("referenceButton")
        self.valve_reference_btn.setMinimumHeight(35)
        self.valve_reference_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        calibration_layout.addWidget(self.valve_reference_btn)

        # Pulse duration control
        pulse_duration_layout = QHBoxLayout()
        pulse_duration_layout.setSpacing(6)

        pulse_label = QLabel("Pulse duration:")
        pulse_label.setObjectName("volumeLabel")

        self.pulse_duration_spinbox = QDoubleSpinBox()
        self.pulse_duration_spinbox.setRange(1, 200)
        self.pulse_duration_spinbox.setValue(30)
        self.pulse_duration_spinbox.setDecimals(0)
        self.pulse_duration_spinbox.setSuffix(" ms")
        self.pulse_duration_spinbox.setToolTip("Sets calibration pulse duration. Accepts values between 1 and 200 ms.")
        self.pulse_duration_spinbox.setMinimumHeight(30)
        # noinspection PyUnresolvedReferences
        self.pulse_duration_spinbox.valueChanged.connect(self._update_pulse_duration)

        pulse_duration_layout.addWidget(pulse_label)
        pulse_duration_layout.addWidget(self.pulse_duration_spinbox)

        # Calibrate button
        self.calibrate_btn = QPushButton("📊 Calibrate")
        self.calibrate_btn.setToolTip("Run valve calibration with specified pulse duration")
        # noinspection PyUnresolvedReferences
        self.calibrate_btn.clicked.connect(self._calibrate)
        self.calibrate_btn.setObjectName("calibrateButton")
        self.calibrate_btn.setMinimumHeight(35)
        self.calibrate_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        pulse_duration_layout.addWidget(self.calibrate_btn)

        calibration_layout.addLayout(pulse_duration_layout)

        # Calibration status label
        self.calibration_status_label = QLabel("Calibration: Idle")
        self.calibration_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_status_label.setFont(status_font)
        self.calibration_status_label.setStyleSheet("QLabel { color: #7f8c8d; font-weight: bold; }")
        calibration_layout.addWidget(self.calibration_status_label)

        main_layout.addWidget(calibration_group)

        # Brake Control Group
        brake_group = QGroupBox("Brake Control")
        brake_layout = QVBoxLayout(brake_group)
        brake_layout.setSpacing(6)

        brake_buttons_layout = QHBoxLayout()

        self.brake_lock_btn = QPushButton("🔒 Lock Brake")
        self.brake_lock_btn.setToolTip("Lock the wheel brake")
        # noinspection PyUnresolvedReferences
        self.brake_lock_btn.clicked.connect(self._brake_lock)
        self.brake_lock_btn.setObjectName("brakeLockButton")

        self.brake_unlock_btn = QPushButton("🔓 Unlock Brake")
        self.brake_unlock_btn.setToolTip("Unlock the wheel brake")
        # noinspection PyUnresolvedReferences
        self.brake_unlock_btn.clicked.connect(self._brake_unlock)
        self.brake_unlock_btn.setObjectName("brakeUnlockButton")

        for button in [self.brake_lock_btn, self.brake_unlock_btn]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            brake_buttons_layout.addWidget(button)

        brake_layout.addLayout(brake_buttons_layout)

        # Brake status - starts locked
        self.brake_status_label = QLabel("Brake Status: 🔒 Locked")
        self.brake_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.brake_status_label.setFont(status_font)
        self.brake_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")
        brake_layout.addWidget(self.brake_status_label)

        main_layout.addWidget(brake_group)

        # Gas Puff Valve Control Group
        gas_valve_group = QGroupBox("Gas Puff Valve Control")
        gas_valve_layout = QVBoxLayout(gas_valve_group)
        gas_valve_layout.setSpacing(6)

        # Open/Close buttons
        gas_valve_buttons_layout = QHBoxLayout()

        self.gas_valve_open_btn = QPushButton("🔓 Open")
        self.gas_valve_open_btn.setToolTip("Open the gas puff valve")
        # noinspection PyUnresolvedReferences
        self.gas_valve_open_btn.clicked.connect(self._gas_valve_open)
        self.gas_valve_open_btn.setObjectName("valveOpenButton")

        self.gas_valve_close_btn = QPushButton("🔒 Close")
        self.gas_valve_close_btn.setToolTip("Close the gas puff valve")
        # noinspection PyUnresolvedReferences
        self.gas_valve_close_btn.clicked.connect(self._gas_valve_close)
        self.gas_valve_close_btn.setObjectName("valveCloseButton")

        for button in [self.gas_valve_open_btn, self.gas_valve_close_btn]:
            button.setMinimumHeight(35)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            gas_valve_buttons_layout.addWidget(button)

        gas_valve_layout.addLayout(gas_valve_buttons_layout)

        # Puff duration control and puff button
        puff_layout = QHBoxLayout()
        puff_layout.setSpacing(6)

        # Duration control on the left (same width as Open button)
        puff_sub_layout = QHBoxLayout()
        puff_sub_layout.setSpacing(6)
        gas_puff_label = QLabel("Puff duration:")
        gas_puff_label.setObjectName("volumeLabel")

        self.gas_puff_duration_spinbox = QDoubleSpinBox()
        self.gas_puff_duration_spinbox.setRange(10, 350)
        self.gas_puff_duration_spinbox.setValue(100)
        self.gas_puff_duration_spinbox.setDecimals(0)
        self.gas_puff_duration_spinbox.setSuffix(" ms")
        self.gas_puff_duration_spinbox.setToolTip("Sets gas puff duration. Accepts values between 10 and 350 ms.")
        self.gas_puff_duration_spinbox.setMinimumHeight(35)
        # noinspection PyUnresolvedReferences
        self.gas_puff_duration_spinbox.valueChanged.connect(self._update_gas_puff_duration)

        puff_sub_layout.addWidget(gas_puff_label)
        puff_sub_layout.addWidget(self.gas_puff_duration_spinbox)

        # Puff button on the right (same width as Close button)
        self.gas_valve_puff_btn = QPushButton("💨 Puff")
        self.gas_valve_puff_btn.setToolTip("Deliver a gas puff with specified duration")
        # noinspection PyUnresolvedReferences
        self.gas_valve_puff_btn.clicked.connect(self._gas_valve_puff)
        self.gas_valve_puff_btn.setObjectName("rewardButton")
        self.gas_valve_puff_btn.setMinimumHeight(35)

        puff_layout.addLayout(puff_sub_layout, stretch=1)
        puff_layout.addWidget(self.gas_valve_puff_btn, stretch=1)

        gas_valve_layout.addLayout(puff_layout)

        # Gas valve status
        self.gas_valve_status_label = QLabel("Valve: Closed")
        self.gas_valve_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gas_valve_status_label.setFont(status_font)
        self.gas_valve_status_label.setStyleSheet("QLabel { color: #7f8c8d; font-weight: bold; }")
        gas_valve_layout.addWidget(self.gas_valve_status_label)

        main_layout.addWidget(gas_valve_group)

        # Terminate Button
        self.terminate_btn = QPushButton("✖ Terminate Maintenance")
        self.terminate_btn.setToolTip("Gracefully end the maintenance runtime")
        # noinspection PyUnresolvedReferences
        self.terminate_btn.clicked.connect(self._terminate_runtime)
        self.terminate_btn.setObjectName("exitButton")
        self.terminate_btn.setMinimumHeight(40)

        main_layout.addWidget(self.terminate_btn)

    def _apply_qt6_styles(self) -> None:
        """Applies optimized styling to all UI elements."""
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

            QPushButton:disabled {
                background-color: #ecf0f1;
                color: #95a5a6;
                border-color: #bdc3c7;
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

            QPushButton#exitButton:disabled {
                background-color: #ecf0f1;
                color: #95a5a6;
                border-color: #bdc3c7;
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

            QPushButton#referenceButton {
                background-color: #9b59b6;
                color: white;
                border-color: #8e44ad;
                font-weight: bold;
            }

            QPushButton#referenceButton:hover {
                background-color: #8e44ad;
                border-color: #7d3c98;
            }

            QPushButton#calibrateButton {
                background-color: #16a085;
                color: white;
                border-color: #138d75;
                font-weight: bold;
            }

            QPushButton#calibrateButton:hover {
                background-color: #138d75;
                border-color: #117a65;
            }

            QPushButton#brakeLockButton {
                background-color: #e74c3c;
                color: white;
                border-color: #c0392b;
                font-weight: bold;
            }

            QPushButton#brakeLockButton:hover {
                background-color: #c0392b;
                border-color: #a93226;
            }

            QPushButton#brakeUnlockButton {
                background-color: #27ae60;
                color: white;
                border-color: #229954;
                font-weight: bold;
            }

            QPushButton#brakeUnlockButton:hover {
                background-color: #229954;
                border-color: #1e8449;
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
        """)

    def _setup_monitoring(self) -> None:
        """Sets up a QTimer to monitor the runtime termination status and valve's state."""
        self.monitor_timer = QTimer(self)
        # noinspection PyUnresolvedReferences
        self.monitor_timer.timeout.connect(self._check_external_state)
        self.monitor_timer.start(100)  # Check every 100ms

    def _check_external_state(self) -> None:
        """Checks for external termination signal and updates valve, calibration, and gas puff status."""
        # noinspection PyBroadException
        try:
            # Checks for termination.
            if self._data_array[_DataArrayIndex.TERMINATION] == 1:
                self.close()

            # Reads valve tracker state (index 1 = calibration state, index 2 = open/close state).
            is_calibrating = float(self._valve_tracker[1]) == 0.0
            water_valve_state = int(self._valve_tracker[2])

            # Reads gas puff tracker state (index 1 = open/close state).
            gas_valve_state = int(self._gas_puff_tracker[1])

            # Detects when water valve closes (state transitions to closed while reward was in progress).
            if self._reward_in_progress and water_valve_state == 0:
                self._reward_in_progress = False
                self.valve_status_label.setText("Valve: Closed")
                self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

            # Detects when calibration or referencing completes (valve_tracker indicates no longer calibrating).
            if (self._calibration_in_progress or self._referencing_in_progress) and not is_calibrating:
                self._calibration_in_progress = False
                self._referencing_in_progress = False
                self.calibration_status_label.setText("Calibration: Idle")
                self.calibration_status_label.setStyleSheet("QLabel { color: #7f8c8d; font-weight: bold; }")

            # Detects when gas puff delivery completes (state transitions to closed while puff was in progress).
            if self._puff_in_progress and gas_valve_state == 0:
                self._puff_in_progress = False
                self.gas_valve_status_label.setText("Valve: Closed")
                self.gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

        except Exception:
            self.close()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        """Handles GUI window close events."""
        with contextlib.suppress(Exception):
            self._data_array[_DataArrayIndex.TERMINATION] = 1
        if event is not None:
            event.accept()

    def _update_reward_volume(self) -> None:
        """Updates the volume used when delivering water rewards to match the current GUI configuration."""
        self._data_array[_DataArrayIndex.REWARD_VOLUME] = int(self.volume_spinbox.value())

    def _update_pulse_duration(self) -> None:
        """Updates the calibration pulse duration to match the current GUI configuration."""
        self._data_array[_DataArrayIndex.CALIBRATION_PULSE_DURATION] = int(self.pulse_duration_spinbox.value())

    def _valve_open(self) -> None:
        """Signals to open the valve."""
        self._data_array[_DataArrayIndex.VALVE_OPEN] = 1
        self.valve_status_label.setText("Valve: Open")
        self.valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _valve_close(self) -> None:
        """Signals to close the valve."""
        self._data_array[_DataArrayIndex.VALVE_CLOSE] = 1
        self.valve_status_label.setText("Valve: Closed")
        self.valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _valve_reward(self) -> None:
        """Signals to deliver a water reward."""
        self._data_array[_DataArrayIndex.VALVE_REWARD] = 1
        self._reward_in_progress = True
        self.valve_status_label.setText("Valve: Delivering Reward")
        self.valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")

    def _valve_reference(self) -> None:
        """Signals to run the valve referencing procedure."""
        self._data_array[_DataArrayIndex.VALVE_REFERENCE] = 1
        self._referencing_in_progress = True
        self.calibration_status_label.setText("Calibration: Referencing")
        self.calibration_status_label.setStyleSheet("QLabel { color: #9b59b6; font-weight: bold; }")

    def _calibrate(self) -> None:
        """Signals to run the valve calibration procedure for the currently set pulse duration."""
        self._data_array[_DataArrayIndex.VALVE_CALIBRATE] = 1
        self._calibration_in_progress = True
        self.calibration_status_label.setText("Calibration: Calibrating")
        self.calibration_status_label.setStyleSheet("QLabel { color: #16a085; font-weight: bold; }")

    def _brake_lock(self) -> None:
        """Signals to lock the brake."""
        self._data_array[_DataArrayIndex.BRAKE_LOCK] = 1
        self.brake_status_label.setText("Brake: 🔒 Locked")
        self.brake_status_label.setStyleSheet("QLabel { color: #e74c3c; font-weight: bold; }")

    def _brake_unlock(self) -> None:
        """Signals to unlock the brake."""
        self._data_array[_DataArrayIndex.BRAKE_UNLOCK] = 1
        self.brake_status_label.setText("Brake: 🔓 Unlocked")
        self.brake_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _update_gas_puff_duration(self) -> None:
        """Updates the gas puff duration to match the current GUI configuration."""
        self._data_array[_DataArrayIndex.GAS_VALVE_PULSE_DURATION] = int(self.gas_puff_duration_spinbox.value())

    def _gas_valve_open(self) -> None:
        """Signals to open the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_OPEN] = 1
        self.gas_valve_status_label.setText("Valve: Open")
        self.gas_valve_status_label.setStyleSheet("QLabel { color: #27ae60; font-weight: bold; }")

    def _gas_valve_close(self) -> None:
        """Signals to close the gas puff valve."""
        self._data_array[_DataArrayIndex.GAS_VALVE_CLOSE] = 1
        self.gas_valve_status_label.setText("Valve: Closed")
        self.gas_valve_status_label.setStyleSheet("QLabel { color: #e67e22; font-weight: bold; }")

    def _gas_valve_puff(self) -> None:
        """Signals to deliver a gas puff."""
        self._data_array[_DataArrayIndex.GAS_VALVE_PULSE] = 1
        self._puff_in_progress = True
        self.gas_valve_status_label.setText("Valve: Puffing")
        self.gas_valve_status_label.setStyleSheet("QLabel { color: #3498db; font-weight: bold; }")

    def _terminate_runtime(self) -> None:
        """Signals to terminate the maintenance runtime."""
        self._data_array[_DataArrayIndex.TERMINATION] = 1
        self.terminate_btn.setText("✖ Termination Requested")
        self.terminate_btn.setEnabled(False)
