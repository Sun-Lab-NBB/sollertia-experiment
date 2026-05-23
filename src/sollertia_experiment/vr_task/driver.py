"""Provides the VRTaskDriver class that encapsulates all communication with the Unity game engine used to run the
Virtual Reality task.
"""

from __future__ import annotations

import json
from enum import IntEnum, StrEnum
from json import dumps
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_communication_interface import MQTTCommunication

from .trial_decomposition import DecomposedTrials, CachedMotifDecomposer, decompose_cue_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sollertia_shared_assets import TaskTemplate, TriggerType

    from .configuration import VRTaskConfiguration


_SETUP_POLLING_DELAY_MS: int = 10
"""The delay, in milliseconds, between consecutive Unity buffer polls during the interactive setup sequence."""

_DISPLAY_ANIMATION_STEP_DELAY_MS: int = 100
"""The delay, in milliseconds, between consecutive position updates sent to Unity during the VR display verification
animation."""

_DISPLAY_ANIMATION_STEP_UNITS: float = 0.1
"""The size, in Unity units, of each position update sent during the VR display verification animation."""

_DISPLAY_SCREENS_WARMUP_DELAY_MS: int = 2000
"""The delay, in milliseconds, applied before driving the display verification animation, allowing the
caller-enabled VR screens to settle before they are required to render."""

_CUE_SEQUENCE_RESPONSE_TIMEOUT_MS: int = 5000
"""The maximum time, in milliseconds, to wait for Unity to respond to a cue sequence request before retrying."""


class VRTaskEventKind(IntEnum):
    """Defines the kinds of Virtual Reality task events produced by the VRTaskDriver per runtime cycle.

    Notes:
        This enumeration intentionally covers only the asynchronous Unity messages surfaced by cycle() for the
        caller to dispatch. Messages that arrive as synchronous handshake replies are resolved internally by the
        driver and are deliberately excluded.
    """

    NONE = 0
    """No Unity message was available in the MQTT buffer during this cycle."""
    STIMULUS_TRIGGERED = 1
    """The animal triggered the current trial's stimulus delivery."""
    TRIGGER_DELAY_REQUESTED = 2
    """Unity requested the acquisition system to apply a brake pulse for the specified duration."""
    UNITY_TERMINATED = 3
    """Unity reported that its runtime has been terminated; the acquisition system must enter an emergency pause."""


@dataclass(frozen=True, slots=True)
class VRTaskEvent:
    """Stores the parsed Unity message produced by VRTaskDriver.cycle() during a single runtime cycle."""

    kind: VRTaskEventKind
    """The kind of Virtual Reality task event that occurred during the cycle."""
    delay_ms: int = 0
    """The brake pulse duration in milliseconds. Populated only for the TRIGGER_DELAY_REQUESTED event."""


@dataclass(slots=True)
class VRTaskState:
    """Tracks the runtime state of the Virtual Reality task environment managed by the Unity game engine.

    This dataclass consolidates all Unity-related state tracking attributes used by the VRTaskDriver to monitor the
    Virtual Reality environment state, manage task guidance modes, and facilitate communication between the
    acquisition system and the Unity game engine over MQTT.
    """

    position: np.float64 = field(default_factory=lambda: np.float64(0.0))
    """The current absolute position of the animal, in Unity units, relative to the origin of the Virtual Reality task
    environment's track."""
    cue_sequence: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    """The sequence of Virtual Reality environment wall cues used by the session's task environment. This array defines
    the visual cues displayed to the animal as it progresses through the virtual track."""
    terminated: bool = False
    """Tracks whether the system has detected that the Unity game engine has unexpectedly terminated its runtime. When
    True, the runtime enters an emergency pause state to allow the user to restart Unity."""
    reinforcing_guidance_enabled: bool = False
    """Tracks the state of the reinforcing trial guidance mode."""
    aversive_guidance_enabled: bool = False
    """Tracks the state of the aversive trial guidance mode."""


class _VRTaskMQTTTopics(StrEnum):
    """Defines the set of MQTT topics used to communicate with the Unity game engine that runs the Virtual Reality
    task environment.

    Notes:
        The catalog mirrors the flat PascalCase contract published by sollertia-unity-tasks' MQTTTopics constant
        set.
    """

    SESSION_START = "SessionStart"
    """Lifecycle marker published by Unity when its MQTT client starts (empty trigger payload)."""
    SESSION_STOP = "SessionStop"
    """Lifecycle marker published by Unity on application quit (empty trigger payload)."""
    MOTION = "Motion"
    """Treadmill movement payload sent from the acquisition runtime to Unity (TreadmillMessage with float
    movement)."""
    LICK = "Lick"
    """Lick-port event published by the acquisition runtime when the animal licks the spout (empty trigger
    payload)."""
    STIMULUS = "Stimulus"
    """Stimulus delivery event published by Unity when a stimulus trigger zone fires (empty trigger payload)."""
    DELAY = "Delay"
    """Brake activation request published by Unity carrying the remaining occupancy duration in milliseconds
    (TriggerDelayMessage with uint delayMilliseconds)."""
    CUE_SEQUENCE_TRIGGER = "CueSequenceTrigger"
    """Request for the active task's flattened cue sequence (empty trigger payload)."""
    CUE_SEQUENCE = "CueSequence"
    """Flattened cue sequence reply sent by Unity in response to CueSequenceTrigger (SequenceMessage with byte
    array cueSequence)."""
    SCENE_NAME_TRIGGER = "SceneNameTrigger"
    """Request for the active Unity scene name (empty trigger payload)."""
    SCENE_NAME = "SceneName"
    """Active Unity scene name reply sent in response to SceneNameTrigger (SceneNameMessage with string name)."""
    REQUIRE_LICK = "RequireLick"
    """Lick-requirement toggle published by the acquisition runtime (BoolMessage with bool value)."""
    REQUIRE_WAIT = "RequireWait"
    """Wait-requirement toggle published by the acquisition runtime (BoolMessage with bool value)."""


class VRTaskDriver:
    """Drives the Unity game engine that runs the Virtual Reality task implemented in sollertia-unity-tasks.

    Encapsulates the MQTT contract with Unity: connection lifecycle, scene handshake, VR display verification, wall
    cue sequence retrieval, per-cycle stimulus pump, guidance toggling, and resume-after-Unity-restart. The driver
    is hardware-agnostic; per-cycle Unity events are surfaced as typed VRTaskEvent values that the caller dispatches
    to acquisition system hardware.

    Args:
        configuration: The runtime configuration that defines the MQTT broker discovery fields.
        task_template: The VR TaskTemplate that defines the cue catalog, corridor geometry, per-trial spatial cue
            sequences, and per-trial trigger types for the active Unity scene.
        expected_scene_name: The Unity scene name the driver enforces during the setup handshake.

    Attributes:
        _configuration: The VRTaskConfiguration instance that defines the MQTT broker discovery fields.
        _task_template: The VR TaskTemplate consumed during cue sequence decomposition.
        _expected_scene_name: The Unity scene name enforced during the setup handshake.
        _mqtt: The MQTTCommunication instance that bidirectionally transfers data between this driver and Unity.
        _state: The VRTaskState instance that tracks the Virtual Reality task environment state.
        _motif_decomposer: The CachedMotifDecomposer used to flatten and cache trial motif data between decomposition
            runs.
        _decomposed_trials: The DecomposedTrials produced by the most recent cue sequence decomposition.
        _polling_timer: The PrecisionTimer used to delay between consecutive Unity buffer polls during the setup
            sequence.
    """

    def __init__(
        self,
        configuration: VRTaskConfiguration,
        *,
        task_template: TaskTemplate,
        expected_scene_name: str,
    ) -> None:
        self._configuration: VRTaskConfiguration = configuration
        self._task_template: TaskTemplate = task_template
        self._expected_scene_name: str = expected_scene_name

        monitored_topics: tuple[_VRTaskMQTTTopics, ...] = (
            _VRTaskMQTTTopics.CUE_SEQUENCE,
            _VRTaskMQTTTopics.SESSION_STOP,
            _VRTaskMQTTTopics.SESSION_START,
            _VRTaskMQTTTopics.SCENE_NAME,
            _VRTaskMQTTTopics.STIMULUS,
            _VRTaskMQTTTopics.DELAY,
        )
        self._mqtt: MQTTCommunication = MQTTCommunication(
            ip=configuration.ip,
            port=configuration.port,
            monitored_topics=monitored_topics,
        )

        self._state: VRTaskState = VRTaskState()
        self._motif_decomposer: CachedMotifDecomposer = CachedMotifDecomposer()
        self._decomposed_trials: DecomposedTrials = DecomposedTrials(
            cumulative_distances=np.zeros(0, dtype=np.float64),
            trial_names=(),
            trigger_types=(),
        )
        self._polling_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    @property
    def state(self) -> VRTaskState:
        """Returns the current Virtual Reality task state tracked by the driver."""
        return self._state

    @property
    def configuration(self) -> VRTaskConfiguration:
        """Returns the runtime configuration used by the driver."""
        return self._configuration

    @property
    def cue_sequence_distances(self) -> NDArray[np.float64]:
        """Returns the cumulative distances, in centimeters, the animal must travel to complete each decomposed
        trial.
        """
        return self._decomposed_trials.cumulative_distances

    @property
    def trial_names(self) -> tuple[str, ...]:
        """Returns the name of each decomposed trial, in sequence order."""
        return self._decomposed_trials.trial_names

    @property
    def trigger_types(self) -> tuple[TriggerType, ...]:
        """Returns the stimulus trigger type of each decomposed trial, in sequence order."""
        return self._decomposed_trials.trigger_types

    def connect(self) -> None:
        """Establishes the MQTT connection to the Unity game engine."""
        self._mqtt.connect()

    def disconnect(self) -> None:
        """Closes the MQTT connection to the Unity game engine."""
        self._mqtt.disconnect()

    def setup(self) -> None:
        """Carries out the interactive Unity setup sequence used at the start of a session.

        Notes:
            Verifies that the Unity scene matches the expected scene name, drives a display verification loop on the
            VR screens, re-arms Unity, and requests the wall cue sequence used by the task. The setup sequence is
            interactive and prompts the user via console messages when a step requires confirmation or retry.

            The caller must enable the VR screens before invoking this method and disable them once it returns, as
            the display verification stage relies on the screens rendering the verification animation.
        """
        self._verify_scene_name()
        self._verify_vr_display()
        self._rearm_unity()
        self.refresh_cue_sequence()
        console.echo(message="Unity setup: Complete.", level=LogLevel.SUCCESS)

    def refresh_cue_sequence(self) -> None:
        """Requests and resolves the Virtual Reality wall cue sequence used by the current Unity scene.

        Notes:
            Re-fetches the cue sequence from Unity, decomposes it into per-trial cumulative distances and stimulus
            parameters, and resets the driver's tracked Unity position to the origin.

        Raises:
            RuntimeError: If the decomposer cannot match any trial motif at some position in the cue sequence.
        """
        self._clear_buffer()
        message = (
            "Requesting Virtual Reality wall cue sequence from Unity. Ensure Unity is armed and the task is running."
        )
        console.echo(message=message, level=LogLevel.INFO)

        while True:
            self._mqtt.send_data(topic=_VRTaskMQTTTopics.CUE_SEQUENCE_TRIGGER)
            self._polling_timer.reset()

            received = False
            while self._polling_timer.elapsed < _CUE_SEQUENCE_RESPONSE_TIMEOUT_MS:
                self._polling_timer.delay(delay=_SETUP_POLLING_DELAY_MS, block=False)
                data = self._mqtt.get_data()
                if data is None:
                    continue
                if data[0] != _VRTaskMQTTTopics.CUE_SEQUENCE:
                    continue

                cue_sequence: NDArray[np.uint8] = np.array(
                    json.loads(data[1].decode("utf-8"))["cueSequence"], dtype=np.uint8
                )
                self._state.cue_sequence = cue_sequence

                self._decomposed_trials = decompose_cue_sequence(
                    cue_sequence=cue_sequence,
                    task_template=self._task_template,
                    motif_decomposer=self._motif_decomposer,
                )

                self._state.position = np.float64(0.0)

                console.echo(message="VR cue sequence: Received.", level=LogLevel.SUCCESS)
                received = True
                break

            if received:
                return

            message = (
                f"The Virtual Reality task driver sent a cue sequence request to Unity via the "
                f"'{_VRTaskMQTTTopics.CUE_SEQUENCE_TRIGGER}' topic but received no response within "
                f"{_CUE_SEQUENCE_RESPONSE_TIMEOUT_MS // 1000} seconds. Ensure Unity is armed and the task is running."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

    def push_position(self, absolute_position: np.float64) -> None:
        """Forwards the latest animal position to Unity as a movement delta.

        Notes:
            The driver internally tracks the last position reported to Unity and only emits an MQTT message when the
            position has changed.

        Args:
            absolute_position: The current absolute position of the animal, in Unity units, relative to the runtime
                onset.
        """
        delta = absolute_position - self._state.position
        if delta != 0:
            self._state.position = absolute_position
            payload = dumps(obj={"movement": float(delta)}).encode("utf-8")
            self._mqtt.send_data(topic=_VRTaskMQTTTopics.MOTION, payload=payload)

    def push_lick_event(self) -> None:
        """Notifies Unity that the animal performed a lick."""
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.LICK, payload=None)

    def set_reinforcing_guidance(self, *, enabled: bool) -> None:
        """Sets the reinforcing trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable reinforcing guidance.
        """
        payload = dumps(obj={"value": enabled}).encode("utf-8")
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.REQUIRE_LICK, payload=payload)
        self._state.reinforcing_guidance_enabled = enabled

    def set_aversive_guidance(self, *, enabled: bool) -> None:
        """Sets the aversive trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable aversive guidance.
        """
        payload = dumps(obj={"value": enabled}).encode("utf-8")
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.REQUIRE_WAIT, payload=payload)
        self._state.aversive_guidance_enabled = enabled

    def cycle(self) -> VRTaskEvent:
        """Consumes the next pending Unity message and returns it as a typed event.

        Notes:
            During each runtime cycle, the driver receives and parses exactly one message stored in the MQTT
            buffer. Callers dispatch the returned event to their own hardware (water valve, gas puff, brake) and
            runtime state (trial advancement, emergency pause).

        Returns:
            The VRTaskEvent describing the Unity message that was just parsed. When the MQTT buffer is empty, the
            event kind is NONE.
        """
        data = self._mqtt.get_data()
        if data is None:
            return VRTaskEvent(kind=VRTaskEventKind.NONE)

        topic, payload = data

        if topic == _VRTaskMQTTTopics.STIMULUS:
            return VRTaskEvent(kind=VRTaskEventKind.STIMULUS_TRIGGERED)

        if topic == _VRTaskMQTTTopics.DELAY:
            delay_payload = json.loads(payload.decode("utf-8"))
            delay_ms = int(delay_payload.get("delayMilliseconds", 0))
            return VRTaskEvent(kind=VRTaskEventKind.TRIGGER_DELAY_REQUESTED, delay_ms=delay_ms)

        if topic == _VRTaskMQTTTopics.SESSION_STOP:
            self._state.terminated = True
            return VRTaskEvent(kind=VRTaskEventKind.UNITY_TERMINATED)

        return VRTaskEvent(kind=VRTaskEventKind.NONE)

    def resume_after_unity_restart(self) -> None:
        """Re-fetches the cue sequence after Unity has been restarted and resets the termination flag.

        Notes:
            When the Unity game cycles, it resets the sequence of VR wall cues. This method re-queries the new wall
            cue sequence to enable accurate tracking of the animal's position in VR after the reset.
        """
        self.refresh_cue_sequence()
        self._state.terminated = False

    def _verify_scene_name(self) -> None:
        """Verifies that the Unity scene matches the expected scene name with infinite retry on mismatch."""
        while True:
            self._clear_buffer()
            message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
            console.echo(message=message, level=LogLevel.INFO)

            self._wait_for_topic(expected_topic=_VRTaskMQTTTopics.SESSION_START)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

            message = "Verifying that the Unity game engine is configured to display the correct scene..."
            console.echo(message=message, level=LogLevel.INFO)

            self._mqtt.send_data(topic=_VRTaskMQTTTopics.SCENE_NAME_TRIGGER)
            payload = self._wait_for_topic(expected_topic=_VRTaskMQTTTopics.SCENE_NAME)
            scene_name: str = json.loads(payload.decode("utf-8"))["name"]

            if scene_name == self._expected_scene_name:
                message = "Unity scene configuration: Confirmed."
                console.echo(message=message, level=LogLevel.SUCCESS)
                return

            message = (
                f"The name of the Virtual Reality scene (task) running in Unity ({scene_name}) does not match the "
                f"scene name expected based on the session's experiment configuration ({self._expected_scene_name}). "
                f"Reconfigure Unity to run the correct VR task and try again."
            )
            console.echo(message=message, level=LogLevel.ERROR)
            input("Enter anything to retry: ")

    def _verify_vr_display(self) -> None:
        """Drives the VR display verification loop until the user confirms the scene renders correctly."""
        self._polling_timer.delay(delay=_DISPLAY_SCREENS_WARMUP_DELAY_MS, block=False)

        message = (
            "Verify that the Virtual Reality scene displays on the VR screens as intended. Disable (end) Unity "
            "runtime when ready to advance to the next preparation step."
        )
        console.echo(message=message, level=LogLevel.INFO)

        while True:
            self._animate_until_termination()

            message = "Did the Virtual Reality display render correctly on the VR screens?"
            console.echo(message=message, level=LogLevel.INFO)

            answer = ""
            while answer not in {"y", "n"}:
                user_input = input("Enter 'yes' or 'no': ").strip().lower()
                answer = user_input[0] if user_input else ""

            if answer == "y":
                return

            message = (
                "Restarting VR display verification. Ensure Unity is properly configured and arm the task "
                "to begin the verification."
            )
            console.echo(message=message, level=LogLevel.WARNING)

            self._clear_buffer()
            message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
            console.echo(message=message, level=LogLevel.INFO)
            self._wait_for_topic(expected_topic=_VRTaskMQTTTopics.SESSION_START)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

    def _animate_until_termination(self) -> None:
        """Sends incremental position updates to Unity until Unity reports termination."""
        while True:
            self._polling_timer.delay(delay=_DISPLAY_ANIMATION_STEP_DELAY_MS, block=False)

            payload = dumps(obj={"movement": _DISPLAY_ANIMATION_STEP_UNITS}).encode("utf-8")
            self._mqtt.send_data(topic=_VRTaskMQTTTopics.MOTION, payload=payload)

            data = self._mqtt.get_data()
            if data is None:
                continue

            if data[0] == _VRTaskMQTTTopics.SESSION_STOP:
                message = "Unity termination: Detected."
                console.echo(message=message, level=LogLevel.INFO)
                return

    def _rearm_unity(self) -> None:
        """Re-arms Unity after successful display verification."""
        self._clear_buffer()
        message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
        console.echo(message=message, level=LogLevel.INFO)
        self._wait_for_topic(expected_topic=_VRTaskMQTTTopics.SESSION_START)
        message = "Unity state transition: Confirmed. Unity is now armed."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _wait_for_topic(self, expected_topic: str) -> bytes | bytearray:
        """Blocks until Unity sends a message on the specified topic and returns the payload."""
        while True:
            self._polling_timer.delay(delay=_SETUP_POLLING_DELAY_MS, block=False)
            data = self._mqtt.get_data()
            if data is not None:
                topic, payload = data
                if topic == expected_topic:
                    return payload

    def _clear_buffer(self) -> None:
        """Drains all pending messages from the MQTT buffer used to communicate with Unity."""
        while self._mqtt.has_data:
            _ = self._mqtt.get_data()
