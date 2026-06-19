"""Provides the VRTaskDriver class that encapsulates all communication with the Unity game engine used to run the
Virtual Reality task.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
import json
from typing import TYPE_CHECKING
import threading
from dataclasses import field, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_communication_interface import MQTTCommunication

from .bridge import UnityBridgeError, UnityBridgeClient
from ..cross_system import wait_for_enter
from .trial_decomposition import DecomposedTrials, CachedMotifDecomposer, decompose_cue_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sollertia_shared_assets import TaskTemplate

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

_PLAY_MODE_READINESS_TIMEOUT_MS: int = 60000
"""The maximum time, in milliseconds, to wait for Unity to publish the SessionStart message after the driver requests
Play Mode entry through the bridge. The window is generous because entering Play Mode can trigger a Unity script
recompile before the scene's MQTT client connects."""

_SESSION_STOP_TIMEOUT_MS: int = 5000
"""The maximum time, in milliseconds, to wait for Unity to publish the SessionStop message after the driver requests
Play Mode exit through the bridge."""


class VRTaskEventKind(IntEnum):
    """Defines the kinds of Virtual Reality task events produced by the VRTaskDriver per runtime cycle.

    Notes:
        This enumeration intentionally covers only the asynchronous Unity messages surfaced by cycle() for the
        caller to dispatch. Messages that arrive as synchronous handshake replies are resolved internally by the
        driver and are deliberately excluded.
    """

    NONE = 0
    """No dispatchable Unity event was produced this cycle: either the MQTT buffer was empty, or the consumed message
    was on a topic that cycle() does not surface (a handshake topic)."""
    STIMULUS_TRIGGERED = 1
    """A trial resolved its stimulus outcome. The event's trial_name, delivered, and cause carry the result."""
    TRIGGER_DELAY_REQUESTED = 2
    """Unity requested the acquisition system to apply a brake pulse for the specified duration."""
    UNITY_TERMINATED = 3
    """Unity reported that its runtime has been terminated; the acquisition system must enter an emergency pause."""


class StimulusCause(StrEnum):
    """Defines the cause of a trial's stimulus outcome as reported by Unity.

    Unity stamps this on every Stimulus message so the acquisition system can distinguish a self-driven success
    from a guidance-driven outcome without knowing the appetitive-or-aversive stimulus valence.
    """

    BEHAVIOR = "behavior"
    """The animal's own action produced the outcome."""
    GUIDANCE = "guidance"
    """The guidance fallback produced the outcome."""


@dataclass(frozen=True, slots=True)
class VRTaskEvent:
    """Stores the parsed Unity message produced by VRTaskDriver.cycle() during a single runtime cycle."""

    kind: VRTaskEventKind
    """The kind of Virtual Reality task event that occurred during the cycle."""
    delay_ms: int = 0
    """The brake pulse duration in milliseconds. Populated only for the TRIGGER_DELAY_REQUESTED event."""
    trial_name: str = ""
    """The name of the trial that resolved. Populated only for the STIMULUS_TRIGGERED event."""
    delivered: bool = True
    """Determines whether the trial's physical stimulus was delivered. Populated only for the STIMULUS_TRIGGERED
    event; defaults to True so payloads predating the field resolve as a delivery."""
    cause: StimulusCause = StimulusCause.BEHAVIOR
    """The cause of the trial outcome: the animal's behavior or the guidance fallback. Populated only for the
    STIMULUS_TRIGGERED event."""


@dataclass(slots=True)
class VRTaskState:
    """Tracks the runtime state of the Virtual Reality task environment managed by the Unity game engine.

    Notes:
        This is the single source of truth shared between the per-cycle Unity events surfaced by cycle() and the
        interactive setup handshake.
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
    """Lifecycle marker published by Unity when its MQTT client starts (empty trigger payload). Outside the setup
    handshake, a SESSION_START received during cycle() is consumed without producing a typed event."""
    SESSION_STOP = "SessionStop"
    """Lifecycle marker published by Unity on application quit (empty trigger payload)."""
    MOTION = "Motion"
    """Treadmill movement payload sent from the acquisition runtime to Unity (TreadmillMessage with float
    movement)."""
    INTERACTION = "Interaction"
    """Sensor-interaction event published by the acquisition runtime when the animal engages an interaction sensor
    (empty trigger payload). The Mesoscope-VR system resolves this generic event to the lick sensor."""
    STIMULUS = "Stimulus"
    """Trial outcome event published by Unity when a stimulus trigger zone resolves a trial (StimulusMessage with
    string trialName, bool delivered, and string cause reporting whether the physical stimulus fired and whether
    the animal's behavior or the guidance fallback produced the outcome)."""
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
    REQUIRE_INTERACTION = "RequireInteraction"
    """Interaction-requirement toggle published by the acquisition runtime (BoolMessage with bool value)."""
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
        _bridge: The UnityBridgeClient used to drive Unity scene activation and Play Mode through the editor MCP
            Bridge.
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
        )
        self._polling_timer: PrecisionTimer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
        self._bridge: UnityBridgeClient = UnityBridgeClient()

    def __repr__(self) -> str:
        """Returns a string representation of the VRTaskDriver instance."""
        return (
            f"VRTaskDriver(expected_scene_name={self._expected_scene_name}, "
            f"ip={self._configuration.ip}, port={self._configuration.port})"
        )

    @property
    def state(self) -> VRTaskState:
        """Returns the current Virtual Reality task state tracked by the driver."""
        return self._state

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

    def connect(self) -> None:
        """Establishes the MQTT connection to the Unity game engine."""
        self._mqtt.connect()

    def disconnect(self) -> None:
        """Stops the active Unity scene and closes the MQTT and bridge connections to the Unity game engine.

        Notes:
            Exits Play Mode through the bridge before tearing down the connections so the Unity scene does not keep
            running after the session ends. Stopping the scene is a best-effort step: if the bridge is unreachable
            because the editor was closed, the driver logs a warning and proceeds with the teardown.
        """
        self._stop_unity()
        self._mqtt.disconnect()
        self._bridge.close()

    def setup(self) -> None:
        """Carries out the Unity setup sequence used at the start of a session.

        Notes:
            Requires the Unity Editor MCP Bridge to be reachable, opens the expected scene, arms Unity, verifies the
            active scene matches the expected one, prompts the operator to verify the VR display, and requests the
            wall cue sequence used by the task. Scene activation and Play Mode control are issued through the bridge;
            the operator only confirms that the VR display renders correctly.

            The caller must enable the VR screens before invoking this method and disable them once it returns, as
            the display verification stage relies on the screens rendering the active scene.
        """
        self._require_bridge()
        self._activate_scene()
        self._arm_unity()
        self._verify_scene_name()
        self._verify_vr_display()
        self._refresh_cue_sequence()
        console.echo(message="Unity setup: Complete.", level=LogLevel.SUCCESS)

    def push_position(self, absolute_position: np.float64) -> None:
        """Forwards the latest animal position to Unity as a movement delta.

        Notes:
            The driver internally tracks the last absolute position passed to this method and only emits an MQTT
            message when the position has changed.

        Args:
            absolute_position: The current absolute position of the animal, in Unity units, relative to the origin of
                the Virtual Reality task environment's track.
        """
        delta = absolute_position - self._state.position
        if delta:
            self._state.position = absolute_position
            payload = json.dumps(obj={"movement": float(delta)}).encode("utf-8")
            self._mqtt.send_data(topic=_VRTaskMQTTTopics.MOTION, payload=payload)

    def push_lick_event(self) -> None:
        """Notifies Unity that the animal engaged the interaction sensor.

        Notes:
            The Mesoscope-VR system's interaction sensor is the lick port, so this method retains the lick name the
            system layer uses. It publishes on Unity's generic Interaction topic.
        """
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.INTERACTION, payload=None)

    def set_reinforcing_guidance(self, *, enabled: bool) -> None:
        """Sets the reinforcing trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable reinforcing guidance.
        """
        # Unity's RequireInteraction flag is the inverse of guidance: a True value forces the animal to interact to
        # trigger the reward, which is the unguided behavior. Enabling guidance therefore clears the requirement.
        payload = json.dumps(obj={"value": not enabled}).encode("utf-8")
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.REQUIRE_INTERACTION, payload=payload)
        self._state.reinforcing_guidance_enabled = enabled

    def set_aversive_guidance(self, *, enabled: bool) -> None:
        """Sets the aversive trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable aversive guidance.
        """
        # Unity's RequireWait flag is the inverse of guidance: a True value forces the animal to satisfy the occupancy
        # duration on its own, while the occupancy guidance zone only pulses the brake when the requirement is cleared.
        payload = json.dumps(obj={"value": not enabled}).encode("utf-8")
        self._mqtt.send_data(topic=_VRTaskMQTTTopics.REQUIRE_WAIT, payload=payload)
        self._state.aversive_guidance_enabled = enabled

    def cycle(self) -> VRTaskEvent:
        """Consumes the next pending Unity message and returns it as a typed event.

        Notes:
            During each runtime cycle, the driver consumes at most one message from the MQTT buffer per cycle.
            Callers dispatch the returned event to their own hardware (water valve, gas puff, brake) and runtime
            state (trial advancement, emergency pause).

        Returns:
            The VRTaskEvent describing the Unity message that was just parsed. When the MQTT buffer is empty, the
            event kind is NONE. The event kind is also NONE when the consumed message is on a non-surfaced
            (handshake) topic, such as SESSION_START, SCENE_NAME, or CUE_SEQUENCE.
        """
        data = self._mqtt.get_data()
        if data is None:
            return VRTaskEvent(kind=VRTaskEventKind.NONE)

        topic, payload = data

        if topic == _VRTaskMQTTTopics.STIMULUS:
            trial_name = ""
            delivered = True
            cause = StimulusCause.BEHAVIOR
            if payload is not None:
                stimulus_payload = json.loads(payload.decode("utf-8"))
                trial_name = str(stimulus_payload.get("trialName", ""))
                delivered = bool(stimulus_payload.get("delivered", True))
                # Any cause value other than the explicit guidance marker resolves to behavior, so missing or
                # unrecognized values default to the self-driven outcome.
                cause = (
                    StimulusCause.GUIDANCE
                    if stimulus_payload.get("cause") == StimulusCause.GUIDANCE.value
                    else StimulusCause.BEHAVIOR
                )
            return VRTaskEvent(
                kind=VRTaskEventKind.STIMULUS_TRIGGERED,
                trial_name=trial_name,
                delivered=delivered,
                cause=cause,
            )

        if topic == _VRTaskMQTTTopics.DELAY:
            delay_payload = json.loads(payload.decode("utf-8"))
            delay_ms = int(delay_payload.get("delayMilliseconds", 0))
            return VRTaskEvent(kind=VRTaskEventKind.TRIGGER_DELAY_REQUESTED, delay_ms=delay_ms)

        if topic == _VRTaskMQTTTopics.SESSION_STOP:
            self._state.terminated = True
            return VRTaskEvent(kind=VRTaskEventKind.UNITY_TERMINATED)

        return VRTaskEvent(kind=VRTaskEventKind.NONE)

    def resume_after_unity_restart(self) -> None:
        """Re-arms Unity through the bridge after an emergency pause and re-fetches the wall cue sequence.

        Notes:
            An emergency pause occurs when Unity reports that its runtime terminated. This method ensures the editor
            is reachable, re-arms Unity via the bridge, re-queries the regenerated wall cue sequence so the animal's
            Virtual Reality position is tracked accurately after the reset, and clears the termination flag.
        """
        self._require_bridge()
        self._arm_unity()
        self._refresh_cue_sequence()
        self._state.terminated = False

    def _require_bridge(self) -> None:
        """Blocks until the Unity Editor MCP Bridge is reachable, prompting the operator to open Unity if it is not.

        Notes:
            The bridge starts automatically when the Unity Editor loads, so an unreachable bridge means the editor is
            not running. The method enforces that Unity is open before the session proceeds instead of falling back
            to manual scene and Play Mode control.
        """
        while not self._bridge.is_reachable():
            message = (
                "Unable to reach the Unity Editor. Open the Unity project in the editor; its MCP bridge starts "
                "automatically. Run 'sle get unity' to verify the connection."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            wait_for_enter(message="Press Enter once the Unity Editor is running.")
        console.echo(message="Unity bridge: Connected.", level=LogLevel.SUCCESS)

    def _activate_scene(self) -> None:
        """Opens the expected Unity scene through the bridge, persisting any unsaved editor changes.

        Raises:
            UnityBridgeError: If the expected scene cannot be resolved to a project scene path or the bridge refuses
                to open it.
        """
        try:
            scene_path = self._bridge.resolve_scene_path(scene_name=self._expected_scene_name)
            self._bridge.open_scene(scene_path=scene_path)
        except UnityBridgeError as exception:
            message = f"Unable to open the Virtual Reality scene '{self._expected_scene_name}' in Unity. {exception}"
            console.error(message=message, error=UnityBridgeError)
        message = f"Unity scene: Opened '{self._expected_scene_name}'."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _arm_unity(self) -> None:
        """Arms Unity by requesting Play Mode through the bridge and waiting for the scene's MQTT client to connect.

        Notes:
            The bridge triggers Play Mode entry, but the SessionStart message published by the scene's MQTT client
            remains the authoritative signal that Unity is armed and ready to exchange task data. Entering Play Mode
            can trigger a Unity script recompile, so the method retries on a bridge failure or a readiness timeout.
        """
        while True:
            self._clear_buffer()
            try:
                state = self._bridge.enter_play_mode()
                if state == "playing":
                    # Unity was already in Play Mode, so its SessionStart fired earlier and will not repeat.
                    # Restarting Play Mode yields a fresh SessionStart and a clean Virtual Reality origin.
                    self._stop_unity()
                    self._clear_buffer()
                    self._bridge.enter_play_mode()
            except UnityBridgeError as exception:
                message = f"Unable to arm Unity through the bridge. {exception}"
                console.echo(message=message, level=LogLevel.WARNING)
                wait_for_enter(message="Press Enter to retry arming Unity.")
                continue

            if self._wait_for_topic_bounded(
                expected_topic=_VRTaskMQTTTopics.SESSION_START, timeout_ms=_PLAY_MODE_READINESS_TIMEOUT_MS
            ):
                message = "Unity state transition: Confirmed. Unity is now armed."
                console.echo(message=message, level=LogLevel.SUCCESS)
                return

            # Play Mode entry succeeded but the scene never connected. Leaving Play Mode lets the next iteration
            # trigger a fresh entry that re-publishes SessionStart instead of reporting that Unity is already playing.
            self._stop_unity()
            message = (
                "Unity entered Play Mode but its MQTT client did not connect within the expected time. Ensure the "
                "scene's MQTT settings are correct."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            wait_for_enter(message="Press Enter to retry arming Unity.")

    def _stop_unity(self) -> None:
        """Stops Unity Play Mode through the bridge and drains the resulting SessionStop message.

        Notes:
            Draining the SessionStop published when the scene's MQTT client disconnects prevents it from later being
            misread by cycle() as an unexpected Unity termination during the runtime.
        """
        try:
            self._bridge.exit_play_mode()
        except UnityBridgeError as exception:
            message = f"Unable to stop Unity through the bridge. {exception}"
            console.echo(message=message, level=LogLevel.WARNING)
            return
        self._wait_for_topic_bounded(expected_topic=_VRTaskMQTTTopics.SESSION_STOP, timeout_ms=_SESSION_STOP_TIMEOUT_MS)

    def _refresh_cue_sequence(self) -> None:
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
                topic, payload = data
                if topic != _VRTaskMQTTTopics.CUE_SEQUENCE:
                    continue

                cue_sequence: NDArray[np.uint8] = np.array(
                    json.loads(payload.decode("utf-8"))["cueSequence"], dtype=np.uint8
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
            wait_for_enter(message="Press Enter to retry.")

    def _verify_scene_name(self) -> None:
        """Verifies that the armed Unity scene matches the expected scene name.

        Notes:
            The driver opens the expected scene through the bridge before arming Unity, so a mismatch indicates a
            configuration error, such as a scene whose embedded name disagrees with its file name. The SceneName
            reply also confirms the scene's MQTT client is responsive.

        Raises:
            RuntimeError: If the active Unity scene does not match the expected scene name.
        """
        self._clear_buffer()
        message = "Verifying that the Unity game engine is configured to display the correct scene..."
        console.echo(message=message, level=LogLevel.INFO)

        self._mqtt.send_data(topic=_VRTaskMQTTTopics.SCENE_NAME_TRIGGER)
        payload = self._wait_for_topic(expected_topic=_VRTaskMQTTTopics.SCENE_NAME)
        scene_name: str = json.loads(payload.decode("utf-8"))["name"]

        if scene_name != self._expected_scene_name:
            message = (
                f"The name of the Virtual Reality scene (task) running in Unity ({scene_name}) does not match the "
                f"scene name expected based on the session's experiment configuration ({self._expected_scene_name})."
            )
            console.error(message=message, error=RuntimeError)

        message = "Unity scene configuration: Confirmed."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def _verify_vr_display(self) -> None:
        """Animates the VR scene while the operator verifies the display, then cycles Unity into a fresh armed state.

        Notes:
            The driver animates the scene continuously while the operator inspects the VR screens and resolves any
            display issues, which may include starting and stopping the scene through the editor without disrupting
            the background animation. Pressing Enter signals that the display renders correctly, so the operator is
            responsible for fixing any issues before confirming. The driver then reads the current play state through
            the bridge and cycles the scene off and back on so the session begins from a fresh Virtual Reality origin
            regardless of the state the operator left the editor in.
        """
        self._polling_timer.delay(delay=_DISPLAY_SCREENS_WARMUP_DELAY_MS, block=False)

        message = (
            "Verify the Virtual Reality scene on the VR screens. The scene animates continuously; fix any display "
            "issues now, including starting and stopping the scene through the editor, which does not interrupt the "
            "animation."
        )
        console.echo(message=message, level=LogLevel.INFO)
        self._animate_until_satisfied()

        # Regardless of the play state the operator left Unity in, cycles the scene off and back on so the session
        # starts from a fresh, armed Virtual Reality origin.
        state, _ = self._bridge.get_play_state()
        if state == "playing":
            self._stop_unity()
        self._arm_unity()

    def _animate_until_satisfied(self) -> None:
        """Animates the Virtual Reality scene on a background thread until the operator confirms the display.

        Notes:
            The animation runs on a background thread so the foreground can block on the shared wait_for_enter prompt
            while the Virtual Reality scene keeps scrolling on the VR screens. Driving the MQTT client from the thread
            is safe because its publish path and inbound message queue are both thread-safe. The thread stops as soon
            as the operator presses Enter, and the pending MQTT buffer is drained each cycle so the verification
            animation does not accumulate the messages it does not consume.
        """
        stop_animation = threading.Event()

        def animate() -> None:
            while not stop_animation.is_set():
                self._polling_timer.delay(delay=_DISPLAY_ANIMATION_STEP_DELAY_MS, block=False)
                payload = json.dumps(obj={"movement": _DISPLAY_ANIMATION_STEP_UNITS}).encode("utf-8")
                self._mqtt.send_data(topic=_VRTaskMQTTTopics.MOTION, payload=payload)
                self._clear_buffer()

        animation_thread = threading.Thread(target=animate, daemon=True)
        animation_thread.start()
        try:
            wait_for_enter(message="Press Enter once you are satisfied that the display renders correctly.")
        finally:
            stop_animation.set()
            animation_thread.join()

    def _wait_for_topic(self, expected_topic: str) -> bytes | bytearray:
        """Blocks until Unity sends a message on the specified topic and returns the payload."""
        while True:
            self._polling_timer.delay(delay=_SETUP_POLLING_DELAY_MS, block=False)
            data = self._mqtt.get_data()
            if data is not None:
                topic, payload = data
                if topic == expected_topic:
                    return payload

    def _wait_for_topic_bounded(self, expected_topic: str, timeout_ms: int) -> bool:
        """Polls the Unity MQTT buffer for a message on the given topic until the timeout elapses.

        Args:
            expected_topic: The MQTT topic to wait for.
            timeout_ms: The maximum time, in milliseconds, to wait before giving up.

        Returns:
            True if a message on the expected topic arrived within the timeout, False otherwise.
        """
        self._polling_timer.reset()
        while self._polling_timer.elapsed < timeout_ms:
            self._polling_timer.delay(delay=_SETUP_POLLING_DELAY_MS, block=False)
            data = self._mqtt.get_data()
            if data is not None:
                topic, _ = data
                if topic == expected_topic:
                    return True
        return False

    def _clear_buffer(self) -> None:
        """Drains all pending messages from the MQTT buffer used to communicate with Unity."""
        while self._mqtt.has_data:
            _ = self._mqtt.get_data()
