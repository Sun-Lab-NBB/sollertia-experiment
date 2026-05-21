"""Provides the VRTaskDriver class that encapsulates all communication with the Unity game engine used to run the
Virtual Reality task.
"""

from __future__ import annotations

import json
from json import dumps
from typing import TYPE_CHECKING, Protocol

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_communication_interface import MQTTCommunication

from .events import VRTaskEvent, VRTaskState, VRTaskEventKind, VRTaskMQTTTopics
from .trial_decomposition import DecomposedTrials, CachedMotifDecomposer, decompose_cue_sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sollertia_shared_assets import GasPuffTrial, WaterRewardTrial

    from .configuration import LoggingHooks, VRTaskConfiguration


class ScreenPulse(Protocol):
    """Defines the callable signature used by VRTaskDriver to toggle the VR screens during the setup sequence."""

    def __call__(self, *, state: bool) -> None:
        """Toggles the VR screen state."""


_SETUP_POLLING_DELAY_MS: int = 10
"""The delay, in milliseconds, between consecutive Unity buffer polls during the interactive setup sequence."""

_DISPLAY_ANIMATION_STEP_DELAY_MS: int = 100
"""The delay, in milliseconds, between consecutive position updates sent to Unity during the VR display verification
animation."""

_DISPLAY_ANIMATION_STEP_UNITS: float = 0.1
"""The size, in Unity units, of each position update sent during the VR display verification animation."""

_DISPLAY_SCREENS_WARMUP_DELAY_MS: int = 2000
"""The delay, in milliseconds, applied after enabling the VR screens before driving the display verification
animation."""

_CUE_SEQUENCE_RESPONSE_TIMEOUT_MS: int = 5000
"""The maximum time, in milliseconds, to wait for Unity to respond to a cue sequence request before retrying."""


class VRTaskDriver:
    """Drives the Unity game engine that runs the Virtual Reality task.

    Encapsulates the MQTT contract with Unity: connection lifecycle, scene handshake, VR display verification, wall
    cue sequence retrieval, per-cycle stimulus pump, guidance toggling, and resume-after-Unity-restart. The driver
    is hardware-agnostic; per-cycle Unity events are surfaced as typed VRTaskEvent values that the caller dispatches
    to acquisition system hardware.

    Args:
        configuration: The runtime configuration that defines the MQTT broker discovery fields.
        expected_scene_name: The Unity scene name the driver enforces during the setup handshake.
        trial_structures: The mapping of trial names to trial-type configuration objects defined by the experiment.
        logging_hooks: An optional adapter that forwards driver-generated log payloads to the acquisition system's
            DataLogger. When omitted, the driver does not emit any log payloads.

    Attributes:
        _configuration: The VRTaskConfiguration instance that defines the MQTT broker discovery fields.
        _expected_scene_name: The Unity scene name enforced during the setup handshake.
        _trial_structures: The mapping of trial names to trial-type configuration objects.
        _logging_hooks: The optional LoggingHooks adapter that forwards log payloads to the acquisition system.
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
        expected_scene_name: str,
        trial_structures: dict[str, WaterRewardTrial | GasPuffTrial],
        logging_hooks: LoggingHooks | None = None,
    ) -> None:
        self._configuration: VRTaskConfiguration = configuration
        self._expected_scene_name: str = expected_scene_name
        self._trial_structures: dict[str, WaterRewardTrial | GasPuffTrial] = trial_structures
        self._logging_hooks: LoggingHooks | None = logging_hooks

        monitored_topics: tuple[VRTaskMQTTTopics, ...] = (
            VRTaskMQTTTopics.CUE_SEQUENCE,
            VRTaskMQTTTopics.UNITY_TERMINATION,
            VRTaskMQTTTopics.UNITY_STARTUP,
            VRTaskMQTTTopics.UNITY_SCENE,
            VRTaskMQTTTopics.STIMULUS,
            VRTaskMQTTTopics.TRIGGER_DELAY,
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
            reinforcing_rewards=(),
            aversive_puff_durations=(),
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
    def reinforcing_rewards(self) -> tuple[tuple[float, int], ...]:
        """Returns the reward size and tone duration for each decomposed trial."""
        return self._decomposed_trials.reinforcing_rewards

    @property
    def aversive_puff_durations(self) -> tuple[int, ...]:
        """Returns the gas puff duration for each decomposed trial."""
        return self._decomposed_trials.aversive_puff_durations

    def connect(self) -> None:
        """Establishes the MQTT connection to the Unity game engine."""
        self._mqtt.connect()

    def disconnect(self) -> None:
        """Closes the MQTT connection to the Unity game engine."""
        self._mqtt.disconnect()

    def setup(self, *, screen_pulse: ScreenPulse) -> None:
        """Carries out the interactive Unity setup sequence used at the start of a session.

        Notes:
            Verifies that the Unity scene matches the expected scene name, drives a display verification loop on the
            VR screens, re-arms Unity, and requests the wall cue sequence used by the task. The setup sequence is
            interactive and prompts the user via console messages when a step requires confirmation or retry.

        Args:
            screen_pulse: The callable used to enable and disable the VR screens during the display verification
                stage. Invoked with True when the driver needs the screens to render the verification animation, and
                with False once verification is complete.
        """
        self._verify_scene_name()
        self._verify_vr_display(screen_pulse=screen_pulse)
        self._rearm_unity()
        screen_pulse(state=False)
        self.refresh_cue_sequence()
        message = "Unity setup: Complete."
        console.echo(message=message, level=LogLevel.SUCCESS)

    def refresh_cue_sequence(self) -> None:
        """Requests and resolves the Virtual Reality wall cue sequence used by the current Unity scene.

        Notes:
            Re-fetches the cue sequence from Unity, decomposes it into per-trial cumulative distances and stimulus
            parameters, forwards the raw cue sequence to the logging hook if configured, and resets the driver's
            tracked Unity position to the origin.

        Raises:
            RuntimeError: If the decomposer cannot match any trial motif at some position in the cue sequence.
        """
        self._clear_buffer()
        message = (
            "Requesting Virtual Reality wall cue sequence from Unity. Ensure Unity is armed and the task is running."
        )
        console.echo(message=message, level=LogLevel.INFO)

        while True:
            self._mqtt.send_data(topic=VRTaskMQTTTopics.CUE_SEQUENCE_REQUEST)
            self._polling_timer.reset()

            received = False
            while self._polling_timer.elapsed < _CUE_SEQUENCE_RESPONSE_TIMEOUT_MS:
                self._polling_timer.delay(delay=_SETUP_POLLING_DELAY_MS, block=False)
                data = self._mqtt.get_data()
                if data is None:
                    continue
                if data[0] != VRTaskMQTTTopics.CUE_SEQUENCE:
                    continue

                cue_sequence: NDArray[np.uint8] = np.array(
                    json.loads(data[1].decode("utf-8"))["cue_sequence"], dtype=np.uint8
                )
                self._state.cue_sequence = cue_sequence

                if self._logging_hooks is not None:
                    self._logging_hooks.log_cue_sequence(cue_sequence=cue_sequence)

                self._decomposed_trials = decompose_cue_sequence(
                    cue_sequence=cue_sequence,
                    trial_structures=self._trial_structures,
                    motif_decomposer=self._motif_decomposer,
                )

                self._state.position = np.float64(0.0)

                message = "VR cue sequence: Received."
                console.echo(message=message, level=LogLevel.SUCCESS)
                received = True
                break

            if received:
                return

            message = (
                f"The Virtual Reality task driver sent a cue sequence request to Unity via the "
                f"'{VRTaskMQTTTopics.CUE_SEQUENCE_REQUEST}' topic but received no response within "
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
            self._mqtt.send_data(topic=VRTaskMQTTTopics.ENCODER_DATA, payload=payload)

    def push_lick_event(self) -> None:
        """Notifies Unity that the animal performed a lick."""
        self._mqtt.send_data(topic=VRTaskMQTTTopics.LICK_EVENT, payload=None)

    def set_reinforcing_guidance(self, *, enabled: bool) -> None:
        """Sets the reinforcing trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable reinforcing guidance.
        """
        if not enabled:
            self._mqtt.send_data(topic=VRTaskMQTTTopics.DISABLE_LICK_GUIDANCE)
        else:
            self._mqtt.send_data(topic=VRTaskMQTTTopics.ENABLE_LICK_GUIDANCE)

        if self._logging_hooks is not None:
            self._logging_hooks.log_reinforcing_guidance_change(enabled=enabled)

        self._state.reinforcing_guidance_enabled = enabled

    def set_aversive_guidance(self, *, enabled: bool) -> None:
        """Sets the aversive trial guidance mode.

        Args:
            enabled: Determines whether to enable or disable aversive guidance.
        """
        if not enabled:
            self._mqtt.send_data(topic=VRTaskMQTTTopics.DISABLE_OCCUPANCY_GUIDANCE)
        else:
            self._mqtt.send_data(topic=VRTaskMQTTTopics.ENABLE_OCCUPANCY_GUIDANCE)

        if self._logging_hooks is not None:
            self._logging_hooks.log_aversive_guidance_change(enabled=enabled)

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

        if topic == VRTaskMQTTTopics.STIMULUS:
            return VRTaskEvent(kind=VRTaskEventKind.STIMULUS_TRIGGERED)

        if topic == VRTaskMQTTTopics.TRIGGER_DELAY:
            delay_payload = json.loads(payload.decode("utf-8"))
            delay_ms = int(delay_payload.get("delay_ms", 0))
            return VRTaskEvent(kind=VRTaskEventKind.TRIGGER_DELAY_REQUESTED, delay_ms=delay_ms)

        if topic == VRTaskMQTTTopics.UNITY_TERMINATION:
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

            self._wait_for_topic(expected_topic=VRTaskMQTTTopics.UNITY_STARTUP)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

            message = "Verifying that the Unity game engine is configured to display the correct scene..."
            console.echo(message=message, level=LogLevel.INFO)

            self._mqtt.send_data(topic=VRTaskMQTTTopics.UNITY_SCENE_REQUEST)
            payload = self._wait_for_topic(expected_topic=VRTaskMQTTTopics.UNITY_SCENE)
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

    def _verify_vr_display(self, *, screen_pulse: ScreenPulse) -> None:
        """Drives the VR display verification loop until the user confirms the scene renders correctly."""
        screen_pulse(state=True)
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
            self._wait_for_topic(expected_topic=VRTaskMQTTTopics.UNITY_STARTUP)
            message = "Unity state transition: Confirmed. Unity is now armed."
            console.echo(message=message, level=LogLevel.SUCCESS)

    def _animate_until_termination(self) -> None:
        """Sends incremental position updates to Unity until Unity reports termination."""
        while True:
            self._polling_timer.delay(delay=_DISPLAY_ANIMATION_STEP_DELAY_MS, block=False)

            payload = dumps(obj={"movement": _DISPLAY_ANIMATION_STEP_UNITS}).encode("utf-8")
            self._mqtt.send_data(topic=VRTaskMQTTTopics.ENCODER_DATA, payload=payload)

            data = self._mqtt.get_data()
            if data is None:
                continue

            if data[0] == VRTaskMQTTTopics.UNITY_TERMINATION:
                message = "Unity termination: Detected."
                console.echo(message=message, level=LogLevel.INFO)
                return

    def _rearm_unity(self) -> None:
        """Re-arms Unity after successful display verification."""
        self._clear_buffer()
        message = "Arm the Unity task by pressing the 'play' button in the Unity Editor."
        console.echo(message=message, level=LogLevel.INFO)
        self._wait_for_topic(expected_topic=VRTaskMQTTTopics.UNITY_STARTUP)
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
