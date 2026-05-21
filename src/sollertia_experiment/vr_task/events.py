"""Provides the runtime vocabulary used by the Unity Virtual Reality task driver."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VRTaskMQTTTopics(StrEnum):
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


class VRTaskEventKind(IntEnum):
    """Defines the kinds of Virtual Reality task events produced by the VRTaskDriver per runtime cycle."""

    NONE = 0
    """No Unity message was available in the MQTT buffer during this cycle."""
    STIMULUS_TRIGGERED = 1
    """The animal triggered the current trial's stimulus (water reward or gas puff)."""
    TRIGGER_DELAY_REQUESTED = 2
    """Unity requested the acquisition system to apply a brake pulse for the specified duration."""
    UNITY_TERMINATED = 3
    """Unity reported that its runtime has been terminated; the acquisition system must enter an emergency pause."""


@dataclass
class VRTaskState:
    """Tracks the runtime state of the Virtual Reality task environment managed by the Unity game engine.

    This dataclass consolidates all Unity-related state tracking attributes used by the VRTaskDriver to monitor the
    Virtual Reality environment state, manage task guidance modes, and facilitate communication between the
    acquisition system and the Unity game engine over MQTT.
    """

    position: np.float64 = field(default_factory=lambda: np.float64(0.0))
    """The current absolute position of the animal, in Unity units, relative to the origin of the Virtual Reality task
    environment's track."""
    cue_sequence: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(shape=0, dtype=np.uint8))
    """The sequence of Virtual Reality environment wall cues used by the session's task environment. This array defines
    the visual cues displayed to the animal as it progresses through the virtual track."""
    terminated: bool = False
    """Tracks whether the system has detected that the Unity game engine has unexpectedly terminated its runtime. When
    True, the runtime enters an emergency pause state to allow the user to restart Unity."""
    reinforcing_guidance_enabled: bool = False
    """Tracks the state of the reinforcing trial guidance mode."""
    aversive_guidance_enabled: bool = False
    """Tracks the state of the aversive trial guidance mode."""


@dataclass(frozen=True, slots=True)
class VRTaskEvent:
    """Stores the parsed Unity message produced by VRTaskDriver.cycle() during a single runtime cycle."""

    kind: VRTaskEventKind
    """The kind of Virtual Reality task event that occurred during the cycle."""
    delay_ms: int = 0
    """The brake pulse duration in milliseconds. Populated only for the TRIGGER_DELAY_REQUESTED event."""
