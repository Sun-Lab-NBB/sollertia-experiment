"""Provides the runtime vocabulary used by the Unity Virtual Reality task driver.

Bundles the MQTT topic catalog used on the wire with Unity, the mutable state tracked by the driver during a
session, and the typed events the driver surfaces back to the acquisition system per runtime cycle.
"""

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
        The topics defined in this enumeration are used in addition to the topic defined by the hardware module
        interfaces of the acquisition system that drives the Virtual Reality task.
    """

    UNITY_TERMINATION = "Gimbl/Session/Stop"
    """Stops the Unity game session."""
    UNITY_STARTUP = "Gimbl/Session/Start"
    """Starts the Unity game session."""
    CUE_SEQUENCE = "CueSequence/"
    """The topic to which Unity sends the sequence of Virtual Reality cues used by the current game session."""
    CUE_SEQUENCE_REQUEST = "CueSequenceTrigger/"
    """Requests Unity to send the sequence of Virtual Reality cues used by the current game session."""
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
