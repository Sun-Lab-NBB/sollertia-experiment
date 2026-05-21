"""Provides the typed event surface used by the Unity Virtual Reality task driver to report parsed Unity messages
back to the caller.
"""

from enum import IntEnum
from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class VRTaskEvent:
    """Stores the parsed Unity message produced by VRTaskDriver.cycle() during a single runtime cycle."""

    kind: VRTaskEventKind
    """The kind of Virtual Reality task event that occurred during the cycle."""
    delay_ms: int = 0
    """The brake pulse duration in milliseconds. Populated only for the TRIGGER_DELAY_REQUESTED event."""
