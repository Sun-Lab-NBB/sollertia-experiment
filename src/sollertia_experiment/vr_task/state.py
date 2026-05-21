"""Provides the runtime state dataclass tracked by the Unity Virtual Reality task driver."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
