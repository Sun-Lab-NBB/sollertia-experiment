"""Provides the acquisition-system-agnostic interface for the Unity Virtual Reality tasks in
https://github.com/Sun-Lab-NBB/sollertia-unity-tasks.
"""

from .bridge import UnityBridgeError, UnityBridgeClient
from .driver import VRTaskEvent, VRTaskState, VRTaskDriver, StimulusCause, VRTaskEventKind
from .configuration import VRTaskConfiguration, load_vr_task_template
from .trial_decomposition import DecomposedTrials, CachedMotifDecomposer, decompose_cue_sequence

__all__ = [
    "CachedMotifDecomposer",
    "DecomposedTrials",
    "StimulusCause",
    "UnityBridgeClient",
    "UnityBridgeError",
    "VRTaskConfiguration",
    "VRTaskDriver",
    "VRTaskEvent",
    "VRTaskEventKind",
    "VRTaskState",
    "decompose_cue_sequence",
    "load_vr_task_template",
]
