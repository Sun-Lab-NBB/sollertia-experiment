from enum import IntEnum, StrEnum
from dataclasses import field, dataclass

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from sollertia_shared_assets import TaskTemplate as TaskTemplate
from ataraxis_communication_interface import MQTTCommunication

from .configuration import VRTaskConfiguration as VRTaskConfiguration
from .trial_decomposition import (
    DecomposedTrials as DecomposedTrials,
    CachedMotifDecomposer as CachedMotifDecomposer,
    decompose_cue_sequence as decompose_cue_sequence,
)

_SETUP_POLLING_DELAY_MS: int
_DISPLAY_ANIMATION_STEP_DELAY_MS: int
_DISPLAY_ANIMATION_STEP_UNITS: float
_DISPLAY_SCREENS_WARMUP_DELAY_MS: int
_CUE_SEQUENCE_RESPONSE_TIMEOUT_MS: int

class VRTaskEventKind(IntEnum):
    NONE = 0
    STIMULUS_TRIGGERED = 1
    TRIGGER_DELAY_REQUESTED = 2
    UNITY_TERMINATED = 3

@dataclass(frozen=True, slots=True)
class VRTaskEvent:
    kind: VRTaskEventKind
    delay_ms: int = ...

@dataclass(slots=True)
class VRTaskState:
    position: np.float64 = field(default_factory=Incomplete)
    cue_sequence: NDArray[np.uint8] = field(default_factory=Incomplete)
    terminated: bool = ...
    reinforcing_guidance_enabled: bool = ...
    aversive_guidance_enabled: bool = ...

class _VRTaskMQTTTopics(StrEnum):
    SESSION_START = "SessionStart"
    SESSION_STOP = "SessionStop"
    MOTION = "Motion"
    LICK = "Lick"
    STIMULUS = "Stimulus"
    DELAY = "Delay"
    CUE_SEQUENCE_TRIGGER = "CueSequenceTrigger"
    CUE_SEQUENCE = "CueSequence"
    SCENE_NAME_TRIGGER = "SceneNameTrigger"
    SCENE_NAME = "SceneName"
    REQUIRE_LICK = "RequireLick"
    REQUIRE_WAIT = "RequireWait"

class VRTaskDriver:
    _configuration: VRTaskConfiguration
    _task_template: TaskTemplate
    _expected_scene_name: str
    _mqtt: MQTTCommunication
    _state: VRTaskState
    _motif_decomposer: CachedMotifDecomposer
    _decomposed_trials: DecomposedTrials
    _polling_timer: PrecisionTimer
    def __init__(
        self, configuration: VRTaskConfiguration, *, task_template: TaskTemplate, expected_scene_name: str
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def state(self) -> VRTaskState: ...
    @property
    def cue_sequence_distances(self) -> NDArray[np.float64]: ...
    @property
    def trial_names(self) -> tuple[str, ...]: ...
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def setup(self) -> None: ...
    def push_position(self, absolute_position: np.float64) -> None: ...
    def push_lick_event(self) -> None: ...
    def set_reinforcing_guidance(self, *, enabled: bool) -> None: ...
    def set_aversive_guidance(self, *, enabled: bool) -> None: ...
    def cycle(self) -> VRTaskEvent: ...
    def resume_after_unity_restart(self) -> None: ...
    def _refresh_cue_sequence(self) -> None: ...
    def _verify_scene_name(self) -> None: ...
    def _verify_vr_display(self) -> None: ...
    def _animate_until_termination(self) -> None: ...
    def _rearm_unity(self) -> None: ...
    def _wait_for_topic(self, expected_topic: str) -> bytes | bytearray: ...
    def _clear_buffer(self) -> None: ...
