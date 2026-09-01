from enum import IntEnum
from decimal import Decimal
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from ataraxis_time import PrecisionTimer
from sollertia_shared_assets import (
    SessionData as SessionData,
    MesoscopeGasPuffTrial as MesoscopeGasPuffTrial,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeWaterRewardTrial as MesoscopeWaterRewardTrial,
    MesoscopeExperimentDescriptor,
)

from .system import (
    MesoscopeData as MesoscopeData,
    MesoscopePositions as MesoscopePositions,
)
from .runtime_ui import (
    collect_surgery_quality as collect_surgery_quality,
    collect_experimenter_notes as collect_experimenter_notes,
    collect_experimenter_given_water_volume as collect_experimenter_given_water_volume,
)
from ..cross_system import (
    request_text as request_text,
    wait_for_enter as wait_for_enter,
    request_confirmation as request_confirmation,
    request_required_confirmation as request_required_confirmation,
)
from .binding_classes import ZaberMotors as ZaberMotors
from .mesoscope_driver import MesoscopeDriver as MesoscopeDriver

RESPONSE_DELAY: int
_DEFAULT_TOTAL_WATER_VOLUME_ML: float
_THREE_DECIMAL_QUANTUM: Decimal
_REFERENCE_WRITE_PERMISSION_BITS: int
_REFERENCE_RENAME_RETRY_COUNT: int
_REFERENCE_RENAME_RETRY_DELAY_MILLISECONDS: int

class _ResponseDelayTimer:
    _timer: PrecisionTimer | None
    def __init__(self) -> None: ...
    def delay(self, delay: int, *, allow_sleep: bool = False, block: bool = False) -> None: ...
    def reset(self) -> None: ...
    @property
    def elapsed(self) -> int: ...
    def _release(self) -> None: ...

RESPONSE_DELAY_TIMER: _ResponseDelayTimer

class MesoscopeVRLogMessageCodes(IntEnum):
    SYSTEM_STATE = 1
    RUNTIME_STATE = 2
    REINFORCING_GUIDANCE_STATE = 3
    AVERSIVE_GUIDANCE_STATE = 4
    DISTANCE_SNAPSHOT = 5
    MESOSCOPE_ACQUISITION_STATE = 6

@dataclass(slots=True)
class TrialState:
    completed: int = ...
    distances: NDArray[np.float64] = field(default_factory=Incomplete)
    reinforcing_guided_trials: int = ...
    reinforcing_failed_trials: int = ...
    reinforcing_recovery_threshold: int = ...
    reinforcing_recovery_trials: int = ...
    reinforcing_rewarded: bool = ...
    reinforcing_rewards: tuple[tuple[float, int], ...] = ...
    aversive_guided_trials: int = ...
    aversive_failed_trials: int = ...
    aversive_recovery_threshold: int = ...
    aversive_recovery_trials: int = ...
    aversive_succeeded: bool = ...
    aversive_puff_durations: tuple[int, ...] = ...
    trial_structures: dict[str, MesoscopeWaterRewardTrial | MesoscopeGasPuffTrial] = field(default_factory=dict)
    def trial_completed(self, traveled_distance: float) -> bool: ...
    def is_current_trial_aversive(self) -> bool: ...
    def advance_trial(self) -> int: ...

@dataclass(frozen=True, slots=True)
class _PreviousSessionWaterContext:
    animal_weight_g: float
    received_water_volume_ml: float

def generate_mesoscope_position_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, mesoscope_driver: MesoscopeDriver
) -> None: ...
def generate_zaber_snapshot(
    session_data: SessionData, mesoscope_data: MesoscopeData, zaber_motors: ZaberMotors
) -> None: ...
def setup_zaber_motors(zaber_motors: ZaberMotors) -> None: ...
def reset_zaber_motors(zaber_motors: ZaberMotors) -> None: ...
def setup_mesoscope(
    session_data: SessionData, mesoscope_data: MesoscopeData, mesoscope_driver: MesoscopeDriver
) -> None: ...
def finalize_session_descriptor(
    descriptor: MesoscopeExperimentDescriptor
    | LickTrainingDescriptor
    | RunTrainingDescriptor
    | WindowCheckingDescriptor,
    session_data: SessionData,
    mesoscope_data: MesoscopeData,
) -> None: ...
def _publish_reference_pair(
    motion_estimator_source: Path, roi_source: Path, motion_estimator_destination: Path, roi_destination: Path
) -> None: ...
def _restore_replaced_file(staged_path: Path, destination: Path) -> None: ...
def _stage_reference_file(source: Path, destination: Path) -> Path: ...
def _publish_staged_file(staged_path: Path, destination: Path) -> None: ...
def _resolve_previous_session_water_context(persistent_data_path: Path) -> _PreviousSessionWaterContext | None: ...
def _prompt_red_dot_alignment(previous_value: float) -> float: ...
def _validate_red_dot_response(response: str) -> bool | str: ...
def _floor_to_three_decimals(value: float) -> float: ...
