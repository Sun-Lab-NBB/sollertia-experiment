from pathlib import Path

from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import ExperimentState as ExperimentState

from .system import (
    RUN_TRAINING_THRESHOLD_LIMITS as RUN_TRAINING_THRESHOLD_LIMITS,
    MesoscopeData as MesoscopeData,
    ZaberPositions as ZaberPositions,
    MesoscopeVRStates as MesoscopeVRStates,
    MesoscopePositions as MesoscopePositions,
    MesoscopeSystemConfiguration as MesoscopeSystemConfiguration,
    get_system_configuration as get_system_configuration,
)
from ..cross_system import (
    BrakeInterface as BrakeInterface,
    WaterValveInterface as WaterValveInterface,
    GasPuffValveInterface as GasPuffValveInterface,
    get_version_data as get_version_data,
    get_project_experiments as get_project_experiments,
)
from .maintenance_ui import MaintenanceControlUI as MaintenanceControlUI
from .binding_classes import (
    ZaberMotors as ZaberMotors,
    VideoSystems as VideoSystems,
)
from .system_controller import _MesoscopeVRSystem as _MesoscopeVRSystem
from .data_preprocessing import (
    purge_session as purge_session,
    preprocess_session_data as preprocess_session_data,
)
from .acquisition_components import (
    _RESPONSE_DELAY as _RESPONSE_DELAY,
    _setup_mesoscope as _setup_mesoscope,
    _reset_zaber_motors as _reset_zaber_motors,
    _setup_zaber_motors as _setup_zaber_motors,
    _response_delay_timer as _response_delay_timer,
    _generate_zaber_snapshot as _generate_zaber_snapshot,
    _verify_descriptor_update as _verify_descriptor_update,
    _generate_mesoscope_position_snapshot as _generate_mesoscope_position_snapshot,
)

_RENDERING_SEPARATION_DELAY: int
_MICROLITERS_PER_MILLILITER: float

def window_checking_logic(experimenter: str, project_name: str, animal_id: str) -> None: ...
def lick_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    reward_size: float | None = None,
    reward_tone_duration: int | None = None,
    minimum_reward_delay: int | None = None,
    maximum_reward_delay: int | None = None,
    maximum_water_volume: float | None = None,
    maximum_training_time: int | None = None,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def run_training_logic(
    experimenter: str,
    project_name: str,
    animal_id: str,
    animal_weight: float,
    reward_size: float | None = None,
    reward_tone_duration: int | None = None,
    initial_speed_threshold: float | None = None,
    initial_duration_threshold: float | None = None,
    speed_increase_step: float | None = None,
    duration_increase_step: float | None = None,
    increase_threshold: float | None = None,
    maximum_water_volume: float | None = None,
    maximum_training_time: int | None = None,
    maximum_idle_time: float | None = None,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def experiment_logic(
    experimenter: str,
    project_name: str,
    experiment_name: str,
    animal_id: str,
    animal_weight: float,
    maximum_unconsumed_rewards: int | None = None,
) -> None: ...
def maintenance_logic() -> None: ...
def _verify_project_configured(
    session_description: str, system_configuration: MesoscopeSystemConfiguration, project_name: str, animal_id: str
) -> Path: ...
def _verify_animal_project_membership(
    session_description: str, system_configuration: MesoscopeSystemConfiguration, project_name: str, animal_id: str
) -> None: ...
