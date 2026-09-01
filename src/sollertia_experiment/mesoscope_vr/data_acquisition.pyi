from pathlib import Path

from numpy.typing import NDArray as NDArray
from sollertia_shared_assets import ExperimentState as ExperimentState
from ataraxis_communication_interface import MicroControllerInterface

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
    BEHAVIOR_LOGGER_NAME as BEHAVIOR_LOGGER_NAME,
    BrakeInterface as BrakeInterface,
    WaterValveInterface as WaterValveInterface,
    GasPuffValveInterface as GasPuffValveInterface,
    wait_for_enter as wait_for_enter,
    get_version_data as get_version_data,
    run_shutdown_step as run_shutdown_step,
    get_project_experiments as get_project_experiments,
    request_required_confirmation as request_required_confirmation,
)
from .maintenance_ui import MaintenanceControlUI as MaintenanceControlUI
from .binding_classes import (
    ZaberMotors as ZaberMotors,
    VideoSystems as VideoSystems,
)
from .mesoscope_driver import MesoscopeDriver as MesoscopeDriver
from .system_controller import MesoscopeVRSystem as MesoscopeVRSystem
from .data_preprocessing import (
    purge_session as purge_session,
    preprocess_session_data as preprocess_session_data,
)
from .acquisition_components import (
    RESPONSE_DELAY as RESPONSE_DELAY,
    RESPONSE_DELAY_TIMER as RESPONSE_DELAY_TIMER,
    setup_mesoscope as setup_mesoscope,
    reset_zaber_motors as reset_zaber_motors,
    setup_zaber_motors as setup_zaber_motors,
    generate_zaber_snapshot as generate_zaber_snapshot,
    finalize_session_descriptor as finalize_session_descriptor,
    generate_mesoscope_position_snapshot as generate_mesoscope_position_snapshot,
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
def _stop_actor_controller(controller: MicroControllerInterface) -> None: ...
def _verify_project_configured(
    session_description: str, system_configuration: MesoscopeSystemConfiguration, project_name: str, animal_id: str
) -> Path: ...
def _verify_animal_project_membership(
    session_description: str, system_configuration: MesoscopeSystemConfiguration, project_name: str, animal_id: str
) -> None: ...
def _park_maintenance_motors(zaber_motors: ZaberMotors) -> None: ...
