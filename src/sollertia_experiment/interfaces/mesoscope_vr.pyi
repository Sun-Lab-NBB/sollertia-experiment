from pathlib import Path
from dataclasses import dataclass

import click
from _typeshed import Incomplete

from ..mesoscope_vr import (
    purge_session as purge_session,
    experiment_logic as experiment_logic,
    maintenance_logic as maintenance_logic,
    run_training_logic as run_training_logic,
    lick_training_logic as lick_training_logic,
    window_checking_logic as window_checking_logic,
    check_mesoscope_bridge as check_mesoscope_bridge,
    preprocess_session_data as preprocess_session_data,
    get_system_configuration as get_system_configuration,
    migrate_animal_between_projects as migrate_animal_between_projects,
    create_system_configuration_file as create_system_configuration_file,
    create_experiment_configuration_file as create_experiment_configuration_file,
)

_CONTEXT_SETTINGS: dict[str, int]

@dataclass(frozen=True, slots=True)
class _SharedSessionParameters:
    user: str | None
    project: str | None
    animal: str | None
    animal_weight: float | None
    def require_user(self) -> str: ...
    def require_project(self) -> str: ...
    def require_animal(self) -> str: ...
    def require_animal_weight(self) -> float: ...

_pass_shared_parameters: Incomplete

def mesoscope() -> None: ...
def configure() -> None: ...
def configure_system() -> None: ...
def configure_experiment(
    project: str,
    experiment: str,
    template: str,
    state_count: int,
    reward_size: float,
    reward_tone_duration: int,
    puff_duration: int,
    *,
    force: bool,
) -> None: ...
def maintain() -> None: ...
def check_bridge() -> None: ...
@click.pass_context
def run(
    context: click.Context, user: str | None, project: str | None, animal: str | None, animal_weight: float | None
) -> None: ...
@_pass_shared_parameters
def window_checking(shared: _SharedSessionParameters) -> None: ...
@_pass_shared_parameters
def lick_training(
    shared: _SharedSessionParameters,
    maximum_time: int | None,
    minimum_delay: int | None,
    maximum_delay: int | None,
    maximum_volume: float | None,
    unconsumed_rewards: int | None,
) -> None: ...
@_pass_shared_parameters
def run_training(
    shared: _SharedSessionParameters,
    maximum_time: int | None,
    initial_speed: float | None,
    initial_duration: float | None,
    increase_threshold: float | None,
    speed_step: float | None,
    duration_step: float | None,
    maximum_volume: float | None,
    maximum_idle_time: float | None,
    unconsumed_rewards: int | None,
) -> None: ...
@_pass_shared_parameters
def run_experiment(shared: _SharedSessionParameters, experiment: str, unconsumed_rewards: int | None) -> None: ...
def preprocess(session_path: Path) -> None: ...
def delete(session_path: Path) -> None: ...
def migrate(source: str, destination: str, animal: str) -> None: ...
