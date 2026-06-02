from pathlib import Path

import click

from ..mesoscope_vr import (
    purge_session as purge_session,
    experiment_logic as experiment_logic,
    maintenance_logic as maintenance_logic,
    run_training_logic as run_training_logic,
    lick_training_logic as lick_training_logic,
    window_checking_logic as window_checking_logic,
    preprocess_session_data as preprocess_session_data,
    get_system_configuration as get_system_configuration,
    migrate_animal_between_projects as migrate_animal_between_projects,
    create_system_configuration_file as create_system_configuration_file,
)

CONTEXT_SETTINGS: dict[str, int]

def mesoscope() -> None: ...
def configure() -> None: ...
def maintain() -> None: ...
@click.pass_context
def run(ctx: click.Context, user: str, project: str, animal: str, animal_weight: float) -> None: ...
@click.pass_context
def window_checking(ctx: click.Context) -> None: ...
@click.pass_context
def lick_training(
    ctx: click.Context,
    maximum_time: int | None,
    minimum_delay: int | None,
    maximum_delay: int | None,
    maximum_volume: float | None,
    unconsumed_rewards: int | None,
) -> None: ...
@click.pass_context
def run_training(
    ctx: click.Context,
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
@click.pass_context
def run_experiment(ctx: click.Context, experiment: str, unconsumed_rewards: int | None) -> None: ...
def preprocess(session_path: Path) -> None: ...
def delete(session_path: Path) -> None: ...
def migrate(source: str, destination: str, animal: str) -> None: ...
