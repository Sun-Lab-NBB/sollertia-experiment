"""Provides the 'sle run' subcommand for running the data acquisition and system maintenance sessions supported by the
data acquisition system managed by the host-machine.
"""

import click

from ..mesoscope_vr import (
    experiment_logic,
    maintenance_logic,
    run_training_logic,
    lick_training_logic,
    window_checking_logic,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}  # pragma: no cover


@click.group("run", context_settings=CONTEXT_SETTINGS)
def run() -> None:  # pragma: no cover
    """Runs data acquisition and system maintenance sessions supported by the data acquisition system managed by the
    host-machine.
    """


@run.command("maintenance")
def maintain_acquisition_system() -> None:
    """Runs the data acquisition system maintenance session.

    Calling this command exposes a GUI for directly interfacing with a small subset of the managed data acquisition
    system's components that require frequent maintenance. It does not collect any data during runtime and does
    not interface with the remote data storage infrastructure accessible to the data acquisition system. It is
    designed to perform minor (day-to-day) maintenance tasks that do not require disassembling the system's components.
    """
    maintenance_logic()


@run.group("session")
@click.option(
    "-u",
    "--user",
    type=str,
    required=True,
    help="The ID of the user supervising the session.",
)
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project to which the animal belongs.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The ID of the animal undergoing the session.",
)
@click.option(
    "-w",
    "--animal_weight",
    type=float,
    required=True,
    help="The weight of the animal, in grams, at the beginning of the session.",
)
@click.pass_context
def session(ctx: click.Context, user: str, project: str, animal: str, animal_weight: float) -> None:  # pragma: no cover
    """Runs the specified data acquisition session for the target animal and project combination."""
    # Store common parameters in the context dictionary to be accessible from the subcommands.
    ctx.ensure_object(dict)
    ctx.obj["user"] = user
    ctx.obj["project"] = project
    ctx.obj["animal"] = animal
    ctx.obj["animal_weight"] = animal_weight


# noinspection PyUnresolvedReferences
@session.command("check-window")
@click.pass_context
def check_window(ctx: click.Context) -> None:
    """Runs the cranial window quality checking session.

    The primary purpose of the cranial window quality checking session is to ensure that the animal is suitable for
    collecting high-quality brain activity data. Additionally, the session is used to generate the animal-specific data
    acquisition system configuration reused during all future data acquisition sessions to fine-tune the system
    to work for the target animal.
    """
    window_checking_logic(
        experimenter=ctx.obj["user"],
        project_name=ctx.obj["project"],
        animal_id=ctx.obj["animal"],
    )


# noinspection PyUnresolvedReferences
@session.command("lick-training")
@click.option(
    "-t",
    "--maximum_time",
    type=int,
    help="The maximum time to run the training session, in minutes. Defaults to 20 minutes.",
)
@click.option(
    "-min",
    "--minimum_delay",
    type=int,
    help=(
        "The minimum number of seconds that has to pass between two consecutive reward deliveries during training. "
        "Defaults to 6 seconds."
    ),
)
@click.option(
    "-max",
    "--maximum_delay",
    type=int,
    help=(
        "The maximum number of seconds that can pass between two consecutive reward deliveries during training. "
        "Defaults to 18 seconds."
    ),
)
@click.option(
    "-v",
    "--maximum_volume",
    type=float,
    help="The maximum volume of water, in milliliters, that can be delivered during training. Defaults to 1.0 mL.",
)
@click.option(
    "-ur",
    "--unconsumed_rewards",
    type=int,
    help=(
        "The maximum number of rewards that can be delivered without the animal consuming them. If the unconsumed "
        "reward count exceeds this threshold, the system stops delivering new water rewards until the animal consumes "
        "the already delivered rewards. Setting this argument to 0 disables the reward consumption tracking. "
        "Defaults to 1."
    ),
)
@click.pass_context
def lick_training(
    ctx: click.Context,
    maximum_time: int | None,
    minimum_delay: int | None,
    maximum_delay: int | None,
    maximum_volume: float | None,
    unconsumed_rewards: int | None,
) -> None:
    """Runs the lick training session.

    Lick training is the first phase of preparing the animal for experiment sessions, and is usually
    carried out over the first two days of the pre-experiment training sequence. This session teaches the animal to
    operate the lick-port and associate licking at the port with water delivery.
    """
    lick_training_logic(
        experimenter=ctx.obj["user"],
        project_name=ctx.obj["project"],
        animal_id=ctx.obj["animal"],
        animal_weight=ctx.obj["animal_weight"],
        minimum_reward_delay=minimum_delay,
        maximum_reward_delay=maximum_delay,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
        maximum_unconsumed_rewards=unconsumed_rewards,
    )


# noinspection PyUnresolvedReferences
@session.command("run-training")
@click.option(
    "-t",
    "--maximum_time",
    type=int,
    help="The maximum time to run the training session, in minutes. Defaults to 40 minutes.",
)
@click.option(
    "-is",
    "--initial_speed",
    type=float,
    help=(
        "The initial speed, in centimeters per second, the animal must maintain to obtain water rewards. "
        "Defaults to 0.8 cm/s."
    ),
)
@click.option(
    "-id",
    "--initial_duration",
    type=float,
    help=(
        "The initial duration, in seconds, the animal must maintain above-threshold running speed to obtain water "
        "rewards. Defaults to 1.5 seconds."
    ),
)
@click.option(
    "-it",
    "--increase_threshold",
    type=float,
    help=(
        "The volume of water delivered to the animal, in milliliters, after which the speed and duration thresholds "
        "are increased by the specified step-sizes. This is used to make the training progressively harder for the "
        "animal over the course of the training session. Defaults to 0.1 mL."
    ),
)
@click.option(
    "-ss",
    "--speed_step",
    type=float,
    help=(
        "The amount, in centimeters per second, to increase the speed threshold each time the animal receives the "
        "volume of water specified by the 'increase-threshold' parameter. Defaults to 0.05 cm/s."
    ),
)
@click.option(
    "-ds",
    "--duration_step",
    type=float,
    help=(
        "The amount, in seconds, to increase the duration threshold each time the animal receives the volume of water "
        "specified by the 'increase-threshold' parameter. Defaults to 0.1 seconds."
    ),
)
@click.option(
    "-v",
    "--maximum_volume",
    type=float,
    help="The maximum volume of water, in milliliters, that can be delivered during training. Defaults to 1.0 mL.",
)
@click.option(
    "-mit",
    "--maximum_idle_time",
    type=float,
    help=(
        "The maximum time, in seconds, the animal is allowed to maintain the speed that is below the speed threshold "
        "and still receive the water reward. Setting this argument to 0 forces the animal to maintain the "
        "above-threshold speed at all times. Defaults to 0.3 seconds."
    ),
)
@click.option(
    "-ur",
    "--unconsumed_rewards",
    type=int,
    help=(
        "The maximum number of rewards that can be delivered without the animal consuming them. If the unconsumed "
        "reward count exceeds this threshold, the system stops delivering new water rewards until the animal consumes "
        "the already delivered rewards. Setting this argument to 0 disables the reward consumption tracking. "
        "Defaults to 1."
    ),
)
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
) -> None:
    """Runs the run training session.

    Run training is the second phase of preparing the animal for experiment sessions, and is usually carried out over
    the five days following the lick training sessions. This session teaches the animal to run on the wheel treadmill
    while being head-fixed and associate getting water rewards with running on the treadmill. Over the course of
    training, the task requirements are adjusted to prepare the animal to perform as many laps as possible during
    experiment sessions lasting ~60 minutes.
    """
    run_training_logic(
        experimenter=ctx.obj["user"],
        project_name=ctx.obj["project"],
        animal_id=ctx.obj["animal"],
        animal_weight=ctx.obj["animal_weight"],
        initial_speed_threshold=initial_speed,
        initial_duration_threshold=initial_duration,
        speed_increase_step=speed_step,
        duration_increase_step=duration_step,
        increase_threshold=increase_threshold,
        maximum_water_volume=maximum_volume,
        maximum_training_time=maximum_time,
        maximum_unconsumed_rewards=unconsumed_rewards,
        maximum_idle_time=maximum_idle_time,
    )


# noinspection PyUnresolvedReferences
@session.command("experiment")
@click.option(
    "-e",
    "--experiment",
    type=str,
    required=True,
    help="The name of the experiment to carry out during runtime.",
)
@click.option(
    "-ur",
    "--unconsumed_rewards",
    type=int,
    help=(
        "The maximum number of rewards that can be delivered without the animal consuming them. If the unconsumed "
        "reward count exceeds this threshold, the system stops delivering new water rewards until the animal consumes "
        "the already delivered rewards. Setting this argument to 0 disables the reward consumption tracking."
    ),
)
@click.pass_context
def run_experiment(ctx: click.Context, experiment: str, unconsumed_rewards: int | None) -> None:
    """Runs the specified experiment session.

    Experiment runtimes are carried out after the lick and run training sessions. This command runs any experiment
    configuration supported by the data acquisition system managed by the host-machine. To create a new experiment
    configuration for the local data-acquisition system, use the 'sle configure experiment' subcommand.
    """
    experiment_logic(
        experimenter=ctx.obj["user"],
        project_name=ctx.obj["project"],
        experiment_name=experiment,
        animal_id=ctx.obj["animal"],
        animal_weight=ctx.obj["animal_weight"],
        maximum_unconsumed_rewards=unconsumed_rewards,
    )
