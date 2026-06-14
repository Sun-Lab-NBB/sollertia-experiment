"""Provides the 'sle mesoscope' command group for configuring, running, and managing the Mesoscope-VR data acquisition
system.

This module combines all Mesoscope-VR-specific interfaces into a single command group: system configuration
('configure'), system maintenance ('maintain'), data acquisition sessions ('run'), and session data management
('preprocess', 'delete', 'migrate'). The general, hardware-agnostic discovery commands are exposed separately via the
'sle get' command group.
"""

from pathlib import Path

import click
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import SessionData, get_data_root

from ..mesoscope_vr import (
    purge_session,
    experiment_logic,
    maintenance_logic,
    run_training_logic,
    lick_training_logic,
    window_checking_logic,
    check_mesoscope_bridge,
    preprocess_session_data,
    get_system_configuration,
    migrate_animal_between_projects,
    create_system_configuration_file,
    create_experiment_configuration_file,
)

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("mesoscope", context_settings=CONTEXT_SETTINGS)
def mesoscope() -> None:  # pragma: no cover
    """Configures, runs, and manages the Mesoscope-VR data acquisition system.

    This command group exposes every Mesoscope-VR-specific runtime: generating the system configuration file,
    performing system maintenance, running data acquisition sessions, and managing the data collected by the system.
    """


@mesoscope.group("configure")
def configure() -> None:  # pragma: no cover
    """Generates Mesoscope-VR configuration files.

    Exposes two configuration targets: 'system' creates the data acquisition system configuration file that binds the
    host-machine to the Mesoscope-VR system, and 'experiment' creates per-experiment configuration files from Unity
    task templates.
    """


@configure.command("system")
def configure_system() -> None:  # pragma: no cover
    """Creates the Mesoscope-VR data acquisition system configuration file under the working directory."""
    create_system_configuration_file()


@configure.command("experiment")
@click.option(
    "-p",
    "--project",
    type=str,
    required=True,
    help="The name of the project for which to generate the new experiment configuration file.",
)
@click.option(
    "-e",
    "--experiment",
    type=str,
    required=True,
    help="The name of the experiment for which to create the configuration file.",
)
@click.option(
    "-t",
    "--template",
    type=str,
    required=True,
    help="The name of the task template to use (filename without .yaml extension).",
)
@click.option(
    "-sc",
    "--state-count",
    type=int,
    default=1,
    show_default=True,
    help="The number of runtime states supported by the experiment.",
)
@click.option(
    "--reward-size",
    type=float,
    default=5.0,
    show_default=True,
    help="Default water reward volume in microliters for lick-type trials.",
)
@click.option(
    "--reward-tone-duration",
    type=int,
    default=300,
    show_default=True,
    help="Default reward tone duration in milliseconds for lick-type trials.",
)
@click.option(
    "--puff-duration",
    type=int,
    default=100,
    show_default=True,
    help="Default gas puff duration in milliseconds for occupancy-type trials.",
)
def configure_experiment(
    project: str,
    experiment: str,
    template: str,
    state_count: int,
    reward_size: float,
    reward_tone_duration: int,
    puff_duration: int,
) -> None:  # pragma: no cover
    """Creates a Mesoscope-VR experiment configuration from a task template under the configured data root."""
    create_experiment_configuration_file(
        project=project,
        experiment=experiment,
        template=template,
        state_count=state_count,
        reward_size=reward_size,
        reward_tone_duration=reward_tone_duration,
        puff_duration=puff_duration,
    )


@mesoscope.command("maintain")
def maintain() -> None:
    """Runs the data acquisition system maintenance session.

    Calling this command exposes a GUI for directly interfacing with a small subset of the managed data acquisition
    system's components that require frequent maintenance. It does not collect any data during runtime and does
    not interface with the remote data storage infrastructure accessible to the data acquisition system. It is
    designed to perform minor (day-to-day) maintenance tasks that do not require disassembling the system's components.
    """
    maintenance_logic()


@mesoscope.command("check-bridge")
def check_bridge() -> None:
    """Checks whether the ScanImagePC's runAcquisition control loop is reachable for Mesoscope imaging sessions.

    The runAcquisition function is a lock-in command loop the operator launches once on the ScanImagePC; it arms and
    commands the Mesoscope over MQTT for the entire runtime. An unreachable bridge means it is not running.
    """
    try:
        reachable, status = check_mesoscope_bridge()
    except Exception as exception:
        message = f"Unable to check the mesoscope control interface. {exception}"
        console.echo(message=message, level=LogLevel.WARNING)
        return
    console.echo(message=status, level=LogLevel.SUCCESS if reachable else LogLevel.WARNING)


@mesoscope.group("run")
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
    "--animal-weight",
    type=float,
    required=True,
    help="The weight of the animal, in grams, at the beginning of the session.",
)
@click.pass_context
def run(ctx: click.Context, user: str, project: str, animal: str, animal_weight: float) -> None:  # pragma: no cover
    """Runs the specified data acquisition session for the target animal and project combination."""
    # Stores common parameters in the context dictionary to be accessible from the subcommands.
    ctx.ensure_object(dict)
    ctx.obj["user"] = user
    ctx.obj["project"] = project
    ctx.obj["animal"] = animal
    ctx.obj["animal_weight"] = animal_weight


@run.command("window-checking")
@click.pass_context
def window_checking(ctx: click.Context) -> None:
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


@run.command("lick-training")
@click.option(
    "-t",
    "--maximum-time",
    type=int,
    help="The maximum time to run the training session, in minutes. Defaults to 20 minutes.",
)
@click.option(
    "-min",
    "--minimum-delay",
    type=int,
    help=(
        "The minimum number of seconds that has to pass between two consecutive reward deliveries during training. "
        "Defaults to 6 seconds."
    ),
)
@click.option(
    "-max",
    "--maximum-delay",
    type=int,
    help=(
        "The maximum number of seconds that can pass between two consecutive reward deliveries during training. "
        "Defaults to 18 seconds."
    ),
)
@click.option(
    "-v",
    "--maximum-volume",
    type=float,
    help="The maximum volume of water, in milliliters, that can be delivered during training. Defaults to 1.0 mL.",
)
@click.option(
    "-ur",
    "--unconsumed-rewards",
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


@run.command("run-training")
@click.option(
    "-t",
    "--maximum-time",
    type=int,
    help="The maximum time to run the training session, in minutes. Defaults to 40 minutes.",
)
@click.option(
    "-is",
    "--initial-speed",
    type=float,
    help=(
        "The initial speed, in centimeters per second, the animal must maintain to obtain water rewards. "
        "Defaults to 0.8 cm/s."
    ),
)
@click.option(
    "-id",
    "--initial-duration",
    type=float,
    help=(
        "The initial duration, in seconds, the animal must maintain above-threshold running speed to obtain water "
        "rewards. Defaults to 1.5 seconds."
    ),
)
@click.option(
    "-it",
    "--increase-threshold",
    type=float,
    help=(
        "The volume of water delivered to the animal, in milliliters, after which the speed and duration thresholds "
        "are increased by the specified step-sizes. This is used to make the training progressively harder for the "
        "animal over the course of the training session. Defaults to 0.1 mL."
    ),
)
@click.option(
    "-ss",
    "--speed-step",
    type=float,
    help=(
        "The amount, in centimeters per second, to increase the speed threshold each time the animal receives the "
        "volume of water specified by the 'increase-threshold' parameter. Defaults to 0.05 cm/s."
    ),
)
@click.option(
    "-ds",
    "--duration-step",
    type=float,
    help=(
        "The amount, in seconds, to increase the duration threshold each time the animal receives the volume of water "
        "specified by the 'increase-threshold' parameter. Defaults to 0.1 seconds."
    ),
)
@click.option(
    "-v",
    "--maximum-volume",
    type=float,
    help="The maximum volume of water, in milliliters, that can be delivered during training. Defaults to 1.0 mL.",
)
@click.option(
    "-mit",
    "--maximum-idle-time",
    type=float,
    help=(
        "The maximum time, in seconds, the animal is allowed to maintain the speed that is below the speed threshold "
        "and still receive the water reward. Setting this argument to 0 forces the animal to maintain the "
        "above-threshold speed at all times. Defaults to 0.3 seconds."
    ),
)
@click.option(
    "-ur",
    "--unconsumed-rewards",
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


@run.command("experiment")
@click.option(
    "-e",
    "--experiment",
    type=str,
    required=True,
    help="The name of the experiment to carry out during runtime.",
)
@click.option(
    "-ur",
    "--unconsumed-rewards",
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
    configuration for the local data-acquisition system, use the 'sle mesoscope configure experiment' subcommand.
    """
    experiment_logic(
        experimenter=ctx.obj["user"],
        project_name=ctx.obj["project"],
        experiment_name=experiment,
        animal_id=ctx.obj["animal"],
        animal_weight=ctx.obj["animal_weight"],
        maximum_unconsumed_rewards=unconsumed_rewards,
    )


@mesoscope.command("preprocess")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    prompt="Enter the path to the target data acquisition session's directory: ",
    help="The path to the data acquisition session's directory to preprocess.",
)
def preprocess(session_path: Path) -> None:
    """Preprocesses the target session's data stored on the data acquisition system's host-machine."""
    system_configuration = get_system_configuration()
    data_root = get_data_root()

    # Prevents using this command on sessions that are not stored on the local host-machine, but accessible to its
    # filesystem. Specifically, prevents working with sessions stored on long-term storage destinations.
    message = (
        f"Unable to preprocess the session's directory stored at the {session_path} path. The session's directory must "
        f"be located inside the data root of the {system_configuration.name} data acquisition system "
        f"({data_root})."
    )
    if not session_path.is_relative_to(data_root):
        console.error(message=message, error=FileNotFoundError)

    session_data = SessionData.load(session_path=session_path)
    preprocess_session_data(session_data)


@mesoscope.command("delete")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    prompt="Enter the path to the target data acquisition session's directory: ",
    help="The path to the data acquisition session's directory to remove.",
)
def delete(session_path: Path) -> None:
    """Removes the target session's data from all destinations accessible to the data acquisition system.

    This is an extremely dangerous command that can potentially delete valuable data if used carelessly. This command
    removes the session's data from all machines of the data acquisition system and all long-term storage destinations
    accessible to the data acquisition system.
    """
    system_configuration = get_system_configuration()
    data_root = get_data_root()

    # Ensures that the command can only target sessions stored on the local host-machine. While this does not make the
    # command safe, it reduces the risk of accidentally removing valid scientific data.
    message = (
        f"Unable to delete the session's directory stored at the {session_path} path. The session's directory must "
        f"be located inside the data root of the {system_configuration.name} data acquisition system "
        f"({data_root})."
    )
    if not session_path.is_relative_to(data_root):
        console.error(message=message, error=FileNotFoundError)

    # Removes all data of the target session from all data acquisition and long-term storage machines accessible to the
    # host-machine.
    session_data = SessionData.load(session_path=session_path)
    purge_session(session_data)


@mesoscope.command("migrate")
@click.option(
    "-s",
    "--source",
    type=str,
    required=True,
    help="The name of the project from which to migrate the data.",
)
@click.option(
    "-d",
    "--destination",
    type=str,
    required=True,
    help="The name of the project to which to migrate the data.",
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=True,
    help="The ID of the animal whose data to migrate.",
)
def migrate(source: str, destination: str, animal: str) -> None:
    """Transfers all sessions for the specified animal from the source project to the target project."""
    migrate_animal_between_projects(source_project=source, target_project=destination, animal=animal)
