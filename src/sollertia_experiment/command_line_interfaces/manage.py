"""Provides the 'sle manage' subcommand for managing the data accessible to the data acquisition system managed by
the host-machine.
"""

from pathlib import Path

import click
from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, get_data_root

from .mcp_servers import run_manage_server
from ..mesoscope_vr import (
    purge_session,
    preprocess_session_data,
    get_system_configuration,
    migrate_animal_between_projects,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}  # pragma: no cover


@click.group("manage", context_settings=CONTEXT_SETTINGS)
def manage() -> None:  # pragma: no cover
    """Manages the data accessible to the data acquisition system managed by the local host-machine."""


@manage.command("preprocess")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    prompt="Enter the path to the target data acquisition session's directory: ",
    help="The path to the data acquisition session's directory to preprocess.",
)
def preprocess_session(session_path: Path) -> None:
    """Preprocesses the target session's data stored on the data acquisition system's host-machine."""
    system_configuration = get_system_configuration()  # Retrieves the system configuration data.
    data_root = get_data_root()

    # Prevent using this command on sessions that are not stored on the local host-machine, but accessible to its
    # filesystem. Specifically, prevents working with sessions stored on long-term storage destinations.
    message = (
        f"Unable to preprocess the session's directory stored at the {session_path} path. The session's directory must "
        f"be located inside the data root of the {system_configuration.name} data acquisition system "
        f"({data_root})."
    )
    if not session_path.is_relative_to(data_root):
        console.error(message=message, error=FileNotFoundError)

    # Loads the SessionData instance for the processed session.
    session_data = SessionData.load(session_path=session_path)
    preprocess_session_data(session_data)  # Runs the preprocessing logic.


@manage.command("delete")
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    prompt="Enter the path to the target data acquisition session's directory: ",
    help="The path to the data acquisition session's directory to remove.",
)
def delete_session(session_path: Path) -> None:
    """Removes the target session's data from all destinations accessible to the data acquisition system.

    This is an extremely dangerous command that can potentially delete valuable data if used carelessly. This command
    removes the session's data from all machines of the data acquisition system and all long-term storage destinations
    accessible to the data acquisition system.
    """
    system_configuration = get_system_configuration()  # Retrieves the system configuration data.
    data_root = get_data_root()

    # Ensures that the command can only target sessions stored on the local host-machine. While this does not make the
    # command safe, it reduces the risk of accidentally removing valid scientific data.
    message = (
        f"Unable to preprocess the session's directory stored at the {session_path} path. The session's directory must "
        f"be located inside the data root of the {system_configuration.name} data acquisition system "
        f"({data_root})."
    )
    if not session_path.is_relative_to(data_root):
        console.error(message=message, error=FileNotFoundError)

    # Removes all data of the target session from all data acquisition and long-term storage machines accessible to the
    # host-machine.
    session_data = SessionData.load(session_path=session_path)
    purge_session(session_data)


@manage.command("migrate")
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
def migrate_animal(source: str, destination: str, animal: str) -> None:
    """Transfers all sessions for the specified animal from the source project to the target project."""
    migrate_animal_between_projects(source_project=source, target_project=destination, animal=animal)


@manage.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=str,
    default="stdio",
    show_default=True,
    help="The MCP transport type ('stdio', 'sse', or 'streamable-http').",
)
def start_manage_mcp_server(transport: str) -> None:  # pragma: no cover
    """Starts the MCP server for agentic access to 'sle manage' tools."""
    run_manage_server(transport=transport)  # type: ignore[arg-type]
