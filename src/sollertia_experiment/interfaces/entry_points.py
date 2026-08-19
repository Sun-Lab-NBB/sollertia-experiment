"""Provides the consolidated 'sle' CLI entry point for the sollertia-experiment library.

The warning filter is applied at module level before any other imports to ensure deprecation warnings from
dependencies are suppressed during the import phase.
"""

import os
from typing import Literal
import warnings

warnings.warn_explicit = warnings.warn = lambda *_, **__: None

# Silences the benign Qt teardown warnings (e.g., "QObject::killTimer: Timers cannot be stopped from another thread")
# that OpenCV's Qt backend writes to stderr when the camera-preview windows are destroyed in the video acquisition
# subprocesses. The variable is set before any subprocess is spawned, so every child inherits it; setdefault preserves
# any value the operator has already exported.
os.environ.setdefault("QT_LOGGING_RULES", "default.warning=false")

import click  # noqa: E402
from ataraxis_base_utilities import LogLevel, console  # noqa: E402

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""Ensures that displayed Click help messages are formatted according to the lab standard."""


@click.group("sle", context_settings=CONTEXT_SETTINGS)
def sle_cli() -> None:  # pragma: no cover
    """Top-level entry point for the sollertia-experiment library.

    Exposes two operational command groups: 'get' for general, hardware-agnostic acquisition system discovery, and
    'mesoscope' for configuring, running, and managing the Mesoscope-VR data acquisition system. The 'mcp' command
    starts a single MCP server that exposes the tools backing both groups to AI agents.
    """


@sle_cli.command("mcp", context_settings=CONTEXT_SETTINGS)
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "streamable-http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="The MCP transport type to use.",
)
def mcp(transport: Literal["stdio", "streamable-http"]) -> None:  # pragma: no cover
    """Starts the MCP server for agentic access to the 'sle get' and 'sle mesoscope' tools."""
    from .mcp_server import run_server  # noqa: PLC0415

    # The stdio transport carries the JSON-RPC message stream over stdout, which is also where the console writes
    # every message up to the WARNING level. Silencing the console keeps library output out of that stream, as a
    # single logged line renders the message it interleaves with unparsable for the connected client.
    if transport == "stdio":
        console.disable()
    else:
        console.echo(
            message=f"Starting the sollertia-experiment MCP server with the {transport} transport.",
            level=LogLevel.INFO,
        )

    run_server(transport=transport)


def _register_subcommands() -> None:
    """Imports and registers all subcommand groups on the top-level 'sle' Click group.

    The imports are deferred to this helper to keep the module's import surface minimal when only metadata is
    needed by the wheel build or by tools like importlib.metadata.
    """
    from .get import get  # noqa: PLC0415
    from .mesoscope_vr import mesoscope  # noqa: PLC0415

    sle_cli.add_command(cmd=get)
    sle_cli.add_command(cmd=mesoscope)


_register_subcommands()
