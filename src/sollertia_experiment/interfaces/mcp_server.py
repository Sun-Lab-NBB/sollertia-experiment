"""Provides the MCP server for agentic interaction with the sollertia-experiment acquisition system functionality."""

from __future__ import annotations

from typing import Literal
from pathlib import Path
import importlib

from .mcp_instance import mcp

__all__ = ["run_server"]


def run_server(transport: Literal["stdio", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration) and 'streamable-http' for HTTP-based
            communication.
    """
    # Delegates to the MCPServer run loop, which blocks until the transport connection is closed. For 'stdio' this
    # means the server runs until the parent process closes stdin. For 'streamable-http' it runs an HTTP server that
    # accepts connections until explicitly terminated.
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        mcp.run(transport=transport, json_response=True)
        return

    mcp.run(transport=transport)


def _register_tool_modules() -> None:
    """Imports every ``*_tools`` module in this package so its ``@mcp.tool()`` decorators register on import.

    Tool modules register their MCP tools purely as an import side effect. Discovering them by the ``_tools`` filename
    suffix means each acquisition system's ``<system>_tools.py`` module, alongside the hardware-agnostic ``get_tools``
    module, registers automatically, so adding a new system requires no edit to this module.
    """
    package_name = __name__.rpartition(".")[0]
    for module_path in sorted(Path(__file__).parent.glob("*_tools.py")):
        importlib.import_module(f"{package_name}.{module_path.stem}")


_register_tool_modules()
