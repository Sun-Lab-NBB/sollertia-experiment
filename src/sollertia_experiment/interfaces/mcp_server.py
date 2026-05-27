"""Provides the MCP server for agentic interaction with the sollertia-experiment acquisition system functionality."""

from __future__ import annotations

from typing import Literal

# noinspection PyUnusedImports
from . import (
    get_tools,  # noqa: F401 - imported to trigger MCP tool registration.
    mesoscope_vr_tools,  # noqa: F401 - imported to trigger MCP tool registration.
)
from .mcp_instance import mcp

__all__ = ["run_server"]


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport protocol to use. Supported values are 'stdio' for standard input/output
            communication (recommended for Claude Desktop integration), 'sse' for Server-Sent Events,
            and 'streamable-http' for HTTP-based communication.
    """
    mcp.run(transport=transport)
