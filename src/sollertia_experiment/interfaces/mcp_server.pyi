from typing import Literal

__all__ = ["run_server"]

def run_server(transport: Literal["stdio", "streamable-http"] = "stdio") -> None: ...
