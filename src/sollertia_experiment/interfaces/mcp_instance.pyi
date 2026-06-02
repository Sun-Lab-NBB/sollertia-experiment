from typing import Any
from pathlib import Path

from _typeshed import Incomplete
from ataraxis_data_structures import YamlConfig as YamlConfig

mcp: Incomplete

def serialize(value: Any) -> Any: ...
def describe_dataclass(cls, *, seen: frozenset[type] | None = None) -> dict[str, Any]: ...
def write_yaml_validated(
    file_path: Path,
    payload: dict[str, Any],
    validator_cls: type[YamlConfig],
    *,
    overwrite: bool = False,
    use_save_method: bool = False,
) -> dict[str, Any]: ...
def read_yaml(file_path: Path, validator_cls: type[YamlConfig]) -> dict[str, Any]: ...
def probe_writable(path: Path) -> str | None: ...
