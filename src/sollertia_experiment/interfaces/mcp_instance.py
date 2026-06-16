"""Provides the shared FastMCP instance and helper functions for the sollertia-experiment MCP tool modules."""

from __future__ import annotations

from enum import Enum
import uuid
from typing import TYPE_CHECKING, Any, get_type_hints
from pathlib import Path
import contextlib
from dataclasses import MISSING, fields, is_dataclass

import yaml  # type: ignore[import-untyped]
from mcp.server.fastmcp import FastMCP
from ataraxis_base_utilities import ensure_directory_exists

if TYPE_CHECKING:
    from ataraxis_data_structures import YamlConfig

mcp = FastMCP(name="sollertia-experiment", json_response=True)
"""The shared FastMCP server instance on which all tool modules register their tools via ``@mcp.tool()``."""


def serialize(value: Any) -> Any:
    """Recursively converts a dataclass, Path, Enum, mapping, or sequence into JSON-friendly Python.

    Args:
        value: The object to convert. Dataclasses, paths, enumerations, mappings, and sequences are converted
            recursively, while all other values are returned unchanged.

    Returns:
        The JSON-serializable representation of the input value.
    """
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_definition.name: serialize(value=getattr(value, field_definition.name))
            for field_definition in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): serialize(value=item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize(value=item) for item in value]
    return value


def describe_dataclass(cls: type, *, seen: frozenset[type] | None = None) -> dict[str, Any]:
    """Returns a structured schema description of a dataclass type, recursively describing nested dataclasses.

    Args:
        cls: The dataclass type to describe.
        seen: The set of dataclass types already visited, used to guard against infinite recursion on
            self-referential schemas.

    Returns:
        A mapping describing the class name and each field's type, default value, and nested schema.
    """
    seen = frozenset() if seen is None else seen
    if cls in seen:
        return {"class": cls.__name__, "recursive_reference": True}
    if not is_dataclass(cls):
        type_name = cls.__name__ if isinstance(cls, type) else str(cls)
        return {"type": type_name}

    next_seen = seen | {cls}
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    schema: dict[str, Any] = {"class": cls.__name__, "fields": {}}
    for field_definition in fields(cls):
        type_hint = hints.get(field_definition.name, field_definition.type)
        type_name = type_hint.__name__ if isinstance(type_hint, type) else str(type_hint).replace("typing.", "")
        field_schema: dict[str, Any] = {"type": type_name}
        if field_definition.default is not MISSING:
            field_schema["default"] = serialize(value=field_definition.default)
        elif field_definition.default_factory is not MISSING:
            try:
                field_schema["default"] = serialize(value=field_definition.default_factory())
            except Exception:
                field_schema["required"] = True
        else:
            field_schema["required"] = True
        if isinstance(type_hint, type) and is_dataclass(type_hint):
            field_schema["nested"] = describe_dataclass(cls=type_hint, seen=next_seen)
        schema["fields"][field_definition.name] = field_schema
    return schema


def write_yaml_validated(
    file_path: Path,
    payload: dict[str, Any],
    validator_cls: type[YamlConfig],
    *,
    overwrite: bool = False,
    use_save_method: bool = False,
) -> dict[str, Any]:
    """Writes a payload as YAML and validates it by round-tripping through ``validator_cls``.

    Args:
        file_path: The path to the YAML file to write.
        payload: The data to serialize into the YAML file.
        validator_cls: The YamlConfig subclass used to validate the written payload.
        overwrite: Determines whether to replace an existing file at the target path.
        use_save_method: Determines whether to persist via the instance's save() method instead of to_yaml().

    Returns:
        A mapping with the written file path and serialized data on success, or an error description on failure.
    """
    if file_path.exists() and not overwrite:
        return {"error": f"File already exists: {file_path}. Pass overwrite=True to replace."}

    ensure_directory_exists(path=file_path.parent)
    temp_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        temp_path.write_text(yaml.safe_dump(data=payload, sort_keys=False))
        instance = validator_cls.from_yaml(file_path=temp_path)
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()
    except Exception as exception:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        return {"error": f"Validation failed for {validator_cls.__name__}: {exception}"}
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()

    try:
        if use_save_method and hasattr(instance, "save"):
            instance.save(path=file_path)
        else:
            instance.to_yaml(file_path=file_path)
    except Exception as exception:
        return {"error": f"Failed to persist {validator_cls.__name__} to {file_path}: {exception}"}

    return {"file_path": str(file_path), "data": serialize(value=instance)}


def read_yaml(file_path: Path, validator_cls: type[YamlConfig]) -> dict[str, Any]:
    """Loads a YAML file via ``validator_cls`` and returns its serialized form.

    Args:
        file_path: The path to the YAML file to load.
        validator_cls: The YamlConfig subclass used to parse and validate the file.

    Returns:
        A mapping with the file path and serialized data on success, or an error description on failure.
    """
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    try:
        instance = validator_cls.from_yaml(file_path=file_path)
    except Exception as exception:
        return {"error": f"Failed to load {file_path} as {validator_cls.__name__}: {exception}"}
    return {"file_path": str(file_path), "data": serialize(value=instance)}


def probe_writable(path: Path) -> str | None:
    """Probes write access to a directory by creating and removing a uniquely-named temporary file.

    Args:
        path: The directory whose write access is probed.

    Returns:
        None when the directory is writable, or a human-readable reason describing why it is not.
    """
    probe = path.joinpath(f".sollertia_experiment_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.touch()
        probe.unlink()
    except OSError as exception:
        return str(exception)
    return None
