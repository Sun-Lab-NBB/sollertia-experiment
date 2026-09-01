"""Provides the shared MCP server instance and helper functions for the sollertia-experiment MCP tool modules."""

from __future__ import annotations

from enum import Enum
import uuid
from types import UnionType
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints
from pathlib import Path
import contextlib
from dataclasses import MISSING, fields, is_dataclass

import yaml
from mcp.server import MCPServer
from ataraxis_base_utilities import ensure_directory_exists

if TYPE_CHECKING:
    from ataraxis_data_structures import YamlConfig

_MAPPING_ARGUMENT_COUNT: int = 2
"""The number of type arguments a mapping annotation carries, which is its key type followed by its value type."""

mcp: MCPServer = MCPServer(name="sollertia-experiment")
"""The shared MCP server instance on which all tool modules register their tools via ``@mcp.tool()``."""


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

    Notes:
        Validation rejects a payload key that names no field of ``validator_cls`` and a built field whose value
        violates its declared annotation, since the deserializer drops the unknown key and keeps the mismatched value
        as written.

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

    unknown_keys = _unknown_payload_keys(payload=payload, cls=validator_cls)
    if unknown_keys:
        return {
            "error": (
                f"Validation failed for {validator_cls.__name__}: the payload carries the unknown key(s) "
                f"{', '.join(unknown_keys)}."
            )
        }

    ensure_directory_exists(path=file_path.parent)
    temporary_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        temporary_path.write_text(yaml.safe_dump(data=payload, sort_keys=False))
        instance = validator_cls.from_yaml(file_path=temporary_path)
        if hasattr(instance, "__post_init__"):
            instance.__post_init__()
    except Exception as exception:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        return {"error": f"Validation failed for {validator_cls.__name__}: {exception}"}
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()

    mismatches = _field_type_mismatches(instance=instance, prefix="")
    if mismatches:
        return {"error": f"Validation failed for {validator_cls.__name__}: {', '.join(mismatches)}."}

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


def _unknown_payload_keys(payload: dict[str, Any], cls: type) -> list[str]:
    """Returns the dotted names of the payload keys that name no field of the target dataclass.

    Notes:
        The walk descends into a nested mapping whenever the field it fills is annotated with a dataclass type, so a
        key misspelled inside a nested section is reported with the full path that locates it in the payload.

    Args:
        payload: The mapping intended to build an instance of the target dataclass.
        cls: The dataclass type the payload is checked against.

    Returns:
        The dotted name of every payload key the dataclass does not declare, in payload order.
    """
    if not is_dataclass(cls):
        return []

    # An annotation naming a type this module cannot resolve leaves the field without a nested walk rather than
    # failing the whole check.
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    field_names = {field_definition.name for field_definition in fields(cls)}
    unknown_keys: list[str] = []
    for key, item in payload.items():
        if key not in field_names:
            unknown_keys.append(key)
            continue
        annotation = hints.get(key)
        nested_class = next(
            (
                candidate
                for candidate in (annotation, *get_args(annotation))
                if isinstance(candidate, type) and is_dataclass(candidate)
            ),
            None,
        )
        if nested_class is not None and isinstance(item, dict):
            unknown_keys.extend(f"{key}.{name}" for name in _unknown_payload_keys(payload=item, cls=nested_class))
    return unknown_keys


def _field_type_mismatches(instance: Any, prefix: str) -> list[str]:
    """Returns a description of every field of the dataclass instance whose value violates its declared annotation.

    Args:
        instance: The dataclass instance whose field values are checked.
        prefix: The dotted path that locates the instance inside the payload it was built from, empty for the root.

    Returns:
        One description per violating field, naming the dotted field path, the type of the stored value, and the
        declared annotation.
    """
    try:
        hints = get_type_hints(type(instance))
    except Exception:
        hints = {}

    mismatches: list[str] = []
    for field_definition in fields(instance):
        value = getattr(instance, field_definition.name)
        name = f"{prefix}{field_definition.name}"
        annotation = hints.get(field_definition.name)
        if annotation is not None and not _annotation_matches(value=value, annotation=annotation):
            expected = annotation.__name__ if isinstance(annotation, type) else str(annotation).replace("typing.", "")
            mismatches.append(f"{name} is {type(value).__name__}, expected {expected}")
            continue
        if is_dataclass(value) and not isinstance(value, type):
            mismatches.extend(_field_type_mismatches(instance=value, prefix=f"{name}."))
    return mismatches


def _annotation_matches(value: Any, annotation: Any) -> bool:
    """Determines whether the value satisfies the declared type annotation.

    Notes:
        A parameterized tuple, list, or mapping annotation is checked against its origin type and its element types,
        and every other parameterized annotation is checked against its origin type alone. An integer satisfies a float
        annotation, following the numeric tower the typing specification defines, except a boolean, which is rejected
        despite subclassing int. An annotation the runtime cannot
        reduce to a type is treated as satisfied, since rejecting it would refuse a value the dataclass itself accepts.

    Args:
        value: The value to check.
        annotation: The declared annotation the value is checked against.

    Returns:
        True when the value satisfies the annotation.
    """
    if annotation is Any or annotation is object:
        return True

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is UnionType:
        return any(_annotation_matches(value=value, annotation=argument) for argument in arguments)

    if origin is None:
        if annotation is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, annotation) if isinstance(annotation, type) else True

    # A special form such as Literal reduces to an origin that isinstance() cannot take, so a value annotated that way
    # is accepted without a container check.
    if not isinstance(origin, type):
        return True

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if not arguments:
            return True
        if arguments[-1] is Ellipsis:
            return all(_annotation_matches(value=item, annotation=arguments[0]) for item in value)
        return len(arguments) == len(value) and all(
            _annotation_matches(value=item, annotation=argument)
            for item, argument in zip(value, arguments, strict=True)
        )

    if origin is list:
        if not isinstance(value, list):
            return False
        return not arguments or all(_annotation_matches(value=item, annotation=arguments[0]) for item in value)

    if origin is dict:
        if not isinstance(value, dict):
            return False
        return len(arguments) != _MAPPING_ARGUMENT_COUNT or all(
            _annotation_matches(value=key, annotation=arguments[0])
            and _annotation_matches(value=item, annotation=arguments[1])
            for key, item in value.items()
        )

    return isinstance(value, origin)
