"""YAML loader for a single library file: file path → :class:`LibrarySpec`.

Schema-validates the structure (required keys present, types correct, body
references syntactically resolvable as strings) and produces a
:class:`LibrarySpec`. Includes are *parsed* (kept as a tuple of path strings
on the resulting spec) but not *resolved* — that happens in the include
linker (B3). Nothing in this module instantiates runtime ``Body`` /
``Constraint`` / ``ConstraintTransition`` objects; that's the validator's
(B5) job once the full library is assembled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

from comb.spec.library import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    LibrarySpec,
    PoseSpec,
    TransitionSpec,
)


class LibraryLoadError(Exception):
    """Raised when a library YAML file is malformed.

    The error message includes a dotted source path (e.g.
    ``arm.lib.yaml:bodies.link_a.pose``) pointing at the field that failed
    validation, so debugging spec files doesn't require staring at line
    numbers.
    """


def load_library_file(path: str | Path) -> LibrarySpec:
    """Read ``path`` and parse it into a :class:`LibrarySpec`."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LibraryLoadError(f"{file_path}: failed to parse YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise LibraryLoadError(
            f"{file_path}: top-level must be a mapping, got " f"{type(data).__name__}"
        )
    return _parse_library(data, source=str(file_path))


_T = TypeVar("_T")


def _parse_library(data: Mapping[str, Any], *, source: str) -> LibrarySpec:
    return LibrarySpec(
        name=_require_str(data, "name", source=source),
        includes=_parse_str_tuple(
            data.get("includes", []), source=f"{source}:includes"
        ),
        bodies=_parse_named_mapping(
            data.get("bodies", {}), _parse_body, source=f"{source}:bodies"
        ),
        constraints=_parse_named_mapping(
            data.get("constraints", {}),
            _parse_constraint,
            source=f"{source}:constraints",
        ),
        transitions=_parse_named_mapping(
            data.get("transitions", {}),
            _parse_transition,
            source=f"{source}:transitions",
        ),
    )


def _parse_body(data: Mapping[str, Any], *, source: str) -> BodySpec:
    return BodySpec(
        visual_geometry=_parse_geometry(
            _require(data, "visual_geometry", source=source),
            source=f"{source}.visual_geometry",
        ),
        collision_geometry=_parse_geometry(
            _require(data, "collision_geometry", source=source),
            source=f"{source}.collision_geometry",
        ),
        pose=_parse_pose(
            _require(data, "pose", source=source), source=f"{source}.pose"
        ),
        anchored=_parse_bool(data.get("anchored", False), source=f"{source}.anchored"),
    )


def _parse_geometry(data: Any, *, source: str) -> GeometrySpec:
    if not isinstance(data, Mapping):
        raise LibraryLoadError(f"{source}: expected mapping, got {type(data).__name__}")
    shape = _require_str(data, "shape", source=source)
    parameters = {
        key: _parse_float(value, source=f"{source}.{key}")
        for key, value in data.items()
        if key != "shape"
    }
    return GeometrySpec(shape=shape, parameters=parameters)


def _parse_pose(data: Any, *, source: str) -> PoseSpec:
    if not isinstance(data, Mapping):
        raise LibraryLoadError(f"{source}: expected mapping, got {type(data).__name__}")
    values = {
        _require_string_key(key, source=source): _parse_float(
            value, source=f"{source}.{key}"
        )
        for key, value in data.items()
    }
    return PoseSpec(values=values)


def _parse_constraint(data: Mapping[str, Any], *, source: str) -> ConstraintSpec:
    return ConstraintSpec(
        type=_require_str(data, "type", source=source),
        body1=_require_str(data, "body1", source=source),
        body2=_require_str(data, "body2", source=source),
        fixed_parameters=_parse_float_mapping(
            data.get("fixed_parameters", {}), source=f"{source}.fixed_parameters"
        ),
        initial_parameters=_parse_float_mapping(
            data.get("initial_parameters", {}),
            source=f"{source}.initial_parameters",
        ),
    )


def _parse_transition(data: Mapping[str, Any], *, source: str) -> TransitionSpec:
    trigger_data = _require(data, "trigger", source=source)
    if not isinstance(trigger_data, Mapping):
        raise LibraryLoadError(
            f"{source}.trigger: expected mapping, got " f"{type(trigger_data).__name__}"
        )
    add_data = data.get("add", [])
    if not isinstance(add_data, list):
        raise LibraryLoadError(
            f"{source}.add: expected list, got {type(add_data).__name__}"
        )
    remove_data = data.get("remove", [])
    if not isinstance(remove_data, list):
        raise LibraryLoadError(
            f"{source}.remove: expected list, got {type(remove_data).__name__}"
        )
    return TransitionSpec(
        trigger=_parse_constraint(trigger_data, source=f"{source}.trigger"),
        tolerance=_parse_float(
            _require(data, "tolerance", source=source),
            source=f"{source}.tolerance",
        ),
        add=tuple(
            _parse_generator_call(item, source=f"{source}.add[{i}]")
            for i, item in enumerate(add_data)
        ),
        remove=tuple(
            _parse_str_in_seq(item, source=f"{source}.remove[{i}]")
            for i, item in enumerate(remove_data)
        ),
    )


def _parse_generator_call(data: Any, *, source: str) -> GeneratorCallSpec:
    if not isinstance(data, Mapping):
        raise LibraryLoadError(f"{source}: expected mapping, got {type(data).__name__}")
    args_data = data.get("args", {})
    if not isinstance(args_data, Mapping):
        raise LibraryLoadError(
            f"{source}.args: expected mapping, got {type(args_data).__name__}"
        )
    return GeneratorCallSpec(
        generator=_require_str(data, "generator", source=source),
        args={_require_string_key(k, source=source): v for k, v in args_data.items()},
    )


# --- helpers ---


def _require(data: Mapping[str, Any], key: str, *, source: str) -> Any:
    if key not in data:
        raise LibraryLoadError(f"{source}: missing required key {key!r}")
    return data[key]


def _require_str(data: Mapping[str, Any], key: str, *, source: str) -> str:
    value = _require(data, key, source=source)
    if not isinstance(value, str):
        raise LibraryLoadError(
            f"{source}.{key}: expected string, got {type(value).__name__}"
        )
    return value


def _require_string_key(key: Any, *, source: str) -> str:
    if not isinstance(key, str):
        raise LibraryLoadError(
            f"{source}: keys must be strings, got {type(key).__name__}"
        )
    return key


def _parse_str_in_seq(value: Any, *, source: str) -> str:
    if not isinstance(value, str):
        raise LibraryLoadError(f"{source}: expected string, got {type(value).__name__}")
    return value


def _parse_str_tuple(data: Any, *, source: str) -> tuple[str, ...]:
    if not isinstance(data, list):
        raise LibraryLoadError(f"{source}: expected list, got {type(data).__name__}")
    return tuple(
        _parse_str_in_seq(item, source=f"{source}[{i}]") for i, item in enumerate(data)
    )


def _parse_float(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise LibraryLoadError(f"{source}: expected number, got bool")
    if not isinstance(value, (int, float)):
        raise LibraryLoadError(f"{source}: expected number, got {type(value).__name__}")
    return float(value)


def _parse_float_mapping(data: Any, *, source: str) -> dict[str, float]:
    if not isinstance(data, Mapping):
        raise LibraryLoadError(f"{source}: expected mapping, got {type(data).__name__}")
    return {
        _require_string_key(key, source=source): _parse_float(
            value, source=f"{source}.{key}"
        )
        for key, value in data.items()
    }


def _parse_bool(value: Any, *, source: str) -> bool:
    if not isinstance(value, bool):
        raise LibraryLoadError(f"{source}: expected bool, got {type(value).__name__}")
    return value


def _parse_named_mapping(
    data: Any, item_parser: Callable[..., _T], *, source: str
) -> dict[str, _T]:
    if not isinstance(data, Mapping):
        raise LibraryLoadError(f"{source}: expected mapping, got {type(data).__name__}")
    result: dict[str, _T] = {}
    for key, value in data.items():
        name = _require_string_key(key, source=source)
        if not isinstance(value, Mapping):
            raise LibraryLoadError(
                f"{source}.{name}: expected mapping, got {type(value).__name__}"
            )
        result[name] = item_parser(value, source=f"{source}.{name}")
    return result
