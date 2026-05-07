"""YAML loaders for library files.

Two entry points:

* :func:`load_library_file` parses a single file into a :class:`LibrarySpec`,
  preserving its ``includes`` field as a tuple of path strings.
* :func:`load_library` parses a file *and* recursively follows its
  ``includes`` into one merged :class:`LibrarySpec` whose ``includes`` is
  empty. Use this for end-to-end loading; the single-file version is the
  building block.

Schema validation only — generator and constraint type names pass through
as strings. Semantic validation (does ``body1`` exist? is the generator
registered?) lands in the validator (B5). Nothing here instantiates runtime
``Body`` / ``Constraint`` / ``ConstraintTransition`` objects.
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


def load_library(path: str | Path) -> LibrarySpec:
    """Load a library file and recursively resolve its ``includes``.

    Returns a single merged :class:`LibrarySpec` whose ``includes`` is empty
    and whose ``bodies`` / ``constraints`` / ``transitions`` are the union
    across the include graph. The merged library's ``name`` is the root
    file's name; included libraries' names are discarded.

    Includes are resolved relative to the file declaring them. The graph
    must be a DAG: cycles, including self-includes, raise. Diamond loads
    (two paths to the same file) deduplicate by absolute path. Names must
    be globally unique across the merged set — same body / constraint /
    transition name in two libraries is an error.
    """
    root = Path(path).resolve()
    if not root.exists():
        raise LibraryLoadError(f"library file not found: {root}")
    loaded: dict[Path, LibrarySpec] = {}
    _load_recursive(root, loaded, stack=[])
    return _merge_libraries(root, loaded)


def _load_recursive(
    path: Path,
    loaded: dict[Path, LibrarySpec],
    stack: list[Path],
) -> None:
    # Check the stack first: ``loaded`` is populated before children are
    # processed, so a cycle ``A -> B -> A`` would otherwise be silenced by
    # the dedup short-circuit on the second visit to A.
    if path in stack:
        cycle_start = stack.index(path)
        chain = " -> ".join(str(p) for p in stack[cycle_start:]) + f" -> {path}"
        raise LibraryLoadError(f"include cycle: {chain}")
    if path in loaded:
        return
    stack.append(path)
    spec = load_library_file(path)
    loaded[path] = spec
    for include in spec.includes:
        included = (path.parent / include).resolve()
        if not included.exists():
            raise LibraryLoadError(
                f"{path}: include not found: {include!r} (resolved to {included})"
            )
        _load_recursive(included, loaded, stack)
    stack.pop()


def _merge_libraries(root: Path, loaded: dict[Path, LibrarySpec]) -> LibrarySpec:
    bodies: dict[str, BodySpec] = {}
    constraints: dict[str, ConstraintSpec] = {}
    transitions: dict[str, TransitionSpec] = {}
    body_origin: dict[str, Path] = {}
    constraint_origin: dict[str, Path] = {}
    transition_origin: dict[str, Path] = {}

    for src_path, spec in loaded.items():
        _merge_into(spec.bodies, bodies, body_origin, src_path, kind="body")
        _merge_into(
            spec.constraints,
            constraints,
            constraint_origin,
            src_path,
            kind="constraint",
        )
        _merge_into(
            spec.transitions,
            transitions,
            transition_origin,
            src_path,
            kind="transition",
        )

    return LibrarySpec(
        name=loaded[root].name,
        includes=(),
        bodies=bodies,
        constraints=constraints,
        transitions=transitions,
    )


def _merge_into(
    incoming: Mapping[str, _T],
    target: dict[str, _T],
    origin: dict[str, Path],
    src_path: Path,
    *,
    kind: str,
) -> None:
    for name, value in incoming.items():
        if name in target:
            raise LibraryLoadError(
                f"{kind} name {name!r} declared in both "
                f"{origin[name]} and {src_path}"
            )
        target[name] = value
        origin[name] = src_path


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
