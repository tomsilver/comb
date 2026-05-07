"""Spec-level validator: structural checks against a parsed library / task.

The YAML loaders (:mod:`comb.spec.load`) handle *shape* validation —
required keys present, types correct. The validator here adds *semantic*
validation:

* every ``ConstraintSpec.type`` exists in the constraint registry, and its
  ``fixed_parameters`` / ``initial_parameters`` keys match the class's
  expected names;
* every body / constraint name reference (in constraints, transitions,
  goals, initial-mode configuration) resolves;
* every ``GeneratorCallSpec.generator`` exists in
  :data:`comb.generators.GENERATORS_2D`, and its body-typed args reference
  declared bodies;
* geometries match a known shape and have the right parameter keys;
* poses match the expected SE(2) keyset;
* every body is reachable from some anchored body via the active
  constraint graph (so no body's pose is left undetermined at t=0).

Validation halts on the first error (no warning collection). Use
:class:`SpecValidationError` to catch.
"""

from __future__ import annotations

from comb.constraints import CONSTRAINT_TYPES_2D
from comb.generators import GENERATORS_2D
from comb.spec.library import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    LibrarySpec,
    PoseSpec,
    TransitionSpec,
)
from comb.spec.task import TaskSpec


class SpecValidationError(Exception):
    """Raised when a parsed spec has unresolved references or wrong shapes."""


_POSE_KEYS_2D = frozenset({"x", "y", "theta"})

_GEOMETRY_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "rectangle": frozenset({"width", "height"}),
}
_GEOMETRY_OPTIONAL_KEYS: dict[str, frozenset[str]] = {
    "rectangle": frozenset({"offset_x", "offset_y"}),
}

# The arg names in a GeneratorCallSpec that we treat as body-name
# references. Other args (e.g. ``body2_offset`` in ``point_pin_2d``) carry
# arbitrary configuration and aren't validated against ``lib.bodies``.
_BODY_ARG_KEYS = frozenset({"body1", "body2"})


def validate_library(lib: LibrarySpec) -> None:
    """Verify a library is structurally well-formed.

    Raises :class:`SpecValidationError` on the first unresolved reference,
    unknown type / generator, or wrong-shape geometry / pose / parameter
    set.
    """
    body_names = set(lib.bodies)
    constraint_names = set(lib.constraints)

    for name, body in lib.bodies.items():
        _validate_body(body, source=f"bodies.{name}")

    for name, constraint in lib.constraints.items():
        _validate_constraint(constraint, body_names, source=f"constraints.{name}")

    for name, transition in lib.transitions.items():
        _validate_transition(
            transition, body_names, constraint_names, source=f"transitions.{name}"
        )


def validate_task(task: TaskSpec, lib: LibrarySpec) -> None:
    """Verify a task is structurally well-formed against ``lib``.

    Validates ``initial_mode`` references against the library's bodies and
    constraints, validates each goal constraint, and checks that every
    body is reachable from some anchored body via the *active* constraint
    graph (the subset listed in ``initial_mode.active_constraints``, or
    all library constraints if that field is ``None``).
    """
    body_names = set(lib.bodies)
    constraint_names = set(lib.constraints)

    if task.initial_mode.active_constraints is not None:
        for i, name in enumerate(task.initial_mode.active_constraints):
            if name not in constraint_names:
                raise SpecValidationError(
                    f"initial_mode.active_constraints[{i}]: {name!r} "
                    f"is not a declared constraint"
                )

    for body_name, pose in task.initial_mode.body_poses.items():
        if body_name not in body_names:
            raise SpecValidationError(
                f"initial_mode.body_poses.{body_name}: not a declared body"
            )
        _validate_pose(pose, source=f"initial_mode.body_poses.{body_name}")

    for cname, params in task.initial_mode.configuration.items():
        if cname not in constraint_names:
            raise SpecValidationError(
                f"initial_mode.configuration.{cname}: not a declared constraint"
            )
        spec = lib.constraints[cname]
        cls = CONSTRAINT_TYPES_2D[spec.type]
        expected = set(cls.parameter_names())
        got = set(params)
        extra = got - expected
        if extra:
            raise SpecValidationError(
                f"initial_mode.configuration.{cname}: unexpected keys "
                f"{sorted(extra)}; valid: {sorted(expected)}"
            )

    for i, goal in enumerate(task.goal):
        _validate_constraint(goal, body_names, source=f"goal[{i}]")

    _validate_connectivity(task, lib)


def _validate_body(body: BodySpec, *, source: str) -> None:
    _validate_geometry(body.visual_geometry, source=f"{source}.visual_geometry")
    _validate_geometry(body.collision_geometry, source=f"{source}.collision_geometry")
    _validate_pose(body.pose, source=f"{source}.pose")


def _validate_geometry(geometry: GeometrySpec, *, source: str) -> None:
    if geometry.shape not in _GEOMETRY_REQUIRED_KEYS:
        raise SpecValidationError(
            f"{source}: unknown shape {geometry.shape!r}; valid: "
            f"{sorted(_GEOMETRY_REQUIRED_KEYS)}"
        )
    required = _GEOMETRY_REQUIRED_KEYS[geometry.shape]
    optional = _GEOMETRY_OPTIONAL_KEYS.get(geometry.shape, frozenset())
    keys = set(geometry.parameters)
    missing = required - keys
    if missing:
        raise SpecValidationError(
            f"{source}: missing geometry parameters {sorted(missing)}"
        )
    extra = keys - required - optional
    if extra:
        raise SpecValidationError(
            f"{source}: unexpected geometry parameters {sorted(extra)}"
        )


def _validate_pose(pose: PoseSpec, *, source: str) -> None:
    keys = set(pose.values)
    if keys != _POSE_KEYS_2D:
        raise SpecValidationError(
            f"{source}: expected SE(2) pose keys {sorted(_POSE_KEYS_2D)}, "
            f"got {sorted(keys)}"
        )


def _validate_constraint(
    constraint: ConstraintSpec, body_names: set[str], *, source: str
) -> None:
    if constraint.type not in CONSTRAINT_TYPES_2D:
        raise SpecValidationError(
            f"{source}: unknown constraint type {constraint.type!r}; "
            f"valid: {sorted(CONSTRAINT_TYPES_2D)}"
        )
    cls = CONSTRAINT_TYPES_2D[constraint.type]
    if constraint.body1 not in body_names:
        raise SpecValidationError(
            f"{source}.body1: {constraint.body1!r} is not a declared body"
        )
    if constraint.body2 not in body_names:
        raise SpecValidationError(
            f"{source}.body2: {constraint.body2!r} is not a declared body"
        )

    expected_fixed = set(cls.fixed_parameter_names())
    got_fixed = set(constraint.fixed_parameters)
    if got_fixed != expected_fixed:
        raise SpecValidationError(
            f"{source}.fixed_parameters: expected keys {sorted(expected_fixed)}, "
            f"got {sorted(got_fixed)}"
        )

    expected_initial = set(cls.parameter_names())
    got_initial = set(constraint.initial_parameters)
    extra_initial = got_initial - expected_initial
    if extra_initial:
        raise SpecValidationError(
            f"{source}.initial_parameters: unexpected keys "
            f"{sorted(extra_initial)}; valid: {sorted(expected_initial)}"
        )


def _validate_transition(
    transition: TransitionSpec,
    body_names: set[str],
    constraint_names: set[str],
    *,
    source: str,
) -> None:
    _validate_constraint(transition.trigger, body_names, source=f"{source}.trigger")
    if transition.trigger.initial_parameters:
        raise SpecValidationError(
            f"{source}.trigger: must have no mutable parameters; got "
            f"{sorted(transition.trigger.initial_parameters)}"
        )
    if transition.tolerance <= 0:
        raise SpecValidationError(
            f"{source}.tolerance: must be positive, got {transition.tolerance}"
        )
    for i, call in enumerate(transition.add):
        _validate_generator_call(call, body_names, source=f"{source}.add[{i}]")
    for i, name in enumerate(transition.remove):
        if name not in constraint_names:
            raise SpecValidationError(
                f"{source}.remove[{i}]: {name!r} is not a declared constraint"
            )


def _validate_generator_call(
    call: GeneratorCallSpec, body_names: set[str], *, source: str
) -> None:
    if call.generator not in GENERATORS_2D:
        raise SpecValidationError(
            f"{source}: unknown generator {call.generator!r}; "
            f"valid: {sorted(GENERATORS_2D)}"
        )
    for key, value in call.args.items():
        if key in _BODY_ARG_KEYS:
            if not isinstance(value, str):
                raise SpecValidationError(
                    f"{source}.args.{key}: expected body-name string, got "
                    f"{type(value).__name__}"
                )
            if value not in body_names:
                raise SpecValidationError(
                    f"{source}.args.{key}: {value!r} is not a declared body"
                )


def _validate_connectivity(task: TaskSpec, lib: LibrarySpec) -> None:
    active = task.initial_mode.active_constraints
    active_names = tuple(lib.constraints) if active is None else active

    adj: dict[str, set[str]] = {body: set() for body in lib.bodies}
    for cname in active_names:
        constraint = lib.constraints[cname]
        adj[constraint.body1].add(constraint.body2)
        adj[constraint.body2].add(constraint.body1)

    anchored = {name for name, body in lib.bodies.items() if body.anchored}
    if not anchored:
        raise SpecValidationError(
            "no anchored bodies — every body must reach an anchor "
            "via the active constraint graph"
        )

    visited: set[str] = set()
    stack = list(anchored)
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(adj[cur] - visited)

    unreachable = set(lib.bodies) - visited
    if unreachable:
        raise SpecValidationError(
            f"bodies not reachable from any anchor via active constraints: "
            f"{sorted(unreachable)}"
        )
