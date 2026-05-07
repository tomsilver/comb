"""Build runtime objects from a parsed and validated spec.

After :func:`comb.spec.load_library` and :func:`comb.spec.validate_library`
return a clean :class:`LibrarySpec`, :func:`instantiate_library` constructs
the actual ``Body`` / ``Constraint`` / ``ConstraintTransition`` instances
the planner runs against. :func:`instantiate_task` then assembles those
into a :class:`comb.system.System` with the task's initial mode applied.

This module assumes the spec has already passed validation (so unknown
type names, missing keys, etc. don't occur). It still raises informative
errors for cases the validator can't easily catch — chiefly geometry
shapes whose YAML key names differ from the Python class's field names.

SE(2) only for now (matches the rest of the spec ecosystem).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Geometry, Rectangle
from comb.constraints import (
    CONSTRAINT_TYPES_2D,
    Constraint,
    ConstraintConfiguration,
    ConstraintParameters,
)
from comb.generators import GENERATORS_2D
from comb.mode import Mode, ModeState
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
from comb.system import System
from comb.transitions import ConstraintTransition


class SpecInstantiationError(Exception):
    """Raised when a validated spec can't be turned into runtime objects."""


@dataclass(frozen=True)
class InstantiatedLibrary:
    """Runtime objects built from a :class:`LibrarySpec`, keyed by spec name.

    Carries the original spec too — useful both for picking up defaults
    (e.g. ``initial_parameters``) at task-instantiation time and for the
    plan serializer's name-mapping arguments.
    """

    spec: LibrarySpec
    bodies: Mapping[str, Body[SE2]]
    anchored_bodies: tuple[Body[SE2], ...]
    constraints: Mapping[str, Constraint[SE2]]
    transitions: Mapping[str, ConstraintTransition[SE2]]


@dataclass(frozen=True)
class InstantiatedTask:
    """A library + a runtime ``System`` and goal built from a :class:`TaskSpec`."""

    library: InstantiatedLibrary
    system: System[SE2]
    goal: tuple[Constraint[SE2], ...]


def instantiate_library(library: LibrarySpec) -> InstantiatedLibrary:
    """Build runtime objects from a validated :class:`LibrarySpec`."""
    bodies: dict[str, Body[SE2]] = {
        name: _instantiate_body(name, spec) for name, spec in library.bodies.items()
    }
    anchored_bodies = tuple(
        bodies[name] for name, spec in library.bodies.items() if spec.anchored
    )
    constraints: dict[str, Constraint[SE2]] = {
        name: _instantiate_constraint(spec, bodies)
        for name, spec in library.constraints.items()
    }
    transitions: dict[str, ConstraintTransition[SE2]] = {
        name: _instantiate_transition(spec, bodies, constraints)
        for name, spec in library.transitions.items()
    }
    return InstantiatedLibrary(
        spec=library,
        bodies=bodies,
        anchored_bodies=anchored_bodies,
        constraints=constraints,
        transitions=transitions,
    )


def instantiate_task(task: TaskSpec, library: InstantiatedLibrary) -> InstantiatedTask:
    """Build a runtime ``System`` and goal from a :class:`TaskSpec`."""
    initial = task.initial_mode

    active_names: tuple[str, ...]
    if initial.active_constraints is None:
        active_names = tuple(library.constraints)
    else:
        active_names = initial.active_constraints
    active_constraints = [library.constraints[name] for name in active_names]

    body_poses: dict[Body[SE2], SE2] = {
        body: body.pose for body in library.bodies.values()
    }
    for body_name, pose_spec in initial.body_poses.items():
        body_poses[library.bodies[body_name]] = _se2_from_pose_spec(pose_spec)

    configuration = _build_initial_configuration(
        active_constraints=active_constraints,
        active_names=active_names,
        library=library,
        configuration_overrides=initial.configuration,
    )

    mode: Mode[SE2] = Mode(
        bodies=list(library.bodies.values()),
        constraints=active_constraints,
        configuration=configuration,
        body_poses=BodyPoses(body_poses),
        anchored_bodies=list(library.anchored_bodies),
    )
    system: System[SE2] = System(
        mode=mode,
        transitions=tuple(library.transitions.values()),
    )
    goal = tuple(_instantiate_constraint(g, library.bodies) for g in task.goal)
    return InstantiatedTask(library=library, system=system, goal=goal)


# --- helpers ---


def _instantiate_body(name: str, spec: BodySpec) -> Body[SE2]:
    return Body(
        name=name,
        pose=_se2_from_pose_spec(spec.pose),
        visual_geometry=_instantiate_geometry(spec.visual_geometry),
        collision_geometry=_instantiate_geometry(spec.collision_geometry),
    )


def _instantiate_geometry(spec: GeometrySpec) -> Geometry[SE2]:
    if spec.shape == "rectangle":
        try:
            return Rectangle(
                size_x=float(spec.parameters["width"]),
                size_y=float(spec.parameters["height"]),
                offset_x=float(spec.parameters.get("offset_x", 0.0)),
                offset_y=float(spec.parameters.get("offset_y", 0.0)),
            )
        except KeyError as exc:
            raise SpecInstantiationError(
                f"rectangle missing parameter {exc.args[0]!r}; "
                f"got {sorted(spec.parameters)}"
            ) from exc
    raise SpecInstantiationError(f"cannot instantiate shape {spec.shape!r}")


def _instantiate_constraint(
    spec: ConstraintSpec, bodies: Mapping[str, Body[SE2]]
) -> Constraint[SE2]:
    cls = CONSTRAINT_TYPES_2D[spec.type]
    fp_names = cls.fixed_parameter_names()
    try:
        fp_values = np.array([float(spec.fixed_parameters[name]) for name in fp_names])
    except KeyError as exc:
        raise SpecInstantiationError(
            f"{spec.type} fixed_parameters missing {exc.args[0]!r}; "
            f"expected {list(fp_names)}"
        ) from exc
    return cls(  # type: ignore[call-arg]
        body1=bodies[spec.body1],
        body2=bodies[spec.body2],
        fixed_parameters=ConstraintParameters(values=fp_values, names=fp_names),
    )


def _instantiate_transition(
    spec: TransitionSpec,
    bodies: Mapping[str, Body[SE2]],
    constraints: Mapping[str, Constraint[SE2]],
) -> ConstraintTransition[SE2]:
    trigger = _instantiate_constraint(spec.trigger, bodies)
    add = _build_combined_add(spec.add, bodies)
    remove = tuple(constraints[name] for name in spec.remove)
    return ConstraintTransition(
        trigger=trigger,
        tolerance=spec.tolerance,
        add=add,
        remove=remove,
    )


def _build_combined_add(
    calls: tuple[GeneratorCallSpec, ...], bodies: Mapping[str, Body[SE2]]
) -> Callable[[ModeState[SE2]], list[Constraint[SE2]]]:
    closures = [_build_generator_closure(call, bodies) for call in calls]

    def combined(state: ModeState[SE2]) -> list[Constraint[SE2]]:
        result: list[Constraint[SE2]] = []
        for closure in closures:
            result.extend(closure(state))
        return result

    return combined


def _build_generator_closure(
    call: GeneratorCallSpec, bodies: Mapping[str, Body[SE2]]
) -> Callable[[ModeState[SE2]], list[Constraint[SE2]]]:
    factory = GENERATORS_2D[call.generator]
    kwargs: dict[str, object] = {}
    for key, value in call.args.items():
        if key in ("body1", "body2"):
            if not isinstance(value, str):
                raise SpecInstantiationError(
                    f"generator {call.generator}: arg {key} must be a body "
                    f"name string, got {type(value).__name__}"
                )
            kwargs[key] = bodies[value]
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return factory(**kwargs)


def _build_initial_configuration(
    *,
    active_constraints: list[Constraint[SE2]],
    active_names: Iterable[str],
    library: InstantiatedLibrary,
    configuration_overrides: Mapping[str, Mapping[str, float]],
) -> ConstraintConfiguration:
    config = ConstraintConfiguration()
    name_by_constraint = {
        constraint: name for name, constraint in library.constraints.items()
    }
    active_by_name = dict(zip(active_names, active_constraints))
    for name, constraint in active_by_name.items():
        names = constraint.parameter_names()
        if not names:
            continue
        defaults = _library_initial_parameters(name, library)
        overrides = configuration_overrides.get(name, {})
        values = np.array(
            [float(overrides.get(p, defaults.get(p, 0.0))) for p in names]
        )
        config[constraint] = ConstraintParameters(values=values, names=names)

    # Provide a useful error if a configuration override targets a non-active
    # constraint — the validator allows this (an override is an override; we
    # don't require activity), but quietly dropping it would surprise users.
    unknown_overrides = set(configuration_overrides) - set(active_by_name)
    if unknown_overrides:
        # Verify the names at least point at *some* library constraint so we
        # can give a clearer error than "unknown".
        truly_unknown = unknown_overrides - set(name_by_constraint.values())
        if truly_unknown:
            raise SpecInstantiationError(
                f"initial_mode.configuration references constraints not in "
                f"the library: {sorted(truly_unknown)}"
            )
    return config


def _library_initial_parameters(
    name: str, library: InstantiatedLibrary
) -> Mapping[str, float]:
    return library.spec.constraints[name].initial_parameters


def _se2_from_pose_spec(spec: PoseSpec) -> SE2:
    v = spec.values
    return SE2(float(v["x"]), float(v["y"]), float(v["theta"]))


# Helpers exposed for tests / advanced users.

__all__ = [
    "InstantiatedLibrary",
    "InstantiatedTask",
    "SpecInstantiationError",
    "instantiate_library",
    "instantiate_task",
]
