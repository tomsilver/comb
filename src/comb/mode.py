"""Mode: a container for bodies, constraints, configuration, and body poses.

The ``Mode`` is the central object that downstream code (rendering, forward
kinematics, collision detection, simulation, optimization) operates on.

It is generic in the pose type, so a ``Mode[SE2]`` is statically distinct
from a ``Mode[SE3]`` — useful e.g. for choosing a 2D vs 3D renderer.

``body_poses`` holds the current pose for each body in the mode. Bodies for
which no entry is provided are auto-populated from each ``Body.pose``.

``anchored_bodies`` are bodies whose poses are fixed under solving (e.g. a
floor, a robot base bolted to the world). The solver only updates poses of
non-anchored bodies; without at least one anchor, the SE(2)/SE(3) gauge is
ambiguous and the solver will refuse to run.

``ModeState`` is an immutable snapshot of the mode's mutable state — the
``(configuration, body_poses)`` pair that ``solve()`` returns. It is the
natural value type for trajectories that drive a mode through time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from comb.bodies import Body, BodyPoses, PoseT, interpolate_body_poses
from comb.constraints import Configuration, Constraint, interpolate_configuration


@dataclass
class Mode(Generic[PoseT]):
    """A kinematic mode: bodies + constraints + configuration + body poses.

    On construction we (i) auto-populate missing entries in ``body_poses`` from
    each ``Body.pose``, then (ii) check that every constraint's bodies are in
    the mode and every constraint with mutable parameters has an entry in
    the configuration. If you mutate ``bodies`` or ``constraints`` after
    construction, call :meth:`validate` to re-check.
    """

    bodies: list[Body[PoseT]]
    constraints: list[Constraint[PoseT]]
    configuration: Configuration = field(default_factory=Configuration)
    body_poses: BodyPoses[PoseT] = field(default_factory=BodyPoses)
    anchored_bodies: list[Body[PoseT]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for body in self.bodies:
            if body not in self.body_poses:
                self.body_poses[body] = body.pose
        self.validate()

    def validate(self) -> None:
        """Check that constraint bodies are in the mode and the config is complete."""
        body_ids = {id(b) for b in self.bodies}
        for constraint in self.constraints:
            if (
                id(constraint.body1) not in body_ids
                or id(constraint.body2) not in body_ids
            ):
                raise ValueError(
                    f"{type(constraint).__name__} references bodies "
                    f"({constraint.body1.name}, {constraint.body2.name}) not in "
                    f"the mode"
                )
            if constraint.parameter_names() and constraint not in self.configuration:
                raise ValueError(
                    f"Configuration is missing an entry for "
                    f"{type(constraint).__name__} between "
                    f"{constraint.body1.name} and {constraint.body2.name}"
                )
        for body in self.anchored_bodies:
            if id(body) not in body_ids:
                raise ValueError(f"Anchored body {body.name!r} is not in the mode")

    def snapshot(self) -> ModeState[PoseT]:
        """An independent ``ModeState`` capturing this mode's current state.

        The returned ``Configuration`` and ``BodyPoses`` are fresh containers, so
        later mutation of the mode doesn't leak into the snapshot.
        """
        return ModeState(
            configuration=Configuration(
                {c: self.configuration[c] for c in self.configuration}
            ),
            body_poses=BodyPoses({b: self.body_poses[b] for b in self.bodies}),
        )

    def apply(self, state: ModeState[PoseT]) -> None:
        """Push ``state``'s contents into this mode's mutable state in place."""
        for body in self.bodies:
            self.body_poses[body] = state.body_poses[body]
        for constraint in state.configuration:
            self.configuration[constraint] = state.configuration[constraint]


@dataclass(frozen=True)
class ModeState(Generic[PoseT]):
    """Immutable snapshot of a mode's mutable state: configuration + body poses."""

    configuration: Configuration
    body_poses: BodyPoses[PoseT]


def interpolate_mode_state(
    start: ModeState[PoseT], end: ModeState[PoseT], s: float
) -> ModeState[PoseT]:
    """Interpolate two mode states by interpolating each piece independently.

    Use as the ``interpolate`` argument to ``trajectories.linear_segment`` to
    build a ``Trajectory[ModeState[PoseT]]`` between two solver checkpoints.
    """
    return ModeState(
        configuration=interpolate_configuration(
            start.configuration, end.configuration, s
        ),
        body_poses=interpolate_body_poses(start.body_poses, end.body_poses, s),
    )
