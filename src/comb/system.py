"""System: a container for bodies, constraints, configuration, and body poses.

The ``System`` is the central object that downstream code (rendering, forward
kinematics, collision detection, simulation, optimization) operates on.

It is generic in the pose type, so a ``System[SE2]`` is statically distinct
from a ``System[SE3]`` — useful e.g. for choosing a 2D vs 3D renderer.

``body_poses`` holds the current pose for each body in the system. Bodies for
which no entry is provided are auto-populated from each ``Body.pose``.

``anchored_bodies`` are bodies whose poses are fixed under solving (e.g. a
floor, a robot base bolted to the world). The solver only updates poses of
non-anchored bodies; without at least one anchor, the SE(2)/SE(3) gauge is
ambiguous and the solver will refuse to run.
"""

from dataclasses import dataclass, field
from typing import Generic

from comb.bodies import Body, BodyPoses, PoseT
from comb.constraints import Configuration, Constraint


@dataclass
class System(Generic[PoseT]):
    """A kinematic system: bodies + constraints + configuration + body poses.

    On construction we (i) auto-populate missing entries in ``body_poses`` from
    each ``Body.pose``, then (ii) check that every constraint's bodies are in
    the system and every constraint with mutable parameters has an entry in
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
        """Check that constraint bodies are in the system and the config is complete."""
        body_ids = {id(b) for b in self.bodies}
        for constraint in self.constraints:
            if (
                id(constraint.body1) not in body_ids
                or id(constraint.body2) not in body_ids
            ):
                raise ValueError(
                    f"{type(constraint).__name__} references bodies "
                    f"({constraint.body1.name}, {constraint.body2.name}) not in "
                    f"the system"
                )
            if constraint.parameter_names() and constraint not in self.configuration:
                raise ValueError(
                    f"Configuration is missing an entry for "
                    f"{type(constraint).__name__} between "
                    f"{constraint.body1.name} and {constraint.body2.name}"
                )
        for body in self.anchored_bodies:
            if id(body) not in body_ids:
                raise ValueError(f"Anchored body {body.name!r} is not in the system")
