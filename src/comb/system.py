"""System: a container for bodies, constraints, and a configuration.

The ``System`` is the central object that downstream code (rendering, forward
kinematics, collision detection, simulation, optimization) operates on.

It is generic in the pose type, so a ``System[SE2]`` is statically distinct
from a ``System[SE3]`` — useful e.g. for choosing a 2D vs 3D renderer.
"""

from dataclasses import dataclass, field
from typing import Generic

from comb.bodies import Body, PoseT
from comb.constraints import Configuration, Constraint


@dataclass
class System(Generic[PoseT]):
    """A kinematic system: bodies + constraints + current configuration.

    On construction we check that (i) every constraint's two bodies are present
    in the system and (ii) every constraint that has mutable parameters has an
    entry in the configuration. If you mutate ``bodies`` or ``constraints``
    after construction, call :meth:`validate` to re-check.
    """

    bodies: list[Body[PoseT]]
    constraints: list[Constraint[PoseT]]
    configuration: Configuration = field(default_factory=Configuration)

    def __post_init__(self) -> None:
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
