"""A 2D door on a hinge: an anchored wall and a swinging door, nothing else.

The wall is a small anchored body marking where the hinge attaches; the
door's body frame sits at the hinge so the door swings about its near edge.
The hinge angle is bounded to ``[0, max_angle]`` (default ``[0, π]``) via
``HingeJoint2D``.
"""

import math

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    HingeJoint2D,
)
from comb.mode import Mode
from comb.system import System


class Door2D:
    """An assembled 2D door swinging on a hinge in an anchored wall."""

    def __init__(
        self,
        door_width: float = 0.8,
        door_thickness: float = 0.05,
        max_angle: float = math.pi,
        initial_angle: float = 0.0,
    ) -> None:
        self.wall = Body(
            name="wall",
            pose=SE2(),
            visual_geometry=Rectangle(0.1, 0.1),
            collision_geometry=Rectangle(0.1, 0.1),
        )
        # The door's frame sits at the hinge; the rectangle is offset along
        # +x by half its width so visually it extends away from the hinge.
        self.door = Body(
            name="door",
            pose=SE2(0.0, 0.0, initial_angle),
            visual_geometry=Rectangle(
                door_width, door_thickness, offset_x=door_width / 2
            ),
            collision_geometry=Rectangle(
                door_width, door_thickness, offset_x=door_width / 2
            ),
        )
        self.hinge = HingeJoint2D(
            body1=self.wall,
            body2=self.door,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, 0.0, max_angle]),
                names=HingeJoint2D.fixed_parameter_names(),
            ),
        )
        config = ConstraintConfiguration(
            {
                self.hinge: ConstraintParameters(
                    values=np.array([initial_angle]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE2] = Mode(
            bodies=[self.wall, self.door],
            constraints=[self.hinge],
            configuration=config,
            anchored_bodies=[self.wall],
        )
        self.system: System[SE2] = System(mode=self.mode)
