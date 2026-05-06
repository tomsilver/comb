"""A single 2D revolute joint connecting an anchored base to a link in the plane.

The link's body frame sits at the joint pivot (the base origin) so the link
rotates *around the joint*, not around its midpoint. The link's rectangle
geometry is offset along +x by ``link_length / 2`` so visually it extends
from the joint outward.
"""

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    RevoluteJoint2D,
)
from comb.mode import Mode
from comb.system import System


class SingleRevolute2D:
    """An assembled single-revolute SE(2) example with handles for its pieces."""

    def __init__(self, link_length: float = 0.5, initial_angle: float = 0.0) -> None:
        self.base = Body(
            name="base",
            pose=SE2(),
            visual_geometry=Rectangle(0.1, 0.1),
            collision_geometry=Rectangle(0.1, 0.1),
        )
        self.link = Body(
            name="link",
            pose=SE2(0.0, 0.0, initial_angle),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        self.joint = RevoluteJoint2D(
            body1=self.base,
            body2=self.link,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        config = ConstraintConfiguration(
            {
                self.joint: ConstraintParameters(
                    values=np.array([initial_angle]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE2] = Mode(
            bodies=[self.base, self.link],
            constraints=[self.joint],
            configuration=config,
            anchored_bodies=[self.base],
        )
        self.system: System[SE2] = System(mode=self.mode)
