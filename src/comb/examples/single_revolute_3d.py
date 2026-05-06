"""A single 3D revolute joint connecting an anchored base to a link.

The joint rotates about the world z-axis with origin at the base. Construct
with a non-zero ``initial_angle`` to start somewhere other than identity.
"""

import numpy as np
from spatialmath import SE3

from comb.bodies import Body, Box
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    RevoluteJoint3D,
)
from comb.mode import Mode
from comb.system import System


class SingleRevolute3D:
    """An assembled single-revolute SE(3) example with handles for its pieces."""

    def __init__(self, initial_angle: float = 0.0) -> None:
        self.base = Body(
            name="base",
            pose=SE3(),
            visual_geometry=Box(0.2, 0.2, 0.2),
            collision_geometry=Box(0.2, 0.2, 0.2),
        )
        self.link = Body(
            name="link",
            pose=SE3.AngVec(initial_angle, [0, 0, 1]),
            visual_geometry=Box(1.0, 0.05, 0.05),
            collision_geometry=Box(1.0, 0.05, 0.05),
        )
        self.joint = RevoluteJoint3D(
            body1=self.base,
            body2=self.link,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                names=RevoluteJoint3D.fixed_parameter_names(),
            ),
        )
        config = ConstraintConfiguration(
            {
                self.joint: ConstraintParameters(
                    values=np.array([initial_angle]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE3] = Mode(
            bodies=[self.base, self.link],
            constraints=[self.joint],
            configuration=config,
            anchored_bodies=[self.base],
        )
        self.system: System[SE3] = System(mode=self.mode)
