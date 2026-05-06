"""A two-link 3D arm: anchored base, two revolute joints about z, two links.

Joint origins are offset along x by ``link_length`` in the parent frame, so at
both angles = 0 the links extend straight out along the x-axis.
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


class TwoLinkArm3D:
    """An assembled two-link SE(3) arm with handles for its pieces."""

    def __init__(self, link_length: float = 1.0) -> None:
        self.base = Body(
            name="base",
            pose=SE3(),
            visual_geometry=Box(0.2, 0.2, 0.2),
            collision_geometry=Box(0.2, 0.2, 0.2),
        )
        link_a_pose = self.base.pose * SE3.Trans([link_length, 0.0, 0.0])
        link_b_pose = link_a_pose * SE3.Trans([link_length, 0.0, 0.0])
        self.link_a = Body(
            name="link_a",
            pose=link_a_pose,
            visual_geometry=Box(link_length, 0.05, 0.05),
            collision_geometry=Box(link_length, 0.05, 0.05),
        )
        self.link_b = Body(
            name="link_b",
            pose=link_b_pose,
            visual_geometry=Box(link_length, 0.05, 0.05),
            collision_geometry=Box(link_length, 0.05, 0.05),
        )
        self.joint_ab = RevoluteJoint3D(
            body1=self.base,
            body2=self.link_a,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, 1.0, link_length, 0.0, 0.0]),
                names=RevoluteJoint3D.fixed_parameter_names(),
            ),
        )
        self.joint_bc = RevoluteJoint3D(
            body1=self.link_a,
            body2=self.link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, 1.0, link_length, 0.0, 0.0]),
                names=RevoluteJoint3D.fixed_parameter_names(),
            ),
        )
        config = ConstraintConfiguration(
            {
                self.joint_ab: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_bc: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE3] = Mode(
            bodies=[self.base, self.link_a, self.link_b],
            constraints=[self.joint_ab, self.joint_bc],
            configuration=config,
            anchored_bodies=[self.base],
        )
        self.system: System[SE3] = System(mode=self.mode)
