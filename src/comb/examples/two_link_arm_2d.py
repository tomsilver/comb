"""A two-link 2D arm: anchored base, two revolute joints, two links in the plane.

Each link's body frame sits at its joint pivot (the parent end of the link),
not at the link's geometric center, so rotating a joint pivots the link
around its near end. The link's rectangle geometry is offset along +x by
``link_length / 2`` so it visually extends away from the joint.
"""

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, Rectangle
from comb.constraints import Configuration, ConstraintParameters, RevoluteJoint2D
from comb.mode import Mode
from comb.system import System


class TwoLinkArm2D:
    """An assembled two-link SE(2) arm with handles for its pieces."""

    def __init__(self, link_length: float = 1.0) -> None:
        self.base = Body(
            name="base",
            pose=SE2(),
            visual_geometry=Rectangle(0.2, 0.2),
            collision_geometry=Rectangle(0.2, 0.2),
        )
        # link_a's frame coincides with the base joint pivot.
        self.link_a = Body(
            name="link_a",
            pose=SE2(),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        # link_b's frame sits at the elbow (link_a's far end at angle=0).
        self.link_b = Body(
            name="link_b",
            pose=SE2(link_length, 0.0, 0.0),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        # Shoulder joint at the base origin.
        self.joint_ab = RevoluteJoint2D(
            body1=self.base,
            body2=self.link_a,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        # Elbow joint at link_a's far end (link_length along +x in link_a's frame).
        self.joint_bc = RevoluteJoint2D(
            body1=self.link_a,
            body2=self.link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([link_length, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        config = Configuration(
            {
                self.joint_ab: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_bc: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE2] = Mode(
            bodies=[self.base, self.link_a, self.link_b],
            constraints=[self.joint_ab, self.joint_bc],
            configuration=config,
            anchored_bodies=[self.base],
        )
        self.system: System[SE2] = System(mode=self.mode)
