"""A 2D two-link arm plus a small free body initially fixed to the world.

The block starts rigidly pinned to an anchored ``world`` body via a
``FixedJoint2D``. ``pickup_transition`` is a ``ConstraintTransition`` whose
trigger fires when the arm's tip (the far end of ``arm.link_b``, at offset
``(link_length, 0)`` in its frame) coincides with the block's body frame:
applying it swaps the world→block pin for a rigid arm.link_b→block
attachment, so subsequent planning carries the block along with the arm.

Smallest workable example for "pick up an object" experiments — plan to
bring the tip to the block, fire ``pickup_transition``, then plan with
the resulting mode (block now coupled to the arm) to a placement target.
"""

from __future__ import annotations

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import ConstraintParameters, FixedJoint2D, PointEquality2D
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.mode import Mode
from comb.system import System
from comb.transitions import ConstraintTransition, rigid_attachment_2d


class TwoLinkArmWithObject2D:
    """An assembled 2D arm + initially-fixed block + pickup transition."""

    def __init__(
        self,
        link_length: float = 1.0,
        block_pose: SE2 | None = None,
        block_size: float = 0.15,
        pickup_tolerance: float = 0.05,
    ) -> None:
        if block_pose is None:
            block_pose = SE2(0.5, 1.0, 0.0)

        self.arm = TwoLinkArm2D(link_length=link_length)
        self.world = Body(
            name="world",
            pose=SE2(),
            visual_geometry=Rectangle(0.0, 0.0),
            collision_geometry=Rectangle(0.0, 0.0),
        )
        self.block = Body(
            name="block",
            pose=block_pose,
            visual_geometry=Rectangle(block_size, block_size),
            collision_geometry=Rectangle(block_size, block_size),
        )
        # Pin the block rigidly to the world at its initial pose.
        self.world_to_block = FixedJoint2D(
            body1=self.world,
            body2=self.block,
            fixed_parameters=ConstraintParameters(
                values=np.array(
                    [
                        float(block_pose.t[0]),
                        float(block_pose.t[1]),
                        float(block_pose.theta()),
                    ]
                ),
                names=FixedJoint2D.fixed_parameter_names(),
            ),
        )
        self.mode: Mode[SE2] = Mode(
            bodies=self.arm.mode.bodies + [self.world, self.block],
            constraints=list(self.arm.mode.constraints) + [self.world_to_block],
            configuration=self.arm.mode.configuration,
            body_poses=BodyPoses(
                {b: self.arm.mode.body_poses[b] for b in self.arm.mode.bodies}
                | {self.world: SE2(), self.block: block_pose}
            ),
            anchored_bodies=self.arm.mode.anchored_bodies + [self.world],
        )
        # Trigger: arm tip (offset (link_length, 0) in link_b's frame) coincident
        # with the block's body frame (offset (0, 0) in the block's frame).
        self.pickup_trigger = PointEquality2D(
            body1=self.block,
            body2=self.arm.link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, link_length, 0.0]),
                names=PointEquality2D.fixed_parameter_names(),
            ),
        )
        self.pickup_transition: ConstraintTransition[SE2] = ConstraintTransition(
            trigger=self.pickup_trigger,
            tolerance=pickup_tolerance,
            add=rigid_attachment_2d(self.arm.link_b, self.block),
            remove=(self.world_to_block,),
        )
        self.system: System[SE2] = System(
            mode=self.mode,
            transitions=(self.pickup_transition,),
        )
