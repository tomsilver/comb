"""Two fixed-base 2D arms, an object initially within reach of only one arm.

The scene has:

* arm A — fixed base at the origin, two-link arm extending into the workspace
* arm B — fixed base at ``separation`` along +x, mirrored so its arm extends
  back toward arm A
* an object near arm A, initially pinned to the world

Two transitions are bundled in the system:

* ``pickup_transition`` — fires when arm A's tip is at the object. Removes
  the world→object pin and adds a rigid attachment from arm A's ``link_b``
  to the object, so subsequent motion of arm A drags the object along.
* ``handover_transition`` — fires when arm B's tip is at the object (which
  by then is being held by arm A in the workspace overlap). Adds a rigid
  attachment from arm B's ``link_b`` to the object, and *dynamically*
  removes whatever rigid attachment from arm A's ``link_b`` is currently in
  the mode (the one created by ``pickup_transition``). This is the use case
  for the callable ``ConstraintTransition.remove`` form: the constraint to
  remove was constructed by an earlier transition's ``add``, so we don't
  have a stable reference at construction time — we look it up at apply
  time by matching bodies.
"""

import math
from collections.abc import Callable

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import (
    Constraint,
    ConstraintConfiguration,
    ConstraintParameters,
    FixedJoint2D,
    PointEquality2D,
    RevoluteJoint2D,
)
from comb.mode import Mode, ModeState
from comb.system import System
from comb.transitions import ConstraintTransition


class DualArmHandover2D:
    """An assembled two-arm scene with pickup + handover transitions."""

    def __init__(
        self,
        link_length: float = 1.0,
        separation: float = 3.5,
        object_pose: SE2 | None = None,
        object_size: float = 0.15,
        attach_tolerance: float = 0.05,
    ) -> None:
        if object_pose is None:
            object_pose = SE2(1.0, 0.0, 0.0)
        self.world = Body(
            name="world",
            pose=SE2(),
            visual_geometry=Rectangle(0.0, 0.0),
            collision_geometry=Rectangle(0.0, 0.0),
        )

        # --- arm A: base at origin, links extending along +x at zero angles ---
        arm_a_base_pose = SE2(0.0, 0.0, 0.0)
        self.arm_a_base = Body(
            name="arm_a_base",
            pose=arm_a_base_pose,
            visual_geometry=Rectangle(0.2, 0.2),
            collision_geometry=Rectangle(0.2, 0.2),
        )
        self.arm_a_link_a = Body(
            name="arm_a_link_a",
            pose=arm_a_base_pose,
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        self.arm_a_link_b = Body(
            name="arm_a_link_b",
            pose=arm_a_base_pose * SE2(link_length, 0.0, 0.0),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )

        # --- arm B: base at (separation, 0) facing -x ---
        arm_b_base_pose = SE2(separation, 0.0, math.pi)
        self.arm_b_base = Body(
            name="arm_b_base",
            pose=arm_b_base_pose,
            visual_geometry=Rectangle(0.2, 0.2),
            collision_geometry=Rectangle(0.2, 0.2),
        )
        self.arm_b_link_a = Body(
            name="arm_b_link_a",
            pose=arm_b_base_pose,
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        self.arm_b_link_b = Body(
            name="arm_b_link_b",
            pose=arm_b_base_pose * SE2(link_length, 0.0, 0.0),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )

        self.object_body = Body(
            name="object",
            pose=object_pose,
            visual_geometry=Rectangle(object_size, object_size),
            collision_geometry=Rectangle(object_size, object_size),
        )

        # --- arm joints ---
        self.joint_a_ab = RevoluteJoint2D(
            body1=self.arm_a_base,
            body2=self.arm_a_link_a,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        self.joint_a_bc = RevoluteJoint2D(
            body1=self.arm_a_link_a,
            body2=self.arm_a_link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([link_length, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        self.joint_b_ab = RevoluteJoint2D(
            body1=self.arm_b_base,
            body2=self.arm_b_link_a,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        self.joint_b_bc = RevoluteJoint2D(
            body1=self.arm_b_link_a,
            body2=self.arm_b_link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([link_length, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )

        # Object initially pinned to world at object_pose.
        self.world_to_object = FixedJoint2D(
            body1=self.world,
            body2=self.object_body,
            fixed_parameters=ConstraintParameters(
                values=np.array(
                    [
                        float(object_pose.t[0]),
                        float(object_pose.t[1]),
                        float(object_pose.theta()),
                    ]
                ),
                names=FixedJoint2D.fixed_parameter_names(),
            ),
        )

        config = ConstraintConfiguration(
            {
                self.joint_a_ab: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_a_bc: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_b_ab: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_b_bc: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE2] = Mode(
            bodies=[
                self.world,
                self.arm_a_base,
                self.arm_a_link_a,
                self.arm_a_link_b,
                self.arm_b_base,
                self.arm_b_link_a,
                self.arm_b_link_b,
                self.object_body,
            ],
            constraints=[
                self.joint_a_ab,
                self.joint_a_bc,
                self.joint_b_ab,
                self.joint_b_bc,
                self.world_to_object,
            ],
            configuration=config,
            body_poses=BodyPoses(
                {
                    self.world: SE2(),
                    self.arm_a_base: arm_a_base_pose,
                    self.arm_a_link_a: arm_a_base_pose,
                    self.arm_a_link_b: arm_a_base_pose * SE2(link_length, 0.0, 0.0),
                    self.arm_b_base: arm_b_base_pose,
                    self.arm_b_link_a: arm_b_base_pose,
                    self.arm_b_link_b: arm_b_base_pose * SE2(link_length, 0.0, 0.0),
                    self.object_body: object_pose,
                }
            ),
            anchored_bodies=[self.world, self.arm_a_base, self.arm_b_base],
        )

        # --- triggers ---
        self.pickup_trigger = PointEquality2D(
            body1=self.object_body,
            body2=self.arm_a_link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, link_length, 0.0]),
                names=PointEquality2D.fixed_parameter_names(),
            ),
        )
        self.handover_trigger = PointEquality2D(
            body1=self.object_body,
            body2=self.arm_b_link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0, link_length, 0.0]),
                names=PointEquality2D.fixed_parameter_names(),
            ),
        )

        # --- transitions ---
        self.pickup_transition: ConstraintTransition[SE2] = ConstraintTransition(
            trigger=self.pickup_trigger,
            tolerance=attach_tolerance,
            add=_make_rigid_attachment_factory(self.arm_a_link_b, self.object_body),
            remove=(self.world_to_object,),
        )

        # Handover: trigger on arm B's tip at object, add B→object grip, and
        # dynamically remove the A→object grip (whichever FixedJoint2D from
        # arm_a_link_b to object is currently in the mode).
        def _remove_a_grip(mode: Mode[SE2]) -> list[Constraint[SE2]]:
            return [
                c
                for c in mode.constraints
                if isinstance(c, FixedJoint2D)
                and c.body1 is self.arm_a_link_b
                and c.body2 is self.object_body
            ]

        self.handover_transition: ConstraintTransition[SE2] = ConstraintTransition(
            trigger=self.handover_trigger,
            tolerance=attach_tolerance,
            add=_make_rigid_attachment_factory(self.arm_b_link_b, self.object_body),
            remove=_remove_a_grip,
        )

        self.system: System[SE2] = System(
            mode=self.mode,
            transitions=(self.pickup_transition, self.handover_transition),
        )


def _make_rigid_attachment_factory(
    body1: Body[SE2], body2: Body[SE2]
) -> Callable[[ModeState[SE2]], list[Constraint[SE2]]]:
    """Build an ``add`` callable that captures body2's relative pose to body1."""

    def add(state: ModeState[SE2]) -> list[Constraint[SE2]]:
        rel = state.body_poses[body1].inv() * state.body_poses[body2]
        return [
            FixedJoint2D(
                body1=body1,
                body2=body2,
                fixed_parameters=ConstraintParameters(
                    values=np.array(
                        [
                            float(rel.t[0]),
                            float(rel.t[1]),
                            float(rel.theta()),
                        ]
                    ),
                    names=FixedJoint2D.fixed_parameter_names(),
                ),
            )
        ]

    return add
