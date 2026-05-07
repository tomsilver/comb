"""A 2D mobile-base robot with a 2-link arm and a door it can open.

The world contains a mobile base, a 2-link arm mounted on it, and a door.
The door starts pinned to the world (no hinge — it's just stuck in place).
A bundled ``attach_transition`` fires when the arm's tip reaches the door's
handle: applying it (a) removes the world→door pin, (b) adds a
``HingeJoint2D`` so the door can swing about a fixed hinge point in the
world, and (c) adds a ``PointEquality2D`` enforcing that the arm tip stays
on the handle. From there, moving the arm drags the door open or closed.

The grip is point-on-point (``PointEquality2D``) rather than full rigid
(``FixedJoint2D``) — the door's orientation isn't slaved to the arm's, just
its handle position. That keeps the post-attach system solvable: with the
hinge fixing the door's pose given a single angle parameter and the arm
chain fixing ``link_b`` given the base + joint angles, a 2-residual coupling
is the right amount to "robot grip moves the door around its hinge."
"""

import math

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    FixedJoint2D,
    HingeJoint2D,
    PlanarJoint2D,
    PointEquality2D,
    RevoluteJoint2D,
)
from comb.mode import Mode
from comb.system import System
from comb.transitions import ConstraintTransition


class MobileArmDoor2D:
    """An assembled mobile robot + door that the robot can attach to and swing."""

    def __init__(
        self,
        link_length: float = 0.5,
        base_size: float = 0.3,
        door_length: float = 1.0,
        door_thickness: float = 0.05,
        hinge_origin: tuple[float, float] = (2.0, 0.0),
        max_door_angle: float = math.pi,
        attach_tolerance: float = 0.05,
    ) -> None:
        self.door_length = door_length
        self.world = Body(
            name="world",
            pose=SE2(),
            visual_geometry=Rectangle(0.0, 0.0),
            collision_geometry=Rectangle(0.0, 0.0),
        )
        self.base = Body(
            name="base",
            pose=SE2(),
            visual_geometry=Rectangle(base_size, base_size),
            collision_geometry=Rectangle(base_size, base_size),
        )
        # Arm: link_a's frame at the base joint pivot, link_b's frame at link_a's
        # far end. Each link's rectangle visually extends along +x from its frame.
        self.link_a = Body(
            name="link_a",
            pose=SE2(),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        self.link_b = Body(
            name="link_b",
            pose=SE2(link_length, 0.0, 0.0),
            visual_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
            collision_geometry=Rectangle(link_length, 0.05, offset_x=link_length / 2),
        )
        # Door's body frame sits at the hinge point; the rectangle extends
        # along +x by ``door_length`` so the door visually goes from hinge
        # (frame origin) to handle (far end).
        door_pose = SE2(hinge_origin[0], hinge_origin[1], 0.0)
        self.door = Body(
            name="door",
            pose=door_pose,
            visual_geometry=Rectangle(
                door_length, door_thickness, offset_x=door_length / 2
            ),
            collision_geometry=Rectangle(
                door_length, door_thickness, offset_x=door_length / 2
            ),
        )

        # --- initial constraints ---
        self.base_joint = PlanarJoint2D(
            body1=self.world,
            body2=self.base,
            fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
        )
        self.joint_ab = RevoluteJoint2D(
            body1=self.base,
            body2=self.link_a,
            fixed_parameters=ConstraintParameters(
                values=np.array([0.0, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        self.joint_bc = RevoluteJoint2D(
            body1=self.link_a,
            body2=self.link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([link_length, 0.0]),
                names=RevoluteJoint2D.fixed_parameter_names(),
            ),
        )
        # Initially the door is pinned to the world at door_pose — no hinge yet.
        self.world_to_door = FixedJoint2D(
            body1=self.world,
            body2=self.door,
            fixed_parameters=ConstraintParameters(
                values=np.array(
                    [
                        float(door_pose.t[0]),
                        float(door_pose.t[1]),
                        float(door_pose.theta()),
                    ]
                ),
                names=FixedJoint2D.fixed_parameter_names(),
            ),
        )

        config = ConstraintConfiguration(
            {
                self.base_joint: ConstraintParameters(
                    values=np.array([0.0, 0.0, 0.0]),
                    names=PlanarJoint2D.parameter_names(),
                ),
                self.joint_ab: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
                self.joint_bc: ConstraintParameters(
                    values=np.array([0.0]), names=("angle",)
                ),
            }
        )
        self.mode: Mode[SE2] = Mode(
            bodies=[self.world, self.base, self.link_a, self.link_b, self.door],
            constraints=[
                self.base_joint,
                self.joint_ab,
                self.joint_bc,
                self.world_to_door,
            ],
            configuration=config,
            body_poses=BodyPoses(
                {
                    self.world: SE2(),
                    self.base: SE2(),
                    self.link_a: SE2(),
                    self.link_b: SE2(link_length, 0.0, 0.0),
                    self.door: door_pose,
                }
            ),
            anchored_bodies=[self.world],
        )

        # --- attach transition ---
        # Trigger / post-attach grip: arm tip at door handle.
        self.attach_trigger = PointEquality2D(
            body1=self.door,
            body2=self.link_b,
            fixed_parameters=ConstraintParameters(
                values=np.array([door_length, 0.0, link_length, 0.0]),
                names=PointEquality2D.fixed_parameter_names(),
            ),
        )
        self.hinge_origin = hinge_origin
        self.max_door_angle = max_door_angle

        def attach_constraints(_state):
            # The grip (same PointEquality2D used as the trigger) becomes a
            # system constraint after attachment, enforcing that the arm tip
            # stays on the handle.
            hinge = HingeJoint2D(
                body1=self.world,
                body2=self.door,
                fixed_parameters=ConstraintParameters(
                    values=np.array(
                        [
                            float(hinge_origin[0]),
                            float(hinge_origin[1]),
                            0.0,
                            float(max_door_angle),
                        ]
                    ),
                    names=HingeJoint2D.fixed_parameter_names(),
                ),
            )
            return [self.attach_trigger, hinge]

        self.attach_transition: ConstraintTransition[SE2] = ConstraintTransition(
            trigger=self.attach_trigger,
            tolerance=attach_tolerance,
            add=attach_constraints,
            remove=(self.world_to_door,),
        )
        self.system: System[SE2] = System(
            mode=self.mode,
            transitions=(self.attach_transition,),
        )
