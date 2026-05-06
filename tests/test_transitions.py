"""Tests for constraint-triggered transitions."""

import math

import numpy as np
import pytest
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    FixedJoint2D,
    PointEquality2D,
    RevoluteJoint2D,
)
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.mode import Mode, ModeState
from comb.solver import solve
from comb.transitions import ConstraintTransition, RigidAttachment2D


def _world_body() -> Body[SE2]:
    return Body(
        name="world",
        pose=SE2(),
        visual_geometry=Rectangle(0.0, 0.0),
        collision_geometry=Rectangle(0.0, 0.0),
    )


def _free_body(name: str, pose: SE2) -> Body[SE2]:
    return Body(
        name=name,
        pose=pose,
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )


def _proximity_trigger(
    world: Body[SE2],
    body: Body[SE2],
    target_x: float,
    target_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> PointEquality2D:
    """A 'tip-near-point' check, expressed as a PointEquality2D."""
    return PointEquality2D(
        body1=world,
        body2=body,
        fixed_parameters=ConstraintParameters(
            values=np.array([target_x, target_y, offset_x, offset_y]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )


def test_is_enabled_when_trigger_residual_under_tolerance():
    """The transition fires when the trigger's residual norm is below tolerance."""
    world = _world_body()
    obj = _free_body("obj", SE2(1.0, 0.0, 0.0))
    trigger = _proximity_trigger(world, obj, target_x=1.05, target_y=0.0)
    transition = ConstraintTransition(trigger=trigger, tolerance=0.1)
    state = ModeState(
        configuration=ConstraintConfiguration(),
        body_poses=BodyPoses({world: SE2(), obj: SE2(1.0, 0.0, 0.0)}),
    )
    assert transition.is_enabled(state)


def test_is_disabled_when_trigger_residual_over_tolerance():
    """The transition does not fire when the trigger's residual norm is above
    tolerance."""
    world = _world_body()
    obj = _free_body("obj", SE2(2.0, 0.0, 0.0))
    trigger = _proximity_trigger(world, obj, target_x=0.0, target_y=0.0)
    transition = ConstraintTransition(trigger=trigger, tolerance=0.1)
    state = ModeState(
        configuration=ConstraintConfiguration(),
        body_poses=BodyPoses({world: SE2(), obj: SE2(2.0, 0.0, 0.0)}),
    )
    assert not transition.is_enabled(state)


def test_apply_raises_when_not_enabled():
    """Applying a disabled transition raises rather than silently doing nothing."""
    world = _world_body()
    obj = _free_body("obj", SE2(2.0, 0.0, 0.0))
    trigger = _proximity_trigger(world, obj, target_x=0.0, target_y=0.0)
    transition = ConstraintTransition(trigger=trigger, tolerance=0.1)
    mode: Mode[SE2] = Mode(bodies=[world, obj], constraints=[], anchored_bodies=[world])
    state = mode.snapshot()
    with pytest.raises(ValueError, match="not enabled"):
        transition.apply(mode, state)


def test_construction_rejects_trigger_with_mutable_parameters():
    """Triggers with mutable params are rejected (no values to draw from at runtime)."""
    a = _free_body("a", SE2())
    b = _free_body("b", SE2(1.0, 0.0, 0.0))
    revolute = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0]), names=RevoluteJoint2D.fixed_parameter_names()
        ),
    )
    with pytest.raises(ValueError, match="no mutable parameters"):
        ConstraintTransition(trigger=revolute, tolerance=0.1)


def test_construction_rejects_non_positive_tolerance():
    """Tolerance must be positive."""
    world = _world_body()
    obj = _free_body("obj", SE2())
    trigger = _proximity_trigger(world, obj, target_x=0.0, target_y=0.0)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        ConstraintTransition(trigger=trigger, tolerance=0.0)


def test_apply_adds_constraints_from_factory():
    """Constraints returned by ``add`` end up in the new mode."""
    world = _world_body()
    obj = _free_body("obj", SE2(0.5, 0.0, 0.0))
    trigger = _proximity_trigger(world, obj, target_x=0.5, target_y=0.0)

    def add_marker(_state: ModeState[SE2]) -> list:
        return [
            FixedJoint2D(
                body1=world,
                body2=obj,
                fixed_parameters=ConstraintParameters(
                    values=np.array([0.5, 0.0, 0.0]),
                    names=FixedJoint2D.fixed_parameter_names(),
                ),
            )
        ]

    transition = ConstraintTransition(trigger=trigger, tolerance=0.1, add=add_marker)
    mode: Mode[SE2] = Mode(bodies=[world, obj], constraints=[], anchored_bodies=[world])
    new_mode = transition.apply(mode, mode.snapshot())
    assert len(new_mode.constraints) == 1
    assert isinstance(new_mode.constraints[0], FixedJoint2D)


def test_apply_removes_listed_constraints():
    """Constraints listed in ``remove`` are absent from the new mode."""
    world = _world_body()
    obj = _free_body("obj", SE2(0.5, 0.0, 0.0))
    existing_pin = FixedJoint2D(
        body1=world,
        body2=obj,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.5, 0.0, 0.0]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )
    trigger = _proximity_trigger(world, obj, target_x=0.5, target_y=0.0)
    transition = ConstraintTransition(
        trigger=trigger, tolerance=0.1, remove=(existing_pin,)
    )
    mode: Mode[SE2] = Mode(
        bodies=[world, obj],
        constraints=[existing_pin],
        anchored_bodies=[world],
    )
    new_mode = transition.apply(mode, mode.snapshot())
    assert not new_mode.constraints


def test_apply_rejects_remove_constraint_not_in_mode():
    """Asking to remove a constraint that isn't in the mode raises clearly."""
    world = _world_body()
    obj = _free_body("obj", SE2(0.0, 0.0, 0.0))
    trigger = _proximity_trigger(world, obj, target_x=0.0, target_y=0.0)
    bogus = FixedJoint2D(
        body1=world,
        body2=obj,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 0.0]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )
    transition = ConstraintTransition(trigger=trigger, tolerance=0.1, remove=(bogus,))
    mode: Mode[SE2] = Mode(bodies=[world, obj], constraints=[], anchored_bodies=[world])
    with pytest.raises(ValueError, match="not in mode.constraints"):
        transition.apply(mode, mode.snapshot())


def test_canonical_rigid_attachment_to_end_effector():
    """Arm + free object: when the tip is near the object, attach it.

    After attaching, moving the arm should drag the object along — the solver respects
    the new FixedJoint constraint.
    """
    arm = TwoLinkArm2D()
    world = _world_body()
    # Object placed so the arm's tip can reach it at angles roughly (π/2, 0).
    obj = _free_body("obj", SE2(0.0, 2.0, 0.0))
    mode: Mode[SE2] = Mode(
        bodies=arm.mode.bodies + [world, obj],
        constraints=list(arm.mode.constraints),
        configuration=arm.mode.configuration,
        body_poses=BodyPoses(
            {b: arm.mode.body_poses[b] for b in arm.mode.bodies}
            | {world: SE2(), obj: SE2(0.0, 2.0, 0.0)}
        ),
        anchored_bodies=arm.mode.anchored_bodies + [world],
    )
    # Drive the arm to bring its tip near the object.
    near_state = solve(
        mode,
        delta={
            arm.joint_ab: np.array([math.pi / 2]),
            arm.joint_bc: np.array([0.0]),
        },
    )
    # Apply the new state to the mode so the trigger sees the right poses.
    mode.set_state(near_state)

    # Trigger: tip (offset (1, 0) in link_b's frame) coincident with object's frame.
    trigger = PointEquality2D(
        body1=obj,
        body2=arm.link_b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )
    transition = RigidAttachment2D(arm.link_b, obj, trigger=trigger, tolerance=0.05)
    assert transition.is_enabled(mode.snapshot())

    attached_mode = transition.apply(mode, mode.snapshot())
    assert len(attached_mode.constraints) == len(mode.constraints) + 1

    # Now drive the arm to a different configuration and verify the object
    # tracks along — i.e. the new FixedJoint is enforced by the solver.
    obj_pose_at_attach = SE2(attached_mode.body_poses[obj])
    link_b_pose_at_attach = SE2(attached_mode.body_poses[arm.link_b])
    rel_at_attach = link_b_pose_at_attach.inv() * obj_pose_at_attach

    after_move_poses = solve(
        attached_mode,
        delta={arm.joint_ab: np.array([-math.pi / 4])},
    ).body_poses
    rel_after_move = after_move_poses[arm.link_b].inv() * after_move_poses[obj]
    np.testing.assert_allclose(rel_after_move.A, rel_at_attach.A, atol=1e-6)
