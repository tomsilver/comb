"""Tests that each example builds a valid mode and can be solved."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.constraints import ConstraintParameters, HingeJoint2D, PointEquality2D
from comb.examples.door_2d import Door2D
from comb.examples.dual_arm_handover_2d import DualArmHandover2D
from comb.examples.fixed_pair_3d import FixedPair3D
from comb.examples.mobile_arm_door_2d import MobileArmDoor2D
from comb.examples.mobile_base_2d import MobileBase2D
from comb.examples.single_revolute_2d import SingleRevolute2D
from comb.examples.single_revolute_3d import SingleRevolute3D
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.examples.two_link_arm_3d import TwoLinkArm3D
from comb.examples.two_link_arm_with_object_2d import TwoLinkArmWithObject2D
from comb.solver import find_satisfying_state, solve


def _residual_norm(example_mode) -> float:
    norms = []
    for c in example_mode.constraints:
        if c.parameter_names():
            params = example_mode.configuration[c]
        else:
            params = ConstraintParameters(values=np.array([]), names=())
        norms.append(
            np.linalg.norm(c.constraint_function(params, example_mode.body_poses))
        )
    return float(sum(norms))


def test_single_revolute_3d_starts_valid():
    """The single 3D revolute example builds in a valid state."""
    ex = SingleRevolute3D(initial_angle=0.4)
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.mode.anchored_bodies == [ex.base]


def test_single_revolute_3d_solves_delta():
    """Applying a delta to the single 3D revolute example matches FK."""
    ex = SingleRevolute3D()
    new_state = solve(ex.mode, delta={ex.joint: np.array([np.pi / 2])})
    assert new_state.configuration[ex.joint]["angle"] == pytest.approx(
        np.pi / 2, abs=1e-9
    )
    expected_link_pose = ex.base.pose * SE3.AngVec(np.pi / 2, [0, 0, 1])
    np.testing.assert_allclose(
        new_state.body_poses[ex.link].A, expected_link_pose.A, atol=1e-9
    )


def test_two_link_arm_3d_starts_valid():
    """The two-link 3D arm example builds in a valid state."""
    ex = TwoLinkArm3D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.mode.anchored_bodies == [ex.base]


def test_two_link_arm_3d_solves_delta_propagates():
    """A delta on joint_ab propagates through the unchanged joint_bc."""
    ex = TwoLinkArm3D(link_length=1.0)
    new_poses = solve(ex.mode, delta={ex.joint_ab: np.array([np.pi / 2])}).body_poses
    expected_a = (
        ex.base.pose * SE3.Trans([1.0, 0.0, 0.0]) * SE3.AngVec(np.pi / 2, [0, 0, 1])
    )
    expected_b = expected_a * SE3.Trans([1.0, 0.0, 0.0])
    np.testing.assert_allclose(new_poses[ex.link_a].A, expected_a.A, atol=1e-7)
    np.testing.assert_allclose(new_poses[ex.link_b].A, expected_b.A, atol=1e-7)


def test_single_revolute_2d_starts_valid():
    """The single 2D revolute example builds in a valid state."""
    ex = SingleRevolute2D()
    assert _residual_norm(ex.mode) < 1e-12


def test_single_revolute_2d_solves_delta():
    """Applying a delta to the 2D revolute example matches FK."""
    ex = SingleRevolute2D()
    new_poses = solve(ex.mode, delta={ex.joint: np.array([np.pi / 4])}).body_poses
    expected_link = ex.base.pose * SE2(0.0, 0.0, np.pi / 4)
    np.testing.assert_allclose(new_poses[ex.link].A, expected_link.A, atol=1e-9)


def test_two_link_arm_2d_starts_valid():
    """The two-link 2D arm example builds in a valid state."""
    ex = TwoLinkArm2D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.mode.anchored_bodies == [ex.base]


def test_two_link_arm_2d_solves_delta_propagates():
    """A delta on joint_ab propagates through the unchanged joint_bc."""
    ex = TwoLinkArm2D()
    new_poses = solve(ex.mode, delta={ex.joint_ab: np.array([np.pi / 2])}).body_poses
    # Each link's frame sits at its joint pivot, so after rotating joint_ab
    # by 90 deg about z, link_a's frame is still at the base origin but
    # rotated 90 deg, and link_b's frame sits at link_a's far end.
    expected_a = ex.base.pose * SE2(0.0, 0.0, np.pi / 2)
    expected_b = expected_a * SE2(1.0, 0.0, 0.0)
    np.testing.assert_allclose(new_poses[ex.link_a].A, expected_a.A, atol=1e-7)
    np.testing.assert_allclose(new_poses[ex.link_b].A, expected_b.A, atol=1e-7)


def test_mobile_base_2d_starts_valid():
    """The mobile base example builds in a valid state with the base at the origin."""
    ex = MobileBase2D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.mode.anchored_bodies == [ex.world]
    np.testing.assert_array_equal(ex.mode.body_poses[ex.base].t, [0.0, 0.0])


def test_mobile_base_2d_solves_delta_drives_base():
    """A delta on the planar joint moves the base to (tx, ty, theta) in world frame."""
    ex = MobileBase2D()
    new_state = solve(
        ex.mode,
        delta={ex.joint: np.array([1.5, -0.5, np.pi / 3])},
    )
    np.testing.assert_allclose(new_state.body_poses[ex.base].t, [1.5, -0.5], atol=1e-7)
    assert new_state.body_poses[ex.base].theta() == pytest.approx(np.pi / 3, abs=1e-7)
    assert new_state.configuration[ex.joint]["tx"] == pytest.approx(1.5)
    assert new_state.configuration[ex.joint]["ty"] == pytest.approx(-0.5)
    assert new_state.configuration[ex.joint]["theta"] == pytest.approx(np.pi / 3)


def test_door_2d_starts_valid():
    """The door example builds in a valid state with the door closed (angle 0)."""
    ex = Door2D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.mode.anchored_bodies == [ex.wall]
    assert ex.mode.configuration[ex.hinge]["angle"] == 0.0


def test_door_2d_swings_under_delta():
    """Driving the hinge angle rotates the door about the hinge."""
    ex = Door2D()
    new_state = solve(ex.mode, delta={ex.hinge: np.array([np.pi / 2])})
    assert new_state.body_poses[ex.door].theta() == pytest.approx(np.pi / 2, abs=1e-7)
    assert new_state.configuration[ex.hinge]["angle"] == pytest.approx(np.pi / 2)


def test_door_2d_clamps_at_max_angle():
    """A delta past max_angle clamps via the BoundedReal parameter space.

    Builds a door with ``max_angle=π/2`` directly because the bundled YAML library
    hardcodes ``max_angle=π``.
    """
    from comb.bodies import Body, Rectangle  # pylint: disable=import-outside-toplevel
    from comb.constraints import (  # pylint: disable=import-outside-toplevel
        ConstraintConfiguration,
    )
    from comb.mode import Mode  # pylint: disable=import-outside-toplevel

    wall = Body(
        name="wall",
        pose=SE2(),
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )
    door = Body(
        name="door",
        pose=SE2(),
        visual_geometry=Rectangle(0.8, 0.05, offset_x=0.4),
        collision_geometry=Rectangle(0.8, 0.05, offset_x=0.4),
    )
    hinge = HingeJoint2D(
        body1=wall,
        body2=door,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 0.0, np.pi / 2]),
            names=HingeJoint2D.fixed_parameter_names(),
        ),
    )
    mode = Mode(
        bodies=[wall, door],
        constraints=[hinge],
        configuration=ConstraintConfiguration(
            {hinge: ConstraintParameters(values=np.array([0.0]), names=("angle",))}
        ),
        anchored_bodies=[wall],
    )
    new_state = solve(mode, delta={hinge: np.array([10.0])})
    assert new_state.configuration[hinge]["angle"] == pytest.approx(np.pi / 2)


def test_arm_with_object_2d_starts_valid():
    """The arm-with-object example builds valid: arm at zero, block pinned to world."""
    ex = TwoLinkArmWithObject2D()
    assert _residual_norm(ex.mode) < 1e-12
    np.testing.assert_array_equal(ex.mode.body_poses[ex.block].t, [0.5, 1.0])
    assert ex.world_to_block in ex.mode.constraints


def test_arm_with_object_2d_pickup_transition_attaches_block_to_arm():
    """Bring the arm tip to the block, fire pickup, verify the block now follows the
    arm."""
    ex = TwoLinkArmWithObject2D()
    # Drive the arm so its tip is at the block's pose by augmenting with the
    # tip-at-block constraint and IK-solving.
    near_state = find_satisfying_state(ex.mode, [ex.pickup_trigger])
    ex.mode.set_state(near_state)
    assert ex.pickup_transition.is_enabled(ex.mode.snapshot())

    new_mode = ex.pickup_transition.apply(ex.mode, ex.mode.snapshot())
    # The world→block pin is gone; the arm→block attachment is in.
    assert ex.world_to_block not in new_mode.constraints
    assert any(
        isinstance(c, type(ex.world_to_block))
        and c.body1 is ex.arm.link_b
        and c.body2 is ex.block
        for c in new_mode.constraints
    )

    # Move the arm; the block tracks along under the new constraint.
    rel_at_attach = (
        new_mode.body_poses[ex.arm.link_b].inv() * new_mode.body_poses[ex.block]
    )
    after_move_poses = solve(
        new_mode, delta={ex.arm.joint_ab: np.array([-np.pi / 6])}
    ).body_poses
    rel_after_move = after_move_poses[ex.arm.link_b].inv() * after_move_poses[ex.block]
    np.testing.assert_allclose(rel_after_move.A, rel_at_attach.A, atol=1e-6)


def test_fixed_pair_3d_starts_valid():
    """The fixed-pair example builds in a valid state."""
    ex = FixedPair3D(translation=(2.0, 1.0, -0.5))
    assert _residual_norm(ex.mode) < 1e-12


def test_fixed_pair_3d_solves_after_perturbation():
    """If we manually perturb the child's pose, the solver pulls it back."""
    ex = FixedPair3D(translation=(1.0, 2.0, 3.0))
    ex.mode.body_poses[ex.child] = SE3.Trans([5.0, 5.0, 5.0])
    new_poses = solve(ex.mode).body_poses
    expected_child = ex.base.pose * SE3.Trans([1.0, 2.0, 3.0])
    np.testing.assert_allclose(new_poses[ex.child].A, expected_child.A, atol=1e-7)


def test_mobile_arm_door_2d_starts_valid():
    """The mobile-arm-door example builds in a valid state with the door pinned."""
    ex = MobileArmDoor2D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.world_to_door in ex.mode.constraints
    np.testing.assert_array_equal(ex.mode.body_poses[ex.door].t, [2.0, 0.0])


def test_mobile_arm_door_2d_attach_trigger_initially_far():
    """At construction the arm tip is far from the door handle — trigger doesn't
    fire."""
    ex = MobileArmDoor2D()
    assert not ex.attach_transition.is_enabled(ex.mode.snapshot())


def test_mobile_arm_door_2d_attach_swaps_in_hinge_and_grip():
    """Once the tip reaches the handle, applying the transition adds a hinge + grip."""
    ex = MobileArmDoor2D()
    near_state = find_satisfying_state(ex.mode, [ex.attach_trigger])
    ex.mode.set_state(near_state)
    assert ex.attach_transition.is_enabled(ex.mode.snapshot())

    new_mode = ex.attach_transition.apply(ex.mode, ex.mode.snapshot())
    assert ex.world_to_door not in new_mode.constraints
    # Hinge added between world and door.
    hinges = [c for c in new_mode.constraints if isinstance(c, HingeJoint2D)]
    assert len(hinges) == 1
    assert hinges[0].body1 is ex.world
    assert hinges[0].body2 is ex.door
    # The grip constraint (PointEquality2D) is in the post-attach mode.
    grips = [
        c
        for c in new_mode.constraints
        if isinstance(c, PointEquality2D)
        and c.body1 is ex.door
        and c.body2 is ex.link_b
    ]
    assert len(grips) == 1
    # Hinge angle starts at 0 (zero-init by ConstraintTransition.apply).
    assert new_mode.configuration[hinges[0]]["angle"] == 0.0


def test_dual_arm_handover_2d_starts_valid():
    """The dual-arm-handover example builds in a valid state with the object pinned."""
    ex = DualArmHandover2D()
    assert _residual_norm(ex.mode) < 1e-12
    assert ex.world_to_object in ex.mode.constraints
    assert ex.arm_a_base in ex.mode.anchored_bodies
    assert ex.arm_b_base in ex.mode.anchored_bodies
    assert not ex.pickup_transition.is_enabled(ex.mode.snapshot())
    assert not ex.handover_transition.is_enabled(ex.mode.snapshot())


def test_dual_arm_handover_2d_pickup_swaps_object_to_arm_a():
    """Bringing arm A's tip to the object enables pickup; applying it grafts the object
    onto arm A."""
    ex = DualArmHandover2D()
    near_state = find_satisfying_state(ex.mode, [ex.pickup_trigger])
    ex.mode.set_state(near_state)
    assert ex.pickup_transition.is_enabled(ex.mode.snapshot())

    held_mode = ex.pickup_transition.apply(ex.mode, ex.mode.snapshot())
    assert ex.world_to_object not in held_mode.constraints
    a_grips = [
        c
        for c in held_mode.constraints
        if c.body1 is ex.arm_a_link_b and c.body2 is ex.object_body
    ]
    assert len(a_grips) == 1


def test_dual_arm_handover_2d_handover_swaps_object_from_a_to_b():
    """After pickup, bringing arm B to the object fires handover: A grip out, B grip
    in."""
    ex = DualArmHandover2D()
    # Stage 1: pick up with arm A.
    near_state = find_satisfying_state(ex.mode, [ex.pickup_trigger])
    ex.mode.set_state(near_state)
    held_mode = ex.pickup_transition.apply(ex.mode, ex.mode.snapshot())
    a_grip = next(
        c
        for c in held_mode.constraints
        if c.body1 is ex.arm_a_link_b and c.body2 is ex.object_body
    )

    # Stage 2: drive arm A so the object lands somewhere arm B can also reach,
    # plus arm B's tip onto the object — find_satisfying_state handles all of it.
    handover_ready = find_satisfying_state(held_mode, [ex.handover_trigger])
    held_mode.set_state(handover_ready)
    assert ex.handover_transition.is_enabled(held_mode.snapshot())

    after = ex.handover_transition.apply(held_mode, held_mode.snapshot())
    # The A grip is gone (dynamic remove found and pulled it out).
    assert a_grip not in after.constraints
    assert not any(
        c.body1 is ex.arm_a_link_b and c.body2 is ex.object_body
        for c in after.constraints
    )
    # A new B grip is in.
    b_grips = [
        c
        for c in after.constraints
        if c.body1 is ex.arm_b_link_b and c.body2 is ex.object_body
    ]
    assert len(b_grips) == 1
