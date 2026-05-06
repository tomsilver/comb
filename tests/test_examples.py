"""Tests that each example builds a valid mode and can be solved."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.constraints import ConstraintParameters
from comb.examples.fixed_pair_3d import FixedPair3D
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
    ex = SingleRevolute2D(initial_angle=0.3)
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
    ex = TwoLinkArm2D(link_length=1.0)
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
