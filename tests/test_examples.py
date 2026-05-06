"""Tests that each example builds a valid system and can be solved."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.constraints import ConstraintParameters
from comb.examples.fixed_pair_3d import FixedPair3D
from comb.examples.single_revolute_2d import SingleRevolute2D
from comb.examples.single_revolute_3d import SingleRevolute3D
from comb.examples.two_link_arm_3d import TwoLinkArm3D
from comb.solver import solve


def _residual_norm(example_system) -> float:
    norms = []
    for c in example_system.constraints:
        if c.parameter_names():
            params = example_system.configuration[c]
        else:
            params = ConstraintParameters(values=np.array([]), names=())
        norms.append(
            np.linalg.norm(c.constraint_function(params, example_system.body_poses))
        )
    return float(sum(norms))


def test_single_revolute_3d_starts_valid():
    """The single 3D revolute example builds in a valid state."""
    ex = SingleRevolute3D(initial_angle=0.4)
    assert _residual_norm(ex.system) < 1e-12
    assert ex.system.anchored_bodies == [ex.base]


def test_single_revolute_3d_solves_delta():
    """Applying a delta to the single 3D revolute example matches FK."""
    ex = SingleRevolute3D()
    new_config, new_poses = solve(ex.system, delta={ex.joint: np.array([np.pi / 2])})
    assert new_config[ex.joint]["angle"] == pytest.approx(np.pi / 2, abs=1e-9)
    expected_link_pose = ex.base.pose * SE3.AngVec(np.pi / 2, [0, 0, 1])
    np.testing.assert_allclose(new_poses[ex.link].A, expected_link_pose.A, atol=1e-9)


def test_two_link_arm_3d_starts_valid():
    """The two-link 3D arm example builds in a valid state."""
    ex = TwoLinkArm3D()
    assert _residual_norm(ex.system) < 1e-12
    assert ex.system.anchored_bodies == [ex.base]


def test_two_link_arm_3d_solves_delta_propagates():
    """A delta on joint_ab propagates through the unchanged joint_bc."""
    ex = TwoLinkArm3D(link_length=1.0)
    _, new_poses = solve(ex.system, delta={ex.joint_ab: np.array([np.pi / 2])})
    expected_a = (
        ex.base.pose * SE3.Trans([1.0, 0.0, 0.0]) * SE3.AngVec(np.pi / 2, [0, 0, 1])
    )
    expected_b = expected_a * SE3.Trans([1.0, 0.0, 0.0])
    np.testing.assert_allclose(new_poses[ex.link_a].A, expected_a.A, atol=1e-7)
    np.testing.assert_allclose(new_poses[ex.link_b].A, expected_b.A, atol=1e-7)


def test_single_revolute_2d_starts_valid():
    """The single 2D revolute example builds in a valid state."""
    ex = SingleRevolute2D(initial_angle=0.3)
    assert _residual_norm(ex.system) < 1e-12


def test_single_revolute_2d_solves_delta():
    """Applying a delta to the 2D revolute example matches FK."""
    ex = SingleRevolute2D()
    _, new_poses = solve(ex.system, delta={ex.joint: np.array([np.pi / 4])})
    expected_link = ex.base.pose * SE2(0.0, 0.0, np.pi / 4)
    np.testing.assert_allclose(new_poses[ex.link].A, expected_link.A, atol=1e-9)


def test_fixed_pair_3d_starts_valid():
    """The fixed-pair example builds in a valid state."""
    ex = FixedPair3D(translation=(2.0, 1.0, -0.5))
    assert _residual_norm(ex.system) < 1e-12


def test_fixed_pair_3d_solves_after_perturbation():
    """If we manually perturb the child's pose, the solver pulls it back."""
    ex = FixedPair3D(translation=(1.0, 2.0, 3.0))
    ex.system.body_poses[ex.child] = SE3.Trans([5.0, 5.0, 5.0])
    _, new_poses = solve(ex.system)
    expected_child = ex.base.pose * SE3.Trans([1.0, 2.0, 3.0])
    np.testing.assert_allclose(new_poses[ex.child].A, expected_child.A, atol=1e-7)
