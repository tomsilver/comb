"""Tests for solver module."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.bodies import Body, BodyPoses, Box, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    ConstraintParameters,
    FixedJoint3D,
    RevoluteJoint2D,
    RevoluteJoint3D,
)
from comb.mode import Mode
from comb.solver import solve


def _body_3d(name: str, pose: SE3 | None = None) -> Body[SE3]:
    return Body(
        name=name,
        pose=pose if pose is not None else SE3(),
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )


def _body_2d(name: str, pose: SE2 | None = None) -> Body[SE2]:
    return Body(
        name=name,
        pose=pose if pose is not None else SE2(),
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )


def _z_revolute(a: Body[SE3], b: Body[SE3], origin=(0.0, 0.0, 0.0)) -> RevoluteJoint3D:
    return RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, *origin], dtype=float),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )


def _residual_norm(
    mode: Mode, config: ConstraintConfiguration, body_poses: BodyPoses
) -> float:
    """Sum of L2 norms of all constraint residuals — for validation in tests."""
    norms = []
    for c in mode.constraints:
        if c.parameter_names():
            params = config[c]
        else:
            params = ConstraintParameters(values=np.array([]), names=())
        norms.append(np.linalg.norm(c.constraint_function(params, body_poses)))
    return float(sum(norms))


def test_solve_no_delta_keeps_valid_state():
    """Solving with no delta on an already-valid mode returns essentially the same
    state."""
    a = _body_3d("a")
    b = _body_3d("b", pose=SE3.AngVec(0.5, [0, 0, 1]))
    joint = _z_revolute(a, b)
    config = ConstraintConfiguration(
        {joint: ConstraintParameters(values=np.array([0.5]), names=("angle",))}
    )
    mode: Mode[SE3] = Mode(
        bodies=[a, b],
        constraints=[joint],
        configuration=config,
        anchored_bodies=[a],
    )
    new_state = solve(mode)
    assert _residual_norm(mode, new_state.configuration, new_state.body_poses) < 1e-9
    assert new_state.configuration[joint]["angle"] == pytest.approx(0.5, abs=1e-9)


def test_solve_revolute_chain_3d_is_exact():
    """Applying a delta to a single revolute joint in a chain matches FK exactly."""
    a = _body_3d("a")
    b = _body_3d("b")
    joint = _z_revolute(a, b)
    config = ConstraintConfiguration(
        {joint: ConstraintParameters(values=np.array([0.0]), names=("angle",))}
    )
    mode: Mode[SE3] = Mode(
        bodies=[a, b],
        constraints=[joint],
        configuration=config,
        anchored_bodies=[a],
    )

    new_state = solve(mode, delta={joint: np.array([np.pi / 2])})
    assert new_state.configuration[joint]["angle"] == pytest.approx(np.pi / 2, abs=1e-9)
    expected_b_pose = a.pose * SE3.AngVec(np.pi / 2, [0, 0, 1])
    np.testing.assert_allclose(new_state.body_poses[b].A, expected_b_pose.A, atol=1e-9)
    assert _residual_norm(mode, new_state.configuration, new_state.body_poses) < 1e-9


def test_solve_two_link_chain_3d_propagates():
    """A two-link chain: delta on joint1 propagates to body3 through body2."""
    a = _body_3d("a")
    # Initial state must satisfy the constraints with both joint angles at 0.
    b_pose = a.pose * SE3.Trans([1.0, 0.0, 0.0])
    c_pose = b_pose * SE3.Trans([1.0, 0.0, 0.0])
    b = _body_3d("b", pose=b_pose)
    c = _body_3d("c", pose=c_pose)
    joint_ab = _z_revolute(a, b, origin=(1.0, 0.0, 0.0))
    joint_bc = _z_revolute(b, c, origin=(1.0, 0.0, 0.0))
    config = ConstraintConfiguration(
        {
            joint_ab: ConstraintParameters(values=np.array([0.0]), names=("angle",)),
            joint_bc: ConstraintParameters(values=np.array([0.0]), names=("angle",)),
        }
    )
    mode: Mode[SE3] = Mode(
        bodies=[a, b, c],
        constraints=[joint_ab, joint_bc],
        configuration=config,
        anchored_bodies=[a],
    )

    new_state = solve(mode, delta={joint_ab: np.array([np.pi / 2])})
    assert _residual_norm(mode, new_state.configuration, new_state.body_poses) < 1e-7
    # After rotating joint_ab by 90 deg about z (origin at (1,0,0) in a's frame),
    # body c (which is reachable through b) should end up at the propagated pose.
    expected_b = a.pose * SE3.Trans([1.0, 0.0, 0.0]) * SE3.AngVec(np.pi / 2, [0, 0, 1])
    expected_c = expected_b * SE3.Trans([1.0, 0.0, 0.0]) * SE3.AngVec(0.0, [0, 0, 1])
    np.testing.assert_allclose(new_state.body_poses[b].A, expected_b.A, atol=1e-7)
    np.testing.assert_allclose(new_state.body_poses[c].A, expected_c.A, atol=1e-7)


def test_solve_revolute_chain_2d_is_exact():
    """The 2D analogue of the single-revolute chain test."""
    a = _body_2d("a")
    # b's initial pose must satisfy the constraint at angle=0: at the joint origin.
    b = _body_2d("b", pose=SE2(0.5, 0.0, 0.0))
    joint = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.5, 0.0]),
            names=RevoluteJoint2D.fixed_parameter_names(),
        ),
    )
    config = ConstraintConfiguration(
        {joint: ConstraintParameters(values=np.array([0.0]), names=("angle",))}
    )
    mode: Mode[SE2] = Mode(
        bodies=[a, b],
        constraints=[joint],
        configuration=config,
        anchored_bodies=[a],
    )

    new_state = solve(mode, delta={joint: np.array([np.pi / 3])})
    assert new_state.configuration[joint]["angle"] == pytest.approx(np.pi / 3, abs=1e-9)
    expected_b = a.pose * SE2(0.5, 0.0, np.pi / 3)
    np.testing.assert_allclose(new_state.body_poses[b].A, expected_b.A, atol=1e-9)


def test_solve_with_fixed_constraint_only():
    """A mode with only a fixed constraint converges to the implied pose."""
    a = _body_3d("a")
    # b starts at the wrong pose — solver should move it to satisfy the constraint.
    b = _body_3d("b", pose=SE3.Trans([5.0, 5.0, 5.0]))
    fixed = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[fixed], anchored_bodies=[a])
    new_state = solve(mode)
    expected_b = a.pose * SE3.Trans([1.0, 2.0, 3.0])
    np.testing.assert_allclose(new_state.body_poses[b].A, expected_b.A, atol=1e-7)
    assert len(new_state.configuration) == 0


def test_solve_requires_anchored_body():
    """Solve() refuses to run on a mode with no anchored bodies."""
    a = _body_3d("a")
    b = _body_3d("b", pose=SE3.AngVec(0.5, [0, 0, 1]))
    joint = _z_revolute(a, b)
    config = ConstraintConfiguration(
        {joint: ConstraintParameters(values=np.array([0.5]), names=("angle",))}
    )
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[joint], configuration=config)
    with pytest.raises(ValueError, match="anchored_bodies"):
        solve(mode)


def test_solve_does_not_drift_over_many_calls():
    """Repeated solve+apply cycles must not let body poses drift off SE(2)/SE(3).

    Regression test for an issue where many slider adjustments in the GUI would
    eventually push body pose matrices off the manifold, triggering spatialmath validity
    checks inside Twist2/Twist3.
    """
    a = _body_3d("a")
    b = _body_3d("b")
    joint = _z_revolute(a, b)
    config = ConstraintConfiguration(
        {joint: ConstraintParameters(values=np.array([0.0]), names=("angle",))}
    )
    mode: Mode[SE3] = Mode(
        bodies=[a, b],
        constraints=[joint],
        configuration=config,
        anchored_bodies=[a],
    )
    for _ in range(500):
        new_state = solve(mode, delta={joint: np.array([0.013])})
        for c in mode.constraints:
            if c.parameter_names():
                mode.configuration[c] = new_state.configuration[c]
        for body in mode.bodies:
            mode.body_poses[body] = new_state.body_poses[body]
    # If we got here without spatialmath complaining, we're good. As a sanity
    # check, the rotation block should still be orthonormal.
    rot = mode.body_poses[b].R
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_solve_loop_reduces_residual():
    """For an over-constrained loop, solve cannot zero the residual but reduces it."""
    a = _body_3d("a")
    b = _body_3d("b", pose=SE3.Trans([0.5, 0.0, 0.0]))
    # Two competing fixed constraints that can't both be satisfied.
    f1 = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    f2 = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[f1, f2], anchored_bodies=[a])
    initial_residual = _residual_norm(mode, mode.configuration, mode.body_poses)
    new_state = solve(mode)
    final_residual = _residual_norm(mode, new_state.configuration, new_state.body_poses)
    assert final_residual < initial_residual
