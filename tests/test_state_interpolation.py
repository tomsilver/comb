"""Tests for SystemState and interpolators of Configuration / BodyPoses / SystemState.

These primitives are what a future trajectory-producing solver will compose into a
``Trajectory[SystemState[PoseT]]``.
"""

import math

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.bodies import Body, BodyPoses, Box, Rectangle, interpolate_body_poses
from comb.constraints import (
    Configuration,
    ConstraintParameters,
    RevoluteJoint2D,
    RevoluteJoint3D,
    interpolate_configuration,
)
from comb.parameter_spaces import BoundedReal
from comb.system import SystemState, interpolate_system_state
from comb.trajectories import linear_segment


def _body_3d(name: str) -> Body[SE3]:
    return Body(
        name=name,
        pose=SE3(),
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )


def _body_2d(name: str) -> Body[SE2]:
    return Body(
        name=name,
        pose=SE2(),
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )


def _revolute_2d(a: Body[SE2], b: Body[SE2]) -> RevoluteJoint2D:
    return RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0]),
            names=RevoluteJoint2D.fixed_parameter_names(),
        ),
    )


def _revolute_3d(a: Body[SE3], b: Body[SE3]) -> RevoluteJoint3D:
    return RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )


def test_interpolate_configuration_endpoints():
    """At s=0 the result equals start; at s=1 it equals end (for non-circular
    params)."""
    a, b = _body_3d("a"), _body_3d("b")
    joint = _revolute_3d(a, b)
    # Override default Circle with BoundedReal so straight-line interpolation
    # is what we expect at the endpoints.
    bounded_joint = RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=joint.fixed_parameters,
        parameter_space_overrides=(BoundedReal(-2.0, 2.0),),
    )
    start = Configuration(
        {bounded_joint: ConstraintParameters(np.array([0.5]), ("angle",))}
    )
    end = Configuration(
        {bounded_joint: ConstraintParameters(np.array([1.5]), ("angle",))}
    )
    at_zero = interpolate_configuration(start, end, 0.0)
    at_one = interpolate_configuration(start, end, 1.0)
    at_half = interpolate_configuration(start, end, 0.5)
    assert at_zero[bounded_joint]["angle"] == pytest.approx(0.5)
    assert at_one[bounded_joint]["angle"] == pytest.approx(1.5)
    assert at_half[bounded_joint]["angle"] == pytest.approx(1.0)


def test_interpolate_configuration_takes_short_way_on_circle():
    """A revolute joint defaults to Circle, so interpolation crosses ±π via shortest
    arc."""
    a, b = _body_3d("a"), _body_3d("b")
    joint = _revolute_3d(a, b)
    # 3.0 -> -3.0 the short way crosses ±π, not 0.
    start = Configuration({joint: ConstraintParameters(np.array([3.0]), ("angle",))})
    end = Configuration({joint: ConstraintParameters(np.array([-3.0]), ("angle",))})
    mid = interpolate_configuration(start, end, 0.5)
    angle = mid[joint]["angle"]
    # Short way midpoint is ±π (Circle.retract canonicalizes -π to +π).
    assert abs(abs(angle) - math.pi) < 1e-9


def test_interpolate_configuration_clamps_on_bounded_real():
    """BoundedReal stays in range under interpolation, even with overshooting s."""
    a, b = _body_3d("a"), _body_3d("b")
    joint = RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
        parameter_space_overrides=(BoundedReal(0.0, 1.0),),
    )
    start = Configuration({joint: ConstraintParameters(np.array([0.0]), ("angle",))})
    end = Configuration({joint: ConstraintParameters(np.array([1.0]), ("angle",))})
    over = interpolate_configuration(start, end, 1.5)
    assert over[joint]["angle"] == pytest.approx(1.0)


def test_interpolate_configuration_rejects_mismatched_constraint_sets():
    """The two configurations must share the same constraints."""
    a, b, c = _body_3d("a"), _body_3d("b"), _body_3d("c")
    j1 = _revolute_3d(a, b)
    j2 = _revolute_3d(b, c)
    start = Configuration({j1: ConstraintParameters(np.array([0.0]), ("angle",))})
    end = Configuration({j2: ConstraintParameters(np.array([0.0]), ("angle",))})
    with pytest.raises(ValueError, match="matching constraint sets"):
        interpolate_configuration(start, end, 0.5)


def test_interpolate_body_poses_se3():
    """Per-body SE(3) interp at s=0.5 matches spatialmath's interp."""
    a, b = _body_3d("a"), _body_3d("b")
    p_a_start = SE3()
    p_a_end = SE3.Trans(2.0, 0.0, 0.0)
    p_b_start = SE3.Trans(0.0, 0.0, 0.0)
    p_b_end = SE3.Rt(SE3.Rx(1.0).R, [0.0, 0.0, 0.0])
    start = BodyPoses({a: p_a_start, b: p_b_start})
    end = BodyPoses({a: p_a_end, b: p_b_end})
    half = interpolate_body_poses(start, end, 0.5)
    np.testing.assert_allclose(half[a].t, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(half[b].A, p_b_start.interp(p_b_end, 0.5).A, atol=1e-12)


def test_interpolate_body_poses_se2():
    """Per-body SE(2) interp midpoint."""
    a = _body_2d("a")
    start = BodyPoses({a: SE2(0.0, 0.0, 0.0)})
    end = BodyPoses({a: SE2(2.0, 4.0, 1.0)})
    half = interpolate_body_poses(start, end, 0.5)
    np.testing.assert_allclose(half[a].t, [1.0, 2.0], atol=1e-12)
    assert half[a].theta() == pytest.approx(0.5)


def test_interpolate_body_poses_rejects_mismatched_body_sets():
    """Both inputs must hold the same set of bodies."""
    a, b = _body_3d("a"), _body_3d("b")
    start = BodyPoses({a: SE3()})
    end = BodyPoses({b: SE3()})
    with pytest.raises(ValueError, match="matching body sets"):
        interpolate_body_poses(start, end, 0.5)


def test_interpolate_body_poses_ndarray():
    """Ndarray pose values interpolate linearly (degenerate but supported)."""
    body = Body(
        name="ndarr",
        pose=np.zeros(3),
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )
    start = BodyPoses({body: np.array([0.0, 0.0, 0.0])})
    end = BodyPoses({body: np.array([2.0, 4.0, 6.0])})
    half = interpolate_body_poses(start, end, 0.5)
    np.testing.assert_allclose(half[body], [1.0, 2.0, 3.0])


def test_system_state_immutable_snapshot():
    """SystemState bundles configuration and body_poses as a frozen pair."""
    a, b = _body_3d("a"), _body_3d("b")
    joint = _revolute_3d(a, b)
    config = Configuration({joint: ConstraintParameters(np.array([0.5]), ("angle",))})
    poses = BodyPoses({a: SE3(), b: SE3.Trans(1.0, 0.0, 0.0)})
    state: SystemState[SE3] = SystemState(configuration=config, body_poses=poses)
    assert state.configuration is config
    assert state.body_poses is poses
    with pytest.raises(Exception):
        state.configuration = Configuration()  # type: ignore[misc]


def test_interpolate_system_state_combines_pieces():
    """interpolate_system_state interpolates configuration and body_poses together."""
    a, b = _body_2d("a"), _body_2d("b")
    joint = _revolute_2d(a, b)
    cfg_start = Configuration(
        {joint: ConstraintParameters(np.array([0.0]), ("angle",))}
    )
    cfg_end = Configuration({joint: ConstraintParameters(np.array([1.0]), ("angle",))})
    poses_start = BodyPoses({a: SE2(0.0, 0.0, 0.0), b: SE2(1.0, 0.0, 0.0)})
    poses_end = BodyPoses({a: SE2(0.0, 0.0, 0.0), b: SE2(1.0, 0.0, 1.0)})
    start = SystemState(configuration=cfg_start, body_poses=poses_start)
    end = SystemState(configuration=cfg_end, body_poses=poses_end)
    half = interpolate_system_state(start, end, 0.5)
    assert half.configuration[joint]["angle"] == pytest.approx(0.5)
    np.testing.assert_allclose(half.body_poses[a].t, [0.0, 0.0], atol=1e-12)
    assert half.body_poses[b].theta() == pytest.approx(0.5)


def test_system_state_trajectory_via_linear_segment():
    """End-to-end: a Trajectory[SystemState] queries cleanly at any time."""
    a, b = _body_2d("a"), _body_2d("b")
    joint = _revolute_2d(a, b)
    cfg_start = Configuration(
        {joint: ConstraintParameters(np.array([0.0]), ("angle",))}
    )
    cfg_end = Configuration({joint: ConstraintParameters(np.array([1.0]), ("angle",))})
    poses_start = BodyPoses({a: SE2(0.0, 0.0, 0.0), b: SE2(1.0, 0.0, 0.0)})
    poses_end = BodyPoses({a: SE2(0.0, 0.0, 0.0), b: SE2(1.0, 0.0, 1.0)})
    traj = linear_segment(
        SystemState(configuration=cfg_start, body_poses=poses_start),
        SystemState(configuration=cfg_end, body_poses=poses_end),
        duration=2.0,
        interpolate=interpolate_system_state,
    )
    quarter = traj(0.5)
    assert quarter.configuration[joint]["angle"] == pytest.approx(0.25)
    assert quarter.body_poses[b].theta() == pytest.approx(0.25)
    end_state = traj(2.0)
    assert end_state.configuration[joint]["angle"] == pytest.approx(1.0)
