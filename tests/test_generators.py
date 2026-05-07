"""Tests for state-dependent constraint generators."""

import math

import numpy as np
import pytest
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import (
    ConstraintConfiguration,
    FixedJoint2D,
    PointEquality2D,
)
from comb.generators import (
    GENERATORS_2D,
    freeze_pose_2d,
    point_pin_2d,
    rigid_attachment_2d,
)
from comb.mode import ModeState


def _free_body(name: str, pose: SE2) -> Body[SE2]:
    return Body(
        name=name,
        pose=pose,
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )


def _state(poses: dict[Body[SE2], SE2]) -> ModeState[SE2]:
    return ModeState(
        configuration=ConstraintConfiguration(),
        body_poses=BodyPoses(poses),
    )


def test_rigid_attachment_2d_captures_relative_pose():
    """Generated FixedJoint2D's parameters reproduce the current relative transform."""
    a = _free_body("a", SE2(1.0, 0.0, math.pi / 6))
    b = _free_body("b", SE2(2.0, 0.5, math.pi / 3))
    state = _state({a: a.pose, b: b.pose})

    constraints = rigid_attachment_2d(a, b)(state)

    assert len(constraints) == 1
    fixed_joint = constraints[0]
    assert isinstance(fixed_joint, FixedJoint2D)
    assert fixed_joint.body1 is a
    assert fixed_joint.body2 is b
    expected_rel = a.pose.inv() * b.pose
    fp = fixed_joint.fixed_parameters
    np.testing.assert_allclose(fp["tx"], float(expected_rel.t[0]), atol=1e-12)
    np.testing.assert_allclose(fp["ty"], float(expected_rel.t[1]), atol=1e-12)
    np.testing.assert_allclose(fp["theta"], float(expected_rel.theta()), atol=1e-12)


def test_rigid_attachment_2d_residual_zero_at_capture_state():
    """The captured constraint's residual is zero at the state it was built from."""
    a = _free_body("a", SE2(0.7, -0.3, math.pi / 5))
    b = _free_body("b", SE2(1.4, 0.9, -math.pi / 4))
    state = _state({a: a.pose, b: b.pose})

    constraint = rigid_attachment_2d(a, b)(state)[0]

    residual = constraint.constraint_function(
        constraint.fixed_parameters, state.body_poses
    )
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_freeze_pose_2d_pins_body_to_world_at_current_pose():
    """``freeze_pose_2d(world, body)`` produces the same constraint as
    ``rigid_attachment_2d(world, body)``."""
    world = _free_body("world", SE2())
    body = _free_body("body", SE2(0.4, 1.2, math.pi / 7))
    state = _state({world: world.pose, body: body.pose})

    via_freeze = freeze_pose_2d(world, body)(state)[0]
    via_rigid = rigid_attachment_2d(world, body)(state)[0]

    np.testing.assert_allclose(
        via_freeze.fixed_parameters.values, via_rigid.fixed_parameters.values
    )


def test_point_pin_2d_residual_zero_at_capture_state():
    """``point_pin_2d`` builds a PointEquality2D satisfied at the capture state."""
    a = _free_body("a", SE2(0.0, 0.0, math.pi / 4))
    b = _free_body("b", SE2(1.0, 0.5, -math.pi / 6))
    state = _state({a: a.pose, b: b.pose})

    constraint = point_pin_2d(a, b)(state)[0]

    assert isinstance(constraint, PointEquality2D)
    residual = constraint.constraint_function(
        constraint.fixed_parameters, state.body_poses
    )
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_point_pin_2d_with_body2_offset():
    """A non-zero ``body2_offset`` pins that point of body2, not the origin.

    Verified by checking the residual is zero when body2 has been moved so that its
    body2_offset point ends up at body1's origin.
    """
    a = _free_body("a", SE2())
    b = _free_body("b", SE2(2.0, 0.0, 0.0))
    capture_state = _state({a: a.pose, b: b.pose})

    constraint = point_pin_2d(a, b, body2_offset=(0.5, 0.0))(capture_state)[0]

    fp = constraint.fixed_parameters
    assert float(fp["offset_x"]) == pytest.approx(0.5)
    assert float(fp["offset_y"]) == pytest.approx(0.0)
    residual = constraint.constraint_function(fp, capture_state.body_poses)
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-12)


def test_generators_registry_lists_all_public_2d_generators():
    """``GENERATORS_2D`` is the closed registry; updating the module without updating
    the registry should not be silently allowed."""
    assert set(GENERATORS_2D) == {
        "rigid_attachment_2d",
        "freeze_pose_2d",
        "point_pin_2d",
    }
    assert GENERATORS_2D["rigid_attachment_2d"] is rigid_attachment_2d
    assert GENERATORS_2D["freeze_pose_2d"] is freeze_pose_2d
    assert GENERATORS_2D["point_pin_2d"] is point_pin_2d
