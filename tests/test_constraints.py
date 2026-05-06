"""Tests for constraints module."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.bodies import Body, BodyPoses, Box, Rectangle
from comb.constraints import (
    Constraint,
    ConstraintConfiguration,
    ConstraintParameters,
    FixedJoint2D,
    FixedJoint3D,
    PlanarJoint2D,
    PointEquality2D,
    RevoluteJoint2D,
    RevoluteJoint3D,
)
from comb.parameter_spaces import Circle, Real


def _make_body_3d(name: str) -> Body[SE3]:
    return Body(
        name=name,
        pose=SE3(),
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )


def _make_body_2d(name: str) -> Body[SE2]:
    return Body(
        name=name,
        pose=SE2(),
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )


def test_point_equality_2d_residual_zero_when_tip_meets_target():
    """The residual is zero exactly when body2's offset point matches body1's target
    point."""
    world = _make_body_2d("world")
    end_effector = _make_body_2d("ee")
    # Tip in end-effector's frame is at (1, 0). World target is at (3, 4).
    constraint = PointEquality2D(
        body1=world,
        body2=end_effector,
        fixed_parameters=ConstraintParameters(
            values=np.array([3.0, 4.0, 1.0, 0.0]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )
    # Place end-effector at (2, 4, 0): tip at (3, 4). Residual zero.
    poses = BodyPoses({world: SE2(), end_effector: SE2(2.0, 4.0, 0.0)})
    np.testing.assert_allclose(
        constraint.constraint_function(ConstraintParameters(np.array([]), ()), poses),
        [0.0, 0.0],
        atol=1e-12,
    )
    # Move end-effector. Residual is the tip displacement from target.
    poses[end_effector] = SE2(0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        constraint.constraint_function(ConstraintParameters(np.array([]), ()), poses),
        [3.0 - 1.0, 4.0 - 0.0],
        atol=1e-12,
    )


def test_point_equality_2d_uses_body2_orientation():
    """The offset is in body2's frame, so its rotation moves the tip in world coords."""
    world = _make_body_2d("world")
    end_effector = _make_body_2d("ee")
    # Tip at (1, 0) in body2's frame. Target at (0, 1) in world.
    constraint = PointEquality2D(
        body1=world,
        body2=end_effector,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 1.0, 1.0, 0.0]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )
    # Body2 at origin, rotated 90°: tip is at (0, 1). Residual zero.
    poses = BodyPoses({world: SE2(), end_effector: SE2(0.0, 0.0, np.pi / 2)})
    np.testing.assert_allclose(
        constraint.constraint_function(ConstraintParameters(np.array([]), ()), poses),
        [0.0, 0.0],
        atol=1e-12,
    )


def test_planar_joint_2d_structure():
    """PlanarJoint2D has no fixed parameters and three mutable ones, with Circle on
    theta."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = PlanarJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
    )
    assert not joint.fixed_parameter_names()
    assert joint.parameter_names() == ("tx", "ty", "theta")
    spaces = joint.parameter_spaces
    assert isinstance(spaces[0], Real)
    assert isinstance(spaces[1], Real)
    assert isinstance(spaces[2], Circle)


def test_planar_joint_2d_residual_zero_when_relative_transform_matches():
    """The constraint residual is zero when body2 = body1 * SE2(tx, ty, theta)."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = PlanarJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
    )
    params = ConstraintParameters(
        values=np.array([1.0, 2.0, 0.5]),
        names=PlanarJoint2D.parameter_names(),
    )
    poses = BodyPoses(
        {a: SE2(0.0, 0.0, 0.0), b: SE2(1.0, 2.0, 0.5)},
    )
    np.testing.assert_allclose(
        joint.constraint_function(params, poses), [0.0, 0.0, 0.0], atol=1e-12
    )


def _make_revolute_3d(a: Body[SE3], b: Body[SE3]) -> RevoluteJoint3D:
    return RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )


def test_constraint_parameters_lookup():
    """ConstraintParameters supports name-based lookup of values."""
    p = ConstraintParameters(
        values=np.array([1.0, 2.0, 3.0]),
        names=("a", "b", "c"),
    )
    assert p["a"] == 1.0
    assert p["b"] == 2.0
    assert p["c"] == 3.0


def test_constraint_parameters_rejects_length_mismatch():
    """ConstraintParameters rejects values whose length doesn't match names."""
    with pytest.raises(ValueError):
        ConstraintParameters(values=np.array([1.0, 2.0]), names=("a",))


def test_constraint_parameters_rejects_non_1d():
    """ConstraintParameters rejects non-1D arrays."""
    with pytest.raises(ValueError):
        ConstraintParameters(
            values=np.array([[1.0], [2.0]]),
            names=("a", "b"),
        )


def test_constraint_parameters_rejects_duplicate_names():
    """ConstraintParameters rejects duplicate names."""
    with pytest.raises(ValueError):
        ConstraintParameters(
            values=np.array([1.0, 2.0]),
            names=("a", "a"),
        )


def test_constraint_parameters_empty():
    """ConstraintParameters supports the empty case (no parameters)."""
    p = ConstraintParameters(values=np.array([]), names=())
    assert p.values.shape == (0,)


def test_revolute_joint_3d_structure():
    """RevoluteJoint3D stores axis/origin as fixed and angle as mutable."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = _make_revolute_3d(a, b)
    assert joint.body1 is a
    assert joint.body2 is b
    assert joint.fixed_parameters["axis_z"] == 1.0
    assert joint.parameter_names() == ("angle",)


def test_revolute_joint_2d_structure():
    """RevoluteJoint2D stores only origin as fixed; axis is implicit out-of-plane."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.5, -0.25]),
            names=RevoluteJoint2D.fixed_parameter_names(),
        ),
    )
    assert joint.fixed_parameters["origin_x"] == 0.5
    assert joint.fixed_parameters["origin_y"] == -0.25
    assert joint.parameter_names() == ("angle",)


def test_fixed_joint_3d_structure():
    """FixedJoint3D stores an SE(3) transform with no mutable parameters."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    c = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    assert c.fixed_parameters["qw"] == 1.0
    assert not c.parameter_names()


def test_fixed_joint_2d_structure():
    """FixedJoint2D stores an SE(2) transform with no mutable parameters."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    c = FixedJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 2.0, 0.5]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )
    assert c.fixed_parameters["tx"] == 1.0
    assert c.fixed_parameters["theta"] == 0.5
    assert not c.parameter_names()


def test_constraint_rejects_wrong_fixed_parameter_names():
    """A constraint subclass rejects fixed parameters with the wrong names."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    with pytest.raises(ValueError):
        RevoluteJoint3D(
            body1=a,
            body2=b,
            fixed_parameters=ConstraintParameters(
                values=np.array([1.0, 2.0]),
                names=("foo", "bar"),
            ),
        )


def test_constraint_is_abstract():
    """The base Constraint class cannot be instantiated."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    empty = ConstraintParameters(values=np.array([]), names=())
    with pytest.raises(TypeError):
        Constraint(  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated
            body1=a, body2=b, fixed_parameters=empty
        )


def test_configuration_basic_set_and_get():
    """ConstraintConfiguration stores and returns mutable parameters per constraint."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = _make_revolute_3d(a, b)
    config = ConstraintConfiguration()
    config[joint] = ConstraintParameters(values=np.array([0.5]), names=("angle",))
    assert config[joint]["angle"] == 0.5
    assert joint in config
    assert len(config) == 1


def test_configuration_update_in_place():
    """Reassigning parameters for the same constraint replaces the previous value."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = _make_revolute_3d(a, b)
    config = ConstraintConfiguration()
    config[joint] = ConstraintParameters(values=np.array([0.0]), names=("angle",))
    config[joint] = ConstraintParameters(values=np.array([np.pi / 2]), names=("angle",))
    assert config[joint]["angle"] == pytest.approx(np.pi / 2)
    assert len(config) == 1


def test_configuration_rejects_wrong_names():
    """ConstraintConfiguration rejects parameters whose names don't match the
    constraint."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = _make_revolute_3d(a, b)
    config = ConstraintConfiguration()
    with pytest.raises(ValueError):
        config[joint] = ConstraintParameters(
            values=np.array([0.0]), names=("not_angle",)
        )


def test_configuration_distinguishes_constraint_instances():
    """Two structurally identical constraints are kept as separate
    ConstraintConfiguration keys."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint1 = _make_revolute_3d(a, b)
    joint2 = _make_revolute_3d(a, b)
    config = ConstraintConfiguration()
    config[joint1] = ConstraintParameters(values=np.array([0.1]), names=("angle",))
    config[joint2] = ConstraintParameters(values=np.array([0.9]), names=("angle",))
    assert config[joint1]["angle"] == pytest.approx(0.1)
    assert config[joint2]["angle"] == pytest.approx(0.9)
    assert len(config) == 2


def test_configuration_zeros():
    """ConstraintConfiguration.zeros initializes every constraint's parameters to
    zero."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    revolute = _make_revolute_3d(a, b)
    config = ConstraintConfiguration.zeros([revolute])
    np.testing.assert_array_equal(config[revolute].values, [0.0])
    assert config[revolute].names == ("angle",)


def test_configuration_unknown_constraint_raises():
    """Looking up a constraint that was never set raises KeyError."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = _make_revolute_3d(a, b)
    config = ConstraintConfiguration()
    with pytest.raises(KeyError):
        _ = config[joint]


# ---------- relative_transform / constraint_function ---------- #


def _empty_params() -> ConstraintParameters:
    return ConstraintParameters(values=np.array([]), names=())


def test_fixed_joint_2d_relative_transform():
    """FixedJoint2D.relative_transform reproduces the fixed SE(2) parameters."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = FixedJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 2.0, 0.5]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )
    t = joint.relative_transform(_empty_params())
    np.testing.assert_allclose(t.t, [1.0, 2.0])
    assert t.theta() == pytest.approx(0.5)


def test_fixed_joint_3d_relative_transform():
    """FixedJoint3D.relative_transform builds an SE(3) from translation + quaternion."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    # 90-degree rotation about z: qx=0, qy=0, qz=sin(pi/4), qw=cos(pi/4)
    s = np.sin(np.pi / 4)
    joint = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 2.0, 3.0, 0.0, 0.0, s, s]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    t = joint.relative_transform(_empty_params())
    np.testing.assert_allclose(t.t, [1.0, 2.0, 3.0])
    expected_r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(t.R, expected_r, atol=1e-12)


def test_revolute_joint_2d_relative_transform():
    """RevoluteJoint2D.relative_transform places body2 at origin, rotated by angle."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.5, -0.25]),
            names=RevoluteJoint2D.fixed_parameter_names(),
        ),
    )
    params = ConstraintParameters(values=np.array([np.pi / 2]), names=("angle",))
    t = joint.relative_transform(params)
    np.testing.assert_allclose(t.t, [0.5, -0.25])
    assert t.theta() == pytest.approx(np.pi / 2)


def test_revolute_joint_3d_relative_transform():
    """RevoluteJoint3D.relative_transform translates to origin then rotates about
    axis."""
    a, b = _make_body_3d("a"), _make_body_3d("b")
    joint = RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0]),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )
    params = ConstraintParameters(values=np.array([np.pi / 2]), names=("angle",))
    t = joint.relative_transform(params)
    np.testing.assert_allclose(t.t, [1.0, 2.0, 3.0])
    expected_r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(t.R, expected_r, atol=1e-12)


def test_constraint_function_zero_when_satisfied_2d():
    """RevoluteJoint2D.constraint_function returns ~0 when body poses satisfy it."""
    angle = 0.7
    origin = (0.5, -0.25)
    a = _make_body_2d("a")
    # Place body2 exactly where the joint dictates: body1.pose * relative_transform.
    expected_b_pose = a.pose * SE2(origin[0], origin[1], angle)
    b = Body[SE2](
        name="b",
        pose=expected_b_pose,
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )
    joint = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array(origin), names=RevoluteJoint2D.fixed_parameter_names()
        ),
    )
    residual = joint.constraint_function(
        ConstraintParameters(values=np.array([angle]), names=("angle",)),
        BodyPoses({a: a.pose, b: b.pose}),
    )
    assert residual.shape == (3,)
    np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)


def test_constraint_function_zero_when_satisfied_3d():
    """RevoluteJoint3D.constraint_function returns ~0 when body poses satisfy it."""
    angle = 0.4
    origin = (1.0, 2.0, 3.0)
    axis = (0.0, 0.0, 1.0)
    a = _make_body_3d("a")
    expected_relative = SE3.Trans(list(origin)) * SE3.AngVec(angle, list(axis))
    b = Body[SE3](
        name="b",
        pose=a.pose * expected_relative,
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )
    joint = RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array(axis + origin),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )
    residual = joint.constraint_function(
        ConstraintParameters(values=np.array([angle]), names=("angle",)),
        BodyPoses({a: a.pose, b: b.pose}),
    )
    assert residual.shape == (6,)
    np.testing.assert_allclose(residual, np.zeros(6), atol=1e-12)


def test_constraint_function_nonzero_when_violated():
    """When the body poses don't satisfy the joint, the residual is nonzero."""
    a, b = _make_body_2d("a"), _make_body_2d("b")
    joint = RevoluteJoint2D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0]),
            names=RevoluteJoint2D.fixed_parameter_names(),
        ),
    )
    # Bodies both at identity, but the joint's angle is nonzero -> violated.
    residual = joint.constraint_function(
        ConstraintParameters(values=np.array([0.5]), names=("angle",)),
        BodyPoses({a: a.pose, b: b.pose}),
    )
    assert np.linalg.norm(residual) > 1e-6


def test_fixed_joint_constraint_function_zero_when_satisfied():
    """FixedJoint3D residual is ~0 when body2 is exactly the fixed transform of
    body1."""
    a = _make_body_3d("a")
    transform = SE3.Trans([1.0, 2.0, 3.0])
    b = Body[SE3](
        name="b",
        pose=a.pose * transform,
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )
    joint = FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )
    residual = joint.constraint_function(
        _empty_params(), BodyPoses({a: a.pose, b: b.pose})
    )
    np.testing.assert_allclose(residual, np.zeros(6), atol=1e-12)
