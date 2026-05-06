"""Tests for constraints module."""

import numpy as np
import pytest
from spatialmath import SE3

from comb.bodies import Body, Sphere
from comb.constraints import (
    Configuration,
    Constraint,
    ConstraintParameters,
    FixedConstraint,
    PlanarJoint,
    PrismaticJoint,
    RevoluteJoint,
)


def _make_body(name: str) -> Body:
    return Body(
        name=name,
        pose=SE3(),
        visual_geometry=Sphere(0.1),
        collision_geometry=Sphere(0.1),
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


def _make_revolute(a: Body, b: Body) -> RevoluteJoint:
    return RevoluteJoint(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            names=RevoluteJoint.fixed_parameter_names(),
        ),
    )


def test_revolute_joint_structure():
    """RevoluteJoint stores axis/origin as fixed parameters."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    assert joint.body1 is a
    assert joint.body2 is b
    assert joint.fixed_parameters["axis_z"] == 1.0
    assert joint.parameter_names() == ("angle",)


def test_prismatic_joint_structure():
    """PrismaticJoint stores axis/origin as fixed parameters."""
    a, b = _make_body("a"), _make_body("b")
    joint = PrismaticJoint(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            names=PrismaticJoint.fixed_parameter_names(),
        ),
    )
    assert joint.fixed_parameters["axis_x"] == 1.0
    assert joint.parameter_names() == ("offset",)


def test_planar_joint_structure():
    """PlanarJoint has no fixed parameters and (x, y, theta) as mutable params."""
    a, b = _make_body("a"), _make_body("b")
    joint = PlanarJoint(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
    )
    assert joint.parameter_names() == ("x", "y", "theta")


def test_fixed_constraint_structure():
    """FixedConstraint stores an SE(3) transform with no mutable parameters."""
    a, b = _make_body("a"), _make_body("b")
    c = FixedConstraint(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedConstraint.fixed_parameter_names(),
        ),
    )
    assert c.fixed_parameters["qw"] == 1.0
    assert not c.parameter_names()


def test_constraint_rejects_wrong_fixed_parameter_names():
    """A constraint subclass rejects fixed parameters with the wrong names."""
    a, b = _make_body("a"), _make_body("b")
    with pytest.raises(ValueError):
        RevoluteJoint(
            body1=a,
            body2=b,
            fixed_parameters=ConstraintParameters(
                values=np.array([1.0, 2.0]),
                names=("foo", "bar"),
            ),
        )


def test_constraint_is_abstract():
    """The base Constraint class cannot be instantiated."""
    a, b = _make_body("a"), _make_body("b")
    empty = ConstraintParameters(values=np.array([]), names=())
    with pytest.raises(TypeError):
        Constraint(  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated
            body1=a, body2=b, fixed_parameters=empty
        )


def test_configuration_basic_set_and_get():
    """Configuration stores and returns mutable parameters per constraint."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    config = Configuration()
    config[joint] = ConstraintParameters(values=np.array([0.5]), names=("angle",))
    assert config[joint]["angle"] == 0.5
    assert joint in config
    assert len(config) == 1


def test_configuration_update_in_place():
    """Reassigning parameters for the same constraint replaces the previous value."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    config = Configuration()
    config[joint] = ConstraintParameters(values=np.array([0.0]), names=("angle",))
    config[joint] = ConstraintParameters(values=np.array([np.pi / 2]), names=("angle",))
    assert config[joint]["angle"] == pytest.approx(np.pi / 2)
    assert len(config) == 1


def test_configuration_rejects_wrong_names():
    """Configuration rejects parameters whose names don't match the constraint."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    config = Configuration()
    with pytest.raises(ValueError):
        config[joint] = ConstraintParameters(
            values=np.array([0.0]), names=("not_angle",)
        )


def test_configuration_distinguishes_constraint_instances():
    """Two structurally identical constraints are kept as separate Configuration
    keys."""
    a, b = _make_body("a"), _make_body("b")
    joint1 = _make_revolute(a, b)
    joint2 = _make_revolute(a, b)
    config = Configuration()
    config[joint1] = ConstraintParameters(values=np.array([0.1]), names=("angle",))
    config[joint2] = ConstraintParameters(values=np.array([0.9]), names=("angle",))
    assert config[joint1]["angle"] == pytest.approx(0.1)
    assert config[joint2]["angle"] == pytest.approx(0.9)
    assert len(config) == 2


def test_configuration_zeros():
    """Configuration.zeros initializes every constraint's parameters to zero."""
    a, b = _make_body("a"), _make_body("b")
    revolute = _make_revolute(a, b)
    planar = PlanarJoint(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
    )
    config = Configuration.zeros([revolute, planar])
    np.testing.assert_array_equal(config[revolute].values, [0.0])
    np.testing.assert_array_equal(config[planar].values, [0.0, 0.0, 0.0])
    assert config[planar].names == ("x", "y", "theta")


def test_configuration_unknown_constraint_raises():
    """Looking up a constraint that was never set raises KeyError."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    config = Configuration()
    with pytest.raises(KeyError):
        _ = config[joint]
