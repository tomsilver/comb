"""Tests for mode module."""

import numpy as np
import pytest
from spatialmath import SE3

from comb.bodies import Body, Box
from comb.constraints import (
    Configuration,
    ConstraintParameters,
    FixedJoint3D,
    RevoluteJoint3D,
)
from comb.mode import Mode


def _make_body(name: str) -> Body[SE3]:
    return Body(
        name=name,
        pose=SE3(),
        visual_geometry=Box(0.1, 0.1, 0.1),
        collision_geometry=Box(0.1, 0.1, 0.1),
    )


def _make_revolute(a: Body[SE3], b: Body[SE3]) -> RevoluteJoint3D:
    return RevoluteJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            names=RevoluteJoint3D.fixed_parameter_names(),
        ),
    )


def _make_fixed(a: Body[SE3], b: Body[SE3]) -> FixedJoint3D:
    return FixedJoint3D(
        body1=a,
        body2=b,
        fixed_parameters=ConstraintParameters(
            values=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            names=FixedJoint3D.fixed_parameter_names(),
        ),
    )


def test_mode_basic_construction():
    """A Mode[SE3] holds bodies, constraints, and a configuration."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    config = Configuration.zeros([joint])
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[joint], configuration=config)
    assert mode.bodies == [a, b]
    assert mode.constraints == [joint]
    assert mode.configuration[joint]["angle"] == 0.0


def test_mode_default_empty_configuration():
    """Mode works without an explicit configuration when no constraint is mutable."""
    a, b = _make_body("a"), _make_body("b")
    fixed = _make_fixed(a, b)
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[fixed])
    assert len(mode.configuration) == 0


def test_mode_rejects_constraint_with_unknown_body():
    """A constraint whose bodies are not in the mode is rejected."""
    a, b, c = _make_body("a"), _make_body("b"), _make_body("c")
    joint = _make_revolute(a, b)
    with pytest.raises(ValueError, match="not in the mode"):
        Mode[SE3](
            bodies=[a, c],
            constraints=[joint],
            configuration=Configuration.zeros([joint]),
        )


def test_mode_requires_configuration_for_mutable_constraint():
    """A constraint with mutable parameters must have an entry in the configuration."""
    a, b = _make_body("a"), _make_body("b")
    joint = _make_revolute(a, b)
    with pytest.raises(ValueError, match="Configuration is missing"):
        Mode[SE3](bodies=[a, b], constraints=[joint])


def test_mode_validate_after_mutation():
    """Validate() re-checks invariants after the user mutates bodies or constraints."""
    a, b = _make_body("a"), _make_body("b")
    fixed = _make_fixed(a, b)
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[fixed])
    # Add a new constraint without updating the mode; validate should catch it.
    new_revolute = _make_revolute(a, b)
    mode.constraints.append(new_revolute)
    with pytest.raises(ValueError, match="Configuration is missing"):
        mode.validate()
    # Add the missing config entry; now validate passes.
    mode.configuration[new_revolute] = ConstraintParameters(
        values=np.array([0.0]), names=("angle",)
    )
    mode.validate()


def test_mode_rejects_anchor_not_in_bodies():
    """anchored_bodies must reference bodies that are in the mode."""
    a, b, c = _make_body("a"), _make_body("b"), _make_body("c")
    fixed = _make_fixed(a, b)
    with pytest.raises(ValueError, match="not in the mode"):
        Mode[SE3](bodies=[a, b], constraints=[fixed], anchored_bodies=[c])


def test_mode_accepts_multiple_anchors():
    """Multiple anchored bodies are allowed (e.g. both ends of a chain fixed)."""
    a, b = _make_body("a"), _make_body("b")
    fixed = _make_fixed(a, b)
    mode: Mode[SE3] = Mode(bodies=[a, b], constraints=[fixed], anchored_bodies=[a, b])
    assert mode.anchored_bodies == [a, b]


def test_mode_holds_multiple_constraints():
    """A Mode can hold many constraints sharing bodies."""
    a, b, c = _make_body("a"), _make_body("b"), _make_body("c")
    joint_ab = _make_revolute(a, b)
    joint_bc = _make_revolute(b, c)
    config = Configuration.zeros([joint_ab, joint_bc])
    mode: Mode[SE3] = Mode(
        bodies=[a, b, c], constraints=[joint_ab, joint_bc], configuration=config
    )
    assert len(mode.constraints) == 2
