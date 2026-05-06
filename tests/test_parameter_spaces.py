"""Tests for parameter_spaces module."""

import math

import numpy as np
import pytest

from comb.constraints import RevoluteJoint3D
from comb.parameter_spaces import BoundedReal, Circle, ParameterSpace, Real


def test_real_is_just_addition():
    """Real.retract is +, difference is -, contains is everything."""
    space = Real()
    assert space.retract(2.0, 0.5) == 2.5
    assert space.difference(3.0, 1.0) == 2.0
    assert space.contains(1e9)
    assert space.contains(-1e9)
    assert space.preferred_range((-7.0, 7.0)) == (-7.0, 7.0)


def test_circle_wraps_on_retract():
    """Circle.retract wraps step results back into [-π, π]."""
    space = Circle()
    # Stepping past +π should wrap to negative side.
    assert space.retract(math.pi - 0.1, 0.2) == pytest.approx(-math.pi + 0.1)
    # Stepping past -π should wrap to positive side.
    assert space.retract(-math.pi + 0.1, -0.2) == pytest.approx(math.pi - 0.1)
    # Full revolution returns to start (modulo float).
    assert space.retract(0.5, 2 * math.pi) == pytest.approx(0.5)


def test_circle_difference_is_shortest_signed_angle():
    """Circle.difference picks the short way around, even across the seam."""
    space = Circle()
    # From near -π to near +π → short way is just under -2*0.1 (going forward
    # through -π); this is the wraparound case.
    diff = space.difference(math.pi - 0.1, -math.pi + 0.1)
    assert diff == pytest.approx(-0.2, abs=1e-9)
    # Symmetric direction.
    diff = space.difference(-math.pi + 0.1, math.pi - 0.1)
    assert diff == pytest.approx(0.2, abs=1e-9)
    # Within range, plain subtraction.
    assert space.difference(0.5, 0.2) == pytest.approx(0.3)


def test_circle_preferred_range_is_pi():
    """Circle reports its natural [-π, π] range regardless of caller default."""
    assert Circle().preferred_range((-1.0, 1.0)) == (-math.pi, math.pi)


def test_circle_contains_canonical_range():
    """Circle.contains accepts [-π, π] (both endpoints) and rejects outside."""
    space = Circle()
    assert space.contains(0.0)
    assert space.contains(math.pi)
    assert space.contains(-math.pi)
    assert not space.contains(math.pi + 0.01)
    assert not space.contains(-math.pi - 0.01)


def test_bounded_real_clamps_on_retract():
    """BoundedReal.retract clamps the step result to [lower, upper]."""
    space = BoundedReal(lower=-1.0, upper=1.0)
    assert space.retract(0.5, 1.0) == 1.0  # clamped at upper
    assert space.retract(-0.5, -1.0) == -1.0  # clamped at lower
    assert space.retract(0.0, 0.3) == pytest.approx(0.3)


def test_bounded_real_contains_is_strict():
    """BoundedReal.contains rejects values outside the closed interval."""
    space = BoundedReal(lower=0.0, upper=1.0)
    assert space.contains(0.0)
    assert space.contains(1.0)
    assert space.contains(0.5)
    assert not space.contains(-0.01)
    assert not space.contains(1.01)


def test_bounded_real_preferred_range():
    """BoundedReal reports its own [lower, upper] regardless of caller default."""
    assert BoundedReal(-2.0, 3.0).preferred_range((-100.0, 100.0)) == (-2.0, 3.0)


def test_bounded_real_rejects_inverted_bounds():
    """BoundedReal raises if lower > upper."""
    with pytest.raises(ValueError, match="must be"):
        BoundedReal(lower=1.0, upper=0.0)


def test_parameter_space_is_abstract():
    """The base ParameterSpace cannot be instantiated."""
    with pytest.raises(TypeError):
        ParameterSpace()  # type: ignore[abstract]  # pylint: disable=abstract-class-instantiated


def test_retract_then_difference_roundtrips_real():
    """For Real, difference recovers the tangent that retract would apply."""
    space = Real()
    point, tangent = 1.0, 0.7
    new_point = space.retract(point, tangent)
    assert space.difference(new_point, point) == pytest.approx(tangent)


def test_retract_then_difference_roundtrips_circle_within_range():
    """For Circle (no wraparound), difference recovers the tangent."""
    space = Circle()
    point, tangent = 0.5, 0.3
    new_point = space.retract(point, tangent)
    assert space.difference(new_point, point) == pytest.approx(tangent)


def test_revolute_joint_3d_default_is_circle():
    """RevoluteJoint3D's default parameter_spaces is (Circle(),)."""
    spaces = RevoluteJoint3D.default_parameter_spaces()
    assert spaces == (Circle(),)
    assert isinstance(spaces[0], Circle)


def test_circle_wraps_large_positive_steps():
    """Many small +deltas across the seam stay canonical."""
    space = Circle()
    point = 0.0
    for _ in range(1000):
        point = space.retract(point, 0.05)
    # After 50 rad of stepping (~7.96 revolutions), point should still be
    # in [-π, π].
    assert -math.pi <= point <= math.pi
    np.testing.assert_allclose(point, ((50.0 + math.pi) % (2 * math.pi)) - math.pi)
