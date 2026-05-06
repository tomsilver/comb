"""Tests for trajectories module."""

import numpy as np
import pytest
from spatialmath import SE2, SE3

from comb.trajectories import (
    Trajectory,
    concatenate,
    constant,
    interpolate_array,
    interpolate_se2,
    interpolate_se3,
    linear_segment,
    piecewise_linear,
)


def test_trajectory_call_clamps_time():
    """``__call__`` clamps to ``[0, duration]`` so float drift is forgiven."""
    traj: Trajectory[float] = Trajectory(lambda t: t, 1.0)
    assert traj(-0.1) == 0.0
    assert traj(0.5) == 0.5
    assert traj(1.5) == 1.0


def test_trajectory_negative_duration_rejected():
    """Negative durations are not meaningful and should raise."""
    with pytest.raises(ValueError, match="duration must be non-negative"):
        Trajectory(lambda _t: 0.0, -1.0)


def test_constant_returns_same_value_everywhere():
    """``constant`` returns its value at every time in ``[0, duration]``."""
    traj = constant(np.array([1.0, 2.0, 3.0]), duration=4.0)
    assert traj.duration == 4.0
    np.testing.assert_array_equal(traj(0.0), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(traj(2.0), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(traj(4.0), [1.0, 2.0, 3.0])


def test_linear_segment_array_interpolates():
    """A linear segment of arrays hits both endpoints and the midpoint."""
    start = np.array([0.0, 0.0])
    end = np.array([10.0, -2.0])
    traj = linear_segment(start, end, duration=2.0, interpolate=interpolate_array)
    np.testing.assert_array_equal(traj(0.0), start)
    np.testing.assert_array_equal(traj(2.0), end)
    np.testing.assert_allclose(traj(1.0), [5.0, -1.0])


def test_linear_segment_zero_duration_rejected():
    """A linear segment needs positive duration to define a fraction."""
    with pytest.raises(ValueError, match="duration must be positive"):
        linear_segment(0.0, 1.0, duration=0.0, interpolate=lambda a, b, s: a)


def test_linear_segment_se3_uses_shortest_path():
    """SE(3) segment endpoints match start and end exactly; midpoint halfway."""
    start = SE3()
    end = SE3.Rt(SE3.Rx(1.0).R, [2.0, 4.0, 6.0])
    traj = linear_segment(start, end, duration=1.0, interpolate=interpolate_se3)
    np.testing.assert_allclose(traj(0.0).A, start.A, atol=1e-12)
    np.testing.assert_allclose(traj(1.0).A, end.A, atol=1e-12)
    np.testing.assert_allclose(traj(0.5).t, [1.0, 2.0, 3.0], atol=1e-12)


def test_linear_segment_se2_midpoint():
    """SE(2) segment midpoint translates and rotates halfway."""
    start = SE2(0.0, 0.0, 0.0)
    end = SE2(2.0, 4.0, 1.0)
    traj = linear_segment(start, end, duration=2.0, interpolate=interpolate_se2)
    mid = traj(1.0)
    np.testing.assert_allclose(mid.t, [1.0, 2.0], atol=1e-12)
    assert mid.theta() == pytest.approx(0.5)


def test_concatenate_dispatches_to_correct_segment():
    """A query lands on the segment whose interval contains the time."""
    a = constant(1.0, duration=1.0)
    b = constant(2.0, duration=2.0)
    c = constant(3.0, duration=0.5)
    traj = concatenate([a, b, c])
    assert traj.duration == pytest.approx(3.5)
    assert traj(0.5) == 1.0
    assert traj(1.5) == 2.0
    assert traj(3.25) == 3.0
    # Boundary times go to the earlier segment (``time <= start + duration``).
    assert traj(1.0) == 1.0
    assert traj(3.0) == 2.0


def test_concatenate_empty_rejected():
    """Empty concatenation has no defined duration and should raise."""
    with pytest.raises(ValueError, match="at least one"):
        concatenate([])


def test_concatenate_single_returns_input():
    """A single-element concat is the input trajectory itself."""
    traj = constant(7.0, 1.0)
    assert concatenate([traj]) is traj


def test_concatenate_of_concatenate_is_correct():
    """Nested concats dispatch recursively without flattening."""
    inner = concatenate([constant(1.0, 1.0), constant(2.0, 1.0)])
    outer = concatenate([inner, constant(3.0, 1.0)])
    assert outer.duration == pytest.approx(3.0)
    assert outer(0.5) == 1.0
    assert outer(1.5) == 2.0
    assert outer(2.5) == 3.0


def test_sub_reindexes_to_zero():
    """``sub(a, b)`` returns a trajectory whose t=0 maps to the parent's t=a."""
    traj = linear_segment(
        np.array([0.0]), np.array([10.0]), duration=10.0, interpolate=interpolate_array
    )
    sliced = traj.sub(2.0, 5.0)
    assert sliced.duration == pytest.approx(3.0)
    np.testing.assert_allclose(sliced(0.0), [2.0])
    np.testing.assert_allclose(sliced(3.0), [5.0])
    np.testing.assert_allclose(sliced(1.5), [3.5])


def test_sub_rejects_invalid_ranges():
    """Out-of-range or inverted slices raise."""
    traj = constant(0.0, 1.0)
    with pytest.raises(ValueError):
        traj.sub(-0.1, 0.5)
    with pytest.raises(ValueError):
        traj.sub(0.5, 1.5)
    with pytest.raises(ValueError):
        traj.sub(0.6, 0.4)


def test_enumerate_yields_grid_times_and_endpoint():
    """Default enumerate yields the regular grid plus the endpoint when off-grid."""
    traj = linear_segment(
        0.0, 10.0, duration=1.0, interpolate=lambda a, b, s: a + s * (b - a)
    )
    points = list(traj.enumerate(0.3))
    times = [t for t, _ in points]
    values = [v for _, v in points]
    assert times == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])
    assert values == pytest.approx([0.0, 3.0, 6.0, 9.0, 10.0])


def test_enumerate_skips_endpoint_when_on_grid():
    """When ``duration`` is a multiple of ``dt``, the endpoint is already in the
    grid."""
    traj = constant(1.0, duration=1.0)
    times = [t for t, _ in traj.enumerate(0.25)]
    assert times == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_enumerate_can_omit_endpoint():
    """``include_endpoint=False`` returns only the regular grid."""
    traj = linear_segment(
        0.0, 10.0, duration=1.0, interpolate=lambda a, b, s: a + s * (b - a)
    )
    times = [t for t, _ in traj.enumerate(0.3, include_endpoint=False)]
    assert times == pytest.approx([0.0, 0.3, 0.6, 0.9])


def test_enumerate_rejects_non_positive_dt():
    """``dt`` must be > 0 to make progress."""
    traj = constant(0.0, 1.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        list(traj.enumerate(0.0))
    with pytest.raises(ValueError, match="dt must be positive"):
        list(traj.enumerate(-0.1))


def test_piecewise_linear_through_waypoints():
    """``piecewise_linear`` interpolates linearly between successive waypoints."""
    points = [np.array([0.0]), np.array([1.0]), np.array([0.0])]
    traj = piecewise_linear(points, dt=1.0, interpolate=interpolate_array)
    assert traj.duration == pytest.approx(2.0)
    np.testing.assert_allclose(traj(0.0), [0.0])
    np.testing.assert_allclose(traj(0.5), [0.5])
    np.testing.assert_allclose(traj(1.0), [1.0])
    np.testing.assert_allclose(traj(1.5), [0.5])
    np.testing.assert_allclose(traj(2.0), [0.0])


def test_piecewise_linear_requires_two_points():
    """A trajectory needs at least one segment, i.e. two waypoints."""
    with pytest.raises(ValueError, match="at least 2 points"):
        piecewise_linear([np.array([0.0])], dt=1.0, interpolate=interpolate_array)


def test_enumerate_long_trajectory_avoids_drift():
    """``i * dt`` is used instead of accumulation so float error doesn't compound."""
    traj = constant(0.0, duration=1000.0)
    times = [t for t, _ in traj.enumerate(0.1)]
    # 1000/0.1 = 10000 grid steps -> 10001 samples; endpoint already on grid.
    assert len(times) == 10001
    assert times[-1] == pytest.approx(1000.0)
    # Drift check: the final gap should match dt to high precision.
    assert times[-1] - times[-2] == pytest.approx(0.1, abs=1e-12)


def test_concatenate_with_se3_segments():
    """End-to-end: concat two SE(3) linear segments and query through the join."""
    a_start = SE3()
    a_end = SE3.Trans(1.0, 0.0, 0.0)
    b_end = SE3.Trans(1.0, 1.0, 0.0)
    seg_a = linear_segment(a_start, a_end, duration=1.0, interpolate=interpolate_se3)
    seg_b = linear_segment(a_end, b_end, duration=1.0, interpolate=interpolate_se3)
    traj = concatenate([seg_a, seg_b])
    np.testing.assert_allclose(traj(0.0).t, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj(0.5).t, [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj(1.0).t, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj(1.5).t, [1.0, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj(2.0).t, [1.0, 1.0, 0.0], atol=1e-12)


def test_sub_then_enumerate():
    """Sub-trajectory enumeration walks the parent's slice."""
    traj = linear_segment(
        0.0, 10.0, duration=10.0, interpolate=lambda a, b, s: a + s * (b - a)
    )
    sliced = traj.sub(4.0, 7.0)
    samples = list(sliced.enumerate(1.0))
    assert [t for t, _ in samples] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert [v for _, v in samples] == pytest.approx([4.0, 5.0, 6.0, 7.0])


def test_enumerate_zero_duration_yields_single_sample():
    """A zero-duration trajectory has one sample at t=0."""
    traj = constant(42.0, duration=0.0)
    samples = list(traj.enumerate(0.1))
    assert samples == [(0.0, 42.0)]
