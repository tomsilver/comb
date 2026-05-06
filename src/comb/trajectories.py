"""Continuous-time trajectories generic in the value type.

A ``Trajectory[T]`` is a function from a time in ``[0, duration]`` to a value
of type ``T``, packaged with that finite duration. The same machinery works
for any ``T`` — ``np.ndarray``, ``SE2`` / ``SE3`` poses, a ``ConstraintConfiguration``,
a ``BodyPoses``, etc. — by supplying the appropriate interpolator.

Construction primitives:

* :func:`constant` — hold a single value for some duration.
* :func:`linear_segment` — interpolate between two endpoints via a pluggable
  ``interpolate(start, end, s)`` callable. Common interpolators are provided
  for ``np.ndarray`` (:func:`interpolate_array`), ``SE2``
  (:func:`interpolate_se2`), and ``SE3`` (:func:`interpolate_se3`).
* :func:`piecewise_linear` — convenience wrapper that strings together linear
  segments through a sequence of waypoints.

Other interpolation strategies (splines, minimum-jerk, ...) drop in the same
way: a constructor that builds the appropriate ``fn`` and returns a
``Trajectory[T]``. The base type stays the same, and operations on it
(``__call__``, :meth:`Trajectory.sub`, :meth:`Trajectory.enumerate`,
:func:`concatenate`) work uniformly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from spatialmath import SE2, SE3

T = TypeVar("T")


@dataclass(frozen=True)
class Trajectory(Generic[T]):
    """A continuous-time mapping from ``[0, duration]`` to a value of type ``T``.

    ``fn`` is queried with a clamped time, so callers can pass values slightly
    outside the range (e.g. due to float drift) without raising.
    """

    fn: Callable[[float], T]
    duration: float

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

    def __call__(self, time: float) -> T:
        return self.fn(max(0.0, min(time, self.duration)))

    def sub(self, start: float, end: float) -> Trajectory[T]:
        """Slice ``[start, end]``; the new trajectory is re-indexed to start at 0."""
        if start < 0 or end > self.duration or start > end:
            raise ValueError(
                f"sub({start}, {end}) is outside [0, {self.duration}] or inverted"
            )
        offset = start
        parent_fn = self.fn
        return Trajectory(lambda t: parent_fn(offset + t), end - start)

    def enumerate(
        self, dt: float, *, include_endpoint: bool = True
    ) -> Iterator[tuple[float, T]]:
        """Yield ``(time, value)`` pairs at intervals of ``dt`` over ``[0, duration]``.

        Times are computed as ``i * dt`` (not accumulated) to avoid drift on
        long trajectories. If ``include_endpoint`` is True (default) and the
        last grid time falls strictly before ``duration``, also emits
        ``(duration, value)`` so the endpoint is never silently skipped.
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        n = int(math.floor(self.duration / dt))
        last = -math.inf
        for i in range(n + 1):
            t = i * dt
            last = t
            yield t, self(t)
        if include_endpoint and last < self.duration:
            yield self.duration, self(self.duration)


def constant(value: T, duration: float) -> Trajectory[T]:
    """A trajectory that holds ``value`` for ``duration`` seconds."""
    return Trajectory(lambda _t: value, duration)


def linear_segment(
    start: T,
    end: T,
    duration: float,
    interpolate: Callable[[T, T, float], T],
) -> Trajectory[T]:
    """Interpolate from ``start`` to ``end`` over ``duration`` via ``interpolate``.

    ``interpolate(start, end, s)`` is called with ``s`` in ``[0, 1]``.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    def fn(t: float) -> T:
        return interpolate(start, end, t / duration)

    return Trajectory(fn, duration)


def concatenate(trajectories: Sequence[Trajectory[T]]) -> Trajectory[T]:
    """Stitch ``trajectories`` end-to-end; total duration is the sum of parts.

    A query at time ``t`` is dispatched to whichever segment covers ``t``
    (linear scan; fine for typical segment counts).
    """
    if not trajectories:
        raise ValueError("concatenate() requires at least one trajectory")
    if len(trajectories) == 1:
        return trajectories[0]
    segments = tuple(trajectories)
    starts: list[float] = []
    cum = 0.0
    for traj in segments:
        starts.append(cum)
        cum += traj.duration
    total = cum
    seg_starts = tuple(starts)

    def fn(time: float) -> T:
        for i, seg in enumerate(segments):
            if time <= seg_starts[i] + seg.duration:
                return seg(time - seg_starts[i])
        last = segments[-1]
        return last(last.duration)

    return Trajectory(fn, total)


def piecewise_linear(
    points: Sequence[T],
    dt: float,
    interpolate: Callable[[T, T, float], T],
) -> Trajectory[T]:
    """A piecewise-linear trajectory through ``points``, each leg taking ``dt``."""
    if len(points) < 2:
        raise ValueError(
            f"piecewise_linear() requires at least 2 points, got {len(points)}"
        )
    segments = [
        linear_segment(points[i], points[i + 1], dt, interpolate)
        for i in range(len(points) - 1)
    ]
    return concatenate(segments)


def interpolate_array(start: np.ndarray, end: np.ndarray, s: float) -> np.ndarray:
    """Linear interpolation between two numpy arrays."""
    return start + s * (end - start)


def interpolate_se2(start: SE2, end: SE2, s: float) -> SE2:
    """SE(2) interpolation via spatialmath ``SE2.interp`` (shortest twist path)."""
    return start.interp(end, s)


def interpolate_se3(start: SE3, end: SE3, s: float) -> SE3:
    """SE(3) interpolation via spatialmath ``SE3.interp`` (shortest twist path)."""
    return start.interp(end, s)
