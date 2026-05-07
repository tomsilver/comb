"""Plan validator: replay a :class:`Plan` and check correctness.

A :class:`Plan` is internally consistent when:

* ``sample_times`` is monotonically non-decreasing;
* every transition event sits on a sample time, in time order;
* at the sample immediately before each event, the transition's trigger
  residual is below the transition's own ``tolerance``;
* at every checkpoint, every *active* constraint's residual norm is below
  ``tolerance`` — where the active set evolves as transitions fire (start
  with ``system.mode.constraints``; each event removes / adds via the
  transition's own ``remove`` / ``add`` callables);
* the final checkpoint satisfies every goal constraint.

Runtime drift *between* checkpoints (mid-segment, where the trajectory's
linear interpolation may leave the manifold) is not checked here — that's a
"smaller interval" tuning concern, not a plan-correctness concern.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from spatialmath import SE2, SE3, Twist2, Twist3

from comb.bodies import BodyPoses, PoseT
from comb.constraints import Constraint, ConstraintParameters
from comb.mode import Mode, ModeState
from comb.system import System

if TYPE_CHECKING:
    from comb.planners import Plan


class PlanValidationError(Exception):
    """Raised when a plan fails internal-consistency checks."""


def validate_plan(
    plan: Plan[PoseT],
    system: System[PoseT],
    *,
    goal: Iterable[Constraint[PoseT]] = (),
    tolerance: float = 1e-6,
    max_segment_twist: float | None = None,
) -> None:
    """Verify that ``plan`` is a valid execution of ``system``.

    Replays the plan: starts with ``system.mode.constraints`` as the active
    set, applies each transition event in turn (verifying its trigger fires
    at the prior checkpoint), and checks that every active constraint's
    residual stays within ``tolerance`` at every sample time. ``goal`` is
    a final-state set the last checkpoint must additionally satisfy.

    If ``max_segment_twist`` is given, every body's twist-norm distance
    between adjacent checkpoints must be at most that bound — typically
    sourced from a task's :class:`comb.spec.GranularitySpec`.

    Raises :class:`PlanValidationError` on the first failure.
    """
    times = plan.sample_times
    if not times:
        raise PlanValidationError("plan must have at least one sample time")
    for prev_t, next_t in zip(times, times[1:]):
        if next_t < prev_t:
            raise PlanValidationError(
                f"sample_times must be non-decreasing, got {prev_t:g} then {next_t:g}"
            )
    for prev_event, next_event in zip(plan.events, plan.events[1:]):
        if next_event.time < prev_event.time:
            raise PlanValidationError(
                f"events must be in time order, got t={prev_event.time:g} "
                f"then t={next_event.time:g}"
            )

    goal_constraints = list(goal)
    active_mode = _copy_mode(system.mode)
    events = list(plan.events)
    event_idx = 0

    for i, t in enumerate(times):
        # Apply any events that fire at this sample time. (Per A1 convention,
        # an event's time matches the time of the post-transition state.)
        while event_idx < len(events) and events[event_idx].time <= t:
            event = events[event_idx]
            if event.time != t:
                raise PlanValidationError(
                    f"event at t={event.time:g} doesn't match a sample time"
                )
            if i == 0:
                raise PlanValidationError(
                    f"event at t={t:g} has no preceding state to check the trigger"
                )
            prev_state = plan.trajectory(times[i - 1])
            trigger_residual = event.transition.trigger_residual(prev_state)
            if trigger_residual > event.transition.tolerance:
                raise PlanValidationError(
                    f"event at t={t:g}: trigger residual {trigger_residual:g} "
                    f"exceeds transition tolerance {event.transition.tolerance:g}"
                )
            try:
                active_mode = event.transition.apply(active_mode, prev_state)
            except ValueError as exc:
                raise PlanValidationError(
                    f"event at t={t:g}: transition.apply failed: {exc}"
                ) from exc
            event_idx += 1

        state = plan.trajectory(t)
        for constraint in active_mode.constraints:
            residual_norm = _residual_norm(constraint, state)
            if residual_norm > tolerance:
                raise PlanValidationError(
                    f"checkpoint t={t:g}: residual of "
                    f"{type(constraint).__name__}({constraint.body1.name}, "
                    f"{constraint.body2.name}) is {residual_norm:g} > "
                    f"tolerance {tolerance:g}"
                )

    if event_idx < len(events):
        leftover = events[event_idx]
        raise PlanValidationError(
            f"event at t={leftover.time:g} is past the last sample time "
            f"({times[-1]:g})"
        )

    if max_segment_twist is not None:
        for prev_t, next_t in zip(times, times[1:]):
            prev_state = plan.trajectory(prev_t)
            next_state = plan.trajectory(next_t)
            distance = _max_pose_distance(prev_state.body_poses, next_state.body_poses)
            if distance > max_segment_twist:
                raise PlanValidationError(
                    f"segment t=[{prev_t:g}, {next_t:g}]: max body twist "
                    f"{distance:g} > granularity {max_segment_twist:g}"
                )

    final_state = plan.trajectory(times[-1])
    for goal_constraint in goal_constraints:
        residual_norm = _residual_norm(goal_constraint, final_state)
        if residual_norm > tolerance:
            raise PlanValidationError(
                f"goal {type(goal_constraint).__name__}"
                f"({goal_constraint.body1.name}, "
                f"{goal_constraint.body2.name}) not satisfied at final "
                f"t={times[-1]:g}: residual {residual_norm:g} > "
                f"tolerance {tolerance:g}"
            )


def _copy_mode(mode: Mode[PoseT]) -> Mode[PoseT]:
    snapshot = mode.snapshot()
    return Mode(
        bodies=list(mode.bodies),
        constraints=list(mode.constraints),
        configuration=snapshot.configuration,
        body_poses=snapshot.body_poses,
        anchored_bodies=list(mode.anchored_bodies),
    )


def _max_pose_distance(prev: BodyPoses[PoseT], curr: BodyPoses[PoseT]) -> float:
    """Largest twist-norm distance between matching body poses in two snapshots."""
    return max(
        (_pose_distance(prev[body], curr[body]) for body in prev if body in curr),
        default=0.0,
    )


def _pose_distance(a: Any, b: Any) -> float:
    """Twist-norm distance between two SE(2) / SE(3) poses (or L2 for ndarrays)."""
    if isinstance(a, SE2):
        return float(np.linalg.norm(Twist2(a.inv() * b).A))
    if isinstance(a, SE3):
        return float(np.linalg.norm(Twist3(a.inv() * b).A))
    if isinstance(a, np.ndarray):
        return float(np.linalg.norm(b - a))
    raise TypeError(f"cannot compute distance for pose type {type(a).__name__}")


def _residual_norm(constraint: Constraint[PoseT], state: ModeState[PoseT]) -> float:
    if constraint.parameter_names():
        try:
            params = state.configuration[constraint]
        except KeyError as exc:
            raise PlanValidationError(
                f"state configuration is missing an entry for "
                f"{type(constraint).__name__}({constraint.body1.name}, "
                f"{constraint.body2.name})"
            ) from exc
    else:
        params = ConstraintParameters(np.array([]), ())
    residual = constraint.constraint_function(params, state.body_poses)
    return float(np.linalg.norm(residual))
