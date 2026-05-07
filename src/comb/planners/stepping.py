"""A planner that searches sequences of modes via BFS, stepping within each.

``SteppingPlanner.plan`` performs a breadth-first search over modes:

1. From the system's current mode and state, try planning directly to a
   state satisfying ``final_constraints`` (via :func:`find_satisfying_state`
   for the goal, then solver-bounded stepping toward it).
2. If that fails, for each transition in ``system.transitions``, try planning
   to a state where the transition's trigger holds. If reachable, apply the
   transition and queue the new mode.
3. Repeat until the goal is reached or ``max_modes`` is exceeded.

Within a mode, stepping calls :func:`solve` in a loop with the same
``interval`` bound on per-checkpoint body twist distance, so adjacent
checkpoints stay close on the constraint manifold and linear interpolation
between them stays near-valid.

The flat list of solver checkpoints across all visited modes is returned as
a :class:`Plan`: every mode contributes its entry state and every subsequent
stepping checkpoint. Two adjacent checkpoints across a transition share
body poses but differ in configuration (the new mode's constraint set). The
:class:`TransitionEvent` records the time of the post-transition state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
from spatialmath import SE2, SE3, Twist2, Twist3

from comb.bodies import BodyPoses, PoseT
from comb.constraints import Constraint, ConstraintConfiguration
from comb.mode import Mode, ModeState, interpolate_mode_state
from comb.planners import Plan, Planner, PlanningError, TransitionEvent
from comb.solver import UnsatisfiableConstraints, find_satisfying_state, solve
from comb.system import System
from comb.trajectories import concatenate, constant, linear_segment
from comb.transitions import ConstraintTransition


class _WithinModeFailure(PlanningError):
    """Internal: a within-mode planning attempt failed."""


@dataclass(frozen=True)
class SteppingPlanner(Planner):
    """BFS-over-modes planner with solver-bounded stepping within each mode.

    ``interval`` bounds the max twist-norm distance any body may move between
    adjacent checkpoints; ``max_substeps`` and ``max_modes`` are budgets per
    mode and across the BFS. ``convergence_tolerance`` and ``min_step_scale``
    are inner-loop knobs that rarely need tuning.
    """

    interval: float
    convergence_tolerance: float = 1e-6
    max_substeps: int = 1000
    max_modes: int = 100
    min_step_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")

    def plan(
        self,
        system: System[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
        horizon: float,
    ) -> Plan[PoseT]:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        finals = list(final_constraints)

        initial_mode = _internal_copy(system.mode)
        initial_state = initial_mode.snapshot()
        queue: deque[_BfsNode[PoseT]] = deque(
            [_BfsNode(initial_mode, initial_state, [], [])]
        )

        for _ in range(self.max_modes):
            if not queue:
                break
            node = queue.popleft()

            # 1. Can we reach the goal from this mode?
            try:
                tail = self._states_within_mode(node.mode, node.entry_state, finals)
                full_states = node.states_before_entry + tail
                return self._build_plan(full_states, node.events, horizon)
            except _WithinModeFailure:
                pass

            # 2. Otherwise try each transition that's reachable from here.
            for transition in system.transitions:
                try:
                    approach = self._states_within_mode(
                        node.mode, node.entry_state, [transition.trigger]
                    )
                except _WithinModeFailure:
                    continue
                try:
                    next_mode = transition.apply(node.mode, approach[-1])
                except ValueError:
                    continue
                # ``approach`` starts at this node's entry state; concatenating
                # it onto ``states_before_entry`` makes the child's
                # ``states_before_entry`` end at the trigger state (in the old
                # mode). The child's own entry_state will sit immediately
                # after, marking the post-transition slot in the flat list.
                child_before = node.states_before_entry + approach
                event_idx = len(child_before)
                queue.append(
                    _BfsNode(
                        mode=next_mode,
                        entry_state=next_mode.snapshot(),
                        states_before_entry=child_before,
                        events=node.events + [(event_idx, transition)],
                    )
                )

        raise PlanningError("SteppingPlanner found no plan reaching final_constraints")

    def _states_within_mode(
        self,
        mode: Mode[PoseT],
        start_state: ModeState[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
    ) -> list[ModeState[PoseT]]:
        """Solver checkpoints from ``start_state`` to a state satisfying constraints.

        Raises :class:`_WithinModeFailure` if the goal is infeasible from
        ``start_state``, ``max_substeps`` is exceeded, or stepping can't
        satisfy ``interval`` even with ``min_step_scale``.
        """
        work_mode = _internal_copy(mode)
        work_mode.set_state(start_state)
        try:
            goal = find_satisfying_state(work_mode, final_constraints)
        except UnsatisfiableConstraints as e:
            raise _WithinModeFailure(str(e)) from e

        states: list[ModeState[PoseT]] = [work_mode.snapshot()]
        while not _at_target(
            states[-1].configuration, goal.configuration, self.convergence_tolerance
        ):
            if len(states) - 1 >= self.max_substeps:
                raise _WithinModeFailure(
                    f"max_substeps={self.max_substeps} exceeded within mode"
                )
            next_state = self._take_step(work_mode, states[-1], goal)
            work_mode.set_state(next_state)
            states.append(next_state)
        return states

    def _take_step(
        self,
        work_mode: Mode[PoseT],
        current: ModeState[PoseT],
        goal: ModeState[PoseT],
    ) -> ModeState[PoseT]:
        """One scaled step toward ``goal``, halving until it satisfies ``interval``."""
        delta = _delta_toward(current.configuration, goal.configuration)
        scale = 1.0
        while True:
            new_state = solve(work_mode, delta={c: scale * d for c, d in delta.items()})
            distance = _max_pose_distance(current.body_poses, new_state.body_poses)
            if distance <= self.interval:
                return new_state
            if scale < self.min_step_scale:
                raise _WithinModeFailure(
                    f"Cannot reduce step scale below {self.min_step_scale} "
                    f"while keeping max pose distance ({distance:g}) ≤ "
                    f"interval ({self.interval:g})"
                )
            scale /= 2

    def _build_plan(
        self,
        states: list[ModeState[PoseT]],
        event_records: list[tuple[int, ConstraintTransition[PoseT]]],
        horizon: float,
    ) -> Plan[PoseT]:
        n_segments = len(states) - 1
        if n_segments == 0:
            sample_times: tuple[float, ...] = (0.0,)
            trajectory = constant(states[0], horizon)
        else:
            sub_dt = horizon / n_segments
            sample_times = tuple(i * sub_dt for i in range(len(states)))
            trajectory = concatenate(
                [
                    linear_segment(
                        states[i],
                        states[i + 1],
                        duration=sub_dt,
                        interpolate=interpolate_mode_state,
                    )
                    for i in range(n_segments)
                ]
            )
        events = tuple(
            TransitionEvent(time=sample_times[idx], transition=t)
            for idx, t in event_records
        )
        return Plan(trajectory=trajectory, events=events, sample_times=sample_times)


@dataclass(frozen=True)
class _BfsNode(Generic[PoseT]):
    """One entry in the BFS queue.

    ``states_before_entry`` is the flat list of solver checkpoints that
    precede this node's ``entry_state`` in the eventual plan. ``events`` is
    the list of ``(state_index, transition)`` records for transitions fired
    on the path leading to this node, where ``state_index`` is the position
    of the post-transition state (this node's entry, for the most recent
    event) in the final flat state list.
    """

    mode: Mode[PoseT]
    entry_state: ModeState[PoseT]
    states_before_entry: list[ModeState[PoseT]]
    events: list[tuple[int, ConstraintTransition[PoseT]]]


def _internal_copy(mode: Mode[PoseT]) -> Mode[PoseT]:
    """A Mode whose mutable state is independent of ``mode``'s."""
    state = mode.snapshot()
    return Mode(
        bodies=list(mode.bodies),
        constraints=list(mode.constraints),
        configuration=state.configuration,
        body_poses=state.body_poses,
        anchored_bodies=list(mode.anchored_bodies),
    )


def _delta_toward(
    current: ConstraintConfiguration, target: ConstraintConfiguration
) -> dict[Constraint, np.ndarray]:
    """Per-constraint geodesic tangent that retracts ``current`` toward ``target``."""
    return {
        c: np.array(
            [
                space.difference(
                    float(target[c].values[i]), float(current[c].values[i])
                )
                for i, space in enumerate(c.parameter_spaces)
            ]
        )
        for c in target
        if c in current
    }


def _at_target(
    current: ConstraintConfiguration,
    target: ConstraintConfiguration,
    tolerance: float,
) -> bool:
    """Whether all target parameters are within tolerance of current (geodesic)."""
    return all(
        abs(space.difference(float(target[c].values[i]), float(current[c].values[i])))
        <= tolerance
        for c in target
        if c in current
        for i, space in enumerate(c.parameter_spaces)
    )


def _max_pose_distance(a: BodyPoses[PoseT], b: BodyPoses[PoseT]) -> float:
    """Largest twist-norm distance between corresponding poses in ``a`` and ``b``."""
    return max((_pose_distance(a[body], b[body]) for body in a), default=0.0)


def _pose_distance(a: Any, b: Any) -> float:
    """Twist-norm distance between two poses (SE2/SE3) or L2 (ndarray)."""
    if isinstance(a, SE2):
        return float(np.linalg.norm(Twist2(a.inv() * b).A))
    if isinstance(a, SE3):
        return float(np.linalg.norm(Twist3(a.inv() * b).A))
    if isinstance(a, np.ndarray):
        return float(np.linalg.norm(b - a))
    raise TypeError(f"Cannot compute distance for pose type {type(a).__name__}")
