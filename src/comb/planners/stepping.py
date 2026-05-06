"""A planner that searches sequences of modes via BFS, stepping within each.

``SteppingPlanner.plan`` performs a breadth-first search over modes:

1. From the system's current mode and state, try planning directly to a
   state satisfying ``final_constraints`` (via :func:`find_satisfying_state`
   for the goal, then solver-bounded stepping toward it).
2. If that fails, for each transition in ``system.transitions``, try planning
   to a state where the transition's trigger holds. If reachable, apply the
   transition (yielding a new mode), and add it to the BFS queue.
3. Repeat until the goal is reached or ``max_modes`` is exceeded.

Within each mode, stepping uses :func:`solve` in a loop with the same
``interval`` bound on per-checkpoint body twist distance, so adjacent
checkpoints stay close on the constraint manifold and linear interpolation
between them stays near-valid.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from spatialmath import SE2, SE3, Twist2, Twist3

from comb.bodies import BodyPoses, PoseT
from comb.constraints import Constraint, ConstraintConfiguration
from comb.mode import Mode, ModeState, interpolate_mode_state
from comb.planners import Planner
from comb.solver import find_satisfying_state, solve
from comb.system import System
from comb.trajectories import Trajectory, concatenate, constant, linear_segment


@dataclass(frozen=True)
class SteppingPlanner(Planner):
    """BFS-over-modes planner with solver-bounded stepping within each mode.

    Hyperparameters
    ---------------
    interval
        Maximum twist-norm distance any body may move between adjacent
        checkpoints inside a mode.
    convergence_tolerance
        Per-parameter tolerance for "at goal" within a mode.
    max_substeps
        Per-mode safety cap on solver checkpoints; raises if exceeded
        *while exploring* a mode (the BFS catches it and tries other paths).
    max_modes
        Total BFS budget across all modes explored.
    min_step_scale
        Smallest allowed step-scale fraction before stepping gives up on
        ``interval`` and raises.
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
    ) -> Trajectory[ModeState[PoseT]]:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        finals = list(final_constraints)

        initial_mode = _internal_copy(system.mode)
        initial_state = initial_mode.snapshot()
        queue: deque[_BfsNode] = deque(
            [
                _BfsNode(
                    mode=initial_mode,
                    entry_state=initial_state,
                    states_so_far=[initial_state],
                )
            ]
        )
        explored = 0

        while queue:
            if explored >= self.max_modes:
                raise RuntimeError(
                    f"SteppingPlanner BFS exceeded max_modes={self.max_modes} "
                    f"without finding a plan"
                )
            node = queue.popleft()
            explored += 1

            # 1. Can we reach the goal from this mode without further transitions?
            within = self._states_within_mode_or_none(
                node.mode, node.entry_state, finals
            )
            if within is not None:
                full_states = node.states_so_far + within[1:]
                return self._build_trajectory(full_states, horizon)

            # 2. Otherwise, try each available transition.
            for transition in system.transitions:
                to_trigger = self._states_within_mode_or_none(
                    node.mode, node.entry_state, [transition.trigger]
                )
                if to_trigger is None:
                    continue
                state_at_trigger = to_trigger[-1]
                try:
                    new_mode = transition.apply(node.mode, state_at_trigger)
                except ValueError:
                    continue
                queue.append(
                    _BfsNode(
                        mode=new_mode,
                        entry_state=new_mode.snapshot(),
                        states_so_far=node.states_so_far + to_trigger[1:],
                    )
                )

        raise RuntimeError("SteppingPlanner found no plan reaching final_constraints")

    def _states_within_mode_or_none(
        self,
        mode: Mode[PoseT],
        start_state: ModeState[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
    ) -> list[ModeState[PoseT]] | None:
        """Try planning within ``mode`` from ``start_state`` to satisfy the constraints.

        Returns the list of solver checkpoints on success, ``None`` if the
        goal is unreachable from this state or stepping fails to converge.
        """
        try:
            return self._states_within_mode(mode, start_state, final_constraints)
        except RuntimeError:
            return None

    def _states_within_mode(
        self,
        mode: Mode[PoseT],
        start_state: ModeState[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
    ) -> list[ModeState[PoseT]]:
        """Solver checkpoints from ``start_state`` to a state satisfying constraints."""
        work_mode = _internal_copy(mode)
        work_mode.set_state(start_state)

        goal_state = find_satisfying_state(work_mode, final_constraints)

        state = work_mode.snapshot()
        states: list[ModeState[PoseT]] = [state]

        while not _at_target(
            state.configuration,
            goal_state.configuration,
            self.convergence_tolerance,
        ):
            if len(states) - 1 >= self.max_substeps:
                raise RuntimeError(
                    f"max_substeps={self.max_substeps} exceeded within mode"
                )
            delta = _delta_toward(state.configuration, goal_state.configuration)
            scale = 1.0
            while True:
                scaled = {c: scale * d for c, d in delta.items()}
                new_state = solve(work_mode, delta=scaled)
                distance = _max_pose_distance(state.body_poses, new_state.body_poses)
                if distance <= self.interval:
                    break
                if scale < self.min_step_scale:
                    raise RuntimeError(
                        f"Cannot reduce step scale below {self.min_step_scale} "
                        f"while keeping max pose distance ({distance:g}) ≤ "
                        f"interval ({self.interval:g})"
                    )
                scale /= 2
            work_mode.set_state(new_state)
            states.append(new_state)
            state = new_state

        return states

    def _build_trajectory(
        self, states: list[ModeState[PoseT]], horizon: float
    ) -> Trajectory[ModeState[PoseT]]:
        n_segments = len(states) - 1
        if n_segments == 0:
            return constant(states[0], horizon)
        sub_dt = horizon / n_segments
        segments = [
            linear_segment(
                states[i],
                states[i + 1],
                duration=sub_dt,
                interpolate=interpolate_mode_state,
            )
            for i in range(n_segments)
        ]
        return concatenate(segments)


@dataclass
class _BfsNode:  # pylint: disable=too-few-public-methods
    """One entry in the BFS queue: a mode, its entry state, and the path so far."""

    mode: Mode
    entry_state: ModeState
    states_so_far: list[ModeState]


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
    delta: dict[Constraint, np.ndarray] = {}
    for constraint in target:
        if constraint not in current:
            continue
        cur_vals = current[constraint].values
        tgt_vals = target[constraint].values
        spaces = constraint.parameter_spaces
        delta[constraint] = np.array(
            [
                spaces[i].difference(float(tgt_vals[i]), float(cur_vals[i]))
                for i in range(len(spaces))
            ]
        )
    return delta


def _at_target(
    current: ConstraintConfiguration,
    target: ConstraintConfiguration,
    tolerance: float,
) -> bool:
    for constraint in target:
        if constraint not in current:
            continue
        cur_vals = current[constraint].values
        tgt_vals = target[constraint].values
        spaces = constraint.parameter_spaces
        for i, space in enumerate(spaces):
            d = space.difference(float(tgt_vals[i]), float(cur_vals[i]))
            if abs(d) > tolerance:
                return False
    return True


def _max_pose_distance(a: BodyPoses[PoseT], b: BodyPoses[PoseT]) -> float:
    return max((_pose_distance(a[body], b[body]) for body in a), default=0.0)


def _pose_distance(a: Any, b: Any) -> float:
    if isinstance(a, SE2):
        return float(np.linalg.norm(Twist2(a.inv() * b).A))
    if isinstance(a, SE3):
        return float(np.linalg.norm(Twist3(a.inv() * b).A))
    if isinstance(a, np.ndarray):
        return float(np.linalg.norm(b - a))
    raise TypeError(f"Cannot compute distance for pose type {type(a).__name__}")
