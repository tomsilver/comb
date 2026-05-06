"""A planner that walks toward a goal state via solver-bounded substeps.

``SteppingPlanner.plan`` first solves the augmented system
(``system.constraints + final_constraints``) via :func:`find_satisfying_state`
to find a goal state where all constraints are satisfied, then marches the
system from its current state toward that goal in many small substeps. Each
substep advances the joint parameters by some scaled fraction of the
remaining delta and calls :func:`solve` to update body poses; the scale is
halved as needed so no body's pose moves more than ``interval`` (twist-norm
distance) between adjacent substeps.

The result is a ``Trajectory[SystemState[PoseT]]`` whose checkpoints are
valid solver outputs and whose adjacent checkpoints are bounded in pose
distance, so linear interpolation between them stays close to the constraint
manifold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from spatialmath import SE2, SE3, Twist2, Twist3

from comb.bodies import BodyPoses, PoseT
from comb.constraints import Configuration, Constraint
from comb.planners import Planner
from comb.solver import find_satisfying_state, solve
from comb.system import System, SystemState, interpolate_system_state
from comb.trajectories import Trajectory, concatenate, constant, linear_segment


@dataclass(frozen=True)
class SteppingPlanner(Planner):
    """Manifold-following planner: solver-in-a-loop, bounded by ``interval``.

    Hyperparameters
    ---------------
    interval
        Maximum twist-norm distance any body may move between adjacent
        checkpoints. Smaller values produce denser checkpoints, hugging the
        constraint manifold more closely under linear interpolation.
    convergence_tolerance
        Per-parameter tolerance for "at goal" — the loop terminates when
        every parameter is within this of the goal value.
    max_substeps
        Safety cap; the planner raises ``RuntimeError`` if exceeded.
    min_step_scale
        Smallest allowed step-scale fraction before the planner gives up on
        satisfying ``interval`` and raises.
    """

    interval: float
    convergence_tolerance: float = 1e-6
    max_substeps: int = 1000
    min_step_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")

    def plan(
        self,
        system: System[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
        horizon: float,
    ) -> Trajectory[SystemState[PoseT]]:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        goal_cfg, goal_poses = find_satisfying_state(system, final_constraints)
        goal_state = SystemState(configuration=goal_cfg, body_poses=goal_poses)

        work_system = _internal_copy(system)
        state = work_system.snapshot()
        states: list[SystemState[PoseT]] = [state]

        while not _at_target(
            state.configuration, goal_state.configuration, self.convergence_tolerance
        ):
            if len(states) - 1 >= self.max_substeps:
                raise RuntimeError(
                    f"Stepping planner exceeded max_substeps={self.max_substeps} "
                    f"before reaching goal; tighten interval or relax tolerance"
                )
            delta = _delta_toward(state.configuration, goal_state.configuration)
            scale = 1.0
            while True:
                scaled = {c: scale * d for c, d in delta.items()}
                new_cfg, new_poses = solve(work_system, delta=scaled)
                new_state = SystemState(configuration=new_cfg, body_poses=new_poses)
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
            work_system.apply(new_state)
            states.append(new_state)
            state = new_state

        n_segments = len(states) - 1
        if n_segments == 0:
            return constant(states[0], horizon)
        sub_dt = horizon / n_segments
        segments = [
            linear_segment(
                states[i],
                states[i + 1],
                duration=sub_dt,
                interpolate=interpolate_system_state,
            )
            for i in range(n_segments)
        ]
        return concatenate(segments)


def _internal_copy(system: System[PoseT]) -> System[PoseT]:
    """A System whose mutable state is independent of ``system``'s."""
    state = system.snapshot()
    return System(
        bodies=list(system.bodies),
        constraints=list(system.constraints),
        configuration=state.configuration,
        body_poses=state.body_poses,
        anchored_bodies=list(system.anchored_bodies),
    )


def _delta_toward(
    current: Configuration, target: Configuration
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


def _at_target(current: Configuration, target: Configuration, tolerance: float) -> bool:
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
