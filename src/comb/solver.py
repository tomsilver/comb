"""Solver: find a close valid configuration after applying a parameter delta.

Given a ``System`` with a (presumably valid) configuration and body poses, plus
a ``delta`` mapping a subset of constraints to parameter offsets, ``solve``
returns a new ``(Configuration, BodyPoses)`` that drives the stacked constraint
residuals toward zero — exact for kinematic trees, least-squares for loops or
over-constrained systems.

Semantics: parameters in ``delta`` are set to ``current + delta``; parameters
not in ``delta`` are held at their current values; only body poses are
optimized to satisfy the constraints. Bodies in ``system.anchored_bodies``
keep their poses fixed; at least one anchor is required to remove the global
SE(2)/SE(3) gauge.

The implementation is a simple Gauss-Newton iteration with a finite-difference
Jacobian. Body poses are updated via SE(2)/SE(3) twist exponentials.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from spatialmath import SE2, SE3

from comb.bodies import Body, BodyPoses, PoseT
from comb.constraints import Configuration, Constraint, ConstraintParameters
from comb.system import System

_FD_EPSILON = 1e-7


def solve(
    system: System[PoseT],
    delta: Mapping[Constraint[PoseT], np.ndarray] | None = None,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-9,
    step_tolerance: float = 1e-12,
    max_step_norm: float = 1.0,
) -> tuple[Configuration, BodyPoses[PoseT]]:
    """Find a close valid configuration after applying delta to driven parameters.

    ``delta`` maps a subset of constraints to a numpy delta vector matching
    each constraint's ``parameter_names()``. Parameters in ``delta`` are
    treated as driven (target = current + delta); other parameters stay fixed
    at their current values. The solver then updates body poses to drive the
    stacked constraint residuals toward zero.
    """
    delta = dict(delta or {})

    twist_dim, exp_map = _twist_dim_and_exp(system)

    bodies = list(system.bodies)
    if not system.anchored_bodies:
        raise ValueError(
            "solve() requires system.anchored_bodies to be non-empty: without "
            "an anchor, the SE(2)/SE(3) gauge is ambiguous and pose updates "
            "would be distributed across bodies rather than propagated through "
            "joints"
        )
    anchored_ids = {id(b) for b in system.anchored_bodies}
    movable_bodies = [b for b in bodies if id(b) not in anchored_ids]
    body_offsets = {b: i * twist_dim for i, b in enumerate(movable_bodies)}
    n_vars = len(movable_bodies) * twist_dim

    # Initialize current body poses and parameter values.
    body_pose_curr: dict[Body[PoseT], Any] = {b: system.body_poses[b] for b in bodies}
    params_curr: dict[Constraint[PoseT], np.ndarray] = {}
    for constraint in system.constraints:
        if constraint.parameter_names():
            current = system.configuration[constraint].values.astype(float).copy()
            if constraint in delta:
                current = current + np.asarray(delta[constraint], dtype=float)
            params_curr[constraint] = current

    def evaluate_residuals() -> np.ndarray:
        body_poses_obj = BodyPoses(body_pose_curr)
        residuals = []
        for c in system.constraints:
            if c.parameter_names():
                params = ConstraintParameters(params_curr[c], c.parameter_names())
            else:
                params = ConstraintParameters(np.array([]), ())
            residuals.append(c.constraint_function(params, body_poses_obj))
        if not residuals:
            return np.array([])
        return np.concatenate(residuals)

    def perturb(var_idx: int, eps: float) -> Callable[[], None]:
        """Apply a perturbation to variable ``var_idx``; return an undo callback."""
        body_idx, twist_idx = divmod(var_idx, twist_dim)
        body = movable_bodies[body_idx]
        saved = body_pose_curr[body]
        twist = np.zeros(twist_dim)
        twist[twist_idx] = eps
        body_pose_curr[body] = saved * exp_map(twist)
        return lambda: body_pose_curr.__setitem__(body, saved)

    if n_vars == 0:
        return _build_outputs(system, params_curr, body_pose_curr)

    prev_residual_norm = float("inf")
    for _ in range(max_iterations):
        residuals = evaluate_residuals()
        if residuals.size == 0:
            break
        residual_norm = float(np.linalg.norm(residuals))
        if residual_norm < residual_tolerance:
            break
        # Stop once residual stops improving — avoids endless oscillation when
        # constraints can't all be satisfied (e.g. over-constrained loops),
        # which would otherwise let body pose matrices drift off the manifold.
        if prev_residual_norm - residual_norm < residual_tolerance:
            break
        prev_residual_norm = residual_norm

        jacobian = np.empty((residuals.size, n_vars))
        for v in range(n_vars):
            undo = perturb(v, _FD_EPSILON)
            r_pert = evaluate_residuals()
            undo()
            jacobian[:, v] = (r_pert - residuals) / _FD_EPSILON

        step, *_ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        # Clip step to keep body pose updates from drifting off the manifold
        # when residuals can't be driven to zero.
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step_norm:
            step = step * (max_step_norm / step_norm)
        if step_norm < step_tolerance:
            break

        for body, off in body_offsets.items():
            twist = step[off : off + twist_dim]
            body_pose_curr[body] = body_pose_curr[body] * exp_map(twist)

    return _build_outputs(system, params_curr, body_pose_curr)


def _twist_dim_and_exp(
    system: System[PoseT],
) -> tuple[int, Callable[[np.ndarray], Any]]:
    """Pick twist dimension and exponential map by inspecting a body's current pose."""
    if not system.bodies:
        raise ValueError("Cannot solve a system with no bodies")
    sample = system.body_poses[system.bodies[0]]
    if isinstance(sample, SE2):
        return 3, SE2.Exp
    if isinstance(sample, SE3):
        return 6, SE3.Exp
    raise TypeError(
        f"solver only supports SE(2) and SE(3) body poses, got {type(sample).__name__}"
    )


def _build_outputs(
    system: System[PoseT],
    params: dict[Constraint[PoseT], np.ndarray],
    poses: dict[Body[PoseT], Any],
) -> tuple[Configuration, BodyPoses[PoseT]]:
    config = Configuration()
    for constraint in system.constraints:
        if constraint.parameter_names():
            config[constraint] = ConstraintParameters(
                values=params[constraint], names=constraint.parameter_names()
            )
    return config, BodyPoses(poses)
