"""Solver: find a close valid state for a system.

Two functions:

* :func:`solve` — apply a parameter delta and update body poses to satisfy
  the system's constraints. Joint parameters are *fixed* at ``current+delta``;
  only body poses are optimized. Used by the GUI (slider tick) and by the
  stepping planner's inner loop.

* :func:`find_satisfying_state` — joint parameters AND body poses are both
  optimization variables. Finds a state that satisfies the system's
  constraints plus any number of *extra* constraints. Useful for finding goal
  states from posed constraints (e.g. "end-effector at world pose X").

Both use a finite-difference Jacobian. ``solve`` is plain Gauss-Newton
(simple, fast, fine when the start is on or near the manifold).
``find_satisfying_state`` uses Levenberg-Marquardt damping so it stays
robust at kinematic singularities where pure Gauss-Newton's Jacobian is
rank-deficient and would stall. Body poses update via SE(2)/SE(3) twist
exponentials; parameters update via each ``ParameterSpace.retract`` so
circular angles wrap and bounded reals clamp.
"""

from collections.abc import Callable, Iterable, Mapping
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
                # Apply each delta component via the parameter's own retract,
                # so circular angles wrap and bounded reals clamp.
                delta_arr = np.asarray(delta[constraint], dtype=float)
                spaces = constraint.parameter_spaces
                current = np.array(
                    [
                        spaces[i].retract(float(current[i]), float(delta_arr[i]))
                        for i in range(len(current))
                    ]
                )
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


def find_satisfying_state(  # pylint: disable=too-many-locals,too-many-statements
    system: System[PoseT],
    extra_constraints: Iterable[Constraint[PoseT]] = (),
    *,
    max_iterations: int = 200,
    residual_tolerance: float = 1e-6,
    step_tolerance: float = 1e-12,
    max_step_norm: float = 1.0,
    initial_damping: float = 1e-3,
) -> tuple[Configuration, BodyPoses[PoseT]]:
    """Find a state satisfying ``system.constraints + extra_constraints``.

    Joint parameters and body poses are *both* optimization variables. Initial
    values come from ``system.configuration`` / ``system.body_poses``; any
    parameterized constraint in ``extra_constraints`` not present in the
    system's configuration starts at zero.

    Uses Levenberg-Marquardt: each step solves
    ``(JᵀJ + λI) δ = -Jᵀr`` with adaptive ``λ`` (decreased when a step
    reduces the residual, increased when it doesn't). This stays robust at
    kinematic singularities where pure Gauss-Newton's Jacobian is
    rank-deficient and would stall.

    Raises ``RuntimeError`` if the residual can't be driven below
    ``residual_tolerance`` within ``max_iterations`` — typically meaning the
    augmented constraint set is unsatisfiable from the given initial state.
    """
    extras = list(extra_constraints)
    all_constraints = list(system.constraints) + extras
    twist_dim, exp_map = _twist_dim_and_exp(system)
    if not system.anchored_bodies:
        raise ValueError(
            "find_satisfying_state() requires system.anchored_bodies to be "
            "non-empty: without an anchor, the SE(2)/SE(3) gauge is ambiguous"
        )

    bodies = list(system.bodies)
    anchored_ids = {id(b) for b in system.anchored_bodies}
    movable_bodies = [b for b in bodies if id(b) not in anchored_ids]
    body_offsets = {b: i * twist_dim for i, b in enumerate(movable_bodies)}
    n_pose_vars = len(movable_bodies) * twist_dim

    parameterized = [c for c in all_constraints if c.parameter_names()]
    param_offsets: dict[Constraint[PoseT], tuple[int, int]] = {}
    cursor = n_pose_vars
    for c in parameterized:
        n = len(c.parameter_names())
        param_offsets[c] = (cursor, n)
        cursor += n
    n_vars = cursor

    body_pose_curr: dict[Body[PoseT], Any] = {b: system.body_poses[b] for b in bodies}
    params_curr: dict[Constraint[PoseT], np.ndarray] = {}
    for c in parameterized:
        if c in system.configuration:
            params_curr[c] = system.configuration[c].values.astype(float).copy()
        else:
            params_curr[c] = np.zeros(len(c.parameter_names()))

    def evaluate_residuals() -> np.ndarray:
        body_poses_obj = BodyPoses(body_pose_curr)
        residuals = []
        for c in all_constraints:
            if c.parameter_names():
                params = ConstraintParameters(params_curr[c], c.parameter_names())
            else:
                params = ConstraintParameters(np.array([]), ())
            residuals.append(c.constraint_function(params, body_poses_obj))
        if not residuals:
            return np.array([])
        return np.concatenate(residuals)

    def perturb(var_idx: int, eps: float) -> Callable[[], None]:
        if var_idx < n_pose_vars:
            body_idx, twist_idx = divmod(var_idx, twist_dim)
            body = movable_bodies[body_idx]
            saved = body_pose_curr[body]
            twist = np.zeros(twist_dim)
            twist[twist_idx] = eps
            body_pose_curr[body] = saved * exp_map(twist)
            return lambda: body_pose_curr.__setitem__(body, saved)
        for c, (off, n) in param_offsets.items():
            if off <= var_idx < off + n:
                pidx = var_idx - off
                saved = params_curr[c].copy()
                new = saved.copy()
                space = c.parameter_spaces[pidx]
                new[pidx] = space.retract(float(saved[pidx]), eps)
                params_curr[c] = new
                return lambda: params_curr.__setitem__(c, saved)
        raise IndexError(f"var_idx={var_idx} out of variable range")

    def apply_step(step: np.ndarray) -> None:
        for body, off in body_offsets.items():
            twist = step[off : off + twist_dim]
            body_pose_curr[body] = body_pose_curr[body] * exp_map(twist)
        for c, (off, n) in param_offsets.items():
            tangent = step[off : off + n]
            spaces = c.parameter_spaces
            params_curr[c] = np.array(
                [
                    spaces[i].retract(float(params_curr[c][i]), float(tangent[i]))
                    for i in range(n)
                ]
            )

    final_residual_norm = float(np.linalg.norm(evaluate_residuals()))
    if n_vars > 0:
        damping = initial_damping
        for _ in range(max_iterations):
            if final_residual_norm < residual_tolerance:
                break
            residuals = evaluate_residuals()
            if residuals.size == 0:
                break

            jacobian = np.empty((residuals.size, n_vars))
            for v in range(n_vars):
                undo = perturb(v, _FD_EPSILON)
                r_pert = evaluate_residuals()
                undo()
                jacobian[:, v] = (r_pert - residuals) / _FD_EPSILON

            # Levenberg-Marquardt: stack a sqrt(λ)·I block onto J so the
            # least-squares solution is (JᵀJ + λI)⁻¹ Jᵀ(-r).
            damped_j = np.vstack([jacobian, np.sqrt(damping) * np.eye(n_vars)])
            damped_r = np.concatenate([-residuals, np.zeros(n_vars)])
            step, *_ = np.linalg.lstsq(damped_j, damped_r, rcond=None)

            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step_norm:
                step = step * (max_step_norm / step_norm)
            if step_norm < step_tolerance:
                break

            saved_poses = dict(body_pose_curr)
            saved_params = {c: v.copy() for c, v in params_curr.items()}
            apply_step(step)
            new_residual_norm = float(np.linalg.norm(evaluate_residuals()))
            if new_residual_norm < final_residual_norm:
                final_residual_norm = new_residual_norm
                damping = max(damping / 10.0, 1e-9)
            else:
                # Revert and try with stronger damping next iteration.
                body_pose_curr.update(saved_poses)
                params_curr.clear()
                params_curr.update(saved_params)
                damping = min(damping * 10.0, 1e6)

    if final_residual_norm > residual_tolerance:
        raise RuntimeError(
            f"find_satisfying_state failed to converge: residual norm "
            f"{final_residual_norm:g} > tolerance {residual_tolerance:g}; the "
            f"augmented constraint set may be unsatisfiable from the initial state"
        )

    config = Configuration()
    for c in system.constraints:
        if c.parameter_names():
            config[c] = ConstraintParameters(
                values=params_curr[c], names=c.parameter_names()
            )
    return config, BodyPoses({b: _sanitize_pose(p) for b, p in body_pose_curr.items()})


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
    return config, BodyPoses({b: _sanitize_pose(p) for b, p in poses.items()})


def _sanitize_pose(pose: Any) -> Any:
    """Project a pose back onto SE(2)/SE(3) to remove accumulated float drift.

    Without this, repeated calls to solve() would let body pose matrices slowly
    drift off the manifold, eventually triggering spatialmath's validity checks
    inside Twist2/Twist3 conversion.
    """
    if isinstance(pose, SE2):
        return SE2(float(pose.t[0]), float(pose.t[1]), float(pose.theta()))
    if isinstance(pose, SE3):
        # Re-orthonormalize the rotation block via SVD.
        u, _, vt = np.linalg.svd(pose.R)
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        return SE3.Rt(rot, np.asarray(pose.t))
    return pose
