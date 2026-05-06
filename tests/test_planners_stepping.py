"""Tests for the manifold-following SteppingPlanner."""

import math

import numpy as np
import pytest
from spatialmath import SE2, Twist2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import (
    Constraint,
    ConstraintParameters,
    FixedJoint2D,
    PointEquality2D,
)
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.examples.two_link_arm_with_object_2d import TwoLinkArmWithObject2D
from comb.mode import Mode, ModeState
from comb.planners import PlanningError
from comb.planners.stepping import SteppingPlanner
from comb.solver import solve
from comb.system import System


def _world_body() -> Body[SE2]:
    return Body(
        name="world",
        pose=SE2(),
        visual_geometry=Rectangle(0.0, 0.0),
        collision_geometry=Rectangle(0.0, 0.0),
    )


def _pin(world: Body[SE2], body: Body[SE2], pose: SE2) -> FixedJoint2D:
    return FixedJoint2D(
        body1=world,
        body2=body,
        fixed_parameters=ConstraintParameters(
            values=np.array([float(pose.t[0]), float(pose.t[1]), float(pose.theta())]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )


def _arm_with_world(arm: TwoLinkArm2D) -> tuple[Mode[SE2], Body[SE2]]:
    """Augment the arm with an anchored world body so final_constraints can pin to
    it."""
    world = _world_body()
    augmented = Mode(
        bodies=arm.mode.bodies + [world],
        constraints=list(arm.mode.constraints),
        configuration=arm.mode.configuration,
        body_poses=BodyPoses(
            {b: arm.mode.body_poses[b] for b in arm.mode.bodies} | {world: SE2()}
        ),
        anchored_bodies=arm.mode.anchored_bodies + [world],
    )
    return augmented, world


def _reachable_link_b_pose(arm: TwoLinkArm2D, ab: float, bc: float) -> SE2:
    """The pose ``link_b`` would have at the given joint angles.

    The 2-link arm is 2-DoF, so a full SE(2) goal pose is generally over-constrained.
    Constructing the goal from a known joint config guarantees reachability.
    """
    poses = solve(
        arm.mode,
        delta={
            arm.joint_ab: np.array([ab]),
            arm.joint_bc: np.array([bc]),
        },
    ).body_poses
    return poses[arm.link_b]


def _se2_distance(a: SE2, b: SE2) -> float:
    return float(np.linalg.norm(Twist2(a.inv() * b).A))


def _max_residual(mode: Mode, state: ModeState) -> float:
    """Largest constraint residual norm at ``state`` (zero at solver checkpoints)."""

    def _norm(constraint: Constraint) -> float:
        if constraint.parameter_names():
            params = state.configuration[constraint]
        else:
            params = ConstraintParameters(np.array([]), ())
        return float(
            np.linalg.norm(constraint.constraint_function(params, state.body_poses))
        )

    return max((_norm(c) for c in mode.constraints), default=0.0)


def test_plan_reaches_goal_pose():
    """The trajectory ends at a state satisfying the final constraint."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal)]
    traj = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    end = traj(traj.duration)
    assert _se2_distance(end.body_poses[arm.link_b], goal) < 1e-3


def test_plan_does_not_mutate_mode():
    """The user's mode is unchanged after planning."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    initial_ab = float(mode.configuration[arm.joint_ab]["angle"])
    initial_link_a = SE2(mode.body_poses[arm.link_a])
    goal = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=math.pi / 4)
    SteppingPlanner(interval=0.1).plan(
        System(mode=mode), [_pin(world, arm.link_b, goal)], 1.0
    )
    assert float(mode.configuration[arm.joint_ab]["angle"]) == initial_ab
    np.testing.assert_allclose(mode.body_poses[arm.link_a].A, initial_link_a.A)


def test_plan_respects_interval_between_checkpoints():
    """Adjacent checkpoints' max body twist distance is bounded by interval."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 3, bc=-math.pi / 4)
    interval = 0.15
    traj = SteppingPlanner(interval=interval).plan(
        System(mode=mode), [_pin(world, arm.link_b, goal)], horizon=1.0
    )
    samples = [traj(i / 200 * traj.duration) for i in range(201)]
    max_local = max(
        _se2_distance(prev.body_poses[body], nxt.body_poses[body])
        for prev, nxt in zip(samples, samples[1:])
        for body in arm.mode.bodies
    )
    assert max_local <= interval + 1e-6


def test_plan_horizon_is_exact():
    """Reported duration equals the requested horizon."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 6, bc=math.pi / 6)
    traj = SteppingPlanner(interval=0.2).plan(
        System(mode=mode), [_pin(world, arm.link_b, goal)], horizon=3.7
    )
    assert traj.duration == pytest.approx(3.7)


def test_plan_already_at_goal_returns_constant():
    """If the mode already satisfies final constraints, the trajectory is constant."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    # Pin link_b at its current pose — already satisfied.
    final = [_pin(world, arm.link_b, SE2(mode.body_poses[arm.link_b]))]
    traj = SteppingPlanner(interval=0.1).plan(System(mode=mode), final, horizon=1.0)
    assert traj.duration == pytest.approx(1.0)
    s0, s_mid, s_end = traj(0.0), traj(0.5), traj(1.0)
    for body in arm.mode.bodies:
        np.testing.assert_allclose(s0.body_poses[body].A, s_mid.body_poses[body].A)
        np.testing.assert_allclose(s0.body_poses[body].A, s_end.body_poses[body].A)


def test_plan_rejects_non_positive_interval():
    """Interval must be positive to bound segment size."""
    with pytest.raises(ValueError, match="interval must be positive"):
        SteppingPlanner(interval=0.0)


def test_plan_rejects_non_positive_horizon():
    """Horizon must be positive."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 6, bc=0.0)
    with pytest.raises(ValueError, match="horizon must be positive"):
        SteppingPlanner(interval=0.1).plan(
            System(mode=mode), [_pin(world, arm.link_b, goal)], horizon=0.0
        )


def test_plan_raises_when_max_substeps_exceeded():
    """A reachable goal with a tiny interval can't fit in a small substep budget."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 3, bc=-math.pi / 4)
    planner = SteppingPlanner(interval=1e-3, max_substeps=5)
    with pytest.raises(PlanningError, match="no plan"):
        planner.plan(System(mode=mode), [_pin(world, arm.link_b, goal)], horizon=1.0)


def test_plan_raises_when_goal_unreachable():
    """An unreachable goal with no transitions to try yields a no-plan error."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    # Two-link arm has reach <= 2.0; pose at (10, 0) is unreachable.
    final = [_pin(world, arm.link_b, SE2(10.0, 0.0, 0.0))]
    with pytest.raises(PlanningError, match="no plan"):
        SteppingPlanner(interval=0.1).plan(System(mode=mode), final, horizon=1.0)


def test_smaller_interval_yields_path_closer_to_manifold():
    """A finer ``interval`` keeps mid-segment states closer to the constraint
    manifold."""
    arm = TwoLinkArm2D()
    mode, world = _arm_with_world(arm)
    goal = _reachable_link_b_pose(arm, ab=math.pi / 3, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal)]
    traj_coarse = SteppingPlanner(interval=0.5).plan(
        System(mode=mode), final, horizon=1.0
    )
    traj_fine = SteppingPlanner(interval=0.05).plan(
        System(mode=mode), final, horizon=1.0
    )
    n = 50
    coarse_avg = sum(
        _max_residual(arm.mode, traj_coarse(i / n * traj_coarse.duration))
        for i in range(1, n)
    ) / (n - 1)
    fine_avg = sum(
        _max_residual(arm.mode, traj_fine(i / n * traj_fine.duration))
        for i in range(1, n)
    ) / (n - 1)
    assert fine_avg < coarse_avg


def test_plan_uses_transitions_via_bfs_to_pick_and_place():
    """The planner discovers the pickup transition automatically via BFS.

    Goal: block at a placement target. Initial mode pins the block to the
    world, so the goal isn't reachable directly; the only way is to fire the
    bundled `pickup_transition`, then drive the now-attached block to the
    target. The user just hands the planner the system and the goal — they
    don't tell it about the transition.
    """
    ex = TwoLinkArmWithObject2D()
    placement_xy = (-0.6, 1.2)
    goal = PointEquality2D(
        body1=ex.world,
        body2=ex.block,
        fixed_parameters=ConstraintParameters(
            values=np.array([placement_xy[0], placement_xy[1], 0.0, 0.0]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )
    trajectory = SteppingPlanner(interval=0.1).plan(ex.system, [goal], horizon=2.0)
    end_state = trajectory(trajectory.duration)
    np.testing.assert_allclose(
        end_state.body_poses[ex.block].t,
        [placement_xy[0], placement_xy[1]],
        atol=1e-3,
    )
