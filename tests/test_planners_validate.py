"""Tests for the plan validator."""

import math
from dataclasses import replace

import numpy as np
import pytest
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import ConstraintParameters, FixedJoint2D, PointEquality2D
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.examples.two_link_arm_with_object_2d import TwoLinkArmWithObject2D
from comb.mode import Mode, ModeState
from comb.planners import PlanValidationError, TransitionEvent, validate_plan
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


def _augment(arm: TwoLinkArm2D) -> tuple[Mode[SE2], Body[SE2]]:
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


def _pin(world: Body[SE2], body: Body[SE2], pose: SE2) -> FixedJoint2D:
    return FixedJoint2D(
        body1=world,
        body2=body,
        fixed_parameters=ConstraintParameters(
            values=np.array([float(pose.t[0]), float(pose.t[1]), float(pose.theta())]),
            names=FixedJoint2D.fixed_parameter_names(),
        ),
    )


def _reachable_link_b_pose(arm: TwoLinkArm2D, ab: float, bc: float) -> SE2:
    poses = solve(
        arm.mode,
        delta={arm.joint_ab: np.array([ab]), arm.joint_bc: np.array([bc])},
    ).body_poses
    return poses[arm.link_b]


# --- happy paths ---


def test_validate_plan_from_stepping_planner_passes() -> None:
    """A plan straight from SteppingPlanner satisfies its own residuals."""
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    validate_plan(plan, System(mode=mode), goal=final, tolerance=1e-3)


def test_validate_plan_with_transition_passes() -> None:
    """A pickup plan replays cleanly: trigger fires, post-transition residuals zero."""
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
    plan = SteppingPlanner(interval=0.1).plan(ex.system, [goal], horizon=2.0)
    validate_plan(plan, ex.system, goal=[goal], tolerance=1e-3)


def test_validate_plan_no_goal_passes() -> None:
    """The default empty goal makes the goal check a no-op."""
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 6, bc=0.0)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    validate_plan(plan, System(mode=mode))


# --- error paths ---


def test_unsatisfied_goal_rejected() -> None:
    """A goal the plan doesn't actually reach raises with a residual report."""
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    # Different goal than the plan was built for — far away, won't be satisfied.
    bogus_goal = [_pin(world, arm.link_b, SE2(10.0, 0.0, 0.0))]
    with pytest.raises(PlanValidationError, match="goal .* not satisfied"):
        validate_plan(plan, System(mode=mode), goal=bogus_goal, tolerance=1e-3)


def test_non_monotonic_sample_times_rejected() -> None:
    """A plan whose sample_times go backwards is structurally invalid."""
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    bad = replace(plan, sample_times=tuple(reversed(plan.sample_times)))
    with pytest.raises(
        PlanValidationError, match="sample_times must be non-decreasing"
    ):
        validate_plan(bad, System(mode=mode))


def test_event_past_last_sample_time_rejected() -> None:
    """An event scheduled after the trajectory ends is rejected."""
    ex = TwoLinkArmWithObject2D()
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)
    # Plant a fake event past the trajectory's end. Its transition is borrowed
    # from the pickup example — the validator never tries to fire it (it's
    # past the last sample), so the borrow is harmless.
    late_event = TransitionEvent(
        time=plan.sample_times[-1] + 1.0, transition=ex.pickup_transition
    )
    bad = replace(plan, events=(late_event,))
    with pytest.raises(PlanValidationError, match="past the last sample time"):
        validate_plan(bad, System(mode=mode))


def test_event_not_at_sample_time_rejected() -> None:
    """An event time that doesn't match any checkpoint is rejected."""
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
    plan = SteppingPlanner(interval=0.1).plan(ex.system, [goal], horizon=2.0)
    original = plan.events[0]
    # Move the event to a time strictly between two sample times.
    midway = (plan.sample_times[1] + plan.sample_times[2]) / 2
    bad = replace(
        plan,
        events=(TransitionEvent(time=midway, transition=original.transition),),
    )
    with pytest.raises(PlanValidationError, match="doesn't match a sample time"):
        validate_plan(bad, ex.system)


def test_trigger_not_satisfied_at_pre_transition_state_rejected() -> None:
    """Moving the pickup event earlier than when the tip actually reaches the block
    makes the trigger residual exceed the transition's tolerance."""
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
    plan = SteppingPlanner(interval=0.1).plan(ex.system, [goal], horizon=2.0)
    real_event = plan.events[0]
    early = TransitionEvent(time=plan.sample_times[1], transition=real_event.transition)
    bad = replace(plan, events=(early,))
    with pytest.raises(PlanValidationError, match="trigger residual"):
        validate_plan(bad, ex.system)


def test_event_at_first_sample_rejected() -> None:
    """A transition at t=0 has no pre-transition state to check the trigger against, so
    it's a structural error."""
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
    plan = SteppingPlanner(interval=0.1).plan(ex.system, [goal], horizon=2.0)
    bad_event = TransitionEvent(
        time=plan.sample_times[0], transition=ex.pickup_transition
    )
    bad = replace(plan, events=(bad_event,))
    with pytest.raises(PlanValidationError, match="no preceding state"):
        validate_plan(bad, ex.system)


def test_constraint_residual_too_large_rejected() -> None:
    """Tightening the tolerance below what the planner achieves catches drift."""
    arm = TwoLinkArm2D()
    mode, world = _augment(arm)
    goal_pose = _reachable_link_b_pose(arm, ab=math.pi / 3, bc=-math.pi / 4)
    final = [_pin(world, arm.link_b, goal_pose)]
    plan = SteppingPlanner(interval=0.2).plan(System(mode=mode), final, horizon=1.0)

    # Inject a noisy state by replacing the trajectory with one that perturbs
    # every body pose by a noticeable amount. We do it by wrapping the
    # trajectory's sampling.
    def _noisy_trajectory(t: float) -> ModeState[SE2]:
        clean = plan.trajectory(t)
        perturbed_poses = BodyPoses(
            {b: clean.body_poses[b] * SE2(0.5, 0.0, 0.0) for b in clean.body_poses}
        )
        return ModeState(configuration=clean.configuration, body_poses=perturbed_poses)

    noisy_trajectory = replace(plan.trajectory, fn=_noisy_trajectory)
    noisy_plan = replace(plan, trajectory=noisy_trajectory)
    with pytest.raises(PlanValidationError, match="residual"):
        validate_plan(noisy_plan, System(mode=mode), tolerance=1e-3)
