"""Round-trip tests for ``plan_to_yaml`` / ``plan_from_yaml``."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import Constraint, ConstraintParameters, PointEquality2D
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.examples.two_link_arm_with_object_2d import TwoLinkArmWithObject2D
from comb.mode import Mode
from comb.planners import Plan
from comb.planners.stepping import SteppingPlanner
from comb.spec import (
    PlanSerializationError,
    plan_from_yaml,
    plan_from_yaml_file,
    plan_to_yaml,
    plan_to_yaml_file,
)
from comb.system import System
from comb.trajectories import constant
from comb.transitions import ConstraintTransition

# --- mappings ---


def _arm_mappings(arm: TwoLinkArm2D) -> tuple[
    dict[Body[SE2], str],
    dict[str, Body[SE2]],
    dict[Constraint[SE2], str],
    dict[str, Constraint[SE2]],
]:
    body_to_name: dict[Body[SE2], str] = {
        arm.base: "base",
        arm.link_a: "link_a",
        arm.link_b: "link_b",
    }
    name_to_body: dict[str, Body[SE2]] = {v: k for k, v in body_to_name.items()}
    constraint_to_name: dict[Constraint[SE2], str] = {
        arm.joint_ab: "joint_ab",
        arm.joint_bc: "joint_bc",
    }
    name_to_constraint: dict[str, Constraint[SE2]] = {
        v: k for k, v in constraint_to_name.items()
    }
    return body_to_name, name_to_body, constraint_to_name, name_to_constraint


def _pickup_mappings(ex: TwoLinkArmWithObject2D) -> tuple[
    Mapping[Body[SE2], str],
    Mapping[str, Body[SE2]],
    Mapping[Constraint[SE2], str],
    Mapping[str, Constraint[SE2]],
    Mapping[ConstraintTransition[SE2], str],
    Mapping[str, ConstraintTransition[SE2]],
]:
    body_to_name: dict[Body[SE2], str] = {
        ex.arm.base: "base",
        ex.arm.link_a: "link_a",
        ex.arm.link_b: "link_b",
        ex.world: "world",
        ex.block: "block",
    }
    name_to_body: dict[str, Body[SE2]] = {v: k for k, v in body_to_name.items()}
    constraint_to_name: dict[Constraint[SE2], str] = {
        ex.arm.joint_ab: "joint_ab",
        ex.arm.joint_bc: "joint_bc",
        ex.world_to_block: "world_to_block",
    }
    name_to_constraint: dict[str, Constraint[SE2]] = {
        v: k for k, v in constraint_to_name.items()
    }
    transition_to_name: dict[ConstraintTransition[SE2], str] = {
        ex.pickup_transition: "pickup"
    }
    name_to_transition: dict[str, ConstraintTransition[SE2]] = {
        v: k for k, v in transition_to_name.items()
    }
    return (
        body_to_name,
        name_to_body,
        constraint_to_name,
        name_to_constraint,
        transition_to_name,
        name_to_transition,
    )


# --- helpers ---


def _augment_arm_with_world(arm: TwoLinkArm2D) -> tuple[Mode[SE2], Body[SE2]]:
    world = Body(
        name="world",
        pose=SE2(),
        visual_geometry=Rectangle(0.0, 0.0),
        collision_geometry=Rectangle(0.0, 0.0),
    )
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


def _pin(
    world: Body[SE2], body: Body[SE2], target: tuple[float, float]
) -> PointEquality2D:
    return PointEquality2D(
        body1=world,
        body2=body,
        fixed_parameters=ConstraintParameters(
            values=np.array([target[0], target[1], 1.0, 0.0]),
            names=PointEquality2D.fixed_parameter_names(),
        ),
    )


def _assert_plans_equivalent(original, restored, *, atol: float = 1e-9) -> None:
    assert restored.sample_times == original.sample_times
    assert restored.trajectory.duration == pytest.approx(original.trajectory.duration)
    assert len(restored.events) == len(original.events)
    for original_event, restored_event in zip(original.events, restored.events):
        assert restored_event.time == pytest.approx(original_event.time)
        assert restored_event.transition is original_event.transition
    for t in original.sample_times:
        original_state = original.trajectory(t)
        restored_state = restored.trajectory(t)
        for body in original_state.body_poses:
            np.testing.assert_allclose(
                restored_state.body_poses[body].A,
                original_state.body_poses[body].A,
                atol=atol,
            )
        for c in original_state.configuration:
            np.testing.assert_allclose(
                restored_state.configuration[c].values,
                original_state.configuration[c].values,
                atol=atol,
            )


# --- tests ---


def test_round_trip_within_mode_plan() -> None:
    """A multi-state within-mode plan round-trips with no events."""
    arm = TwoLinkArm2D()
    mode, world = _augment_arm_with_world(arm)
    body_to_name, name_to_body, constraint_to_name, name_to_constraint = _arm_mappings(
        arm
    )
    body_to_name[world] = "world"
    name_to_body["world"] = world

    plan = SteppingPlanner(interval=0.2).plan(
        System(mode=mode), [_pin(world, arm.link_b, (0.5, 1.0))], horizon=1.0
    )

    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=constraint_to_name,
        transition_names={},
    )
    restored = plan_from_yaml(
        text,
        bodies=name_to_body,
        constraints=name_to_constraint,
        transitions={},
    )
    assert not restored.events
    _assert_plans_equivalent(plan, restored)


def test_round_trip_plan_with_transition_event() -> None:
    """A pickup plan emits a transition event; round-tripping preserves it."""
    ex = TwoLinkArmWithObject2D()
    body_to_name, name_to_body, c_to_name, name_to_c, t_to_name, name_to_t = (
        _pickup_mappings(ex)
    )
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
    assert len(plan.events) == 1

    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=c_to_name,
        transition_names=t_to_name,
    )
    restored = plan_from_yaml(
        text,
        bodies=name_to_body,
        constraints=name_to_c,
        transitions=name_to_t,
    )
    _assert_plans_equivalent(plan, restored)
    assert restored.events[0].transition is ex.pickup_transition


def test_event_records_appear_before_state_at_same_time() -> None:
    """Per A1 convention, the event YAML record sorts before its post-transition
    state."""
    ex = TwoLinkArmWithObject2D()
    body_to_name, _, c_to_name, _, t_to_name, _ = _pickup_mappings(ex)
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
    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=c_to_name,
        transition_names=t_to_name,
    )

    # Find the first transition record's index in the dumped YAML, then look
    # at the records that share its time.
    parsed = yaml.safe_load(text)["plan"]
    event_indexes = [i for i, r in enumerate(parsed) if "transition" in r]
    assert len(event_indexes) == 1
    event_idx = event_indexes[0]
    event_t = parsed[event_idx]["t"]
    # The record immediately after must be a state record at the same t.
    after = parsed[event_idx + 1]
    assert "transition" not in after
    assert after["t"] == pytest.approx(event_t)


def test_unknown_constraint_in_input_raises() -> None:
    """Serializing a plan whose configuration references a constraint we have no name
    for fails fast."""
    arm = TwoLinkArm2D()
    mode, world = _augment_arm_with_world(arm)
    plan = SteppingPlanner(interval=0.2).plan(
        System(mode=mode), [_pin(world, arm.link_b, (0.5, 1.0))], horizon=1.0
    )

    body_to_name, _, _, _ = _arm_mappings(arm)
    body_to_name[world] = "world"
    incomplete_constraint_names: dict[Constraint[SE2], str] = {}  # missing joint_ab/bc

    with pytest.raises(PlanSerializationError, match="no entry in constraint_names"):
        plan_to_yaml(
            plan,
            body_names=body_to_name,
            constraint_names=incomplete_constraint_names,
            transition_names={},
        )


def test_unknown_transition_name_in_yaml_raises() -> None:
    """Loading a YAML with a transition name unknown to the caller errors."""
    arm = TwoLinkArm2D()
    mode, world = _augment_arm_with_world(arm)
    body_to_name, name_to_body, c_to_name, name_to_c = _arm_mappings(arm)
    body_to_name[world] = "world"
    name_to_body["world"] = world

    plan = SteppingPlanner(interval=0.2).plan(
        System(mode=mode), [_pin(world, arm.link_b, (0.5, 1.0))], horizon=1.0
    )

    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=c_to_name,
        transition_names={},
    )
    parsed = yaml.safe_load(text)
    parsed["plan"].append({"t": parsed["plan"][-1]["t"], "transition": "ghost"})
    text_with_ghost = yaml.safe_dump(parsed)
    with pytest.raises(PlanSerializationError, match="unknown transition 'ghost'"):
        plan_from_yaml(
            text_with_ghost,
            bodies=name_to_body,
            constraints=name_to_c,
            transitions={},
        )


def test_file_round_trip(tmp_path: Path) -> None:
    """``plan_to_yaml_file`` + ``plan_from_yaml_file`` round-trip via disk."""
    arm = TwoLinkArm2D()
    mode, world = _augment_arm_with_world(arm)
    body_to_name, name_to_body, c_to_name, name_to_c = _arm_mappings(arm)
    body_to_name[world] = "world"
    name_to_body["world"] = world

    plan = SteppingPlanner(interval=0.2).plan(
        System(mode=mode), [_pin(world, arm.link_b, (0.5, 1.0))], horizon=1.0
    )
    out = tmp_path / "plan.yaml"
    plan_to_yaml_file(
        out,
        plan,
        body_names=body_to_name,
        constraint_names=c_to_name,
        transition_names={},
    )
    restored = plan_from_yaml_file(
        out, bodies=name_to_body, constraints=name_to_c, transitions={}
    )
    _assert_plans_equivalent(plan, restored)


def test_duration_field_recovers_constant_trajectory_horizon() -> None:
    """A constant single-state plan round-trips with its full duration preserved.

    The YAML carries an explicit ``duration`` so a one-checkpoint plan (e.g. "already at
    goal" with nonzero horizon) doesn't collapse to a zero-duration trajectory on
    reload.
    """
    arm = TwoLinkArm2D()
    mode, world = _augment_arm_with_world(arm)
    body_to_name, name_to_body, c_to_name, name_to_c = _arm_mappings(arm)
    body_to_name[world] = "world"
    name_to_body["world"] = world

    state = mode.snapshot()
    plan: Plan[SE2] = Plan(
        trajectory=constant(state, 1.5),
        events=(),
        sample_times=(0.0,),
    )

    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=c_to_name,
        transition_names={},
    )
    restored = plan_from_yaml(
        text, bodies=name_to_body, constraints=name_to_c, transitions={}
    )
    assert restored.trajectory.duration == pytest.approx(1.5)
    assert restored.sample_times == (0.0,)
