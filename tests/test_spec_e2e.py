"""End-to-end test of the spec language: yaml → validate → instantiate → plan → validate
plan.

Drives the same pickup-and-place scenario as
``test_planners_stepping.test_plan_uses_transitions_via_bfs_to_pick_and_place``,
but the bodies, constraints, transition, and goal all come from YAML
fixtures rather than the hand-rolled ``TwoLinkArmWithObject2D`` Python
example. Round-trips the resulting plan through YAML and re-validates the
restored copy to exercise the full pipeline.
"""

from pathlib import Path

import numpy as np

from comb.planners import validate_plan
from comb.planners.stepping import SteppingPlanner
from comb.spec import (
    instantiate_library,
    instantiate_task,
    load_library,
    load_task_file,
    plan_from_yaml,
    plan_to_yaml,
    validate_library,
    validate_task,
)

_FIXTURES = Path(__file__).parent / "spec_fixtures"


def test_yaml_pickup_pipeline_runs_end_to_end() -> None:
    """Load YAML, validate, instantiate, plan a pickup, validate the plan."""
    library_spec = load_library(_FIXTURES / "example_two_link_arm_with_object.lib.yaml")
    task_spec = load_task_file(_FIXTURES / "example_pickup_place.task.yaml")

    validate_library(library_spec)
    validate_task(task_spec, library_spec)

    library = instantiate_library(library_spec)
    task = instantiate_task(task_spec, library)

    plan = SteppingPlanner(interval=0.1).plan(task.system, task.goal, horizon=2.0)

    # The plan reaches the goal: block at the placement point.
    end_state = plan.trajectory(plan.trajectory.duration)
    np.testing.assert_allclose(
        end_state.body_poses[library.bodies["block"]].t,
        [-0.6, 1.2],
        atol=1e-3,
    )

    # The pickup transition fired exactly once.
    assert len(plan.events) == 1
    assert plan.events[0].transition is library.transitions["pickup"]

    # The plan's residuals + transitions all check out.
    validate_plan(plan, task.system, goal=task.goal, tolerance=1e-3)


def test_plan_yaml_round_trip_against_yaml_library() -> None:
    """Serialize a planned plan to YAML and reload it using the library's mappings."""
    library_spec = load_library(_FIXTURES / "example_two_link_arm_with_object.lib.yaml")
    task_spec = load_task_file(_FIXTURES / "example_pickup_place.task.yaml")
    validate_library(library_spec)
    validate_task(task_spec, library_spec)

    library = instantiate_library(library_spec)
    task = instantiate_task(task_spec, library)
    plan = SteppingPlanner(interval=0.1).plan(task.system, task.goal, horizon=2.0)

    body_to_name = {body: name for name, body in library.bodies.items()}
    constraint_to_name = {c: name for name, c in library.constraints.items()}
    transition_to_name = {t: name for name, t in library.transitions.items()}

    text = plan_to_yaml(
        plan,
        body_names=body_to_name,
        constraint_names=constraint_to_name,
        transition_names=transition_to_name,
    )
    restored = plan_from_yaml(
        text,
        bodies=library.bodies,
        constraints=library.constraints,
        transitions=library.transitions,
    )

    validate_plan(restored, task.system, goal=task.goal, tolerance=1e-3)
    assert restored.events[0].transition is library.transitions["pickup"]
