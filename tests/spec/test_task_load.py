"""Tests for the YAML task loader."""

from pathlib import Path

import pytest

from comb.spec import (
    ConstraintSpec,
    InitialModeSpec,
    LibraryLoadError,
    PoseSpec,
    TaskSpec,
    load_task_file,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _fx(name: str) -> Path:
    return _FIXTURES / name


def test_load_full_task() -> None:
    """A task with active_constraints, body_poses, configuration, and goal round-
    trips."""
    task = load_task_file(_fx("full_task.yaml"))

    assert isinstance(task, TaskSpec)
    assert task.name == "pickup_at_target"
    assert task.library == "pickup_arm.lib.yaml"
    assert task.initial_mode.active_constraints == (
        "joint_ab",
        "joint_bc",
        "world_to_block",
    )
    assert task.initial_mode.body_poses == {
        "block": PoseSpec(values={"x": 0.5, "y": 1.0, "theta": 0.0}),
    }
    assert task.initial_mode.configuration == {
        "joint_ab": {"angle": 0.0},
        "joint_bc": {"angle": 0.0},
    }
    assert task.goal == (
        ConstraintSpec(
            type="PointEquality2D",
            body1="world",
            body2="block",
            fixed_parameters={
                "target_x": -0.6,
                "target_y": 1.2,
                "offset_x": 0.0,
                "offset_y": 0.0,
            },
        ),
    )


def test_load_minimal_task() -> None:
    """``name`` and ``library`` are mandatory; everything else defaults."""
    task = load_task_file(_fx("minimal_task.yaml"))
    assert task == TaskSpec(
        name="minimal",
        library="minimal.lib.yaml",
        initial_mode=InitialModeSpec(),
    )
    # active_constraints defaults to None, meaning "all library constraints".
    assert task.initial_mode.active_constraints is None


def test_missing_library_raises() -> None:
    """``library`` is required — without it we don't know what to validate against."""
    with pytest.raises(LibraryLoadError, match="missing required key 'library'"):
        load_task_file(_fx("task_missing_library.yaml"))


def test_goal_must_be_list() -> None:
    """``goal`` is a list of constraint specs, not a single mapping."""
    with pytest.raises(LibraryLoadError, match=r"goal: expected list"):
        load_task_file(_fx("task_goal_not_list.yaml"))


def test_goal_items_must_be_mappings() -> None:
    """A scalar in the goal list (e.g. a constraint name) is rejected."""
    with pytest.raises(LibraryLoadError, match=r"goal\[0\]: expected mapping"):
        load_task_file(_fx("task_goal_item_not_mapping.yaml"))


def test_active_constraints_can_be_empty_tuple() -> None:
    """An explicit empty list means "no constraints active" (distinct from "all")."""
    task = load_task_file(_fx("task_initial_mode_active_empty.yaml"))
    assert task.initial_mode.active_constraints == ()


def test_body_poses_must_be_mapping() -> None:
    """``body_poses`` is keyed by body name; a list is rejected."""
    with pytest.raises(LibraryLoadError, match=r"body_poses: expected mapping"):
        load_task_file(_fx("task_initial_mode_body_poses_not_mapping.yaml"))


def test_load_task_with_granularity() -> None:
    """A task with a ``granularity:`` block parses into ``GranularitySpec``."""
    task = load_task_file(_fx("task_with_granularity.yaml"))
    assert task.granularity is not None
    assert task.granularity.max_segment_twist == 0.15


def test_granularity_missing_required_key_raises() -> None:
    """``granularity`` requires ``max_segment_twist``."""
    with pytest.raises(
        LibraryLoadError,
        match=r"granularity: missing required key 'max_segment_twist'",
    ):
        load_task_file(_fx("task_granularity_missing_field.yaml"))


def test_task_default_granularity_is_none() -> None:
    """Tasks without a ``granularity:`` block leave the field as ``None``."""
    task = load_task_file(_fx("minimal_task.yaml"))
    assert task.granularity is None
