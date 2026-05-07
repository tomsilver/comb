"""Tests for the ``comb`` CLI."""

from pathlib import Path

import pytest
import yaml

from comb.cli import main

_FIXTURES = Path(__file__).parent / "spec" / "fixtures"


def test_validate_library_ok(capsys: pytest.CaptureFixture[str]) -> None:
    """``comb validate library`` returns 0 and prints "ok" on a clean library."""
    rc = main(
        [
            "validate",
            "library",
            str(_FIXTURES / "example_two_link_arm_with_object.lib.yaml"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok" in captured.out


def test_validate_library_bad_yaml_returns_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid YAML at the library path is reported on stderr with exit code 1."""
    rc = main(["validate", "library", str(_FIXTURES / "invalid_yaml.yaml")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err.lower()


def test_validate_task_ok(capsys: pytest.CaptureFixture[str]) -> None:
    """``comb validate task`` resolves the library path relative to the task file."""
    rc = main(["validate", "task", str(_FIXTURES / "example_pickup_place.task.yaml")])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok" in captured.out


def test_plan_runs_end_to_end_without_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``comb plan`` succeeds and prints a one-line summary; no output file written."""
    rc = main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "--horizon",
            "2.0",
            "--interval",
            "0.1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "planned" in captured.out
    assert "transitions" in captured.out


def test_plan_writes_yaml_when_output_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``-o`` writes a parseable plan YAML to disk."""
    out = tmp_path / "plan.yaml"
    rc = main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(out),
            "--horizon",
            "2.0",
            "--interval",
            "0.1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert f"wrote plan to {out}" in captured.out
    parsed = yaml.safe_load(out.read_text())
    assert "duration" in parsed
    assert "plan" in parsed
    transition_records = [r for r in parsed["plan"] if "transition" in r]
    assert len(transition_records) == 1
    assert transition_records[0]["transition"] == "pickup"


def test_plan_unreachable_goal_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreachable goal surfaces the planning error on stderr with exit code 1."""
    # Author a task that pins the block at a totally unreachable point.
    task = tmp_path / "unreachable.task.yaml"
    task.write_text(
        "name: unreachable\n"
        f"library: {_FIXTURES / 'example_two_link_arm_with_object.lib.yaml'}\n"
        "initial_mode: {}\n"
        "goal:\n"
        "  - type: PointEquality2D\n"
        "    body1: world\n"
        "    body2: block\n"
        "    fixed_parameters:"
        " {target_x: 100.0, target_y: 100.0, offset_x: 0.0, offset_y: 0.0}\n"
    )
    rc = main(["plan", str(task)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err.lower()


def test_missing_task_file_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    """A nonexistent task file produces a clean error rather than a stack trace."""
    rc = main(["plan", "/tmp/definitely_does_not_exist.task.yaml"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err.lower()


def test_render_writes_gif_for_saved_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``comb plan`` then ``comb render`` produces a non-empty GIF on disk."""
    plan_path = tmp_path / "plan.yaml"
    rc = main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(plan_path),
            "--horizon",
            "2.0",
            "--interval",
            "0.1",
        ]
    )
    assert rc == 0
    capsys.readouterr()  # discard plan output

    gif_path = tmp_path / "plan.gif"
    rc = main(
        [
            "render",
            str(plan_path),
            "--task",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(gif_path),
            "--fps",
            "20",
            "--dt",
            "0.1",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "wrote" in captured.out
    assert gif_path.exists()
    assert gif_path.stat().st_size > 1024


def test_render_missing_plan_returns_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A nonexistent plan file produces a clean error rather than a stack trace."""
    rc = main(
        [
            "render",
            str(tmp_path / "nope.yaml"),
            "--task",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(tmp_path / "out.gif"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err.lower()


def test_validate_plan_round_trip_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``comb plan`` then ``comb validate plan`` returns 0 on the saved YAML."""
    plan_path = tmp_path / "plan.yaml"
    rc = main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(plan_path),
        ]
    )
    assert rc == 0
    capsys.readouterr()  # discard plan output

    rc = main(
        [
            "validate",
            "plan",
            str(plan_path),
            "--task",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok" in captured.out


def test_plan_uses_task_granularity_as_default_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--interval``, the planner uses the task's max_segment_twist.

    Verified by validating the plan against the same granularity bound — if
    the planner had ignored the granularity (used the larger CLI default of
    0.1), validation would reject the plan.
    """
    plan_path = tmp_path / "plan.yaml"
    rc = main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place_with_granularity.task.yaml"),
            "-o",
            str(plan_path),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    rc = main(
        [
            "validate",
            "plan",
            str(plan_path),
            "--task",
            str(_FIXTURES / "example_pickup_place_with_granularity.task.yaml"),
        ]
    )
    assert rc == 0


def test_validate_plan_tight_tolerance_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tolerance below the planner's actual residuals produces exit code 1."""
    plan_path = tmp_path / "plan.yaml"
    main(
        [
            "plan",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "-o",
            str(plan_path),
        ]
    )
    capsys.readouterr()
    rc = main(
        [
            "validate",
            "plan",
            str(plan_path),
            "--task",
            str(_FIXTURES / "example_pickup_place.task.yaml"),
            "--tolerance",
            "1e-12",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "error" in captured.err.lower()
