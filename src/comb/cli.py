"""Command-line interface for the comb spec-language pipeline.

Subcommands:

* ``comb plan TASK [-o OUT]`` — load + validate + instantiate the task,
  plan to its goal with :class:`SteppingPlanner`, validate the resulting
  plan, and (with ``-o``) serialize it to YAML.
* ``comb render PLAN --task TASK -o OUT.gif`` — load a saved plan against
  the task's library, sample its trajectory, and write an animated GIF.
* ``comb validate library LIB`` — structural validation of a library.
* ``comb validate task TASK`` — structural validation of a task against
  its referenced library.
* ``comb validate plan PLAN --task TASK`` — load a saved plan against the
  task's library and run :func:`comb.planners.validate_plan`.

The CLI catches the spec-language and planner exceptions and turns them
into a one-line stderr message + exit code 1, so shell pipelines can
distinguish "no plan" from "wrong arguments" (argparse exits 2 for the
latter automatically). It always resolves a task's ``library`` path
relative to the task file's directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from comb.planners import PlanningError, PlanValidationError, validate_plan
from comb.planners.stepping import SteppingPlanner
from comb.spec import (
    LibraryLoadError,
    LibrarySpec,
    PlanSerializationError,
    SpecInstantiationError,
    SpecValidationError,
    TaskSpec,
    instantiate_library,
    instantiate_task,
    load_library,
    load_task_file,
    plan_from_yaml_file,
    plan_to_yaml_file,
    validate_library,
    validate_task,
)

_PLAN_VALIDATION_TOLERANCE = 1e-3


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comb",
        description="Comb spec-language CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser(
        "plan", help="Plan a task end-to-end and (optionally) write the plan YAML."
    )
    plan_parser.add_argument("task", help="Path to a task YAML file.")
    plan_parser.add_argument(
        "-o",
        "--output",
        help="Path to write the resulting plan YAML. If omitted, the plan is "
        "validated but not saved.",
    )
    plan_parser.add_argument(
        "--horizon",
        type=float,
        default=2.0,
        help="Trajectory horizon in seconds (default: %(default)s).",
    )
    plan_parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="SteppingPlanner per-checkpoint twist-norm bound "
        "(default: %(default)s).",
    )
    plan_parser.set_defaults(func=_cmd_plan)

    render_parser = sub.add_parser(
        "render",
        help="Render a saved plan as an animated GIF.",
    )
    render_parser.add_argument("plan", help="Path to a plan YAML file.")
    render_parser.add_argument(
        "--task",
        required=True,
        help="Task YAML the plan was generated from (resolves bodies, "
        "constraints, transitions, and the renderer's view).",
    )
    render_parser.add_argument(
        "-o", "--output", required=True, help="Path to write the rendered GIF."
    )
    render_parser.add_argument(
        "--fps", type=int, default=20, help="Output frame rate (default: %(default)s)."
    )
    render_parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Trajectory sample spacing in seconds (default: %(default)s).",
    )
    render_parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6.0, 6.0),
        metavar=("W", "H"),
        help="Figure size in inches (default: %(default)s).",
    )
    render_parser.set_defaults(func=_cmd_render)

    validate_parser = sub.add_parser("validate", help="Structural validation.")
    validate_sub = validate_parser.add_subparsers(dest="kind", required=True)
    vlib = validate_sub.add_parser(
        "library", help="Validate a library YAML (and its includes)."
    )
    vlib.add_argument("library", help="Path to a library YAML file.")
    vlib.set_defaults(func=_cmd_validate_library)
    vtask = validate_sub.add_parser(
        "task", help="Validate a task YAML against its referenced library."
    )
    vtask.add_argument("task", help="Path to a task YAML file.")
    vtask.set_defaults(func=_cmd_validate_task)
    vplan = validate_sub.add_parser(
        "plan",
        help="Validate a saved plan YAML against its task and library.",
    )
    vplan.add_argument("plan", help="Path to a plan YAML file.")
    vplan.add_argument(
        "--task",
        required=True,
        help="Task YAML the plan was generated from (resolves the library "
        "and the goal the plan must satisfy).",
    )
    vplan.add_argument(
        "--tolerance",
        type=float,
        default=_PLAN_VALIDATION_TOLERANCE,
        help="Per-checkpoint residual tolerance (default: %(default)s).",
    )
    vplan.set_defaults(func=_cmd_validate_plan)

    return parser


def _cmd_plan(args: argparse.Namespace) -> int:
    try:
        task_spec, library_spec, library_path = _load_task_and_library(args.task)
        validate_library(library_spec)
        validate_task(task_spec, library_spec)
        library = instantiate_library(library_spec)
        task = instantiate_task(task_spec, library)
        planner = SteppingPlanner(interval=args.interval)
        plan = planner.plan(task.system, task.goal, horizon=args.horizon)
        validate_plan(
            plan,
            task.system,
            goal=task.goal,
            tolerance=_PLAN_VALIDATION_TOLERANCE,
        )
    except (
        FileNotFoundError,
        LibraryLoadError,
        SpecValidationError,
        SpecInstantiationError,
        PlanningError,
    ) as exc:
        return _fail(str(exc))

    if args.output:
        try:
            plan_to_yaml_file(
                args.output,
                plan,
                body_names={b: n for n, b in library.bodies.items()},
                constraint_names={c: n for n, c in library.constraints.items()},
                transition_names={t: n for n, t in library.transitions.items()},
            )
        except PlanSerializationError as exc:
            return _fail(str(exc))

    n_segments = max(len(plan.sample_times) - 1, 0)
    print(
        f"planned {n_segments} segments and {len(plan.events)} transitions "
        f"over {plan.trajectory.duration:g}s "
        f"using library {library_path.name}"
    )
    if args.output:
        print(f"wrote plan to {args.output}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    # Force a headless backend before importing pyplot so the CLI works in
    # environments with no display (CI, ssh sessions, etc.).
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    # pylint: disable=import-outside-toplevel
    from matplotlib import animation, pyplot

    from comb.rendering.matplotlib_2d import MatplotlibRenderer2D

    try:
        task_spec, library_spec, _ = _load_task_and_library(args.task)
        validate_library(library_spec)
        validate_task(task_spec, library_spec)
        library = instantiate_library(library_spec)
        task = instantiate_task(task_spec, library)
        plan = plan_from_yaml_file(
            args.plan,
            bodies=library.bodies,
            constraints=library.constraints,
            transitions=library.transitions,
        )
    except (
        FileNotFoundError,
        LibraryLoadError,
        SpecValidationError,
        SpecInstantiationError,
        PlanSerializationError,
    ) as exc:
        return _fail(str(exc))

    samples = list(plan.trajectory.enumerate(args.dt))
    if not samples:
        return _fail(
            f"plan trajectory has zero duration; nothing to render " f"(dt={args.dt:g})"
        )

    figure, axis = pyplot.subplots(figsize=tuple(args.figsize))
    renderer = MatplotlibRenderer2D(ax=axis)
    mode = task.system.mode

    def draw(frame_idx: int) -> list:  # matplotlib expects Iterable[Artist]
        _, state = samples[frame_idx]
        for body in mode.bodies:
            mode.body_poses[body] = state.body_poses[body]
        renderer.render(mode)
        return []

    anim = animation.FuncAnimation(
        figure, draw, frames=len(samples), interval=int(1000 / args.fps)
    )
    try:
        anim.save(args.output, writer=animation.PillowWriter(fps=args.fps))
    finally:
        pyplot.close(figure)

    print(f"wrote {len(samples)} frames at {args.fps} fps to {args.output}")
    return 0


def _cmd_validate_library(args: argparse.Namespace) -> int:
    try:
        library_spec = load_library(args.library)
        validate_library(library_spec)
    except (FileNotFoundError, LibraryLoadError, SpecValidationError) as exc:
        return _fail(str(exc))
    print(f"{args.library}: ok")
    return 0


def _cmd_validate_plan(args: argparse.Namespace) -> int:
    try:
        task_spec, library_spec, _ = _load_task_and_library(args.task)
        validate_library(library_spec)
        validate_task(task_spec, library_spec)
        library = instantiate_library(library_spec)
        task = instantiate_task(task_spec, library)
        plan = plan_from_yaml_file(
            args.plan,
            bodies=library.bodies,
            constraints=library.constraints,
            transitions=library.transitions,
        )
        validate_plan(plan, task.system, goal=task.goal, tolerance=args.tolerance)
    except (
        FileNotFoundError,
        LibraryLoadError,
        SpecValidationError,
        SpecInstantiationError,
        PlanSerializationError,
        PlanValidationError,
    ) as exc:
        return _fail(str(exc))
    print(f"{args.plan}: ok")
    return 0


def _cmd_validate_task(args: argparse.Namespace) -> int:
    try:
        task_spec, library_spec, _ = _load_task_and_library(args.task)
        validate_library(library_spec)
        validate_task(task_spec, library_spec)
    except (FileNotFoundError, LibraryLoadError, SpecValidationError) as exc:
        return _fail(str(exc))
    print(f"{args.task}: ok")
    return 0


def _load_task_and_library(
    task_path_arg: str,
) -> tuple[TaskSpec, LibrarySpec, Path]:
    """Load the task and its referenced library; resolve the library path
    relative to the task file's directory."""
    task_path = Path(task_path_arg)
    task_spec = load_task_file(task_path)
    library_path = (task_path.parent / task_spec.library).resolve()
    library_spec = load_library(library_path)
    return task_spec, library_spec, library_path


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
