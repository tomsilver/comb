"""Watch a SystemState trajectory of the 2D two-link arm.

Run with ``pytest tests/test_trajectory_video.py --make-videos`` to render a gif into
``unit_test_videos/`` so you can eyeball the trajectory. Without the flag, the test
still runs and just verifies that enumerating the trajectory and rendering each sample
is healthy end-to-end.

A waypoint here is a target delta for the joint parameters. Linear interpolation between
two valid ``SystemState``s walks the configuration manifold and the body-pose manifold
independently, so constraint residuals are zero only at the endpoints — the further
apart in pose distance, the more visibly the arm comes apart mid-segment. We avoid that
by stepping each waypoint in ``_SUBSTEPS_PER_WAYPOINT`` substeps and calling ``solve``
at every substep, so each linear segment spans only a small slice of the manifold. This
is the same trick the future trajectory-producing solver will do automatically via its
``interval`` parameter.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

# pylint: disable=wrong-import-position
import numpy as np
import pytest
from matplotlib import animation, pyplot

from comb.bodies import BodyPoses
from comb.constraints import Configuration
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.solver import solve
from comb.system import SystemState, interpolate_system_state
from comb.trajectories import Trajectory, concatenate, linear_segment
from tests.conftest import MAKE_VIDEOS

_SUBSTEPS_PER_WAYPOINT = 16


def _snapshot(arm: TwoLinkArm2D) -> SystemState:
    """Independent SystemState that won't alias the arm's mutable state."""
    return SystemState(
        configuration=Configuration(
            {c: arm.system.configuration[c] for c in arm.system.configuration}
        ),
        body_poses=BodyPoses({b: arm.system.body_poses[b] for b in arm.system.bodies}),
    )


def _apply(arm: TwoLinkArm2D, state: SystemState) -> None:
    """Push a SystemState into the arm's system so the renderer picks it up."""
    for constraint in state.configuration:
        arm.system.configuration[constraint] = state.configuration[constraint]
    for body in arm.system.bodies:
        arm.system.body_poses[body] = state.body_poses[body]


def _trajectory_through_waypoints(
    arm: TwoLinkArm2D,
    waypoints: list[dict],
    duration_per_waypoint: float,
    n_substeps: int,
) -> Trajectory[SystemState]:
    """Solve the way to each waypoint in substeps and stitch the results.

    Each waypoint is a delta on joint parameters (same shape as ``solve``'s ``delta``
    argument). Splitting it into ``n_substeps`` solver calls keeps consecutive
    checkpoints close on the constraint manifold, so the linear segments between them
    stay near-valid.
    """
    state = _snapshot(arm)
    segments = []
    sub_dt = duration_per_waypoint / n_substeps
    for delta in waypoints:
        substep_delta = {
            c: np.asarray(d, dtype=float) / n_substeps for c, d in delta.items()
        }
        for _ in range(n_substeps):
            cfg, poses = solve(arm.system, delta=substep_delta)
            next_state = SystemState(configuration=cfg, body_poses=poses)
            segments.append(
                linear_segment(
                    state,
                    next_state,
                    duration=sub_dt,
                    interpolate=interpolate_system_state,
                )
            )
            _apply(arm, next_state)
            state = next_state
    return concatenate(segments)


def test_two_link_arm_trajectory_video():
    """Build a SystemState trajectory through joint-delta waypoints and render it.

    Each waypoint is split into substeps so the linear segments stay close to the
    constraint manifold and the arm doesn't visibly come apart between checkpoints. This
    anticipates what the future trajectory-producing solver will do automatically.
    """
    arm = TwoLinkArm2D()
    waypoints = [
        # Lift shoulder, swing elbow back.
        {
            arm.joint_ab: np.array([math.pi / 4]),
            arm.joint_bc: np.array([-math.pi / 3]),
        },
        # Lift shoulder more, swing elbow forward.
        {
            arm.joint_ab: np.array([math.pi / 4]),
            arm.joint_bc: np.array([math.pi / 2]),
        },
    ]
    traj = _trajectory_through_waypoints(
        arm,
        waypoints,
        duration_per_waypoint=1.0,
        n_substeps=_SUBSTEPS_PER_WAYPOINT,
    )
    assert traj.duration == pytest.approx(len(waypoints))

    samples = list(traj.enumerate(0.05))

    fig, ax = pyplot.subplots(figsize=(5, 5))
    renderer = MatplotlibRenderer2D(ax=ax)

    def draw(frame_idx: int) -> list:
        _, state = samples[frame_idx]
        _apply(arm, state)
        renderer.render(arm.system)
        # FuncAnimation expects an iterable of Artists for blitting; we redraw
        # the whole axes each frame, so an empty list is sufficient.
        return []

    if not MAKE_VIDEOS:
        # Smoke-check rendering on a few frames so the test is fast in CI.
        for idx in (0, len(samples) // 2, len(samples) - 1):
            draw(idx)
        pyplot.close(fig)
        return

    output_dir = Path("unit_test_videos")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "trajectory_two_link_arm.gif"
    anim = animation.FuncAnimation(fig, draw, frames=len(samples), interval=50)
    anim.save(str(output_path), writer=animation.PillowWriter(fps=20))
    pyplot.close(fig)
    assert output_path.exists()
