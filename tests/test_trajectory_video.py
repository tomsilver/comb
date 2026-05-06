"""Watch a ModeState trajectory of the 2D two-link arm.

Run with ``pytest tests/test_trajectory_video.py --make-videos`` to render a gif into
``unit_test_videos/`` so you can eyeball the trajectory. Without the flag, the test
still runs and just verifies that planning + enumerating + rendering each sample is
healthy end-to-end.

The trajectory is built by ``SteppingPlanner.plan`` for each waypoint (an end-effector
pose pinned via a FixedJoint to a world body), then ``concatenate``-d together. The
``interval`` keeps adjacent checkpoints close on the constraint manifold, so linear
interpolation between them doesn't visibly detach the links.
"""

import math
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

# pylint: disable=wrong-import-position
import numpy as np
import pytest
from matplotlib import animation, pyplot
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, Rectangle
from comb.constraints import Constraint, ConstraintParameters, FixedJoint2D
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.mode import Mode, ModeState
from comb.planners.stepping import SteppingPlanner
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.rendering.overlays import GhostBodies
from comb.solver import solve
from comb.trajectories import Trajectory, concatenate
from tests.conftest import MAKE_VIDEOS

_INTERVAL = 0.1


def _world_body() -> Body[SE2]:
    return Body(
        name="world",
        pose=SE2(),
        visual_geometry=Rectangle(0.0, 0.0),
        collision_geometry=Rectangle(0.0, 0.0),
    )


def _augmented(arm: TwoLinkArm2D, world: Body[SE2]) -> Mode[SE2]:
    return Mode(
        bodies=arm.mode.bodies + [world],
        constraints=list(arm.mode.constraints),
        configuration=arm.mode.configuration,
        body_poses=BodyPoses(
            {b: arm.mode.body_poses[b] for b in arm.mode.bodies} | {world: SE2()}
        ),
        anchored_bodies=arm.mode.anchored_bodies + [world],
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


def _plan_through(
    mode: Mode[SE2],
    waypoints: Iterable[Iterable[Constraint[SE2]]],
    duration_per_segment: float,
) -> tuple[Trajectory[ModeState], list[ModeState]]:
    """Plan to each waypoint in turn; return the joined trajectory and per-waypoint goal
    states.

    Each waypoint is a set of final constraints handed to the planner. The end-of-
    segment ``ModeState`` is the goal state the planner reached, so we return those
    alongside the trajectory for visualization.
    """
    planner = SteppingPlanner(interval=_INTERVAL)
    segments = []
    goal_states: list[ModeState] = []
    for finals in waypoints:
        segment = planner.plan(mode, finals, horizon=duration_per_segment)
        segments.append(segment)
        end_state = segment(segment.duration)
        goal_states.append(end_state)
        mode.set_state(end_state)
    return concatenate(segments), goal_states


def _reachable_link_b_pose(arm: TwoLinkArm2D, ab: float, bc: float) -> SE2:
    """End-effector pose reachable at the given joint angles (constructed via solve)."""
    poses = solve(
        arm.mode,
        delta={
            arm.joint_ab: np.array([ab]),
            arm.joint_bc: np.array([bc]),
        },
    ).body_poses
    return poses[arm.link_b]


def test_two_link_arm_trajectory_video():
    """Plan a multi-waypoint trajectory steering link_b through end-effector poses.

    Goals are reachable poses constructed by querying ``solve`` at known joint
    configurations — picking arbitrary SE(2) poses would be over-constrained for a 2-DoF
    arm.
    """
    arm = TwoLinkArm2D()
    world = _world_body()
    mode = _augmented(arm, world)
    pose_a = _reachable_link_b_pose(arm, ab=math.pi / 4, bc=-math.pi / 3)
    pose_b = _reachable_link_b_pose(arm, ab=math.pi / 2, bc=math.pi / 6)
    waypoints = [
        [_pin(world, arm.link_b, pose_a)],
        [_pin(world, arm.link_b, pose_b)],
    ]
    traj, goal_states = _plan_through(mode, waypoints, duration_per_segment=1.0)
    assert traj.duration == pytest.approx(2.0)

    samples = list(traj.enumerate(0.05))

    fig, ax = pyplot.subplots(figsize=(5, 5))
    renderer = MatplotlibRenderer2D(ax=ax)

    # Persistent ghost overlays for every waypoint goal, in distinct colors.
    ghost_colors = ["tab:orange", "tab:green"]
    ghosts = [
        GhostBodies(
            bodies=arm.mode.bodies,
            body_poses=goal.body_poses,
            color=color,
            alpha=0.25,
        )
        for goal, color in zip(goal_states, ghost_colors)
    ]

    def draw(frame_idx: int) -> list:
        _, state = samples[frame_idx]
        # Renderer draws the original arm mode; just push the moving bodies in.
        for body in arm.mode.bodies:
            arm.mode.body_poses[body] = state.body_poses[body]
        for c in arm.mode.configuration:
            arm.mode.configuration[c] = state.configuration[c]
        renderer.render(arm.mode, overlays=ghosts)
        return []

    if not MAKE_VIDEOS:
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
