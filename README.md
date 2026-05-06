# Comb

![workflow](https://github.com/tomsilver/comb/actions/workflows/ci.yml/badge.svg)

A small kinematic library: bodies, constraints, modes, transitions, planners.
Generic in pose type so the same machinery works for `SE(2)` and `SE(3)`.

## Key terminology

### Static structure (the world's "what")

| Term | What it is |
|---|---|
| `Body[PoseT]` | a named rigid body with a pose and visual / collision geometry |
| `Geometry[PoseT]` | a shape attached to a body |
| `Constraint[PoseT]` | abstract relationship between two bodies; supplies `constraint_function(parameters, body_poses) -> residual` |
| `ParameterSpace` | manifold for one scalar mutable parameter |

### State (the world's "right now")

| Term | What it is |
|---|---|
| `BodyPoses[PoseT]` | mapping `Body → pose` (mutable) |
| `ConstraintConfiguration` | mapping `Constraint → ConstraintParameters`, for constraints with mutable parameters |
| `ConstraintParameters` | named 1D vector of parameter values for one constraint |
| `ModeState[PoseT]` | frozen snapshot of `(configuration, body_poses)` |

### Transitions

| Term | What it is |
|---|---|
| `ConstraintTransition[PoseT]` | trigger constraint + tolerance + `add` callable + `remove` tuple. `apply(mode, state)` returns a new `Mode`. |

### Containers

| Term | What it is |
|---|---|
| `Mode[PoseT]` | bodies + constraints + state + anchored bodies. What solvers and per-mode planners consume. |
| `System[PoseT]` | a `Mode` plus `transitions: tuple[ConstraintTransition, ...]`. What multi-mode reasoning takes. |

### Trajectories

| Term | What it is |
|---|---|
| `Trajectory[T]` | `(fn: Callable[[float], T], duration)` — continuous-time function |
| Constructors | `constant`, `linear_segment`, `piecewise_linear` |
| Composition | `concatenate`, `Trajectory.sub`, `Trajectory.enumerate` |
| Interpolators | `interpolate_array`, `interpolate_se2`, `interpolate_se3`, `interpolate_mode_state` |

### Planning

| Term | What it is |
|---|---|
| `Planner` | abstract: `plan(mode, final_constraints, horizon) -> Trajectory[ModeState]` |

### Rendering

| Term | What it is |
|---|---|
| `Renderer[PoseT]` | abstract: `render(mode, overlays=())`, `draw_body(...)` |
| `Overlay[PoseT]` | abstract: extra content drawn on top of the system |

## Examples

### Body and geometry

```python
from spatialmath import SE2
from comb.bodies import Body, Rectangle

base = Body(
    name="base",
    pose=SE2(),
    visual_geometry=Rectangle(0.2, 0.2),
    collision_geometry=Rectangle(0.2, 0.2),
)
```

### Constraint (a 2D revolute joint)

```python
import numpy as np
from comb.constraints import ConstraintParameters, RevoluteJoint2D

joint = RevoluteJoint2D(
    body1=base,
    body2=link_a,
    fixed_parameters=ConstraintParameters(
        values=np.array([0.0, 0.0]),  # origin_x, origin_y
        names=RevoluteJoint2D.fixed_parameter_names(),
    ),
)
```

### `ParameterSpace`

```python
from comb.parameter_spaces import Circle

space = Circle()
space.retract(3.0, 0.5)        # → wraps to (-π, π]
space.difference(-3.0, 3.0)    # → shortest signed angle
```

### `ConstraintConfiguration` and `BodyPoses`

```python
from comb.bodies import BodyPoses
from comb.constraints import ConstraintConfiguration

config = ConstraintConfiguration({
    joint: ConstraintParameters(np.array([0.5]), ("angle",)),
})
poses = BodyPoses({base: SE2(), link_a: SE2(0, 0, 0.5)})
```

### `Mode` and `System`

```python
from comb.examples.two_link_arm_2d import TwoLinkArm2D

ex = TwoLinkArm2D()
ex.mode    # Mode[SE2] — bodies, constraints, configuration, body_poses, anchored
ex.system  # System[SE2] — wraps mode plus any transitions (none here)
```

### `Trajectory`

```python
from comb.trajectories import linear_segment, concatenate, interpolate_se2

a = linear_segment(SE2(), SE2(1, 0, 0), duration=1.0, interpolate=interpolate_se2)
b = linear_segment(SE2(1, 0, 0), SE2(1, 1, 0), duration=1.0, interpolate=interpolate_se2)
traj = concatenate([a, b])
for t, pose in traj.enumerate(0.1):
    ...
```

### `SteppingPlanner`

```python
from comb.planners.stepping import SteppingPlanner

planner = SteppingPlanner(interval=0.1)
trajectory = planner.plan(mode, final_constraints=[goal], horizon=2.0)
```

### `ConstraintTransition` and `RigidAttachment2D`

```python
from comb.transitions import RigidAttachment2D

# When `pickup_trigger` is satisfied, attach `block` to `arm.link_b` and
# release the world-to-block pin in one step.
pickup = RigidAttachment2D(
    arm.link_b,
    block,
    trigger=pickup_trigger,
    tolerance=0.05,
    detach_from=(world_to_block,),
)

if pickup.is_enabled(mode.snapshot()):
    new_mode = pickup.apply(mode, mode.snapshot())
```

### Rendering with overlays

```python
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.rendering.overlays import PointMarker2D

renderer = MatplotlibRenderer2D()
target_marker = PointMarker2D(x=0.6, y=1.4, marker="*", color="tab:orange")
renderer.render(mode, overlays=[target_marker])
```

## Notebooks

Worked examples in [`notebooks/`](notebooks/):

- `plan_to_target_pose_2d.ipynb` — drive the two-link 2D arm so the tip reaches a world point.
- `pick_and_place_2d.ipynb` — full pick-and-place loop using a `RigidAttachment2D` transition.

## Development

```bash
uv pip install -e ".[develop]"
./run_ci_checks.sh   # black + isort + docformatter + mypy + pylint + pytest
```
