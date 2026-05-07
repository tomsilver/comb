# Comb

![workflow](https://github.com/tomsilver/comb/actions/workflows/ci.yml/badge.svg)

A small kinematic library: bodies, constraints, modes, transitions, planners.
Generic in pose type so the same machinery works for `SE(2)` and `SE(3)`.

Scenes are described in YAML *libraries* (the bodies, constraints, and
transitions that exist) and *tasks* (which constraints are active at the
start, plus the goal the planner must reach). The `comb` CLI runs the
load → validate → instantiate → plan → render pipeline end-to-end.

## Install

```bash
uv pip install -e ".[develop]"
```

## CLI walkthrough

The bundled examples live under `src/comb/examples/yaml/`. The pickup-and-place
demo points an arm + block library at a placement goal:

```bash
comb plan tests/spec_fixtures/example_pickup_place.task.yaml -o /tmp/plan.yaml
# planned 51 segments and 1 transitions over 2s using library example_two_link_arm_with_object.lib.yaml
# wrote plan to /tmp/plan.yaml
```

Render the saved plan as a GIF:

```bash
comb render /tmp/plan.yaml \
    --task tests/spec_fixtures/example_pickup_place.task.yaml \
    -o /tmp/plan.gif
# wrote 41 frames at 20 fps to /tmp/plan.gif
```

Validate inputs and outputs without producing artifacts:

```bash
comb validate library src/comb/examples/yaml/two_link_arm_with_object.lib.yaml
# src/comb/examples/yaml/two_link_arm_with_object.lib.yaml: ok

comb validate task tests/spec_fixtures/example_pickup_place.task.yaml
# tests/spec_fixtures/example_pickup_place.task.yaml: ok

comb validate plan /tmp/plan.yaml \
    --task tests/spec_fixtures/example_pickup_place.task.yaml
# /tmp/plan.yaml: ok
```

`comb plan` accepts `--horizon` and `--interval`; `comb render` accepts
`--fps`, `--dt`, and `--figsize`; `comb validate plan` accepts
`--tolerance`. `comb --help` and `comb <subcommand> --help` cover the
rest.

## Authoring a library

A library declares bodies, constraints, and transitions. Body geometry is
declarative (`shape: rectangle, width, height, offset_x, offset_y`),
constraint types match the Python class names, and transitions name a
trigger constraint plus a list of state-dependent generators that produce
new constraints when the trigger fires.

```yaml
# my_arm.lib.yaml
name: my_arm
bodies:
  base:
    visual_geometry: {shape: rectangle, width: 0.2, height: 0.2}
    collision_geometry: {shape: rectangle, width: 0.2, height: 0.2}
    pose: {x: 0.0, y: 0.0, theta: 0.0}
    anchored: true
  link:
    visual_geometry: {shape: rectangle, width: 0.5, height: 0.05, offset_x: 0.25}
    collision_geometry: {shape: rectangle, width: 0.5, height: 0.05, offset_x: 0.25}
    pose: {x: 0.0, y: 0.0, theta: 0.0}
constraints:
  joint:
    type: RevoluteJoint2D
    body1: base
    body2: link
    fixed_parameters: {origin_x: 0.0, origin_y: 0.0}
    initial_parameters: {angle: 0.0}
```

A task references the library and supplies a goal:

```yaml
# my_task.task.yaml
name: rotate_link
library: my_arm.lib.yaml
initial_mode: {}
goal:
  - type: PointEquality2D
    body1: base
    body2: link
    fixed_parameters: {target_x: 0.0, target_y: 0.5, offset_x: 0.5, offset_y: 0.0}
```

Larger setups can split across files via `includes:` in the library
header; `comb validate library` resolves the include graph and surfaces
collisions and cycles.

## Development

```bash
./run_ci_checks.sh   # black + isort + docformatter + mypy + pylint + pytest
```
