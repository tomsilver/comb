"""YAML round-trip for :class:`comb.planners.Plan` (SE(2) only).

The on-disk format is a single mapping ``{plan: [...]}`` whose value is an
ordered list of records. Each record is either a *state* — a snapshot at a
checkpoint time — or an *event* — a transition firing. Records sort by time;
events come before states at the same time, matching the convention from
A1 that the post-transition state shares its time with the event::

    plan:
      - {t: 0.0,  body_poses: {...}, configuration: {...}}
      - {t: 0.05, body_poses: {...}, configuration: {...}}
      ...
      - {t: 0.50, transition: pickup}
      - {t: 0.50, body_poses: {...}, configuration: {...}}
      ...

Round-trip requires the runtime ``Body`` / ``Constraint`` /
``ConstraintTransition`` instances on both sides — they're identity-keyed
inside :class:`Plan` and the YAML uses spec-language names instead. The
serializer takes object→name mappings; the loader takes name→object
mappings (typically produced by a future spec instantiator).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from spatialmath import SE2

from comb.bodies import Body, BodyPoses
from comb.constraints import Constraint, ConstraintConfiguration, ConstraintParameters
from comb.mode import ModeState, interpolate_mode_state
from comb.planners import Plan, TransitionEvent
from comb.trajectories import concatenate, constant, linear_segment
from comb.transitions import ConstraintTransition


class PlanSerializationError(Exception):
    """Raised when a plan cannot be serialized or deserialized."""


def plan_to_yaml(
    plan: Plan[SE2],
    *,
    body_names: Mapping[Body[SE2], str],
    constraint_names: Mapping[Constraint[SE2], str],
    transition_names: Mapping[ConstraintTransition[SE2], str],
) -> str:
    """Serialize a 2D :class:`Plan` to a YAML string.

    Maps each runtime object to the name passed in the corresponding
    mapping. Constraints not present in ``constraint_names`` are an error
    (we don't want silently-dropped configuration entries); the same goes
    for bodies and transitions.
    """
    records: list[dict[str, Any]] = []

    # State records at each sample time.
    for t in plan.sample_times:
        state = plan.trajectory(t)
        records.append(
            {
                "t": float(t),
                "rank": _STATE_RANK,
                "data": _state_to_dict(
                    state, body_names=body_names, constraint_names=constraint_names
                ),
            }
        )

    # Event records.
    for event in plan.events:
        if event.transition not in transition_names:
            raise PlanSerializationError(
                "transition fired at t="
                f"{event.time:g} has no entry in transition_names"
            )
        records.append(
            {
                "t": float(event.time),
                "rank": _EVENT_RANK,
                "data": {"transition": transition_names[event.transition]},
            }
        )

    # Stable sort: events before states at the same time.
    records.sort(key=lambda r: (r["t"], r["rank"]))
    output = [{"t": r["t"], **r["data"]} for r in records]
    duration = float(plan.trajectory.duration)
    return yaml.safe_dump({"duration": duration, "plan": output}, sort_keys=False)


def plan_from_yaml(
    text: str,
    *,
    bodies: Mapping[str, Body[SE2]],
    constraints: Mapping[str, Constraint[SE2]],
    transitions: Mapping[str, ConstraintTransition[SE2]],
) -> Plan[SE2]:
    """Deserialize a YAML plan string, resolving names against runtime objects."""
    data = yaml.safe_load(text)
    if not isinstance(data, Mapping) or "plan" not in data:
        raise PlanSerializationError("expected top-level mapping with key 'plan'")
    raw_records = data["plan"]
    if not isinstance(raw_records, list):
        raise PlanSerializationError(
            f"'plan' must be a list, got {type(raw_records).__name__}"
        )

    states: list[tuple[float, ModeState[SE2]]] = []
    events: list[TransitionEvent[SE2]] = []
    for i, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise PlanSerializationError(f"plan[{i}]: expected mapping")
        if "t" not in record:
            raise PlanSerializationError(f"plan[{i}]: missing required key 't'")
        t = float(record["t"])
        if "transition" in record:
            transition_name = record["transition"]
            if transition_name not in transitions:
                raise PlanSerializationError(
                    f"plan[{i}]: unknown transition {transition_name!r}"
                )
            events.append(
                TransitionEvent(time=t, transition=transitions[transition_name])
            )
        else:
            states.append(
                (
                    t,
                    _state_from_dict(
                        record,
                        bodies=bodies,
                        constraints=constraints,
                        source=f"plan[{i}]",
                    ),
                )
            )

    if not states:
        raise PlanSerializationError("plan must contain at least one state record")

    sample_times = tuple(t for t, _ in states)
    duration_raw = data.get("duration")
    if duration_raw is None:
        duration = sample_times[-1]
    else:
        duration = float(duration_raw)
    trajectory = _build_trajectory(states, duration=duration)
    return Plan(
        trajectory=trajectory,
        events=tuple(events),
        sample_times=sample_times,
    )


def plan_to_yaml_file(
    path: str | Path,
    plan: Plan[SE2],
    *,
    body_names: Mapping[Body[SE2], str],
    constraint_names: Mapping[Constraint[SE2], str],
    transition_names: Mapping[ConstraintTransition[SE2], str],
) -> None:
    """Convenience wrapper: serialize a plan and write to ``path``."""
    Path(path).write_text(
        plan_to_yaml(
            plan,
            body_names=body_names,
            constraint_names=constraint_names,
            transition_names=transition_names,
        ),
        encoding="utf-8",
    )


def plan_from_yaml_file(
    path: str | Path,
    *,
    bodies: Mapping[str, Body[SE2]],
    constraints: Mapping[str, Constraint[SE2]],
    transitions: Mapping[str, ConstraintTransition[SE2]],
) -> Plan[SE2]:
    """Convenience wrapper: read ``path`` and deserialize the plan inside."""
    text = Path(path).read_text(encoding="utf-8")
    return plan_from_yaml(
        text, bodies=bodies, constraints=constraints, transitions=transitions
    )


# --- helpers ---


_EVENT_RANK = 0
_STATE_RANK = 1


def _state_to_dict(
    state: ModeState[SE2],
    *,
    body_names: Mapping[Body[SE2], str],
    constraint_names: Mapping[Constraint[SE2], str],
) -> dict[str, Any]:
    poses_out: dict[str, dict[str, float]] = {}
    for body in state.body_poses:
        if body not in body_names:
            raise PlanSerializationError(
                f"body {body.name!r} has no entry in body_names"
            )
        poses_out[body_names[body]] = _se2_to_dict(state.body_poses[body])

    config_out: dict[str, dict[str, float]] = {}
    for constraint in state.configuration:
        if constraint not in constraint_names:
            raise PlanSerializationError(
                f"constraint {type(constraint).__name__} between "
                f"{constraint.body1.name!r} and {constraint.body2.name!r} has "
                "no entry in constraint_names"
            )
        params = state.configuration[constraint]
        config_out[constraint_names[constraint]] = {
            name: float(value) for name, value in zip(params.names, params.values)
        }

    return {"body_poses": poses_out, "configuration": config_out}


def _state_from_dict(
    record: Mapping[str, Any],
    *,
    bodies: Mapping[str, Body[SE2]],
    constraints: Mapping[str, Constraint[SE2]],
    source: str,
) -> ModeState[SE2]:
    poses_raw = record.get("body_poses")
    if not isinstance(poses_raw, Mapping):
        raise PlanSerializationError(
            f"{source}.body_poses: expected mapping, got " f"{type(poses_raw).__name__}"
        )
    body_poses = BodyPoses[SE2](
        {
            _resolve_body(name, bodies, source=f"{source}.body_poses"): _se2_from_dict(
                pose, source=f"{source}.body_poses.{name}"
            )
            for name, pose in poses_raw.items()
        }
    )

    config_raw = record.get("configuration", {})
    if not isinstance(config_raw, Mapping):
        raise PlanSerializationError(
            f"{source}.configuration: expected mapping, got "
            f"{type(config_raw).__name__}"
        )
    config = ConstraintConfiguration()
    for cname, params_raw in config_raw.items():
        constraint = _resolve_constraint(
            cname, constraints, source=f"{source}.configuration"
        )
        if not isinstance(params_raw, Mapping):
            raise PlanSerializationError(
                f"{source}.configuration.{cname}: expected mapping, got "
                f"{type(params_raw).__name__}"
            )
        names = constraint.parameter_names()
        try:
            values = np.array([float(params_raw[n]) for n in names])
        except KeyError as exc:
            raise PlanSerializationError(
                f"{source}.configuration.{cname}: missing parameter "
                f"{exc.args[0]!r}; expected {list(names)}"
            ) from exc
        config[constraint] = ConstraintParameters(values=values, names=names)

    return ModeState(configuration=config, body_poses=body_poses)


def _build_trajectory(
    states: list[tuple[float, ModeState[SE2]]], *, duration: float
) -> Any:
    if len(states) == 1:
        return constant(states[0][1], duration)
    segments = []
    for (t0, s0), (t1, s1) in zip(states, states[1:]):
        if t1 < t0:
            raise PlanSerializationError(
                f"plan times must be non-decreasing, got {t0:g} then {t1:g}"
            )
        segments.append(
            linear_segment(
                s0, s1, duration=float(t1 - t0), interpolate=interpolate_mode_state
            )
        )
    return concatenate(segments)


def _resolve_body(
    name: Any, bodies: Mapping[str, Body[SE2]], *, source: str
) -> Body[SE2]:
    if not isinstance(name, str):
        raise PlanSerializationError(
            f"{source}: body name must be a string, got {type(name).__name__}"
        )
    if name not in bodies:
        raise PlanSerializationError(f"{source}: unknown body {name!r}")
    return bodies[name]


def _resolve_constraint(
    name: Any, constraints: Mapping[str, Constraint[SE2]], *, source: str
) -> Constraint[SE2]:
    if not isinstance(name, str):
        raise PlanSerializationError(
            f"{source}: constraint name must be a string, got " f"{type(name).__name__}"
        )
    if name not in constraints:
        raise PlanSerializationError(f"{source}: unknown constraint {name!r}")
    return constraints[name]


def _se2_to_dict(pose: SE2) -> dict[str, float]:
    return {
        "x": float(pose.t[0]),
        "y": float(pose.t[1]),
        "theta": float(pose.theta()),
    }


def _se2_from_dict(value: Any, *, source: str) -> SE2:
    if not isinstance(value, Mapping):
        raise PlanSerializationError(
            f"{source}: expected pose mapping, got {type(value).__name__}"
        )
    expected = {"x", "y", "theta"}
    keys = set(value)
    if keys != expected:
        raise PlanSerializationError(
            f"{source}: expected SE(2) keys {sorted(expected)}, got {sorted(keys)}"
        )
    return SE2(float(value["x"]), float(value["y"]), float(value["theta"]))
