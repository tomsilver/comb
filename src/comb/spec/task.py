"""Parsed-AST data types for a comb task YAML file.

A task is the PDDL-problem analog: it instantiates a library, declares the
mode that holds at t=0, names the goal constraints the planner must
satisfy, and (optionally) declares the *plan granularity* — the maximum
body-twist any plan claiming to solve this task is allowed to traverse
between adjacent checkpoints. The library reference is a path string here;
resolving it against the loaded :class:`LibrarySpec` is the validator's
job.

As with libraries, the dataclasses mirror the YAML schema 1:1. The loader
(:mod:`comb.spec.load`) returns these types unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from comb.spec.library import ConstraintSpec, PoseSpec


@dataclass(frozen=True)
class GranularitySpec:
    """Plan-quality bounds the task imposes on any solving plan.

    ``max_segment_twist`` is the largest body-twist-norm distance any body
    is permitted to traverse between adjacent plan checkpoints. The
    :class:`comb.planners.SteppingPlanner` uses this directly as its
    ``interval``, and the plan validator rejects plans whose adjacent
    sample states violate it.
    """

    max_segment_twist: float


@dataclass(frozen=True)
class InitialModeSpec:
    """The mode active at the start of a task.

    ``active_constraints`` lists the subset of the library's constraint
    names that hold at t=0. ``None`` means "all library constraints" — the
    common case when transitions are the only mechanism for changing the
    constraint set. An empty tuple means "no constraints active" (rare but
    legal).

    ``body_poses`` overrides the library's default pose for any subset of
    bodies; bodies not listed inherit their library pose. ``configuration``
    provides initial values for mutable constraint parameters, keyed by
    constraint name.
    """

    active_constraints: tuple[str, ...] | None = None
    body_poses: Mapping[str, PoseSpec] = field(default_factory=dict)
    configuration: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """A task: instantiate a library, set up the initial mode, name a goal.

    ``library`` is a path string relative to the task file — resolved
    against the filesystem by the validator (B5), not by the loader.
    ``goal`` is a tuple of inline :class:`ConstraintSpec` records the
    planner must drive the system to satisfy.
    """

    name: str
    library: str
    initial_mode: InitialModeSpec
    goal: tuple[ConstraintSpec, ...] = ()
    granularity: GranularitySpec | None = None
