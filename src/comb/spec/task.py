"""Parsed-AST data types for a comb task YAML file.

A task is the PDDL-problem analog: it instantiates a library, declares the
mode that holds at t=0, and names the goal constraints the planner must
satisfy. The library reference is a path string here; resolving it against
the loaded :class:`LibrarySpec` is the validator's (B5) job.

As with libraries, the dataclasses mirror the YAML schema 1:1. The loader
(:mod:`comb.spec.load`) returns these types unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from comb.spec.library import ConstraintSpec, PoseSpec


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
