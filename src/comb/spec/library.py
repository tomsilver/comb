"""Parsed-AST data types for a comb library YAML file.

A library declares bodies, constraints, and transitions, optionally importing
others via ``includes``. The dataclasses here mirror the YAML schema 1:1 —
``LibrarySpec`` is what the loader (B2) returns, before the include resolver
(B3) merges and the validator (B5) checks names. None of these types
instantiate runtime ``Body`` / ``Constraint`` objects; that happens once the
library is fully resolved.

Bodies / constraints / transitions are keyed by name in their parent
mapping; their dataclass form intentionally omits ``name`` so the same
``ConstraintSpec`` can describe a top-level constraint *or* an inline
transition trigger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GeometrySpec:
    """A primitive shape descriptor.

    ``shape`` selects the kind (e.g. ``"rectangle"``, ``"box"``); ``parameters``
    holds shape-specific dimensions (e.g. ``{"width": 1.0, "height": 0.05}``).
    Validation of shape / parameter compatibility happens in the loader.
    """

    shape: str
    parameters: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseSpec:
    """An initial pose declared in YAML.

    Keys depend on the pose family: ``{"x", "y", "theta"}`` for SE(2),
    ``{"x", "y", "z", "qx", "qy", "qz", "qw"}`` for SE(3). The loader
    enforces the keyset; this dataclass is intentionally permissive so it
    can carry both families.
    """

    values: Mapping[str, float]


@dataclass(frozen=True)
class BodySpec:
    """A body declaration.

    Bodies live keyed by name in :class:`LibrarySpec.bodies`; the name is
    the parent-mapping key, not a field here. ``anchored`` mirrors
    :class:`comb.mode.Mode.anchored_bodies` membership.
    """

    visual_geometry: GeometrySpec
    collision_geometry: GeometrySpec
    pose: PoseSpec
    anchored: bool = False


@dataclass(frozen=True)
class ConstraintSpec:
    """A constraint declaration.

    ``type`` names a constraint class (e.g. ``"FixedJoint2D"``,
    ``"RevoluteJoint2D"``); ``body1`` / ``body2`` are body-name references
    resolved against :class:`LibrarySpec.bodies` at validation time.
    Reused unchanged for inline transition triggers.

    ``fixed_parameters`` corresponds to ``Constraint.fixed_parameters``
    (constants); ``initial_parameters`` provides starting values for
    ``Constraint.parameter_names()`` (mutable parameters that the solver
    can move).
    """

    type: str
    body1: str
    body2: str
    fixed_parameters: Mapping[str, float] = field(default_factory=dict)
    initial_parameters: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorCallSpec:
    """A reference to a registered generator with its call-site arguments.

    ``generator`` is a key in :data:`comb.generators.GENERATORS_2D`; ``args``
    holds positional/keyword arguments — body-name references and any extra
    configuration (e.g. ``body2_offset`` for ``point_pin_2d``).
    """

    generator: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionSpec:
    """A transition declaration.

    ``trigger`` is an inline :class:`ConstraintSpec` (unnamed). ``add`` is
    a sequence of generator invocations whose constraints will be inserted
    on apply; ``remove`` is a sequence of top-level constraint names to
    drop. Tolerance has the same meaning as
    :class:`comb.transitions.ConstraintTransition.tolerance`.
    """

    trigger: ConstraintSpec
    tolerance: float
    add: tuple[GeneratorCallSpec, ...] = ()
    remove: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibrarySpec:
    """A parsed library file.

    ``includes`` is a tuple of paths (as written in YAML, relative to the
    declaring file) this library composes with — resolution happens in the
    include linker (B3). ``bodies`` / ``constraints`` / ``transitions``
    are keyed by the names declared inline in the YAML.
    """

    name: str
    includes: tuple[str, ...] = ()
    bodies: Mapping[str, BodySpec] = field(default_factory=dict)
    constraints: Mapping[str, ConstraintSpec] = field(default_factory=dict)
    transitions: Mapping[str, TransitionSpec] = field(default_factory=dict)
