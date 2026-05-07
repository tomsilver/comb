"""Domain / problem specification language for comb scenes.

The spec language is a YAML-based DSL inspired by PDDL (domain vs. problem
split) and URDF (composable body / constraint declarations). A *library*
declares bodies, constraints, and transitions; a *task* (later) instantiates
a library and adds an initial mode plus goal constraints.

This module currently holds the parsed-AST data types (:mod:`comb.spec.library`).
The YAML loader, include resolver, task loader, and spec-level validator
land in subsequent PRs.
"""

from comb.spec.library import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    LibrarySpec,
    PoseSpec,
    TransitionSpec,
)

__all__ = [
    "BodySpec",
    "ConstraintSpec",
    "GeneratorCallSpec",
    "GeometrySpec",
    "LibrarySpec",
    "PoseSpec",
    "TransitionSpec",
]
