"""Bodies: pose type, geometry primitives, and the Body dataclass.

``Body`` and ``Geometry`` are both generic in the pose type, so a ``Body[SE3]``
must carry ``Geometry[SE3]`` (e.g. ``Box``) and a ``Body[SE2]`` must carry
``Geometry[SE2]`` (e.g. ``Rectangle``). Constraints (see ``comb.constraints``)
inherit the same parameterization.
"""

import abc
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import numpy as np
from spatialmath import SE2, SE3

# Conventional pose types. Body, Geometry, and Constraint are generic in PoseT,
# so any type can be used in principle; this alias names the common cases.
Pose: TypeAlias = SE2 | SE3 | np.ndarray

PoseT = TypeVar("PoseT")


class Geometry(abc.ABC, Generic[PoseT]):
    """Visual or collision geometry, tagged by the pose space it lives in."""


@dataclass(frozen=True)
class Box(Geometry[SE3]):
    """A 3D axis-aligned box with the given extents along x, y, and z."""

    size_x: float
    size_y: float
    size_z: float


@dataclass(frozen=True)
class Rectangle(Geometry[SE2]):
    """A 2D axis-aligned rectangle with the given extents along x and y."""

    size_x: float
    size_y: float


@dataclass(frozen=True)
class Body(Generic[PoseT]):
    """A named rigid body with a pose plus visual and collision geometry."""

    name: str
    pose: PoseT
    visual_geometry: Geometry[PoseT]
    collision_geometry: Geometry[PoseT]
