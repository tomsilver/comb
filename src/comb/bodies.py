"""Bodies: pose type, geometry primitives, and the Body dataclass."""

import abc
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from spatialmath import SE2, SE3

# A pose lives in some configuration space. SE2/SE3 cover planar and 3D rigid
# bodies; np.ndarray covers translation-only spaces like R^1, R^2, R^3.
Pose: TypeAlias = SE2 | SE3 | np.ndarray


class Geometry(abc.ABC):
    """Visual or collision geometry. Concrete subclasses are primitives or meshes."""


@dataclass(frozen=True)
class Sphere(Geometry):
    """A sphere with the given radius."""

    radius: float


@dataclass(frozen=True)
class Box(Geometry):
    """An axis-aligned box with the given extents along x, y, and z."""

    size_x: float
    size_y: float
    size_z: float


@dataclass(frozen=True)
class Cylinder(Geometry):
    """A cylinder with the given radius and height (along the local z-axis)."""

    radius: float
    height: float


@dataclass(frozen=True)
class Mesh(Geometry):
    """A triangle mesh: vertices of shape (V, 3) and integer faces of shape (F, 3)."""

    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class Body:
    """A named rigid body with a pose plus visual and collision geometry."""

    name: str
    pose: Pose
    visual_geometry: Geometry
    collision_geometry: Geometry
