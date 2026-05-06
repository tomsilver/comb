"""Bodies: pose type, geometry primitives, the Body dataclass, and BodyPoses.

``Body`` and ``Geometry`` are both generic in the pose type, so a ``Body[SE3]``
must carry ``Geometry[SE3]`` (e.g. ``Box``) and a ``Body[SE2]`` must carry
``Geometry[SE2]`` (e.g. ``Rectangle``). Constraints (see ``comb.constraints``)
inherit the same parameterization.

``BodyPoses[PoseT]`` is a mutable map from Body to its current pose, parallel
to ``Configuration`` (which holds joint parameters). A ``Mode`` holds both,
and a solver updates them together.
"""

import abc
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar

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
    """A 2D axis-aligned rectangle with the given extents along x and y.

    The rectangle's center sits at ``(offset_x, offset_y)`` in the body's
    frame. Default ``(0, 0)`` means the rectangle is centered on the body's
    frame; non-zero values let a link's frame sit at its joint pivot while
    the rectangle visually extends away from it.
    """

    size_x: float
    size_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0


# Body uses identity-based equality and hashing so that distinct instances can
# serve as BodyPoses keys even when fields contain numpy arrays.
@dataclass(frozen=True, eq=False)
class Body(Generic[PoseT]):
    """A named rigid body with a pose plus visual and collision geometry.

    ``pose`` is the body's reference / initial pose. The mode's *current*
    pose for this body lives in ``BodyPoses`` and may be updated by a solver.
    """

    name: str
    pose: PoseT
    visual_geometry: Geometry[PoseT]
    collision_geometry: Geometry[PoseT]


class BodyPoses(Generic[PoseT]):
    """Current pose for each body in a mode.

    Acts like a mutable mapping from ``Body[PoseT]`` to ``PoseT``. Parallel to
    ``Configuration`` (which holds joint parameter values).
    """

    def __init__(self, poses: Mapping[Body[PoseT], PoseT] | None = None) -> None:
        self._poses: dict[Body[PoseT], PoseT] = {}
        if poses is not None:
            for body, pose in poses.items():
                self[body] = pose

    def __getitem__(self, body: Body[PoseT]) -> PoseT:
        return self._poses[body]

    def __setitem__(self, body: Body[PoseT], pose: PoseT) -> None:
        self._poses[body] = pose

    def __contains__(self, body: object) -> bool:
        return body in self._poses

    def __len__(self) -> int:
        return len(self._poses)

    def __iter__(self) -> Iterator[Body[PoseT]]:
        return iter(self._poses)


def interpolate_body_poses(
    start: BodyPoses[PoseT], end: BodyPoses[PoseT], s: float
) -> BodyPoses[PoseT]:
    """Per-body interpolation between two snapshots.

    Dispatches on pose type: ``SE2`` / ``SE3`` use ``interp`` (shortest twist
    path), ``np.ndarray`` is interpolated linearly. Both inputs must hold the
    same set of bodies (compared by identity, like ``BodyPoses`` itself).
    """
    start_ids = {id(b) for b in start}
    end_ids = {id(b) for b in end}
    if start_ids != end_ids:
        raise ValueError(
            "interpolate_body_poses requires matching body sets in start and end"
        )
    result: BodyPoses[PoseT] = BodyPoses()
    for body in start:
        result[body] = _interpolate_pose(start[body], end[body], s)
    return result


def _interpolate_pose(a: Any, b: Any, s: float) -> Any:
    if isinstance(a, SE2):
        return a.interp(b, s)
    if isinstance(a, SE3):
        return a.interp(b, s)
    if isinstance(a, np.ndarray):
        return a + s * (b - a)
    raise TypeError(f"Cannot interpolate pose of type {type(a).__name__}")
