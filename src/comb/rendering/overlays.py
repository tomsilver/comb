"""Built-in renderer overlays.

Overlays add extra content on top of a system's main rendering. Each overlay
is generic in the pose type so it pairs with a matching ``Renderer[PoseT]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from spatialmath import SE2

from comb.bodies import Body, BodyPoses, PoseT
from comb.rendering.base import Overlay, Renderer


@dataclass(frozen=True)
class GhostBodies(Overlay[PoseT], Generic[PoseT]):
    """Faded rendering of a set of bodies at given poses.

    Useful for visualizing goal states, prior states, or any "where this
    could be" overlay alongside the live system.
    """

    bodies: list[Body[PoseT]]
    body_poses: BodyPoses[PoseT]
    color: str = "tab:orange"
    alpha: float = 0.3

    def draw(self, renderer: Renderer[PoseT]) -> None:
        for body in self.bodies:
            if body in self.body_poses:
                renderer.draw_body(
                    body,
                    self.body_poses[body],
                    color=self.color,
                    alpha=self.alpha,
                )


@dataclass(frozen=True)
class PointMarker2D(Overlay[SE2]):
    """A 2D world-point marker (star, circle, ...) drawn over the system.

    Useful for visualizing target positions, waypoints, or any single point
    of interest. Currently requires a renderer that exposes a matplotlib
    ``ax`` attribute (i.e. ``MatplotlibRenderer2D``).
    """

    x: float
    y: float
    marker: str = "*"
    color: str = "tab:orange"
    size: float = 200.0
    edgecolor: str = "black"

    def draw(self, renderer: Renderer[SE2]) -> None:
        ax = getattr(renderer, "ax", None)
        if ax is None:
            raise NotImplementedError(
                "PointMarker2D requires a renderer with an ``ax`` attribute; "
                f"got {type(renderer).__name__}"
            )
        ax.scatter(
            [self.x],
            [self.y],
            marker=self.marker,
            c=self.color,
            s=self.size,
            edgecolors=self.edgecolor,
            zorder=10,
        )
