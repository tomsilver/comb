"""Built-in renderer overlays.

Overlays add extra content on top of a system's main rendering. Each overlay
is generic in the pose type so it pairs with a matching ``Renderer[PoseT]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

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
