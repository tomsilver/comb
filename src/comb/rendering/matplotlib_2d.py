"""Matplotlib-backed 2D renderer for ``System[SE2]``.

Adds support for new geometry types by registering more methods on
``MatplotlibRenderer2D._draw`` via ``singledispatchmethod``.
"""

from functools import singledispatchmethod

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from spatialmath import SE2

from comb.bodies import Geometry, Rectangle
from comb.rendering.base import Renderer
from comb.system import System

_ANCHOR_COLOR = "tab:gray"
_BODY_COLOR = "tab:blue"


class MatplotlibRenderer2D(Renderer[SE2]):
    """Draw an ``System[SE2]`` into a matplotlib ``Axes``.

    If no axes is provided a new figure is created on construction. Anchored
    bodies are drawn in a different color so the user can see what's pinned.
    """

    def __init__(self, ax: Axes | None = None) -> None:
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax

    def render(self, system: System[SE2]) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal")
        anchored = {id(b) for b in system.anchored_bodies}
        for body in system.bodies:
            color = _ANCHOR_COLOR if id(body) in anchored else _BODY_COLOR
            self._draw(body.visual_geometry, system.body_poses[body], color)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.margins(0.1)

    @singledispatchmethod
    def _draw(
        self, geometry: Geometry[SE2], pose: SE2, color: str
    ) -> None:  # pylint: disable=unused-argument
        raise NotImplementedError(
            f"MatplotlibRenderer2D has no drawing for {type(geometry).__name__}; "
            f"register one with @MatplotlibRenderer2D._draw.register"
        )

    @_draw.register
    def _draw_rectangle(self, geometry: Rectangle, pose: SE2, color: str) -> None:
        sx, sy = geometry.size_x, geometry.size_y
        corners_body = np.array(
            [
                [-sx / 2, -sy / 2],
                [+sx / 2, -sy / 2],
                [+sx / 2, +sy / 2],
                [-sx / 2, +sy / 2],
            ]
        )
        theta = float(pose.theta())
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners_world = corners_body @ rotation.T + np.asarray(pose.t)
        polygon = patches.Polygon(
            corners_world,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=1.0,
        )
        self.ax.add_patch(polygon)
