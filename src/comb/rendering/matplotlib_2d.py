"""Matplotlib-backed 2D renderer for ``System[SE2]``.

Adds support for new geometry types by registering more methods on
``MatplotlibRenderer2D._draw`` via ``singledispatchmethod``.

The first call to :meth:`render` locks the axes limits so the view doesn't
jitter as the user varies parameters. The default limit heuristic is a square
centered on the anchored bodies whose half-extent is the sum of all joint
origin offsets plus the largest geometry circumscribed radius. If you need
different limits, pass ``xlim`` / ``ylim`` to the constructor.
"""

from functools import singledispatchmethod

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from spatialmath import SE2

from comb.bodies import Geometry, Rectangle
from comb.constraints import Constraint
from comb.rendering.base import Renderer
from comb.system import System

_ANCHOR_COLOR = "tab:gray"
_BODY_COLOR = "tab:blue"

# Names in fixed_parameters that contribute to a constraint's
# parent->joint translation in the plane.
_X_OFFSET_NAMES = ("origin_x", "tx")
_Y_OFFSET_NAMES = ("origin_y", "ty")


class MatplotlibRenderer2D(Renderer[SE2]):
    """Draw an ``System[SE2]`` into a matplotlib ``Axes``.

    Anchored bodies are drawn in a different color so the user can see what's
    pinned. Axes limits are computed on the first render and reused
    afterwards; pass ``xlim``/``ylim`` to override.
    """

    def __init__(
        self,
        ax: Axes | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self._xlim = xlim
        self._ylim = ylim

    def render(self, system: System[SE2]) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal")
        anchored = {id(b) for b in system.anchored_bodies}
        for body in system.bodies:
            color = _ANCHOR_COLOR if id(body) in anchored else _BODY_COLOR
            self._draw(body.visual_geometry, system.body_poses[body], color)
        if self._xlim is None or self._ylim is None:
            xlim, ylim = self._estimate_bounds(system)
            if self._xlim is None:
                self._xlim = xlim
            if self._ylim is None:
                self._ylim = ylim
        self.ax.set_xlim(self._xlim)
        self.ax.set_ylim(self._ylim)

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

    def _estimate_bounds(
        self, system: System[SE2]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Square bounds centered on the anchors, sized to cover the workspace."""
        radius = self._estimate_workspace_radius(system)
        if system.anchored_bodies:
            anchor_ts = [
                np.asarray(system.body_poses[b].t) for b in system.anchored_bodies
            ]
            center = np.mean(anchor_ts, axis=0)
        elif system.bodies:
            center = np.asarray(system.body_poses[system.bodies[0]].t)
        else:
            center = np.array([0.0, 0.0])
        half = radius * 1.1  # 10% extra padding
        return (
            (float(center[0] - half), float(center[0] + half)),
            (float(center[1] - half), float(center[1] + half)),
        )

    def _estimate_workspace_radius(self, system: System[SE2]) -> float:
        max_geom = max(
            (self._geometry_radius(b.visual_geometry) for b in system.bodies),
            default=0.5,
        )
        origin_sum = sum(self._constraint_offset(c) for c in system.constraints)
        # Floor at max_geom so a system of one body still gets a sensible window.
        return max(origin_sum + max_geom, max_geom)

    @staticmethod
    def _constraint_offset(constraint: Constraint[SE2]) -> float:
        fp = constraint.fixed_parameters
        x = sum(fp[n] for n in _X_OFFSET_NAMES if n in fp.names)
        y = sum(fp[n] for n in _Y_OFFSET_NAMES if n in fp.names)
        return float(np.hypot(x, y))

    @singledispatchmethod
    def _geometry_radius(
        self, geometry: Geometry[SE2]
    ) -> float:  # pylint: disable=unused-argument
        return 0.5  # generic fallback for unknown shapes

    @_geometry_radius.register
    def _rectangle_radius(self, geometry: Rectangle) -> float:
        return float(np.hypot(geometry.size_x / 2, geometry.size_y / 2))
