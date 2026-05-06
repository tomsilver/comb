"""Matplotlib-backed interactive GUI for adjusting a ``Mode[SE2]``.

This GUI is 2D-only and pairs with the matplotlib 2D renderer in
``comb.rendering.matplotlib_2d``. A 3D GUI will likely use a different
library (meshcat, pyvista, ...) and live in a sibling module
(``comb/gui/<backend>_3d.py``); the two will not share much beyond the
high-level "widget drives the solver, then re-render" pattern.

The GUI builds one widget per mutable parameter across all constraints in the
mode. Widget choice is dispatched on the parameter's
:class:`~comb.parameter_spaces.ParameterSpace`: a circular angle gets a
:class:`~comb.gui.widgets.CircularDial`, everything else gets a slider with
the space's preferred range. On widget change the GUI computes the
geodesic-aware delta from the current configuration, calls ``solve``, applies
the resulting configuration and body poses to the mode, and asks the
renderer to redraw. Requires ``mode.anchored_bodies`` to be non-empty.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from spatialmath import SE2

from comb.constraints import Constraint
from comb.gui.widgets import CircularDial
from comb.mode import Mode
from comb.parameter_spaces import Circle, ParameterSpace
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.solver import solve

ParameterWidget = Union[Slider, CircularDial]

_SLIDER_HEIGHT = 0.025
_DIAL_HEIGHT = 0.15
# Inter-widget spacing must be large enough to host the dial's value text,
# which renders just below the dial axes.
_WIDGET_SPACING = 0.045
_BOTTOM_PAD = 0.04
# Gap between the controls strip and the scene axes — needs to be large enough
# that a dial's title doesn't run into the scene's x-axis tick labels.
_SCENE_GAP_ABOVE_CONTROLS = 0.10
_SCENE_TOP = 0.95


class MatplotlibGUI2D:
    """Interactive matplotlib GUI for adjusting a ``Mode[SE2]``'s parameters."""

    def __init__(
        self,
        mode: Mode[SE2],
        slider_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        if not mode.anchored_bodies:
            raise ValueError(
                "MatplotlibGUI2D requires mode.anchored_bodies to be non-empty"
            )
        self.mode = mode
        self.slider_range = slider_range

        widget_specs: list[tuple[Constraint[SE2], int, str, float, ParameterSpace]] = []
        for constraint in mode.constraints:
            for i, name in enumerate(constraint.parameter_names()):
                label = f"{constraint.body1.name}→{constraint.body2.name}.{name}"
                init = float(mode.configuration[constraint].values[i])
                space = constraint.parameter_spaces[i]
                widget_specs.append((constraint, i, label, init, space))

        heights = [
            _DIAL_HEIGHT if isinstance(spec[4], Circle) else _SLIDER_HEIGHT
            for spec in widget_specs
        ]
        controls_height = sum(heights) + max(0, len(heights) - 1) * _WIDGET_SPACING

        self.figure = plt.figure()
        scene_bottom = _BOTTOM_PAD + controls_height + _SCENE_GAP_ABOVE_CONTROLS
        self.scene_ax = self.figure.add_axes(
            (0.1, scene_bottom, 0.85, _SCENE_TOP - scene_bottom)
        )
        self.renderer = MatplotlibRenderer2D(ax=self.scene_ax)

        self.widgets: list[ParameterWidget] = []
        # Stack widgets top-down so first parameter sits closest to the scene.
        y_top = _BOTTOM_PAD + controls_height
        for (constraint, idx, label, init, space), height in zip(widget_specs, heights):
            y_top -= height
            widget = self._make_widget(label, init, space, y_top, height)
            widget.on_changed(self._make_callback(constraint, idx))
            self.widgets.append(widget)
            y_top -= _WIDGET_SPACING

        self.renderer.render(self.mode)

    def _make_widget(
        self,
        label: str,
        init: float,
        space: ParameterSpace,
        y_bottom: float,
        height: float,
    ) -> ParameterWidget:
        if isinstance(space, Circle):
            ax = self.figure.add_axes((0.42, y_bottom, 0.16, height))
            return CircularDial(ax, label, valinit=init)
        ax = self.figure.add_axes((0.35, y_bottom, 0.5, height))
        lo, hi = space.preferred_range(self.slider_range)
        return Slider(ax, label, lo, hi, valinit=init)

    def _make_callback(
        self, constraint: Constraint[SE2], param_idx: int
    ) -> Callable[[float], None]:
        def _on_change(new_value: float) -> None:
            current = self.mode.configuration[constraint].values
            delta = np.zeros_like(current)
            # Use the parameter space's geodesic difference so e.g. dragging
            # a circular angle from near +π to near -π takes the short way.
            space = constraint.parameter_spaces[param_idx]
            delta[param_idx] = space.difference(new_value, float(current[param_idx]))
            new_config, new_poses = solve(self.mode, delta={constraint: delta})
            for c in self.mode.constraints:
                if c.parameter_names():
                    self.mode.configuration[c] = new_config[c]
            for body in self.mode.bodies:
                self.mode.body_poses[body] = new_poses[body]
            self.renderer.render(self.mode)
            self.figure.canvas.draw_idle()

        return _on_change

    def show(self) -> None:
        """Block until the GUI window is closed."""
        plt.show()
