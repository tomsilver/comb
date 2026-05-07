"""Matplotlib-backed interactive GUI for adjusting a ``System[SE2]``.

This GUI is 2D-only and pairs with the matplotlib 2D renderer in
``comb.rendering.matplotlib_2d``. A 3D GUI will likely use a different
library (meshcat, pyvista, ...) and live in a sibling module
(``comb/gui/<backend>_3d.py``); the two will not share much beyond the
high-level "widget drives the solver, then re-render" pattern.

The window is laid out in two columns: the scene fills the left half, and
all parameter widgets and transition buttons stack vertically in the right
column.

The GUI builds one widget per mutable parameter across all constraints in the
current mode, plus one button per ``ConstraintTransition`` in the system.
Each button highlights green when its trigger holds at the current state and
turns gray (and refuses clicks) otherwise. Clicking an enabled button applies
the transition, swaps the GUI's working mode for the result, and rebuilds the
widget set to match the new constraint topology.

Widget choice is dispatched on each parameter's
:class:`~comb.parameter_spaces.ParameterSpace`: a circular angle gets a
:class:`~comb.gui.widgets.CircularDial`, everything else gets a slider with
the space's preferred range. On widget change the GUI computes the
geodesic-aware delta from the current configuration, calls ``solve``, applies
the resulting state, asks the renderer to redraw, and refreshes the
transition buttons' enabled states. Requires ``mode.anchored_bodies`` to be
non-empty.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button, Slider
from spatialmath import SE2

from comb.constraints import Constraint
from comb.gui.widgets import CircularDial
from comb.mode import Mode
from comb.parameter_spaces import Circle, ParameterSpace
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.solver import solve
from comb.system import System
from comb.transitions import ConstraintTransition

ParameterWidget = Union[Slider, CircularDial]

# Wider-than-tall figure so the scene and the controls column can sit side by side.
_FIGSIZE = (12.0, 6.0)

# Scene is the left column; controls fill the right.
_SCENE_LEFT = 0.05
_SCENE_RIGHT = 0.50
_SCENE_BOTTOM = 0.08
_SCENE_TOP = 0.95
_CONTROLS_LEFT = 0.58
_CONTROLS_RIGHT = 0.95
_CONTROLS_TOP = 0.95
_CONTROLS_WIDTH = _CONTROLS_RIGHT - _CONTROLS_LEFT

_SLIDER_HEIGHT = 0.025
_SLIDER_X = _CONTROLS_LEFT + 0.07  # leave room for the label on the slider's left
_SLIDER_WIDTH = _CONTROLS_RIGHT - _SLIDER_X

_DIAL_HEIGHT = 0.15
_DIAL_WIDTH = 0.20  # axes box wider than the dial so the label has room
_DIAL_X = _CONTROLS_LEFT + (_CONTROLS_WIDTH - _DIAL_WIDTH) / 2

_BUTTON_HEIGHT = 0.04
_BUTTON_X = _CONTROLS_LEFT
_BUTTON_WIDTH = _CONTROLS_WIDTH

# Inter-widget spacing must be large enough to host the dial's value text,
# which renders just below the dial axes.
_WIDGET_SPACING = 0.045
_BUTTON_SPACING = 0.015

_BUTTON_ENABLED_COLOR = "tab:green"
_BUTTON_DISABLED_COLOR = "lightgray"


class MatplotlibGUI2D:
    """Interactive matplotlib GUI for adjusting a ``System[SE2]``.

    Tracks the current mode internally (initially ``system.mode``); applying a
    transition rebinds the working mode to the result of
    ``transition.apply(...)`` and rebuilds the widget set.
    """

    def __init__(
        self,
        system: System[SE2],
        slider_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        if not system.mode.anchored_bodies:
            raise ValueError(
                "MatplotlibGUI2D requires mode.anchored_bodies to be non-empty"
            )
        self.system = system
        self.mode: Mode[SE2] = system.mode
        self.slider_range = slider_range

        self.figure = plt.figure(figsize=_FIGSIZE)
        self.scene_ax = self.figure.add_axes(
            (
                _SCENE_LEFT,
                _SCENE_BOTTOM,
                _SCENE_RIGHT - _SCENE_LEFT,
                _SCENE_TOP - _SCENE_BOTTOM,
            )
        )
        self.renderer = MatplotlibRenderer2D(ax=self.scene_ax)

        self.widgets: list[ParameterWidget] = []
        self._widget_axes: list[Axes] = []
        self.transition_buttons: list[Button] = []
        self._button_axes: list[Axes] = []

        self._build_controls()
        self.renderer.render(self.mode)

    # ----- layout -----

    def _build_controls(self) -> None:
        widget_specs = self._widget_specs()
        widget_heights = [
            _DIAL_HEIGHT if isinstance(spec[4], Circle) else _SLIDER_HEIGHT
            for spec in widget_specs
        ]

        # Stack from the top of the right column downward: parameter widgets
        # first, then transition buttons below them.
        y_top = _CONTROLS_TOP
        for (constraint, idx, label, init, space), height in zip(
            widget_specs, widget_heights
        ):
            y_top -= height
            ax, widget = self._make_widget(label, init, space, y_top, height)
            widget.on_changed(self._make_widget_callback(constraint, idx))
            self.widgets.append(widget)
            self._widget_axes.append(ax)
            y_top -= _WIDGET_SPACING
        if widget_specs and self.system.transitions:
            y_top -= _WIDGET_SPACING - _BUTTON_SPACING
        for i, transition in enumerate(self.system.transitions):
            y_top -= _BUTTON_HEIGHT
            ax, button = self._make_transition_button(transition, i, y_top)
            self.transition_buttons.append(button)
            self._button_axes.append(ax)
            y_top -= _BUTTON_SPACING
        self._refresh_transition_buttons()

    def _clear_controls(self) -> None:
        for ax in self._widget_axes:
            self.figure.delaxes(ax)
        for ax in self._button_axes:
            self.figure.delaxes(ax)
        self.widgets.clear()
        self._widget_axes.clear()
        self.transition_buttons.clear()
        self._button_axes.clear()

    def _rebuild_controls(self) -> None:
        self._clear_controls()
        self._build_controls()
        self.figure.canvas.draw_idle()

    def _widget_specs(
        self,
    ) -> list[tuple[Constraint[SE2], int, str, float, ParameterSpace]]:
        specs: list[tuple[Constraint[SE2], int, str, float, ParameterSpace]] = []
        for constraint in self.mode.constraints:
            for i, name in enumerate(constraint.parameter_names()):
                label = f"{constraint.body1.name}→{constraint.body2.name}.{name}"
                init = float(self.mode.configuration[constraint].values[i])
                space = constraint.parameter_spaces[i]
                specs.append((constraint, i, label, init, space))
        return specs

    def _make_widget(
        self,
        label: str,
        init: float,
        space: ParameterSpace,
        y_bottom: float,
        height: float,
    ) -> tuple[Axes, ParameterWidget]:
        if isinstance(space, Circle):
            ax = self.figure.add_axes((_DIAL_X, y_bottom, _DIAL_WIDTH, height))
            return ax, CircularDial(ax, label, valinit=init)
        ax = self.figure.add_axes((_SLIDER_X, y_bottom, _SLIDER_WIDTH, height))
        lo, hi = space.preferred_range(self.slider_range)
        return ax, Slider(ax, label, lo, hi, valinit=init)

    def _make_transition_button(
        self,
        transition: ConstraintTransition[SE2],
        index: int,
        y_bottom: float,
    ) -> tuple[Axes, Button]:
        label = f"#{index}: apply {type(transition).__name__}"
        ax = self.figure.add_axes((_BUTTON_X, y_bottom, _BUTTON_WIDTH, _BUTTON_HEIGHT))
        button = Button(ax, label, color=_BUTTON_DISABLED_COLOR)
        button.on_clicked(self._make_transition_callback(index))
        return ax, button

    # ----- callbacks -----

    def _make_widget_callback(
        self, constraint: Constraint[SE2], param_idx: int
    ) -> Callable[[float], None]:
        def _on_change(new_value: float) -> None:
            current = self.mode.configuration[constraint].values
            delta = np.zeros_like(current)
            # Use the parameter space's geodesic difference so e.g. dragging
            # a circular angle from near +π to near -π takes the short way.
            space = constraint.parameter_spaces[param_idx]
            delta[param_idx] = space.difference(new_value, float(current[param_idx]))
            new_state = solve(self.mode, delta={constraint: delta})
            for c in self.mode.constraints:
                if c.parameter_names():
                    self.mode.configuration[c] = new_state.configuration[c]
            for body in self.mode.bodies:
                self.mode.body_poses[body] = new_state.body_poses[body]
            self.renderer.render(self.mode)
            self._refresh_transition_buttons()
            self.figure.canvas.draw_idle()

        return _on_change

    def _make_transition_callback(self, index: int) -> Callable[[object], None]:
        def _on_click(_event: object) -> None:
            transition = self.system.transitions[index]
            if not self._is_transition_applicable(transition):
                return
            self.mode = transition.apply(self.mode, self.mode.snapshot())
            self._rebuild_controls()
            self.renderer.render(self.mode)
            self.figure.canvas.draw_idle()

        return _on_click

    def _refresh_transition_buttons(self) -> None:
        for transition, button in zip(self.system.transitions, self.transition_buttons):
            color = (
                _BUTTON_ENABLED_COLOR
                if self._is_transition_applicable(transition)
                else _BUTTON_DISABLED_COLOR
            )
            button.color = color
            button.ax.set_facecolor(color)

    def _is_transition_applicable(self, transition: ConstraintTransition[SE2]) -> bool:
        state = self.mode.snapshot()
        if not transition.is_enabled(state):
            return False
        # Transition's `remove` references must still be in the current mode.
        mode_ids = {id(c) for c in self.mode.constraints}
        return all(id(c) in mode_ids for c in transition.remove)

    def show(self) -> None:
        """Block until the GUI window is closed."""
        plt.show()
