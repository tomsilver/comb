"""Matplotlib-backed interactive GUI for adjusting a ``System[SE2]``.

This GUI is 2D-only and pairs with the matplotlib 2D renderer in
``comb.rendering.matplotlib_2d``. A 3D GUI will likely use a different
library (meshcat, pyvista, ...) and live
in a sibling module (``comb/gui/<backend>_3d.py``); the two will not share
much beyond the high-level "slider drives the solver, then re-render" pattern.

The GUI builds one slider per mutable parameter across all constraints in the
system. On slider change it computes the delta from the current configuration,
calls ``solve``, applies the resulting configuration and body poses to the
system, and asks the renderer to redraw. Requires ``system.anchored_bodies``
to be non-empty (consistent with ``solve()``).
"""

from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from spatialmath import SE2

from comb.constraints import Constraint
from comb.rendering.matplotlib_2d import MatplotlibRenderer2D
from comb.solver import solve
from comb.system import System


class MatplotlibGUI2D:
    """Interactive matplotlib GUI for adjusting a ``System[SE2]``'s parameters."""

    def __init__(
        self,
        system: System[SE2],
        slider_range: tuple[float, float] = (-np.pi, np.pi),
    ) -> None:
        if not system.anchored_bodies:
            raise ValueError(
                "MatplotlibGUI2D requires system.anchored_bodies to be non-empty"
            )
        self.system = system
        self.slider_range = slider_range

        slider_specs: list[tuple[Constraint[SE2], int, str, float]] = []
        for constraint in system.constraints:
            for i, name in enumerate(constraint.parameter_names()):
                label = (
                    f"{type(constraint).__name__}"
                    f"({constraint.body1.name}→{constraint.body2.name}).{name}"
                )
                init = float(system.configuration[constraint].values[i])
                slider_specs.append((constraint, i, label, init))

        slider_height = 0.025
        slider_spacing = 0.012
        bottom_pad = 0.03
        slider_area = bottom_pad + len(slider_specs) * (slider_height + slider_spacing)

        self.figure = plt.figure()
        self.scene_ax = self.figure.add_axes(
            (0.1, slider_area + 0.05, 0.85, 0.92 - slider_area)
        )
        self.renderer = MatplotlibRenderer2D(ax=self.scene_ax)

        self.sliders: list[Slider] = []
        for i, (constraint, idx, label, init) in enumerate(slider_specs):
            slider_ax = self.figure.add_axes(
                (
                    0.25,
                    bottom_pad + i * (slider_height + slider_spacing),
                    0.6,
                    slider_height,
                )
            )
            slider = Slider(
                slider_ax,
                label,
                slider_range[0],
                slider_range[1],
                valinit=init,
            )
            slider.on_changed(self._make_callback(constraint, idx))
            self.sliders.append(slider)

        self.renderer.render(self.system)

    def _make_callback(
        self, constraint: Constraint[SE2], param_idx: int
    ) -> Callable[[float], None]:
        def _on_change(new_value: float) -> None:
            current = self.system.configuration[constraint].values
            delta = np.zeros_like(current)
            delta[param_idx] = new_value - current[param_idx]
            new_config, new_poses = solve(self.system, delta={constraint: delta})
            for c in self.system.constraints:
                if c.parameter_names():
                    self.system.configuration[c] = new_config[c]
            for body in self.system.bodies:
                self.system.body_poses[body] = new_poses[body]
            self.renderer.render(self.system)
            self.figure.canvas.draw_idle()

        return _on_change

    def show(self) -> None:
        """Block until the GUI window is closed."""
        plt.show()
