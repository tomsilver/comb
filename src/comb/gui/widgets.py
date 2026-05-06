"""Custom matplotlib widgets used by the GUI.

Right now this module contains :class:`CircularDial`, a click-and-drag dial
that selects an angle in ``[-π, π]``. It mirrors enough of
``matplotlib.widgets.Slider``'s API (``val``, ``set_val``, ``on_changed``)
that the GUI can treat sliders and dials interchangeably.
"""

import math
from collections.abc import Callable

from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


class CircularDial:
    """A click-and-drag circular dial that selects an angle.

    Click anywhere inside the dial to set the angle, or click and drag the
    handle around the rim. Calls registered observers with the new angle.
    """

    def __init__(self, ax: Axes, label: str, valinit: float = 0.0) -> None:
        self.ax = ax
        self._val = float(valinit)
        self._observers: list[Callable[[float], None]] = []
        self._dragging = False

        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=9, pad=2)

        self._outline = patches.Circle(
            (0, 0), 1.0, fill=False, edgecolor="black", linewidth=1.0
        )
        ax.add_patch(self._outline)

        x, y = math.cos(self._val), math.sin(self._val)
        (self._needle,) = ax.plot([0, x], [0, y], color="tab:blue", linewidth=2)
        (self._tip,) = ax.plot([x], [y], "o", color="tab:blue", markersize=6)
        self._valtext = ax.text(
            0.5, -0.18, f"{self._val:.2f}", transform=ax.transAxes, ha="center"
        )

        canvas = ax.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

    @property
    def val(self) -> float:
        return self._val

    def set_val(self, val: float) -> None:
        """Set the dial's value and fire all registered observers."""
        self._val = float(val)
        self._update_visuals()
        for callback in self._observers:
            callback(self._val)

    def on_changed(self, callback: Callable[[float], None]) -> int:
        """Register a callback fired with the new value whenever it changes."""
        self._observers.append(callback)
        return len(self._observers) - 1

    def _update_visuals(self) -> None:
        x, y = math.cos(self._val), math.sin(self._val)
        self._needle.set_data([0, x], [0, y])
        self._tip.set_data([x], [y])
        self._valtext.set_text(f"{self._val:.2f}")
        self.ax.figure.canvas.draw_idle()

    def _on_press(self, event: MouseEvent) -> None:
        if event.inaxes is self.ax and event.button == 1:
            self._dragging = True
            self._update_from_event(event)

    def _on_motion(self, event: MouseEvent) -> None:
        if self._dragging and event.inaxes is self.ax:
            self._update_from_event(event)

    def _on_release(self, event: MouseEvent) -> None:
        del event
        self._dragging = False

    def _update_from_event(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.xdata == 0.0 and event.ydata == 0.0:
            return
        self.set_val(math.atan2(event.ydata, event.xdata))
