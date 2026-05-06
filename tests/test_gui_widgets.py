"""Tests for custom GUI widgets (CircularDial)."""

import math

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backend_bases import MouseButton, MouseEvent  # noqa: E402

from comb.gui.widgets import CircularDial  # noqa: E402


def _click_at(
    ax: Axes,
    xdata: float,
    ydata: float,
    button: MouseButton = MouseButton.LEFT,
) -> None:
    """Synthesize a button_press_event at given data coordinates."""
    canvas = ax.figure.canvas
    transform = ax.transData
    x_pixel, y_pixel = transform.transform((xdata, ydata))
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent(
            "button_press_event",
            canvas,
            x=x_pixel,
            y=y_pixel,
            button=button,
        ),
    )


def test_dial_initial_value_is_set():
    """The dial starts at the requested initial value."""
    fig, ax = pyplot.subplots()
    dial = CircularDial(ax, label="theta", valinit=0.5)
    assert dial.val == 0.5
    pyplot.close(fig)


def test_set_val_fires_observers():
    """Calling set_val triggers all registered on_changed callbacks."""
    fig, ax = pyplot.subplots()
    dial = CircularDial(ax, label="theta", valinit=0.0)
    received: list[float] = []
    dial.on_changed(received.append)
    dial.set_val(1.23)
    assert dial.val == 1.23
    assert received == [1.23]
    pyplot.close(fig)


def test_clicking_inside_dial_updates_value():
    """A left-click inside the dial sets the value to atan2(y, x) of the click."""
    fig, ax = pyplot.subplots()
    dial = CircularDial(ax, label="theta", valinit=0.0)
    received: list[float] = []
    dial.on_changed(received.append)
    _click_at(ax, 0.0, 1.0)  # straight up → π/2
    assert dial.val == pytest_approx_pi_over_two()
    assert received == [pytest_approx_pi_over_two()]
    pyplot.close(fig)


def test_clicking_left_side_gives_pi():
    """Clicking on the negative-x axis sets the value to ±π (same point on S¹)."""
    fig, ax = pyplot.subplots()
    dial = CircularDial(ax, label="theta", valinit=0.0)
    _click_at(ax, -1.0, 0.0)
    assert math.isclose(abs(dial.val), math.pi, abs_tol=1e-9)
    pyplot.close(fig)


def pytest_approx_pi_over_two() -> float:
    """Return the exact float that atan2(1.0, 0.0) produces, for == comparisons."""
    return math.atan2(1.0, 0.0)


def test_set_val_does_not_canonicalize():
    """set_val passes the value through verbatim; periodicity is downstream's job."""
    fig, ax = pyplot.subplots()
    dial = CircularDial(ax, label="theta", valinit=0.0)
    received: list[float] = []
    dial.on_changed(received.append)
    dial.set_val(3 * math.pi)
    np.testing.assert_allclose(dial.val, 3 * math.pi)
    np.testing.assert_allclose(received, np.array([3 * math.pi]))
    pyplot.close(fig)
