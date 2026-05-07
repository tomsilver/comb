"""Tests for the rendering module."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

# pylint: disable=wrong-import-position
import numpy as np
from matplotlib import (
    patches,
    pyplot,
)
from matplotlib.figure import (
    Figure,
)
from spatialmath import SE2

from comb.bodies import (
    Body,
    Rectangle,
)
from comb.examples.single_revolute_2d import (
    SingleRevolute2D,
)
from comb.mode import Mode
from comb.rendering.matplotlib_2d import (
    MatplotlibRenderer2D,
)


def test_render_draws_one_polygon_per_body():
    """Rendering a mode adds one Polygon patch to the axes per body."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.mode)
    polygons = [p for p in ax.patches if isinstance(p, patches.Polygon)]
    assert len(polygons) == len(ex.mode.bodies)
    pyplot.close(fig)


def test_render_clears_previous_state():
    """A second render() replaces the previous patches rather than stacking them."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.mode)
    renderer.render(ex.mode)
    polygons = [p for p in ax.patches if isinstance(p, patches.Polygon)]
    assert len(polygons) == len(ex.mode.bodies)
    pyplot.close(fig)


def test_render_distinguishes_anchored_bodies():
    """Anchored bodies get a different fill color than non-anchored bodies."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.mode)
    colors = {p.get_facecolor() for p in ax.patches if isinstance(p, patches.Polygon)}
    assert len(colors) == 2
    pyplot.close(fig)


def test_render_creates_figure_when_no_axes_given():
    """If no axes is passed, the renderer creates its own figure."""
    renderer = MatplotlibRenderer2D()
    assert isinstance(renderer.ax.figure, Figure)
    pyplot.close(renderer.ax.figure)


def test_render_limits_are_stable_across_calls():
    """Limits don't shift between renders, so the view doesn't jitter on slider
    changes."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.mode)
    xlim_first = ax.get_xlim()
    ylim_first = ax.get_ylim()
    # Mutate the link's pose to simulate a slider change.
    ex.mode.body_poses[ex.link] = SE2(0.0, 0.0, np.pi / 2)
    renderer.render(ex.mode)
    assert ax.get_xlim() == xlim_first
    assert ax.get_ylim() == ylim_first
    pyplot.close(fig)


def test_render_respects_explicit_limits():
    """Xlim/ylim passed to the constructor override the heuristic and stay fixed."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax, xlim=(-5.0, 5.0), ylim=(-3.0, 3.0))
    renderer.render(ex.mode)
    np.testing.assert_allclose(ax.get_xlim(), (-5.0, 5.0))
    np.testing.assert_allclose(ax.get_ylim(), (-3.0, 3.0))
    pyplot.close(fig)


def test_rectangle_is_translated_and_rotated():
    """A rectangle drawn at a non-identity pose ends up at the expected position."""
    body = Body(
        name="b",
        pose=SE2(1.0, 2.0, np.pi / 2),
        visual_geometry=Rectangle(0.4, 0.2),
        collision_geometry=Rectangle(0.4, 0.2),
    )
    mode: Mode[SE2] = Mode(bodies=[body], constraints=[], anchored_bodies=[body])
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(mode)
    polygon = next(p for p in ax.patches if isinstance(p, patches.Polygon))
    centroid = polygon.get_xy()[:-1].mean(axis=0)  # last vertex repeats the first
    np.testing.assert_allclose(centroid, [1.0, 2.0], atol=1e-9)
    pyplot.close(fig)
