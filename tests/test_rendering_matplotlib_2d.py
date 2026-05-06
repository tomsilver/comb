"""Tests for the rendering module."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import pytest  # noqa: E402  pylint: disable=wrong-import-position
from matplotlib import (  # noqa: E402  pylint: disable=wrong-import-position
    patches,
    pyplot,
)
from matplotlib.figure import (  # noqa: E402  pylint: disable=wrong-import-position
    Figure,
)
from spatialmath import SE2  # noqa: E402  pylint: disable=wrong-import-position

from comb.bodies import (  # noqa: E402  pylint: disable=wrong-import-position
    Body,
    Geometry,
    Rectangle,
)
from comb.examples.single_revolute_2d import (  # noqa: E402  pylint: disable=wrong-import-position
    SingleRevolute2D,
)
from comb.rendering.matplotlib_2d import (  # noqa: E402  pylint: disable=wrong-import-position
    MatplotlibRenderer2D,
)
from comb.system import System  # noqa: E402  pylint: disable=wrong-import-position


def test_render_draws_one_polygon_per_body():
    """Rendering a system adds one Polygon patch to the axes per body."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.system)
    polygons = [p for p in ax.patches if isinstance(p, patches.Polygon)]
    assert len(polygons) == len(ex.system.bodies)
    pyplot.close(fig)


def test_render_clears_previous_state():
    """A second render() replaces the previous patches rather than stacking them."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.system)
    renderer.render(ex.system)
    polygons = [p for p in ax.patches if isinstance(p, patches.Polygon)]
    assert len(polygons) == len(ex.system.bodies)
    pyplot.close(fig)


def test_render_distinguishes_anchored_bodies():
    """Anchored bodies get a different fill color than non-anchored bodies."""
    ex = SingleRevolute2D()
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(ex.system)
    colors = {p.get_facecolor() for p in ax.patches if isinstance(p, patches.Polygon)}
    assert len(colors) == 2
    pyplot.close(fig)


def test_render_creates_figure_when_no_axes_given():
    """If no axes is passed, the renderer creates its own figure."""
    renderer = MatplotlibRenderer2D()
    assert isinstance(renderer.ax.figure, Figure)
    pyplot.close(renderer.ax.figure)


def test_render_rejects_unknown_geometry():
    """Rendering a body whose geometry has no registered drawer raises."""

    class MysteryShape(Geometry[SE2]):
        """A shape with no registered drawing implementation."""


    body = Body(
        name="weird",
        pose=SE2(),
        visual_geometry=MysteryShape(),
        collision_geometry=Rectangle(0.1, 0.1),
    )
    system: System[SE2] = System(bodies=[body], constraints=[], anchored_bodies=[body])
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    with pytest.raises(NotImplementedError):
        renderer.render(system)
    pyplot.close(fig)


def test_rectangle_is_translated_and_rotated():
    """A rectangle drawn at a non-identity pose ends up at the expected position."""
    body = Body(
        name="b",
        pose=SE2(1.0, 2.0, np.pi / 2),
        visual_geometry=Rectangle(0.4, 0.2),
        collision_geometry=Rectangle(0.4, 0.2),
    )
    system: System[SE2] = System(bodies=[body], constraints=[], anchored_bodies=[body])
    fig, ax = pyplot.subplots()
    renderer = MatplotlibRenderer2D(ax=ax)
    renderer.render(system)
    polygon = next(p for p in ax.patches if isinstance(p, patches.Polygon))
    centroid = polygon.get_xy()[:-1].mean(axis=0)  # last vertex repeats the first
    np.testing.assert_allclose(centroid, [1.0, 2.0], atol=1e-9)
    pyplot.close(fig)
