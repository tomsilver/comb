"""Tests for the matplotlib 2D ParameterGUI."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

# pylint: disable=wrong-import-position
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from spatialmath import SE2  # noqa: E402

from comb.bodies import Body, Rectangle  # noqa: E402
from comb.examples.single_revolute_2d import SingleRevolute2D  # noqa: E402
from comb.gui.matplotlib_2d import MatplotlibGUI2D  # noqa: E402
from comb.gui.widgets import CircularDial  # noqa: E402
from comb.mode import Mode  # noqa: E402


def test_gui_builds_one_widget_per_mutable_parameter():
    """The GUI creates exactly one widget for each mutable parameter."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.mode)
    assert len(gui.widgets) == 1  # SingleRevolute2D has one mutable param (angle)
    pyplot.close(gui.figure)


def test_gui_uses_circular_dial_for_circle_parameter():
    """A constraint whose parameter lives on Circle gets a CircularDial."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.mode)
    assert isinstance(gui.widgets[0], CircularDial)
    pyplot.close(gui.figure)


def test_gui_refuses_mode_without_anchored_bodies():
    """A mode with no anchored bodies cannot be opened in the GUI."""
    body = Body(
        name="lonely",
        pose=SE2(),
        visual_geometry=Rectangle(0.1, 0.1),
        collision_geometry=Rectangle(0.1, 0.1),
    )
    mode: Mode[SE2] = Mode(bodies=[body], constraints=[])
    with pytest.raises(ValueError, match="anchored_bodies"):
        MatplotlibGUI2D(mode)


def test_gui_widget_drives_solver_and_updates_mode():
    """Setting a widget value runs solve and updates the mode's body poses."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.mode)
    initial_link_pose = ex.mode.body_poses[ex.link]

    gui.widgets[0].set_val(np.pi / 2)

    # ConstraintConfiguration was updated.
    assert ex.mode.configuration[ex.joint]["angle"] == pytest.approx(
        np.pi / 2, abs=1e-9
    )
    # Body pose for the link rotated to match.
    expected_link_pose = ex.base.pose * SE2(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(
        ex.mode.body_poses[ex.link].A, expected_link_pose.A, atol=1e-9
    )
    # Sanity: the link did move.
    assert not np.allclose(ex.mode.body_poses[ex.link].A, initial_link_pose.A)
    pyplot.close(gui.figure)
