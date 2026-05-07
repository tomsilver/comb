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
from comb.examples.two_link_arm_with_object_2d import (  # noqa: E402
    TwoLinkArmWithObject2D,
)
from comb.gui.matplotlib_2d import MatplotlibGUI2D  # noqa: E402
from comb.gui.widgets import CircularDial  # noqa: E402
from comb.mode import Mode  # noqa: E402
from comb.solver import find_satisfying_state  # noqa: E402
from comb.system import System  # noqa: E402


def test_gui_builds_one_widget_per_mutable_parameter():
    """The GUI creates exactly one widget for each mutable parameter."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.system)
    assert len(gui.widgets) == 1  # SingleRevolute2D has one mutable param (angle)
    pyplot.close(gui.figure)


def test_gui_uses_circular_dial_for_circle_parameter():
    """A constraint whose parameter lives on Circle gets a CircularDial."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.system)
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
        MatplotlibGUI2D(System(mode=mode))


def test_gui_widget_drives_solver_and_updates_mode():
    """Setting a widget value runs solve and updates the mode's body poses."""
    ex = SingleRevolute2D()
    gui = MatplotlibGUI2D(ex.system)
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


def test_gui_creates_one_button_per_transition():
    """Each ConstraintTransition in the system gets a button."""
    ex = TwoLinkArmWithObject2D()
    gui = MatplotlibGUI2D(ex.system)
    assert len(gui.transition_buttons) == len(ex.system.transitions) == 1
    pyplot.close(gui.figure)


def test_gui_transition_button_disabled_when_trigger_not_satisfied():
    """At construction the pickup trigger isn't satisfied; the button is gray."""
    ex = TwoLinkArmWithObject2D()
    gui = MatplotlibGUI2D(ex.system)
    button = gui.transition_buttons[0]
    assert button.color == "lightgray"
    pyplot.close(gui.figure)


def test_gui_transition_button_enables_when_trigger_satisfied():
    """Once the arm tip is at the block, the pickup button turns green."""
    ex = TwoLinkArmWithObject2D()
    gui = MatplotlibGUI2D(ex.system)
    # Drive the mode to a state where the pickup trigger fires.
    near_state = find_satisfying_state(ex.mode, [ex.pickup_trigger])
    ex.mode.set_state(near_state)
    gui._refresh_transition_buttons()  # pylint: disable=protected-access
    assert gui.transition_buttons[0].color == "tab:green"
    pyplot.close(gui.figure)


def test_gui_clicking_transition_swaps_mode_and_rebuilds_widgets():
    """Clicking an enabled transition button swaps the mode and the widget set."""
    ex = TwoLinkArmWithObject2D()
    gui = MatplotlibGUI2D(ex.system)
    n_widgets_before = len(gui.widgets)
    near_state = find_satisfying_state(ex.mode, [ex.pickup_trigger])
    ex.mode.set_state(near_state)
    gui._refresh_transition_buttons()  # pylint: disable=protected-access
    # Fire the transition by invoking the same callback the button would.
    gui._make_transition_callback(0)(None)  # pylint: disable=protected-access
    # Mode rebound to the post-transition mode (block now attached to arm).
    assert ex.world_to_block not in gui.mode.constraints
    # Widget count unchanged (joint params still the same), but they're new objects.
    assert len(gui.widgets) == n_widgets_before
    pyplot.close(gui.figure)
