"""Tests for the multi-mode System wrapper."""

import numpy as np
from spatialmath import SE2

from comb.examples.two_link_arm_with_object_2d import TwoLinkArmWithObject2D
from comb.solver import find_satisfying_state
from comb.system import System


def test_system_construction_with_no_transitions():
    """A System with no transitions exposes the mode and an empty transitions tuple."""
    ex = TwoLinkArmWithObject2D()
    system: System[SE2] = System(mode=ex.mode)
    assert system.mode is ex.mode
    assert not system.transitions
    assert not tuple(system.enabled_transitions())


def test_system_enabled_transitions_filters_by_current_state():
    """Only transitions whose triggers currently hold at the mode's state are
    returned."""
    ex = TwoLinkArmWithObject2D()
    system: System[SE2] = System(mode=ex.mode, transitions=(ex.pickup_transition,))
    # At construction the arm is at zero angles; the pickup trigger isn't satisfied.
    assert not tuple(system.enabled_transitions())

    # Drive the arm so the tip coincides with the block, then check again.
    ex.mode.set_state(find_satisfying_state(ex.mode, [ex.pickup_trigger]))
    enabled = tuple(system.enabled_transitions())
    assert enabled == (ex.pickup_transition,)


def test_system_preserves_transitions_order_and_contents():
    """Transitions is just a tuple — order preserved, contents not validated."""
    ex = TwoLinkArmWithObject2D()
    system: System[SE2] = System(
        mode=ex.mode, transitions=(ex.pickup_transition, ex.pickup_transition)
    )
    assert system.transitions == (ex.pickup_transition, ex.pickup_transition)


def test_system_default_transitions_is_empty_tuple():
    """Constructing without transitions yields an empty tuple."""
    ex = TwoLinkArmWithObject2D()
    system: System[SE2] = System(mode=ex.mode)
    assert isinstance(system.transitions, tuple)
    assert len(system.transitions) == 0
    np.testing.assert_array_equal(system.mode.body_poses[ex.block].t, [0.5, 1.0])
