"""Two-link 2D arm plus a small free body initially fixed to the world.

Loads ``comb/examples/yaml/two_link_arm_with_object.lib.yaml`` (which
``includes`` the bare-arm library). Surfaces the bundled ``pickup``
transition that swaps the world→block pin for an arm.link_b→block rigid
attachment when the arm tip reaches the block.
"""

from __future__ import annotations

from spatialmath import SE2

from comb.examples import load_example_default_task, load_example_library
from comb.examples.two_link_arm_2d import TwoLinkArm2D
from comb.mode import Mode
from comb.system import System
from comb.transitions import ConstraintTransition


class TwoLinkArmWithObject2D:
    """Thin wrapper around the YAML library, surfacing named handles."""

    def __init__(self) -> None:
        self.library = load_example_library("two_link_arm_with_object")
        # Arm wrapper that points into the same instances — included libraries
        # share body / constraint identity with the parent.
        self.arm = TwoLinkArm2D.__new__(TwoLinkArm2D)
        self.arm.library = self.library
        self.arm.base = self.library.bodies["base"]
        self.arm.link_a = self.library.bodies["link_a"]
        self.arm.link_b = self.library.bodies["link_b"]
        self.arm.joint_ab = self.library.constraints["joint_ab"]
        self.arm.joint_bc = self.library.constraints["joint_bc"]

        self.world = self.library.bodies["world"]
        self.block = self.library.bodies["block"]
        self.world_to_block = self.library.constraints["world_to_block"]
        self.pickup_transition: ConstraintTransition[SE2] = self.library.transitions[
            "pickup"
        ]
        self.pickup_trigger = self.pickup_transition.trigger

        task = load_example_default_task(self.library)
        self.mode: Mode[SE2] = task.system.mode
        self.system: System[SE2] = task.system
        self.arm.mode = self.mode
        self.arm.system = self.system
