"""A two-link 2D arm: anchored base, two revolute joints, two links in the plane.

Loads ``comb/examples/yaml/two_link_arm.lib.yaml``. Each link's body frame
sits at its joint pivot (the parent end of the link), not at the link's
geometric center, so rotating a joint pivots the link around its near end.
"""

from __future__ import annotations

from spatialmath import SE2

from comb.examples import load_example_default_task, load_example_library
from comb.mode import Mode
from comb.system import System


class TwoLinkArm2D:
    """Thin wrapper around the YAML library, surfacing named handles."""

    def __init__(self) -> None:
        self.library = load_example_library("two_link_arm")
        self.base = self.library.bodies["base"]
        self.link_a = self.library.bodies["link_a"]
        self.link_b = self.library.bodies["link_b"]
        self.joint_ab = self.library.constraints["joint_ab"]
        self.joint_bc = self.library.constraints["joint_bc"]
        task = load_example_default_task(self.library)
        self.mode: Mode[SE2] = task.system.mode
        self.system: System[SE2] = task.system
