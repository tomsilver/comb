"""A single 2D revolute joint connecting an anchored base to a link in the plane.

Loads ``comb/examples/yaml/single_revolute.lib.yaml``. The link's body frame
sits at the joint pivot so the link rotates *around the joint*, not around
its midpoint. The link's rectangle geometry is offset along +x by half its
length so visually it extends from the joint outward.
"""

from __future__ import annotations

from spatialmath import SE2

from comb.examples import load_example_default_task, load_example_library
from comb.mode import Mode
from comb.system import System


class SingleRevolute2D:
    """Thin wrapper around the YAML library, surfacing named handles."""

    def __init__(self) -> None:
        self.library = load_example_library("single_revolute")
        self.base = self.library.bodies["base"]
        self.link = self.library.bodies["link"]
        self.joint = self.library.constraints["joint"]
        task = load_example_default_task(self.library)
        self.mode: Mode[SE2] = task.system.mode
        self.system: System[SE2] = task.system
