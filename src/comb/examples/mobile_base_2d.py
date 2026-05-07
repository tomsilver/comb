"""A 2D mobile base: a single rectangular body driving freely in the plane.

Loads ``comb/examples/yaml/mobile_base.lib.yaml``. Implemented as an
anchored ``world`` plus a ``base`` connected by a ``PlanarJoint2D`` whose
three parameters ``(tx, ty, theta)`` are the base's pose in the world
frame.
"""

from __future__ import annotations

from spatialmath import SE2

from comb.examples import load_example_default_task, load_example_library
from comb.mode import Mode
from comb.system import System


class MobileBase2D:
    """Thin wrapper around the YAML library, surfacing named handles."""

    def __init__(self) -> None:
        self.library = load_example_library("mobile_base")
        self.world = self.library.bodies["world"]
        self.base = self.library.bodies["base"]
        self.joint = self.library.constraints["joint"]
        task = load_example_default_task(self.library)
        self.mode: Mode[SE2] = task.system.mode
        self.system: System[SE2] = task.system
