"""A 2D door swinging on a hinge in an anchored wall.

Loads ``comb/examples/yaml/door.lib.yaml``. The door's body frame sits at
the hinge so the door swings about its near edge; the hinge angle lives in
``[0, π]`` via ``HingeJoint2D``.
"""

from __future__ import annotations

from spatialmath import SE2

from comb.examples import load_example_default_task, load_example_library
from comb.mode import Mode
from comb.system import System


class Door2D:
    """Thin wrapper around the YAML library, surfacing named handles."""

    def __init__(self) -> None:
        self.library = load_example_library("door")
        self.wall = self.library.bodies["wall"]
        self.door = self.library.bodies["door"]
        self.hinge = self.library.constraints["hinge"]
        task = load_example_default_task(self.library)
        self.mode: Mode[SE2] = task.system.mode
        self.system: System[SE2] = task.system
