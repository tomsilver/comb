"""A 2D mobile base: a single rectangular body that drives freely in the plane.

Implemented as an anchored ``world`` body plus a ``base`` body connected by a
``PlanarJoint2D``. The joint's three parameters ``(tx, ty, theta)`` are the
base's pose in the world frame and are what a planner drives.

Smallest workable example for motion-planning experiments — plan from the
current configuration to a target base pose by handing the planner a
``FixedJoint2D(world, base, target_pose)`` as the final constraint.
"""

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, Rectangle
from comb.constraints import Configuration, ConstraintParameters, PlanarJoint2D
from comb.system import System


class MobileBase2D:
    """An assembled 2D mobile base with handles for its pieces."""

    def __init__(self, base_size: float = 0.4) -> None:
        self.world = Body(
            name="world",
            pose=SE2(),
            visual_geometry=Rectangle(0.0, 0.0),
            collision_geometry=Rectangle(0.0, 0.0),
        )
        self.base = Body(
            name="base",
            pose=SE2(),
            visual_geometry=Rectangle(base_size, base_size),
            collision_geometry=Rectangle(base_size, base_size),
        )
        self.joint = PlanarJoint2D(
            body1=self.world,
            body2=self.base,
            fixed_parameters=ConstraintParameters(values=np.array([]), names=()),
        )
        config = Configuration(
            {
                self.joint: ConstraintParameters(
                    values=np.array([0.0, 0.0, 0.0]),
                    names=PlanarJoint2D.parameter_names(),
                )
            }
        )
        self.system: System[SE2] = System(
            bodies=[self.world, self.base],
            constraints=[self.joint],
            configuration=config,
            anchored_bodies=[self.world],
        )
