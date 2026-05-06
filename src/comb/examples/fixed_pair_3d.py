"""A pair of 3D bodies connected by a fixed SE(3) transform.

The base is anchored. The constraint pins the child at a fixed pure-translation
offset (identity rotation) from the base's frame.
"""

import numpy as np
from spatialmath import SE3

from comb.bodies import Body, Box
from comb.constraints import ConstraintParameters, FixedJoint3D
from comb.mode import Mode
from comb.system import System


class FixedPair3D:
    """An assembled SE(3) fixed-joint pair with handles for its pieces."""

    def __init__(
        self, translation: tuple[float, float, float] = (1.0, 0.0, 0.0)
    ) -> None:
        self.base = Body(
            name="base",
            pose=SE3(),
            visual_geometry=Box(0.2, 0.2, 0.2),
            collision_geometry=Box(0.2, 0.2, 0.2),
        )
        self.child = Body(
            name="child",
            pose=SE3.Trans(list(translation)),
            visual_geometry=Box(0.1, 0.1, 0.1),
            collision_geometry=Box(0.1, 0.1, 0.1),
        )
        self.constraint = FixedJoint3D(
            body1=self.base,
            body2=self.child,
            fixed_parameters=ConstraintParameters(
                values=np.array(
                    [
                        translation[0],
                        translation[1],
                        translation[2],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                ),
                names=FixedJoint3D.fixed_parameter_names(),
            ),
        )
        self.mode: Mode[SE3] = Mode(
            bodies=[self.base, self.child],
            constraints=[self.constraint],
            anchored_bodies=[self.base],
        )
        self.system: System[SE3] = System(mode=self.mode)
