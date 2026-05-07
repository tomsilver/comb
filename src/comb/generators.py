"""State-dependent constraint factories for ``ConstraintTransition.add``.

A *generator* takes some configuration (which bodies, optional offsets) and
returns a callable ``(state) -> list[Constraint]`` that captures information
from the moment of transition. This is the canonical pattern for pickup /
attach transitions: the *trigger* says "when these bodies are close enough,
fire", and the generator says "when you fire, freeze whatever relative
geometry is currently true and turn it into a constraint".

This module is the closed registry the YAML domain-language loader will
dispatch on: spec files reference generators by name, and the loader looks
them up here. Adding a new generator means adding both a function and a
``GENERATORS_2D`` (or eventually ``GENERATORS_3D``) entry.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from spatialmath import SE2

from comb.bodies import Body
from comb.constraints import (
    Constraint,
    ConstraintParameters,
    FixedJoint2D,
    PointEquality2D,
)
from comb.mode import ModeState

GeneratorFn2D = Callable[[ModeState[SE2]], list[Constraint[SE2]]]
"""The shape of every generator's output: a state→constraints closure."""


def rigid_attachment_2d(body1: Body[SE2], body2: Body[SE2]) -> GeneratorFn2D:
    """Generate a ``FixedJoint2D`` capturing body2's current pose relative to body1.

    The returned closure, when called with a state, reads
    ``state.body_poses[body1].inv() * state.body_poses[body2]`` and produces a
    single ``FixedJoint2D`` whose fixed parameters reproduce that relative
    transform — so post-transition, the two bodies stay rigidly locked at
    whatever relative pose they happened to be in at the trigger moment.
    """

    def add(state: ModeState[SE2]) -> list[Constraint[SE2]]:
        rel = state.body_poses[body1].inv() * state.body_poses[body2]
        return [
            FixedJoint2D(
                body1=body1,
                body2=body2,
                fixed_parameters=ConstraintParameters(
                    values=np.array(
                        [float(rel.t[0]), float(rel.t[1]), float(rel.theta())]
                    ),
                    names=FixedJoint2D.fixed_parameter_names(),
                ),
            )
        ]

    return add


def freeze_pose_2d(world: Body[SE2], body: Body[SE2]) -> GeneratorFn2D:
    """Pin ``body`` to ``world`` at ``body``'s current pose.

    Mechanically identical to ``rigid_attachment_2d(world, body)``; named
    separately because the spec-language reading of "freeze a body in place"
    is much clearer than "rigidly attach it to the world".
    """
    return rigid_attachment_2d(world, body)


def point_pin_2d(
    body1: Body[SE2],
    body2: Body[SE2],
    *,
    body2_offset: tuple[float, float] = (0.0, 0.0),
) -> GeneratorFn2D:
    """Generate a ``PointEquality2D`` pinning a point on body2 at its current world
    location, expressed as an offset in body1's frame.

    ``body2_offset`` is the point on body2 to pin (defaults to body2's frame
    origin). The body1-side offset is captured from the current state so the
    constraint's residual is zero at transition time. Position-only —
    orientation stays free, unlike ``rigid_attachment_2d``.
    """

    def add(state: ModeState[SE2]) -> list[Constraint[SE2]]:
        offset_in_b2 = SE2(body2_offset[0], body2_offset[1], 0.0)
        point_world = state.body_poses[body2] * offset_in_b2
        point_in_b1 = state.body_poses[body1].inv() * point_world
        return [
            PointEquality2D(
                body1=body1,
                body2=body2,
                fixed_parameters=ConstraintParameters(
                    values=np.array(
                        [
                            float(point_in_b1.t[0]),
                            float(point_in_b1.t[1]),
                            float(body2_offset[0]),
                            float(body2_offset[1]),
                        ]
                    ),
                    names=PointEquality2D.fixed_parameter_names(),
                ),
            )
        ]

    return add


GENERATORS_2D: dict[str, Callable[..., GeneratorFn2D]] = {
    "rigid_attachment_2d": rigid_attachment_2d,
    "freeze_pose_2d": freeze_pose_2d,
    "point_pin_2d": point_pin_2d,
}
"""Closed registry mapping generator names to their factory functions.

The YAML domain-language loader will dispatch on these names; tests verify
the registry is exhaustive (every public ``..._2d`` function appears here).
"""
