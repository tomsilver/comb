"""Constraint-triggered transitions for conditionally modifying a system.

A ``ConstraintTransition`` is a piece of data that says: "when this *trigger*
constraint approximately holds at the current state, we may swap in this
revised constraint set." Triggers are themselves ``Constraint`` objects —
they reuse the same residual the solver minimizes, just consulted here as a
yes/no condition (residual norm below ``tolerance``).

Canonical use is hybrid / mode-switching planning — rigidly attaching an
object to the gripper when the gripper tip is close to it, breaking that
attachment later, contact establishment / breaking, etc.

The transition is just data; how it's used (manual, planner-driven) is up to
the caller. ``apply(system, state)`` returns a *new* ``System`` reflecting the
transition; it never mutates the input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic

import numpy as np
from spatialmath import SE2

from comb.bodies import Body, BodyPoses, PoseT
from comb.constraints import (
    Configuration,
    Constraint,
    ConstraintParameters,
    FixedJoint2D,
)
from comb.system import System, SystemState


@dataclass(frozen=True)
class ConstraintTransition(Generic[PoseT]):
    """A state-conditional change to a system's constraint set.

    The transition is *enabled* when ``trigger.constraint_function``'s residual
    norm at the given state is below ``tolerance``. ``apply(system, state)``
    returns a fresh ``System`` whose constraints are
    ``[c for c in system.constraints if c not in remove] + add(state)``.

    ``add`` is a callable so the post-transition constraints can capture
    information from the moment of transition — most importantly the current
    relative transform between the bodies being rigidly attached.

    The trigger is required to have no mutable parameters (only fixed ones),
    since the trigger isn't part of the system and so has no entry in
    ``state.configuration`` to draw values from.
    """

    trigger: Constraint[PoseT]
    tolerance: float
    add: Callable[[SystemState[PoseT]], Iterable[Constraint[PoseT]]] = field(
        default=lambda _state: ()
    )
    remove: tuple[Constraint[PoseT], ...] = ()

    def __post_init__(self) -> None:
        if self.trigger.parameter_names():
            raise ValueError(
                "ConstraintTransition trigger must have no mutable parameters; "
                f"got {self.trigger.parameter_names()}"
            )
        if self.tolerance <= 0:
            raise ValueError(
                f"ConstraintTransition tolerance must be positive; got {self.tolerance}"
            )

    def trigger_residual(self, state: SystemState[PoseT]) -> float:
        """L2 norm of the trigger constraint's residual at ``state``."""
        residual = self.trigger.constraint_function(
            ConstraintParameters(np.array([]), ()), state.body_poses
        )
        return float(np.linalg.norm(residual))

    def is_enabled(self, state: SystemState[PoseT]) -> bool:
        """Whether the trigger approximately holds at ``state``."""
        return self.trigger_residual(state) < self.tolerance

    def apply(
        self,
        system: System[PoseT],
        state: SystemState[PoseT],
    ) -> System[PoseT]:
        """Return a new ``System`` reflecting this transition.

        Raises ``ValueError`` if the transition is not enabled at ``state``,
        or if any constraint in ``remove`` isn't present in ``system.constraints``.
        """
        if not self.is_enabled(state):
            raise ValueError(
                f"ConstraintTransition not enabled at state: trigger residual "
                f"{self.trigger_residual(state):g} ≥ tolerance {self.tolerance:g}"
            )
        remove_ids = {id(c) for c in self.remove}
        system_ids = {id(c) for c in system.constraints}
        for c in self.remove:
            if id(c) not in system_ids:
                raise ValueError(
                    f"ConstraintTransition.remove references {type(c).__name__} "
                    f"not in system.constraints"
                )

        kept = [c for c in system.constraints if id(c) not in remove_ids]
        added = list(self.add(state))
        new_constraints = kept + added

        new_config = Configuration()
        for c in kept:
            if c in state.configuration:
                new_config[c] = state.configuration[c]
        for c in added:
            if c.parameter_names():
                new_config[c] = ConstraintParameters(
                    values=np.zeros(len(c.parameter_names())),
                    names=c.parameter_names(),
                )

        new_body_poses = BodyPoses({b: state.body_poses[b] for b in system.bodies})

        return System(
            bodies=list(system.bodies),
            constraints=new_constraints,
            configuration=new_config,
            body_poses=new_body_poses,
            anchored_bodies=list(system.anchored_bodies),
        )


def rigid_attachment_2d(
    body1: Body[SE2], body2: Body[SE2]
) -> Callable[[SystemState[SE2]], list[Constraint[SE2]]]:
    """An ``add`` factory that rigidly attaches ``body2`` to ``body1``.

    Returns a callable suitable for ``ConstraintTransition.add``. At apply time
    it captures the *current* relative transform between the two bodies and
    builds a ``FixedJoint2D`` enforcing it — so ``body2`` stays at its current
    pose-relative-to-``body1`` from then on.
    """

    def make(state: SystemState[SE2]) -> list[Constraint[SE2]]:
        rel = state.body_poses[body1].inv() * state.body_poses[body2]
        return [
            FixedJoint2D(
                body1=body1,
                body2=body2,
                fixed_parameters=ConstraintParameters(
                    values=np.array(
                        [
                            float(rel.t[0]),
                            float(rel.t[1]),
                            float(rel.theta()),
                        ]
                    ),
                    names=FixedJoint2D.fixed_parameter_names(),
                ),
            )
        ]

    return make
