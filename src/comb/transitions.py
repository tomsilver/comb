"""Constraint-triggered transitions for conditionally modifying a mode.

A ``ConstraintTransition`` is a piece of data that says: "when this *trigger*
constraint approximately holds at the current state, we may swap in this
revised constraint set." Triggers are themselves ``Constraint`` objects —
they reuse the same residual the solver minimizes, just consulted here as a
yes/no condition (residual norm below ``tolerance``).

Canonical use is hybrid / mode-switching planning — rigidly attaching an
object to the gripper when the gripper tip is close to it, breaking that
attachment later, contact establishment / breaking, etc. Common
state-capturing ``add`` callbacks (rigid attach, point pin, freeze in
place) live in :mod:`comb.generators` so spec-language YAML can name them.

The transition is just data; how it's used (manual, planner-driven) is up to
the caller. ``apply(mode, state)`` returns a *new* ``Mode`` reflecting the
transition; it never mutates the input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic

import numpy as np

from comb.bodies import BodyPoses, PoseT
from comb.constraints import (
    Constraint,
    ConstraintConfiguration,
    ConstraintParameters,
)
from comb.mode import Mode, ModeState


@dataclass(frozen=True)
class ConstraintTransition(Generic[PoseT]):
    """A state-conditional change to a mode's constraint set.

    The transition is *enabled* when ``trigger.constraint_function``'s residual
    norm at the given state is below ``tolerance``. ``apply(mode, state)``
    returns a fresh ``Mode`` whose constraints are
    ``[c for c in mode.constraints if c not in resolved_remove] + add(state)``.

    ``add`` is a callable so the post-transition constraints can capture
    information from the moment of transition — most importantly the current
    relative transform between the bodies being rigidly attached.

    ``remove`` is either a tuple of specific constraint instances or a
    callable ``mode -> iterable of constraints``. The callable form is
    needed when the constraint to remove was created by an *earlier*
    transition (so there's no stable instance to capture at construction
    time): walk the mode's current constraints and pick out the matching
    one.

    The trigger is required to have no mutable parameters (only fixed ones),
    since the trigger isn't part of the mode and so has no entry in
    ``state.configuration`` to draw values from.
    """

    trigger: Constraint[PoseT]
    tolerance: float
    add: Callable[[ModeState[PoseT]], Iterable[Constraint[PoseT]]] = field(
        default=lambda _state: ()
    )
    remove: (
        Callable[[Mode[PoseT]], Iterable[Constraint[PoseT]]]
        | tuple[Constraint[PoseT], ...]
    ) = ()

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

    def trigger_residual(self, state: ModeState[PoseT]) -> float:
        """L2 norm of the trigger constraint's residual at ``state``."""
        residual = self.trigger.constraint_function(
            ConstraintParameters(np.array([]), ()), state.body_poses
        )
        return float(np.linalg.norm(residual))

    def is_enabled(self, state: ModeState[PoseT]) -> bool:
        """Whether the trigger approximately holds at ``state``."""
        return self.trigger_residual(state) < self.tolerance

    def constraints_to_remove(self, mode: Mode[PoseT]) -> tuple[Constraint[PoseT], ...]:
        """Resolve ``remove`` against ``mode``'s current constraints."""
        resolved = self.remove(mode) if callable(self.remove) else self.remove
        return tuple(resolved)

    def apply(
        self,
        mode: Mode[PoseT],
        state: ModeState[PoseT],
    ) -> Mode[PoseT]:
        """Return a new ``Mode`` reflecting this transition.

        Raises ``ValueError`` if the transition is not enabled at ``state``,
        or if any constraint to remove isn't present in ``mode.constraints``.
        """
        if not self.is_enabled(state):
            raise ValueError(
                f"ConstraintTransition not enabled at state: trigger residual "
                f"{self.trigger_residual(state):g} ≥ tolerance {self.tolerance:g}"
            )
        to_remove = self.constraints_to_remove(mode)
        remove_ids = {id(c) for c in to_remove}
        mode_ids = {id(c) for c in mode.constraints}
        for c in to_remove:
            if id(c) not in mode_ids:
                raise ValueError(
                    f"ConstraintTransition.remove references {type(c).__name__} "
                    f"not in mode.constraints"
                )

        kept = [c for c in mode.constraints if id(c) not in remove_ids]
        added = list(self.add(state))
        new_constraints = kept + added

        new_config = ConstraintConfiguration()
        for c in kept:
            if c in state.configuration:
                new_config[c] = state.configuration[c]
        for c in added:
            if c.parameter_names():
                new_config[c] = ConstraintParameters(
                    values=np.zeros(len(c.parameter_names())),
                    names=c.parameter_names(),
                )

        new_body_poses = BodyPoses({b: state.body_poses[b] for b in mode.bodies})

        return Mode(
            bodies=list(mode.bodies),
            constraints=new_constraints,
            configuration=new_config,
            body_poses=new_body_poses,
            anchored_bodies=list(mode.anchored_bodies),
        )
