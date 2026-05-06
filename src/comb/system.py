"""System: a mode plus the transitions that can change it.

A ``Mode`` (see ``comb.mode``) is the per-mode container — bodies, constraints,
configuration, body poses, anchored bodies. Per-mode operations (``solve``,
``find_satisfying_state``, single-mode planners) take a ``Mode`` directly.

A ``System`` is the multi-mode wrapper: a ``Mode`` plus the
``ConstraintTransitions`` that can fire to swap one constraint topology for
another. Higher-level reasoning (task-and-motion planning, mode-aware
simulation) takes a ``System``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Generic

from comb.bodies import PoseT
from comb.mode import Mode
from comb.transitions import ConstraintTransition


@dataclass(frozen=True)
class System(Generic[PoseT]):
    """A mode + the transitions that can fire from it.

    ``mode`` captures what the system looks like *right now* — bodies,
    constraints, configuration, poses. ``transitions`` is the list of mode
    changes available in this scene; each is a ``ConstraintTransition`` that
    can fire when its trigger is satisfied, producing a new ``Mode``. Whether
    ``transitions`` is filtered to "currently applicable" is up to the caller;
    most planners will just iterate it and call ``transition.is_enabled(state)``.
    """

    mode: Mode[PoseT]
    transitions: tuple[ConstraintTransition[PoseT], ...] = field(default_factory=tuple)

    def enabled_transitions(self) -> Iterable[ConstraintTransition[PoseT]]:
        """Transitions whose triggers currently hold at the mode's snapshot."""
        snapshot = self.mode.snapshot()
        return tuple(t for t in self.transitions if t.is_enabled(snapshot))
