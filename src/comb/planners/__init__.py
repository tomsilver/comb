"""Trajectory planners.

Each planner is a strategy class with a shared ``plan`` interface but its own
hyperparameters as constructor arguments. Strategies emit
``Trajectory[ModeState[PoseT]]`` so callers swap planners without rewriting
the call site.
"""

import abc
from collections.abc import Iterable

from comb.bodies import PoseT
from comb.constraints import Constraint
from comb.mode import Mode, ModeState
from comb.trajectories import Trajectory


class Planner(abc.ABC):
    """Strategy interface for trajectory planners.

    A planner returns a trajectory whose start is the mode's current state
    and whose end satisfies the mode's constraints together with
    ``final_constraints``. Hyperparameters (interval, optimization weights,
    sample budgets, ...) live on the subclass; ``plan`` keeps a uniform
    signature so different strategies are interchangeable.
    """

    @abc.abstractmethod
    def plan(
        self,
        mode: Mode[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
        horizon: float,
    ) -> Trajectory[ModeState[PoseT]]:
        """Plan a trajectory ending at a state satisfying ``final_constraints``."""
