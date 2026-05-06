"""Trajectory planners.

Each planner is a strategy class with a shared ``plan`` interface but its own
hyperparameters as constructor arguments. Strategies emit
``Trajectory[SystemState[PoseT]]`` so callers swap planners without rewriting
the call site.
"""

import abc
from collections.abc import Iterable

from comb.bodies import PoseT
from comb.constraints import Constraint
from comb.system import System, SystemState
from comb.trajectories import Trajectory


class Planner(abc.ABC):
    """Strategy interface for trajectory planners.

    A planner returns a trajectory whose start is the system's current state
    and whose end satisfies the system's constraints together with
    ``final_constraints``. Hyperparameters (interval, optimization weights,
    sample budgets, ...) live on the subclass; ``plan`` keeps a uniform
    signature so different strategies are interchangeable.
    """

    @abc.abstractmethod
    def plan(
        self,
        system: System[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
        horizon: float,
    ) -> Trajectory[SystemState[PoseT]]:
        """Plan a trajectory ending at a state satisfying ``final_constraints``."""
