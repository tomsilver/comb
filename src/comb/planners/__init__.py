"""Trajectory planners.

Each planner is a strategy class with a shared ``plan`` interface but its own
hyperparameters as constructor arguments. Strategies search over sequences of
modes (via the ``system.transitions``) and emit ``Trajectory[ModeState[PoseT]]``
so callers swap planners without rewriting the call site.

Planners raise :class:`PlanningError` when they can't find a plan; concrete
strategies may use specific subclasses internally for control flow but
``PlanningError`` is the catch-all for end users.
"""

import abc
from collections.abc import Iterable

from comb.bodies import PoseT
from comb.constraints import Constraint
from comb.mode import ModeState
from comb.system import System
from comb.trajectories import Trajectory


class PlanningError(Exception):
    """Raised when a planner cannot find a trajectory satisfying the goal."""


class Planner(abc.ABC):
    """Strategy interface for trajectory planners.

    A planner returns a trajectory whose start is the system's current mode
    and state and whose end satisfies ``final_constraints``. Mode changes
    during the plan are driven by the ``system.transitions``: the planner is
    free to fire any transition whose trigger it can reach. Hyperparameters
    (BFS budget, step interval, optimization weights, ...) live on the
    subclass; ``plan`` keeps a uniform signature so strategies are
    interchangeable.
    """

    @abc.abstractmethod
    def plan(
        self,
        system: System[PoseT],
        final_constraints: Iterable[Constraint[PoseT]],
        horizon: float,
    ) -> Trajectory[ModeState[PoseT]]:
        """Plan a trajectory ending at a state satisfying ``final_constraints``."""
