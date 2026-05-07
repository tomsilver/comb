"""Trajectory planners.

Each planner is a strategy class with a shared ``plan`` interface but its own
hyperparameters as constructor arguments. Strategies search over sequences of
modes (via the ``system.transitions``) and emit a :class:`Plan` so callers
swap planners without rewriting the call site.

A :class:`Plan` bundles three views of the same searched path:

* ``trajectory`` — a continuous-time :class:`Trajectory` of ``ModeState`` for
  rendering and sampling. Mode boundaries are smoothed over by linear
  interpolation so the trajectory itself is continuous.
* ``events`` — discrete :class:`TransitionEvent` records marking exactly when
  each ``ConstraintTransition`` fired during the search.
* ``sample_times`` — the times of the planner's checkpoints (the per-segment
  knots of the piecewise-linear trajectory). Useful for re-discretizing the
  plan or for the validator to know which states the planner explicitly
  solved for.

Planners raise :class:`PlanningError` when they can't find a plan; concrete
strategies may use specific subclasses internally for control flow but
``PlanningError`` is the catch-all for end users.
"""

import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic

from comb.bodies import PoseT
from comb.constraints import Constraint
from comb.mode import ModeState
from comb.system import System
from comb.trajectories import Trajectory
from comb.transitions import ConstraintTransition


class PlanningError(Exception):
    """Raised when a planner cannot find a trajectory satisfying the goal."""


@dataclass(frozen=True)
class TransitionEvent(Generic[PoseT]):
    """A single ``ConstraintTransition`` firing inside a :class:`Plan`."""

    time: float
    transition: ConstraintTransition[PoseT]


@dataclass(frozen=True)
class Plan(Generic[PoseT]):
    """The output of a planner: continuous trajectory + discrete events + checkpoints.

    The trajectory is always defined for ``t`` in ``[0, trajectory.duration]``;
    ``sample_times`` lists the knots between linearly-interpolated segments
    (so ``sample_times[0] == 0`` and ``sample_times[-1] == trajectory.duration``);
    ``events`` records each transition that fired, in time order. Each event's
    ``time`` matches the ``sample_times`` entry of the first state in the new
    mode (so the state at that time is post-transition).
    """

    trajectory: Trajectory[ModeState[PoseT]]
    events: tuple[TransitionEvent[PoseT], ...]
    sample_times: tuple[float, ...]


class Planner(abc.ABC):
    """Strategy interface for trajectory planners.

    A planner returns a :class:`Plan` whose start is the system's current mode
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
    ) -> Plan[PoseT]:
        """Plan a trajectory ending at a state satisfying ``final_constraints``."""
