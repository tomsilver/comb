"""Parameter spaces: the manifold a single mutable scalar parameter lives in.

Each ``ParameterSpace`` exposes the operations a Riemannian-style solver needs
(``retract`` to step on the manifold, ``difference`` for geodesic-aware
deltas) plus a ``contains`` check for validation and ``preferred_range`` for
UI hints. Concrete spaces:

* :class:`Real` — unbounded ℝ. Default for parameters with no other structure.
* :class:`Circle` — S¹ with canonical range ``[-π, π]``; wraps on retract,
  shortest signed angle on difference.
* :class:`BoundedReal` — closed interval ``[lower, upper]``; clamps on retract,
  rejects out-of-range in ``contains``.
"""

import abc
import math
from dataclasses import dataclass


class ParameterSpace(abc.ABC):
    """Abstract 1D manifold for a single scalar mutable parameter."""

    @abc.abstractmethod
    def retract(self, point: float, tangent: float) -> float:
        """Step from ``point`` by ``tangent``; return a canonical point."""

    @abc.abstractmethod
    def difference(self, target: float, current: float) -> float:
        """The tangent vector that retracts ``current`` to ``target``."""

    @abc.abstractmethod
    def contains(self, point: float) -> bool:
        """Whether ``point`` lies in the space's representational range."""

    @abc.abstractmethod
    def preferred_range(self, default: tuple[float, float]) -> tuple[float, float]:
        """Suggested display / slider range; fall back to ``default`` if none."""


@dataclass(frozen=True)
class Real(ParameterSpace):
    """Unbounded ℝ; trivial Euclidean parameter space."""

    def retract(self, point: float, tangent: float) -> float:
        return point + tangent

    def difference(self, target: float, current: float) -> float:
        return target - current

    def contains(self, point: float) -> bool:
        del point
        return True

    def preferred_range(self, default: tuple[float, float]) -> tuple[float, float]:
        return default


@dataclass(frozen=True)
class Circle(ParameterSpace):
    """S¹ — angles modulo 2π, canonical range ``[-π, π]`` (both endpoints valid)."""

    def retract(self, point: float, tangent: float) -> float:
        return _wrap_to_pi(point + tangent)

    def difference(self, target: float, current: float) -> float:
        return _wrap_to_pi(target - current)

    def contains(self, point: float) -> bool:
        return -math.pi <= point <= math.pi

    def preferred_range(self, default: tuple[float, float]) -> tuple[float, float]:
        del default
        return (-math.pi, math.pi)


@dataclass(frozen=True)
class BoundedReal(ParameterSpace):
    """Closed interval ``[lower, upper]`` of ℝ — a manifold with boundary."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"BoundedReal lower={self.lower} must be <= upper={self.upper}"
            )

    def retract(self, point: float, tangent: float) -> float:
        return min(max(point + tangent, self.lower), self.upper)

    def difference(self, target: float, current: float) -> float:
        return target - current

    def contains(self, point: float) -> bool:
        return self.lower <= point <= self.upper

    def preferred_range(self, default: tuple[float, float]) -> tuple[float, float]:
        del default
        return (self.lower, self.upper)


def _wrap_to_pi(x: float) -> float:
    """Map any real to the canonical range ``(-π, π]`` (representative on S¹)."""
    wrapped = ((x + math.pi) % (2 * math.pi)) - math.pi
    # Python's % keeps the result in (-π, π]; explicit guard for 0/-π edge.
    if wrapped == -math.pi:
        return math.pi
    return wrapped
