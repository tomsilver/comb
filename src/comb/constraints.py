"""Parameterized constraints between bodies, plus a Configuration holding the
current values of all mutable parameters in a kinematic system.

A ``Constraint`` describes the *structure* of a relationship between two bodies
(which type, which bodies, what fixed properties). It is immutable. The current
values of any mutable parameters live in a separate ``Configuration`` keyed by
constraint instance.
"""

import abc
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from comb.bodies import Body


@dataclass(frozen=True)
class ConstraintParameters:
    """A 1D numpy vector of values together with a name for each component."""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                f"values must be a 1D array, got shape {self.values.shape}"
            )
        if self.values.shape[0] != len(self.names):
            raise ValueError(
                f"values has length {self.values.shape[0]} but there are "
                f"{len(self.names)} names"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"names must be unique, got {self.names}")

    def __getitem__(self, name: str) -> float:
        return float(self.values[self.names.index(name)])


# Constraint uses identity-based equality and hashing so that distinct instances
# can serve as Configuration keys even when fields contain numpy arrays.
@dataclass(frozen=True, eq=False)
class Constraint(abc.ABC):
    """A parameterized constraint relating two bodies (immutable structure)."""

    body1: Body
    body2: Body
    fixed_parameters: ConstraintParameters

    def __post_init__(self) -> None:
        expected = self.fixed_parameter_names()
        if self.fixed_parameters.names != expected:
            raise ValueError(
                f"{type(self).__name__} expects fixed parameter names "
                f"{expected}, got {self.fixed_parameters.names}"
            )

    @classmethod
    @abc.abstractmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        """Names of the fixed parameters this constraint expects."""

    @classmethod
    @abc.abstractmethod
    def parameter_names(cls) -> tuple[str, ...]:
        """Names of the mutable configuration parameters this constraint expects."""


@dataclass(frozen=True, eq=False)
class FixedConstraint(Constraint):
    """A fixed SE(3) rigid-body transform from body1's frame to body2's frame."""

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return ("tx", "ty", "tz", "qx", "qy", "qz", "qw")

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True, eq=False)
class RevoluteJoint(Constraint):
    """A revolute joint with axis and origin fixed in body1's frame.

    The mutable parameter is the joint angle in radians.
    """

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return (
            "axis_x",
            "axis_y",
            "axis_z",
            "origin_x",
            "origin_y",
            "origin_z",
        )

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ("angle",)


@dataclass(frozen=True, eq=False)
class PrismaticJoint(Constraint):
    """A prismatic joint with axis and origin fixed in body1's frame.

    The mutable parameter is the linear offset along the axis.
    """

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return (
            "axis_x",
            "axis_y",
            "axis_z",
            "origin_x",
            "origin_y",
            "origin_z",
        )

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ("offset",)


@dataclass(frozen=True, eq=False)
class PlanarJoint(Constraint):
    """A 3-DOF planar joint, e.g. a robot base on a floor (x, y, theta)."""

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ("x", "y", "theta")


class Configuration:
    """Current values of every constraint's mutable parameters in a system.

    Acts like a mutable mapping from ``Constraint`` to ``ConstraintParameters``,
    with validation that the names of the assigned parameters match the
    constraint's declared ``parameter_names()``.
    """

    def __init__(
        self,
        parameters: Mapping[Constraint, ConstraintParameters] | None = None,
    ) -> None:
        self._parameters: dict[Constraint, ConstraintParameters] = {}
        if parameters is not None:
            for constraint, params in parameters.items():
                self[constraint] = params

    @classmethod
    def zeros(cls, constraints: Iterable[Constraint]) -> "Configuration":
        """Build a configuration with all-zero values for the given constraints."""
        config = cls()
        for constraint in constraints:
            names = constraint.parameter_names()
            config[constraint] = ConstraintParameters(
                values=np.zeros(len(names)), names=names
            )
        return config

    def __getitem__(self, constraint: Constraint) -> ConstraintParameters:
        return self._parameters[constraint]

    def __setitem__(self, constraint: Constraint, params: ConstraintParameters) -> None:
        expected = constraint.parameter_names()
        if params.names != expected:
            raise ValueError(
                f"{type(constraint).__name__} expects parameter names "
                f"{expected}, got {params.names}"
            )
        self._parameters[constraint] = params

    def __contains__(self, constraint: object) -> bool:
        return constraint in self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self._parameters)
