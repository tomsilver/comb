"""Parameterized constraints between bodies, plus a Configuration holding the
current values of all mutable parameters in a kinematic system.

A ``Constraint`` describes the *structure* of a relationship between two bodies
(which type, which bodies, what fixed properties). It is immutable. The current
values of any mutable parameters live in a separate ``Configuration`` keyed by
constraint instance.

``Constraint.constraint_function`` is fully generic: it returns an arbitrary
residual vector that is zero when the constraint is satisfied. Joint-type
constraints (relating two body poses by a relative transform) inherit from
``Joint2D`` or ``Joint3D``, which derive ``constraint_function`` from a
subclass-defined ``relative_transform``. Other custom constraints can implement
``constraint_function`` directly.
"""

import abc
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Generic

import numpy as np
from spatialmath import SE2, SE3, Twist2, Twist3, UnitQuaternion

from comb.bodies import Body, BodyPoses, PoseT
from comb.parameter_spaces import Circle, ParameterSpace, Real


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
class Constraint(abc.ABC, Generic[PoseT]):
    """A parameterized constraint relating two bodies that share a pose type.

    The :attr:`parameter_spaces` property declares the manifold each mutable
    parameter lives in (Real, Circle, BoundedReal, ...). It resolves from
    :attr:`parameter_space_overrides` if set, else falls back to
    :meth:`default_parameter_spaces`, which defaults to all ``Real`` and is
    overridden per-class for joints whose parameters have intrinsic structure
    (e.g. an angle on the circle).
    """

    body1: Body[PoseT]
    body2: Body[PoseT]
    fixed_parameters: ConstraintParameters
    parameter_space_overrides: tuple[ParameterSpace, ...] | None = None

    def __post_init__(self) -> None:
        expected = self.fixed_parameter_names()
        if self.fixed_parameters.names != expected:
            raise ValueError(
                f"{type(self).__name__} expects fixed parameter names "
                f"{expected}, got {self.fixed_parameters.names}"
            )
        if len(self.parameter_spaces) != len(self.parameter_names()):
            raise ValueError(
                f"{type(self).__name__} expects {len(self.parameter_names())} "
                f"parameter spaces, got {len(self.parameter_spaces)}"
            )

    @property
    def parameter_spaces(self) -> tuple[ParameterSpace, ...]:
        """Resolved ParameterSpace for each mutable parameter."""
        if self.parameter_space_overrides is not None:
            return self.parameter_space_overrides
        return self.default_parameter_spaces()

    @classmethod
    @abc.abstractmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        """Names of the fixed parameters this constraint expects."""

    @classmethod
    @abc.abstractmethod
    def parameter_names(cls) -> tuple[str, ...]:
        """Names of the mutable configuration parameters this constraint expects."""

    @classmethod
    def default_parameter_spaces(cls) -> tuple[ParameterSpace, ...]:
        """Default ParameterSpace for each mutable parameter; override per-class."""
        return tuple(Real() for _ in cls.parameter_names())

    @abc.abstractmethod
    def constraint_function(
        self,
        parameters: ConstraintParameters,
        body_poses: BodyPoses[PoseT],
    ) -> np.ndarray:
        """Residual vector that is zero when the constraint is satisfied.

        ``body_poses`` provides the current poses for the bodies in the system.
        The shape and meaning of the residual is constraint-specific.
        """


@dataclass(frozen=True, eq=False)
class Joint2D(Constraint[SE2]):
    """A joint between two SE(2) bodies, defined by a relative SE(2) transform."""

    @abc.abstractmethod
    def relative_transform(self, parameters: ConstraintParameters) -> SE2:
        """Transform from body1's frame to body2's frame implied by ``parameters``.

        The constraint imposes
        ``body2.pose == body1.pose * relative_transform(parameters)``.
        """

    def constraint_function(
        self,
        parameters: ConstraintParameters,
        body_poses: BodyPoses[SE2],
    ) -> np.ndarray:
        """SE(2) twist of the pose error (3-vector, zero when satisfied)."""
        expected = self.relative_transform(parameters)
        actual = body_poses[self.body1].inv() * body_poses[self.body2]
        error = expected.inv() * actual
        return np.asarray(Twist2(error).A, dtype=float)


@dataclass(frozen=True, eq=False)
class Joint3D(Constraint[SE3]):
    """A joint between two SE(3) bodies, defined by a relative SE(3) transform."""

    @abc.abstractmethod
    def relative_transform(self, parameters: ConstraintParameters) -> SE3:
        """Transform from body1's frame to body2's frame implied by ``parameters``.

        The constraint imposes
        ``body2.pose == body1.pose * relative_transform(parameters)``.
        """

    def constraint_function(
        self,
        parameters: ConstraintParameters,
        body_poses: BodyPoses[SE3],
    ) -> np.ndarray:
        """SE(3) twist of the pose error (6-vector, zero when satisfied)."""
        expected = self.relative_transform(parameters)
        actual = body_poses[self.body1].inv() * body_poses[self.body2]
        error = expected.inv() * actual
        return np.asarray(Twist3(error).A, dtype=float)


@dataclass(frozen=True, eq=False)
class FixedJoint2D(Joint2D):
    """A fixed SE(2) rigid-body transform from body1's frame to body2's frame."""

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return ("tx", "ty", "theta")

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ()

    def relative_transform(  # pylint: disable=unused-argument
        self, parameters: ConstraintParameters
    ) -> SE2:
        fp = self.fixed_parameters
        return SE2(fp["tx"], fp["ty"], fp["theta"])


@dataclass(frozen=True, eq=False)
class FixedJoint3D(Joint3D):
    """A fixed SE(3) rigid-body transform from body1's frame to body2's frame."""

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return ("tx", "ty", "tz", "qx", "qy", "qz", "qw")

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ()

    def relative_transform(  # pylint: disable=unused-argument
        self, parameters: ConstraintParameters
    ) -> SE3:
        fp = self.fixed_parameters
        # spatialmath UnitQuaternion is scalar-first; our names follow xyzw.
        rotation = UnitQuaternion([fp["qw"], fp["qx"], fp["qy"], fp["qz"]])
        return SE3.Rt(rotation.R, t=[fp["tx"], fp["ty"], fp["tz"]])


@dataclass(frozen=True, eq=False)
class RevoluteJoint2D(Joint2D):
    """A 2D revolute joint with origin fixed in body1's frame.

    The rotation axis is implicit (out of the plane). The mutable parameter is
    the joint angle in radians, defaulting to live on the circle S¹ (wraps).
    """

    @classmethod
    def fixed_parameter_names(cls) -> tuple[str, ...]:
        return ("origin_x", "origin_y")

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        return ("angle",)

    @classmethod
    def default_parameter_spaces(cls) -> tuple[ParameterSpace, ...]:
        return (Circle(),)

    def relative_transform(self, parameters: ConstraintParameters) -> SE2:
        fp = self.fixed_parameters
        return SE2(fp["origin_x"], fp["origin_y"], parameters["angle"])


@dataclass(frozen=True, eq=False)
class RevoluteJoint3D(Joint3D):
    """A revolute joint with axis and origin fixed in body1's frame.

    The mutable parameter is the joint angle in radians, defaulting to live on
    the circle S¹ (wraps).
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

    @classmethod
    def default_parameter_spaces(cls) -> tuple[ParameterSpace, ...]:
        return (Circle(),)

    def relative_transform(self, parameters: ConstraintParameters) -> SE3:
        fp = self.fixed_parameters
        origin = [fp["origin_x"], fp["origin_y"], fp["origin_z"]]
        axis = [fp["axis_x"], fp["axis_y"], fp["axis_z"]]
        return SE3.Trans(origin) * SE3.AngVec(parameters["angle"], axis)


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
        for name, space, value in zip(
            expected, constraint.parameter_spaces, params.values
        ):
            if not space.contains(float(value)):
                raise ValueError(
                    f"Parameter {name}={float(value)} is not in space {space}"
                )
        self._parameters[constraint] = params

    def __contains__(self, constraint: object) -> bool:
        return constraint in self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self._parameters)
