"""Abstract Renderer base class.

A renderer draws (or refreshes) a ``System`` at its current body poses. It is
generic in the pose type, so a renderer for ``System[SE2]`` is statically
distinct from one for ``System[SE3]``. Concrete subclasses dispatch on the
geometry type via ``functools.singledispatchmethod`` so that adding a new
shape only requires registering a method per renderer that supports it.
"""

import abc
from typing import Generic

from comb.bodies import PoseT
from comb.system import System


class Renderer(abc.ABC, Generic[PoseT]):
    """Renders a ``System[PoseT]`` to some output (figure, window, etc.)."""

    @abc.abstractmethod
    def render(self, system: System[PoseT]) -> None:
        """Draw or refresh the system at its current body poses."""
