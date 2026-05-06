"""Abstract Renderer base class plus an Overlay hook for additional content.

A renderer draws (or refreshes) a ``System`` at its current body poses, plus
any number of ``Overlay`` instances that add extra content (ghost goal
states, waypoint markers, paths, ...). It is generic in the pose type, so a
renderer for ``System[SE2]`` is statically distinct from one for
``System[SE3]``. Concrete subclasses dispatch on the geometry type via
``functools.singledispatchmethod`` so that adding a new shape only requires
registering a method per renderer that supports it.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import Generic

from comb.bodies import Body, PoseT
from comb.system import System


class Renderer(abc.ABC, Generic[PoseT]):
    """Renders a ``System[PoseT]`` plus optional overlays to some output."""

    @abc.abstractmethod
    def render(
        self,
        system: System[PoseT],
        overlays: Iterable[Overlay[PoseT]] = (),
    ) -> None:
        """Draw or refresh the system at its current body poses, then overlays."""

    @abc.abstractmethod
    def draw_body(
        self,
        body: Body[PoseT],
        pose: PoseT,
        *,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        """Draw a single body's visual geometry at ``pose``.

        Provided as a primitive so overlays can render bodies at non-system
        poses (e.g. a ghost goal state) without re-implementing per-backend
        geometry dispatch.
        """


class Overlay(abc.ABC, Generic[PoseT]):
    """Extra content drawn after the system bodies have been rendered.

    Subclasses access whatever drawing primitives they need via the renderer
    they're handed (``renderer.draw_body``, ``renderer.ax`` for matplotlib
    backends, etc.).
    """

    @abc.abstractmethod
    def draw(self, renderer: Renderer[PoseT]) -> None:
        """Render this overlay onto ``renderer``."""
