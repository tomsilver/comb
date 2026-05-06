"""Rendering: drawing a Mode's current body poses to some output surface.

Renderers are organized by backend and dimension because the toolkit and
idioms for 2D vs 3D are quite different (matplotlib works well for 2D but a
3D-quality renderer will likely use a different library — meshcat, pyvista,
etc.). The abstract :class:`comb.rendering.base.Renderer` defines the
interface; concrete renderers live in sibling modules and register
per-geometry drawing methods via single dispatch.

The currently implemented renderer is
:class:`comb.rendering.matplotlib_2d.MatplotlibRenderer2D`, which only handles
``Mode[SE2]``.
"""
