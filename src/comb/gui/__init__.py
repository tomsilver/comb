"""Interactive GUIs for adjusting a Mode's parameters.

GUIs are organized by backend and dimension because the toolkit and idioms for
2D vs 3D are quite different (matplotlib widgets work well for 2D but a
3D-quality GUI will likely use a different library — meshcat, pyvista, etc.).
The currently implemented GUI is :class:`comb.gui.matplotlib_2d.MatplotlibGUI2D`,
which only handles ``Mode[SE2]``.
"""
