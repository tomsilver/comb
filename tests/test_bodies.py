"""Tests for bodies module."""

import numpy as np
from spatialmath import SE2, SE3

from comb.bodies import Body, Box, Rectangle


def test_body_se3():
    """A Body[SE3] holds an SE(3) pose and a 3D Geometry[SE3] (e.g. Box)."""
    body: Body[SE3] = Body(
        name="link1",
        pose=SE3(),
        visual_geometry=Box(0.2, 0.2, 0.2),
        collision_geometry=Box(0.2, 0.2, 0.2),
    )
    assert body.name == "link1"
    assert isinstance(body.pose, SE3)
    np.testing.assert_array_equal(body.pose.A, np.eye(4))
    assert isinstance(body.visual_geometry, Box)


def test_body_se2():
    """A Body[SE2] holds an SE(2) pose and a 2D Geometry[SE2] (e.g. Rectangle)."""
    body: Body[SE2] = Body(
        name="base",
        pose=SE2(1.0, 2.0, 0.5),
        visual_geometry=Rectangle(0.4, 0.3),
        collision_geometry=Rectangle(0.4, 0.3),
    )
    assert isinstance(body.pose, SE2)
    np.testing.assert_array_equal(body.pose.t, [1.0, 2.0])
    assert body.pose.theta() == 0.5
    assert isinstance(body.visual_geometry, Rectangle)


def test_box_geometry():
    """Box stores its three extents."""
    b = Box(size_x=1.0, size_y=2.0, size_z=3.0)
    assert (b.size_x, b.size_y, b.size_z) == (1.0, 2.0, 3.0)


def test_rectangle_geometry():
    """Rectangle stores its two extents."""
    r = Rectangle(size_x=0.5, size_y=0.25)
    assert (r.size_x, r.size_y) == (0.5, 0.25)
