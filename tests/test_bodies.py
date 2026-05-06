"""Tests for bodies module."""

import numpy as np
from spatialmath import SE2, SE3

from comb.bodies import Body, Box, Cylinder, Mesh, Sphere


def test_body_with_se3_pose():
    """A Body can hold an SE(3) pose from spatialmath."""
    body = Body(
        name="link1",
        pose=SE3(),
        visual_geometry=Sphere(0.1),
        collision_geometry=Box(0.2, 0.2, 0.2),
    )
    assert body.name == "link1"
    assert isinstance(body.pose, SE3)
    np.testing.assert_array_equal(body.pose.A, np.eye(4))


def test_body_with_se2_pose():
    """A Body can hold an SE(2) pose."""
    body = Body(
        name="base",
        pose=SE2(1.0, 2.0, 0.5),
        visual_geometry=Sphere(0.1),
        collision_geometry=Sphere(0.1),
    )
    assert isinstance(body.pose, SE2)
    np.testing.assert_array_equal(body.pose.t, [1.0, 2.0])
    assert body.pose.theta() == 0.5


def test_body_with_translation_only_pose():
    """A Body can use a numpy vector for R^n translation-only poses."""
    body = Body(
        name="particle",
        pose=np.array([1.0, 2.0]),
        visual_geometry=Sphere(0.05),
        collision_geometry=Sphere(0.05),
    )
    assert isinstance(body.pose, np.ndarray)
    np.testing.assert_array_equal(body.pose, [1.0, 2.0])


def test_primitive_geometry():
    """Primitive geometries store their parameters."""
    s = Sphere(radius=0.5)
    assert s.radius == 0.5
    b = Box(size_x=1.0, size_y=2.0, size_z=3.0)
    assert (b.size_x, b.size_y, b.size_z) == (1.0, 2.0, 3.0)
    c = Cylinder(radius=0.25, height=1.5)
    assert c.radius == 0.25 and c.height == 1.5


def test_mesh_geometry():
    """Mesh stores vertices and faces with the expected shapes."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(vertices=vertices, faces=faces)
    assert mesh.vertices.shape == (3, 3)
    assert mesh.faces.shape == (1, 3)
