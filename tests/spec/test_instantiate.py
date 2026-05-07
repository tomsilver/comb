"""Tests for the spec → runtime instantiator."""

from pathlib import Path

import numpy as np
import pytest
from spatialmath import SE2

from comb.bodies import Rectangle
from comb.constraints import (
    FixedJoint2D,
    PointEquality2D,
    RevoluteJoint2D,
)
from comb.spec import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    InitialModeSpec,
    LibrarySpec,
    PoseSpec,
    SpecInstantiationError,
    TaskSpec,
    TransitionSpec,
    instantiate_library,
    instantiate_task,
    load_library,
    load_task_file,
)
from comb.transitions import ConstraintTransition

_FIXTURES = Path(__file__).parent / "fixtures"


# --- helpers ---


def _rect(width: float = 0.1, height: float = 0.1) -> GeometrySpec:
    return GeometrySpec(
        shape="rectangle", parameters={"width": width, "height": height}
    )


def _zero_pose() -> PoseSpec:
    return PoseSpec(values={"x": 0.0, "y": 0.0, "theta": 0.0})


def _body(*, anchored: bool = False) -> BodySpec:
    return BodySpec(
        visual_geometry=_rect(),
        collision_geometry=_rect(),
        pose=_zero_pose(),
        anchored=anchored,
    )


# --- library instantiation ---


def test_instantiate_library_builds_bodies_constraints_transitions() -> None:
    """A small library with a transition rebuilds the runtime objects."""
    spec = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "arm": _body(),
            "block": _body(),
        },
        constraints={
            "joint": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="world",
                body2="arm",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                initial_parameters={"angle": 0.0},
            ),
            "pin": ConstraintSpec(
                type="FixedJoint2D",
                body1="world",
                body2="block",
                fixed_parameters={"tx": 0.5, "ty": 0.0, "theta": 0.0},
            ),
        },
        transitions={
            "pickup": TransitionSpec(
                trigger=ConstraintSpec(
                    type="PointEquality2D",
                    body1="block",
                    body2="arm",
                    fixed_parameters={
                        "target_x": 0.0,
                        "target_y": 0.0,
                        "offset_x": 1.0,
                        "offset_y": 0.0,
                    },
                ),
                tolerance=0.05,
                add=(
                    GeneratorCallSpec(
                        generator="rigid_attachment_2d",
                        args={"body1": "arm", "body2": "block"},
                    ),
                ),
                remove=("pin",),
            ),
        },
    )

    library = instantiate_library(spec)

    assert set(library.bodies) == {"world", "arm", "block"}
    assert library.bodies["world"].name == "world"
    assert library.bodies["world"] in library.anchored_bodies
    assert library.bodies["arm"] not in library.anchored_bodies
    assert isinstance(library.constraints["joint"], RevoluteJoint2D)
    assert isinstance(library.constraints["pin"], FixedJoint2D)
    assert library.constraints["joint"].body1 is library.bodies["world"]
    assert library.constraints["joint"].body2 is library.bodies["arm"]
    transition = library.transitions["pickup"]
    assert isinstance(transition, ConstraintTransition)
    assert isinstance(transition.trigger, PointEquality2D)
    assert transition.tolerance == 0.05
    assert transition.remove == (library.constraints["pin"],)


def test_rectangle_geometry_translates_width_height_to_size_xy() -> None:
    """YAML's friendlier ``width``/``height`` map to ``Rectangle(size_x, size_y)``."""
    spec = LibrarySpec(
        name="lib",
        bodies={
            "a": BodySpec(
                visual_geometry=GeometrySpec(
                    shape="rectangle",
                    parameters={
                        "width": 1.5,
                        "height": 0.05,
                        "offset_x": 0.75,
                    },
                ),
                collision_geometry=_rect(),
                pose=_zero_pose(),
                anchored=True,
            ),
        },
    )
    library = instantiate_library(spec)
    geom = library.bodies["a"].visual_geometry
    assert isinstance(geom, Rectangle)
    assert (geom.size_x, geom.size_y, geom.offset_x, geom.offset_y) == (
        1.5,
        0.05,
        0.75,
        0.0,
    )


# --- task instantiation ---


def test_instantiate_task_assembles_system_with_overrides() -> None:
    """body_poses overrides reach the runtime mode; configuration overrides too."""
    library_spec = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "arm": _body(),
        },
        constraints={
            "joint": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="world",
                body2="arm",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                initial_parameters={"angle": 0.0},
            ),
        },
    )
    library = instantiate_library(library_spec)

    task_spec = TaskSpec(
        name="task",
        library="lib.yaml",
        initial_mode=InitialModeSpec(
            body_poses={"arm": PoseSpec(values={"x": 1.0, "y": 0.5, "theta": 0.0})},
            configuration={"joint": {"angle": 0.7}},
        ),
    )

    task = instantiate_task(task_spec, library)

    arm = library.bodies["arm"]
    np.testing.assert_allclose(
        task.system.mode.body_poses[arm].A,
        SE2(1.0, 0.5, 0.0).A,
    )
    joint = library.constraints["joint"]
    np.testing.assert_allclose(task.system.mode.configuration[joint].values, [0.7])


def test_instantiate_task_active_constraints_subset() -> None:
    """active_constraints picks the runtime mode's constraint subset."""
    library_spec = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "arm": _body(),
        },
        constraints={
            "primary": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="world",
                body2="arm",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                initial_parameters={"angle": 0.0},
            ),
            "alternative": ConstraintSpec(
                type="FixedJoint2D",
                body1="world",
                body2="arm",
                fixed_parameters={"tx": 0.0, "ty": 0.0, "theta": 0.0},
            ),
        },
    )
    library = instantiate_library(library_spec)

    task_spec = TaskSpec(
        name="task",
        library="lib.yaml",
        initial_mode=InitialModeSpec(active_constraints=("primary",)),
    )
    task = instantiate_task(task_spec, library)

    assert library.constraints["primary"] in task.system.mode.constraints
    assert library.constraints["alternative"] not in task.system.mode.constraints


def test_instantiate_task_default_active_constraints_is_all() -> None:
    """``active_constraints=None`` makes every library constraint active."""
    library_spec = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "arm": _body(),
        },
        constraints={
            "joint": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="world",
                body2="arm",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                initial_parameters={"angle": 0.0},
            ),
        },
    )
    library = instantiate_library(library_spec)
    task = instantiate_task(
        TaskSpec(name="task", library="lib.yaml", initial_mode=InitialModeSpec()),
        library,
    )
    assert library.constraints["joint"] in task.system.mode.constraints


# --- file-based round trip ---


def test_instantiate_library_from_yaml_fixture() -> None:
    """The full YAML fixture loads + instantiates without errors."""
    library_spec = load_library(_FIXTURES / "example_two_link_arm_with_object.lib.yaml")
    library = instantiate_library(library_spec)
    assert {"base", "link_a", "link_b", "world", "block"} <= set(library.bodies)
    assert "pickup" in library.transitions


def test_instantiate_task_from_yaml_fixture() -> None:
    """The matching task instantiates a System whose transition is the library's."""
    library_spec = load_library(_FIXTURES / "example_two_link_arm_with_object.lib.yaml")
    task_spec = load_task_file(_FIXTURES / "example_pickup_place.task.yaml")
    library = instantiate_library(library_spec)
    task = instantiate_task(task_spec, library)
    assert task.system.transitions == (library.transitions["pickup"],)
    assert len(task.goal) == 1


# --- error paths ---


def test_unknown_geometry_shape_raises() -> None:
    """A geometry that the validator might miss still errors at instantiation time."""
    spec = LibrarySpec(
        name="lib",
        bodies={
            "a": BodySpec(
                visual_geometry=GeometrySpec(shape="circle", parameters={}),
                collision_geometry=_rect(),
                pose=_zero_pose(),
                anchored=True,
            ),
        },
    )
    with pytest.raises(
        SpecInstantiationError, match="cannot instantiate shape 'circle'"
    ):
        instantiate_library(spec)
