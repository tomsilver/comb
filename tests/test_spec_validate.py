"""Tests for the spec-level validator."""

import pytest

from comb.spec import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    InitialModeSpec,
    LibrarySpec,
    PoseSpec,
    SpecValidationError,
    TaskSpec,
    TransitionSpec,
    validate_library,
    validate_task,
)

# --- builders ---


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


def _two_body_lib(*, joint_type: str = "PlanarJoint2D") -> LibrarySpec:
    """A minimal library: anchored ``world`` + free ``base``, joined by a constraint."""
    return LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "base": _body(),
        },
        constraints={
            "joint": ConstraintSpec(type=joint_type, body1="world", body2="base"),
        },
    )


# --- library validation ---


def test_validate_minimal_library_succeeds() -> None:
    """A library with one anchored body and a single PlanarJoint2D validates."""
    validate_library(_two_body_lib())


def test_unknown_constraint_type_raises() -> None:
    """Constraint ``type`` must appear in CONSTRAINT_TYPES_2D."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="NonexistentJoint", body1="a", body2="b")
        },
    )
    with pytest.raises(
        SpecValidationError, match=r"unknown constraint type 'NonexistentJoint'"
    ):
        validate_library(lib)


def test_constraint_body_reference_must_exist() -> None:
    """Body1 / body2 must name a declared body."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True)},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="ghost"),
        },
    )
    with pytest.raises(
        SpecValidationError, match=r"body2: 'ghost' is not a declared body"
    ):
        validate_library(lib)


def test_constraint_fixed_parameter_keys_must_match() -> None:
    """Fixed-parameter keys must exactly match the constraint class's expected names."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "pin": ConstraintSpec(
                type="FixedJoint2D",
                body1="a",
                body2="b",
                fixed_parameters={"tx": 0.0, "ty": 0.0},  # missing 'theta'
            )
        },
    )
    with pytest.raises(SpecValidationError, match=r"fixed_parameters: expected keys"):
        validate_library(lib)


def test_constraint_initial_parameter_keys_extras_rejected() -> None:
    """Initial parameters can be a subset of ``parameter_names`` but not a superset."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="a",
                body2="b",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                initial_parameters={"angle": 0.0, "ghost": 1.0},
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"unexpected keys \['ghost'\]"):
        validate_library(lib)


def test_unknown_geometry_shape_raises() -> None:
    """Geometry shape must be one we know how to validate."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "a": BodySpec(
                visual_geometry=GeometrySpec(
                    shape="triangle", parameters={"side": 1.0}
                ),
                collision_geometry=_rect(),
                pose=_zero_pose(),
                anchored=True,
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"unknown shape 'triangle'"):
        validate_library(lib)


def test_geometry_missing_required_parameter() -> None:
    """A rectangle without ``height`` is rejected with a clear message."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "a": BodySpec(
                visual_geometry=GeometrySpec(
                    shape="rectangle", parameters={"width": 1.0}
                ),
                collision_geometry=_rect(),
                pose=_zero_pose(),
                anchored=True,
            ),
        },
    )
    with pytest.raises(
        SpecValidationError, match=r"missing geometry parameters \['height'\]"
    ):
        validate_library(lib)


def test_pose_keys_must_be_se2() -> None:
    """Poses with the wrong keyset are rejected."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "a": BodySpec(
                visual_geometry=_rect(),
                collision_geometry=_rect(),
                pose=PoseSpec(values={"x": 0.0, "y": 0.0}),  # missing theta
                anchored=True,
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"expected SE\(2\) pose keys"):
        validate_library(lib)


def test_unknown_generator_in_transition_raises() -> None:
    """Transition.add must reference a generator in GENERATORS_2D."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="b")
        },
        transitions={
            "t": TransitionSpec(
                trigger=ConstraintSpec(
                    type="PointEquality2D",
                    body1="a",
                    body2="b",
                    fixed_parameters={
                        "target_x": 0.0,
                        "target_y": 0.0,
                        "offset_x": 0.0,
                        "offset_y": 0.0,
                    },
                ),
                tolerance=0.05,
                add=(
                    GeneratorCallSpec(
                        generator="ghost_gen", args={"body1": "a", "body2": "b"}
                    ),
                ),
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"unknown generator 'ghost_gen'"):
        validate_library(lib)


def test_transition_remove_references_unknown_constraint() -> None:
    """Transition.remove names must point at declared constraints."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="b")
        },
        transitions={
            "t": TransitionSpec(
                trigger=ConstraintSpec(
                    type="PointEquality2D",
                    body1="a",
                    body2="b",
                    fixed_parameters={
                        "target_x": 0.0,
                        "target_y": 0.0,
                        "offset_x": 0.0,
                        "offset_y": 0.0,
                    },
                ),
                tolerance=0.05,
                remove=("ghost_constraint",),
            ),
        },
    )
    with pytest.raises(
        SpecValidationError, match=r"'ghost_constraint' is not a declared constraint"
    ):
        validate_library(lib)


def test_transition_trigger_with_mutable_params_rejected() -> None:
    """Triggers must have no mutable parameters (they aren't part of the mode)."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="b")
        },
        transitions={
            "t": TransitionSpec(
                trigger=ConstraintSpec(
                    type="RevoluteJoint2D",
                    body1="a",
                    body2="b",
                    fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
                    initial_parameters={"angle": 0.0},
                ),
                tolerance=0.05,
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"trigger: must have no mutable"):
        validate_library(lib)


def test_transition_tolerance_must_be_positive() -> None:
    """Tolerance ≤ 0 is rejected at the spec level (also enforced at runtime)."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="b")
        },
        transitions={
            "t": TransitionSpec(
                trigger=ConstraintSpec(
                    type="PointEquality2D",
                    body1="a",
                    body2="b",
                    fixed_parameters={
                        "target_x": 0.0,
                        "target_y": 0.0,
                        "offset_x": 0.0,
                        "offset_y": 0.0,
                    },
                ),
                tolerance=0.0,
            ),
        },
    )
    with pytest.raises(SpecValidationError, match=r"tolerance: must be positive"):
        validate_library(lib)


# --- task validation ---


def test_validate_minimal_task_succeeds() -> None:
    """A task that uses all library constraints validates against the library."""
    lib = _two_body_lib()
    task = TaskSpec(name="t", library="lib.yaml", initial_mode=InitialModeSpec())
    validate_task(task, lib)


def test_active_constraint_must_exist() -> None:
    """active_constraints must reference declared library constraints."""
    lib = _two_body_lib()
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(active_constraints=("ghost",)),
    )
    with pytest.raises(
        SpecValidationError,
        match=r"active_constraints\[0\]: 'ghost' is not a declared constraint",
    ):
        validate_task(task, lib)


def test_body_pose_override_must_reference_declared_body() -> None:
    """body_poses keys must name declared library bodies."""
    lib = _two_body_lib()
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(body_poses={"ghost": _zero_pose()}),
    )
    with pytest.raises(
        SpecValidationError, match=r"body_poses\.ghost: not a declared body"
    ):
        validate_task(task, lib)


def test_configuration_constraint_must_exist() -> None:
    """Configuration keys must name declared library constraints."""
    lib = _two_body_lib()
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(configuration={"ghost": {"angle": 0.0}}),
    )
    with pytest.raises(
        SpecValidationError, match=r"configuration\.ghost: not a declared constraint"
    ):
        validate_task(task, lib)


def test_configuration_param_keys_must_match_constraint_class() -> None:
    """Initial-value keys for a constraint can't include names the class doesn't
    expose."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(anchored=True), "b": _body()},
        constraints={
            "joint": ConstraintSpec(
                type="RevoluteJoint2D",
                body1="a",
                body2="b",
                fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
            ),
        },
    )
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(
            configuration={"joint": {"angle": 0.0, "ghost": 1.0}}
        ),
    )
    with pytest.raises(
        SpecValidationError,
        match=r"configuration\.joint: unexpected keys \['ghost'\]",
    ):
        validate_task(task, lib)


def test_goal_constraint_validated_against_library_bodies() -> None:
    """A goal whose body refs don't exist is rejected."""
    lib = _two_body_lib()
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(),
        goal=(
            ConstraintSpec(
                type="PointEquality2D",
                body1="ghost",
                body2="base",
                fixed_parameters={
                    "target_x": 0.0,
                    "target_y": 0.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                },
            ),
        ),
    )
    with pytest.raises(
        SpecValidationError, match=r"goal\[0\]\.body1: 'ghost' is not a declared body"
    ):
        validate_task(task, lib)


# --- connectivity ---


def test_no_anchored_bodies_rejected() -> None:
    """A library with no anchored body cannot have a satisfiable initial mode."""
    lib = LibrarySpec(
        name="lib",
        bodies={"a": _body(), "b": _body()},
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="a", body2="b")
        },
    )
    task = TaskSpec(name="t", library="lib.yaml", initial_mode=InitialModeSpec())
    with pytest.raises(SpecValidationError, match=r"no anchored bodies"):
        validate_task(task, lib)


def test_unreachable_body_detected() -> None:
    """A free-floating body (not connected to any anchor) is rejected."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "base": _body(),
            "free_block": _body(),
        },
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="world", body2="base"),
        },
    )
    task = TaskSpec(name="t", library="lib.yaml", initial_mode=InitialModeSpec())
    with pytest.raises(
        SpecValidationError, match=r"not reachable from any anchor.*free_block"
    ):
        validate_task(task, lib)


def test_active_constraints_subset_used_for_connectivity() -> None:
    """A constraint that's NOT in active_constraints doesn't contribute edges."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "base": _body(),
        },
        constraints={
            "joint": ConstraintSpec(type="PlanarJoint2D", body1="world", body2="base"),
        },
    )
    # Empty active set → base is unreachable.
    task = TaskSpec(
        name="t",
        library="lib.yaml",
        initial_mode=InitialModeSpec(active_constraints=()),
    )
    with pytest.raises(
        SpecValidationError, match=r"not reachable from any anchor.*base"
    ):
        validate_task(task, lib)


def test_two_disjoint_components_each_with_anchor_succeeds() -> None:
    """Two separate connected components are fine if each contains an anchor."""
    lib = LibrarySpec(
        name="lib",
        bodies={
            "world": _body(anchored=True),
            "block": _body(),
            "arm_base": _body(anchored=True),
            "arm_tip": _body(),
        },
        constraints={
            "world_to_block": ConstraintSpec(
                type="FixedJoint2D",
                body1="world",
                body2="block",
                fixed_parameters={"tx": 0.0, "ty": 0.0, "theta": 0.0},
            ),
            "arm": ConstraintSpec(
                type="PlanarJoint2D", body1="arm_base", body2="arm_tip"
            ),
        },
    )
    task = TaskSpec(name="t", library="lib.yaml", initial_mode=InitialModeSpec())
    validate_task(task, lib)
