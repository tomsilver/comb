"""Tests for the parsed-library data types.

The data types are pure dataclasses; coverage here is mostly shape: defaults work,
equality is value-based, frozen-ness is enforced. The YAML loader (B2) and validator
(B5) get richer tests; here we just check the schema compiles into useful objects.
"""

import pytest

from comb.spec import (
    BodySpec,
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    LibrarySpec,
    PoseSpec,
    TransitionSpec,
)


def test_library_spec_defaults_to_empty_mappings():
    """Empty libraries are valid: no bodies, no constraints, no transitions."""
    lib = LibrarySpec(name="empty")
    assert not lib.includes
    assert not lib.bodies
    assert not lib.constraints
    assert not lib.transitions


def test_body_spec_anchored_defaults_to_false():
    """Anchoring is opt-in to keep YAML default minimal."""
    body = BodySpec(
        visual_geometry=GeometrySpec(
            shape="rectangle", parameters={"width": 1.0, "height": 0.05}
        ),
        collision_geometry=GeometrySpec(
            shape="rectangle", parameters={"width": 1.0, "height": 0.05}
        ),
        pose=PoseSpec(values={"x": 0.0, "y": 0.0, "theta": 0.0}),
    )
    assert body.anchored is False


def test_constraint_spec_defaults_to_empty_parameters():
    """Constraints with no fixed/initial parameters (e.g. PlanarJoint2D) parse
    cleanly."""
    constraint = ConstraintSpec(type="PlanarJoint2D", body1="world", body2="base")
    assert constraint.fixed_parameters == {}
    assert constraint.initial_parameters == {}


def test_transition_spec_with_inline_trigger():
    """Trigger is reusing ConstraintSpec — same shape, no name field."""
    trigger = ConstraintSpec(
        type="PointEquality2D",
        body1="block",
        body2="link_b",
        fixed_parameters={
            "target_x": 0.0,
            "target_y": 0.0,
            "offset_x": 1.0,
            "offset_y": 0.0,
        },
    )
    transition = TransitionSpec(
        trigger=trigger,
        tolerance=0.05,
        add=(
            GeneratorCallSpec(
                generator="rigid_attachment_2d",
                args={"body1": "link_b", "body2": "block"},
            ),
        ),
        remove=("world_to_block",),
    )
    assert transition.trigger.type == "PointEquality2D"
    assert transition.add[0].generator == "rigid_attachment_2d"
    assert transition.remove == ("world_to_block",)


def test_specs_are_frozen():
    """Dataclasses are frozen — fields cannot be reassigned after construction."""
    geom = GeometrySpec(shape="rectangle")
    with pytest.raises(AttributeError):
        geom.shape = "box"  # type: ignore[misc]


def test_library_spec_value_equality():
    """Two LibrarySpecs with the same content compare equal (dataclass __eq__)."""

    def _build() -> LibrarySpec:
        return LibrarySpec(
            name="arm",
            bodies={
                "base": BodySpec(
                    visual_geometry=GeometrySpec(
                        shape="rectangle", parameters={"width": 0.2, "height": 0.2}
                    ),
                    collision_geometry=GeometrySpec(
                        shape="rectangle", parameters={"width": 0.2, "height": 0.2}
                    ),
                    pose=PoseSpec(values={"x": 0.0, "y": 0.0, "theta": 0.0}),
                    anchored=True,
                ),
            },
        )

    assert _build() == _build()


def test_generator_call_spec_carries_arbitrary_args():
    """``args`` is intentionally typed as Mapping[str, Any] — body refs and tuples
    coexist."""
    call = GeneratorCallSpec(
        generator="point_pin_2d",
        args={"body1": "door", "body2": "link_b", "body2_offset": (0.5, 0.0)},
    )
    assert call.args["body2_offset"] == (0.5, 0.0)
