"""Tests for the single-file YAML library loader.

Fixture YAML lives in ``tests/spec/fixtures/``. Storing them as files (rather than
triple-quoted strings inline in the test source) keeps docformatter from mistaking them
for docstrings and rewriting their content.
"""

from pathlib import Path

import pytest

from comb.spec import (
    ConstraintSpec,
    GeneratorCallSpec,
    GeometrySpec,
    LibraryLoadError,
    LibrarySpec,
    PoseSpec,
    TransitionSpec,
    load_library_file,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _fx(name: str) -> Path:
    return _FIXTURES / name


def test_load_full_library() -> None:
    """A library with bodies, constraints, and a transition round-trips."""
    lib = load_library_file(_fx("full_library.yaml"))

    assert isinstance(lib, LibrarySpec)
    assert lib.name == "pickup_arm"
    assert lib.includes == ("other.lib.yaml",)
    assert set(lib.bodies) == {"base", "link_b"}
    assert lib.bodies["base"].anchored is True
    assert lib.bodies["link_b"].anchored is False
    assert lib.bodies["base"].visual_geometry == GeometrySpec(
        shape="rectangle", parameters={"width": 0.2, "height": 0.2}
    )
    assert lib.bodies["link_b"].pose == PoseSpec(
        values={"x": 1.0, "y": 0.0, "theta": 0.0}
    )
    assert lib.constraints["joint_ab"] == ConstraintSpec(
        type="RevoluteJoint2D",
        body1="base",
        body2="link_b",
        fixed_parameters={"origin_x": 0.0, "origin_y": 0.0},
        initial_parameters={"angle": 0.0},
    )
    pickup = lib.transitions["pickup"]
    assert isinstance(pickup, TransitionSpec)
    assert pickup.trigger.type == "PointEquality2D"
    assert pickup.tolerance == 0.05
    assert pickup.add == (
        GeneratorCallSpec(
            generator="rigid_attachment_2d",
            args={"body1": "link_b", "body2": "block"},
        ),
    )
    assert pickup.remove == ("world_to_block",)


def test_load_minimal_library() -> None:
    """Only ``name`` is required; everything else defaults to empty."""
    assert load_library_file(_fx("minimal_library.yaml")) == LibrarySpec(name="minimal")


def test_missing_name_raises() -> None:
    """``name`` is the only mandatory top-level key."""
    with pytest.raises(LibraryLoadError, match="missing required key 'name'"):
        load_library_file(_fx("missing_name.yaml"))


def test_top_level_must_be_mapping() -> None:
    """A YAML list at top level is rejected with a clear message."""
    with pytest.raises(LibraryLoadError, match="top-level must be a mapping"):
        load_library_file(_fx("top_level_list.yaml"))


def test_invalid_yaml_raises_load_error() -> None:
    """Yaml parse errors are wrapped, not propagated as PyYAML exceptions."""
    with pytest.raises(LibraryLoadError, match="failed to parse YAML"):
        load_library_file(_fx("invalid_yaml.yaml"))


def test_body_missing_pose_raises() -> None:
    """A body without a pose is rejected; the error path points at the body."""
    with pytest.raises(
        LibraryLoadError, match=r"bodies\.base: missing required key 'pose'"
    ):
        load_library_file(_fx("body_missing_pose.yaml"))


def test_geometry_must_be_mapping() -> None:
    """A scalar where a geometry mapping was expected is caught."""
    with pytest.raises(LibraryLoadError, match=r"visual_geometry: expected mapping"):
        load_library_file(_fx("geometry_not_mapping.yaml"))


def test_pose_value_must_be_numeric() -> None:
    """A pose value that isn't a number is rejected."""
    with pytest.raises(LibraryLoadError, match=r"pose\.x: expected number"):
        load_library_file(_fx("pose_value_not_numeric.yaml"))


def test_anchored_must_be_bool() -> None:
    """``anchored`` is strictly typed; a string ``"true"`` is rejected."""
    with pytest.raises(LibraryLoadError, match=r"anchored: expected bool"):
        load_library_file(_fx("anchored_not_bool.yaml"))


def test_transition_missing_trigger_raises() -> None:
    """A transition without a trigger is rejected."""
    with pytest.raises(
        LibraryLoadError,
        match=r"transitions\.pickup: missing required key 'trigger'",
    ):
        load_library_file(_fx("transition_missing_trigger.yaml"))


def test_transition_add_must_be_list() -> None:
    """``add`` is a list of generator calls, not a mapping."""
    with pytest.raises(LibraryLoadError, match=r"add: expected list"):
        load_library_file(_fx("transition_add_not_list.yaml"))


def test_unknown_generator_name_passes_through() -> None:
    """Generator names aren't validated at load time — they're just strings.

    The validator (B5) checks them against ``GENERATORS_2D``. The loader's job is only
    to enforce schema shape.
    """
    lib = load_library_file(_fx("unknown_generator.yaml"))
    assert lib.transitions["weird"].add[0].generator == "not_a_real_generator"


def test_includes_must_be_list_of_strings() -> None:
    """An ``includes`` entry that isn't a string is rejected with index info."""
    with pytest.raises(LibraryLoadError, match=r"includes\[1\]: expected string"):
        load_library_file(_fx("includes_not_strings.yaml"))
