"""Tests for the include resolver: ``load_library`` over a graph of files."""

from pathlib import Path

import pytest

from comb.spec import LibraryLoadError, load_library

_FIXTURES = Path(__file__).parent / "fixtures"


def test_no_includes_passthrough() -> None:
    """A library with no includes is loaded as-is, with empty includes."""
    lib = load_library(_FIXTURES / "minimal_library.yaml")
    assert lib.name == "minimal"
    assert not lib.includes


def test_simple_include_merges_bodies() -> None:
    """One-level include: bodies from both files appear in the merged spec."""
    lib = load_library(_FIXTURES / "include_simple" / "a.yaml")
    assert lib.name == "a"
    assert not lib.includes
    assert set(lib.bodies) == {"body_from_a", "body_from_b"}


def test_diamond_include_loads_each_file_once() -> None:
    """A → {B,C} → D: D loaded once, no name-collision error from double-include."""
    lib = load_library(_FIXTURES / "include_diamond" / "a.yaml")
    assert set(lib.bodies) == {"body_a", "body_b", "body_c", "body_d"}


def test_cycle_detected() -> None:
    """A → B → A raises with a clear cycle message."""
    with pytest.raises(LibraryLoadError, match="include cycle"):
        load_library(_FIXTURES / "include_cycle" / "a.yaml")


def test_self_include_detected() -> None:
    """A library that includes itself is also a cycle."""
    with pytest.raises(LibraryLoadError, match="include cycle"):
        load_library(_FIXTURES / "include_self_cycle" / "self.yaml")


def test_body_name_collision_detected() -> None:
    """Two libs declaring the same body name → error mentioning both files."""
    with pytest.raises(
        LibraryLoadError,
        match=r"body name 'foo' declared in both",
    ):
        load_library(_FIXTURES / "include_body_collision" / "a.yaml")


def test_constraint_name_collision_detected() -> None:
    """Two libs declaring the same constraint name → error."""
    with pytest.raises(
        LibraryLoadError,
        match=r"constraint name 'joint' declared in both",
    ):
        load_library(_FIXTURES / "include_constraint_collision" / "a.yaml")


def test_include_not_found() -> None:
    """A missing include path is reported clearly."""
    with pytest.raises(LibraryLoadError, match="include not found"):
        load_library(_FIXTURES / "include_missing" / "a.yaml")


def test_include_paths_resolved_relative_to_declaring_file() -> None:
    """An include like ``sub/inner.yaml`` resolves relative to the declaring file."""
    lib = load_library(_FIXTURES / "include_relative" / "outer.yaml")
    assert "body_from_inner" in lib.bodies


def test_root_file_not_found() -> None:
    """A missing root file raises ``LibraryLoadError``, not ``FileNotFoundError``."""
    with pytest.raises(LibraryLoadError, match="library file not found"):
        load_library(_FIXTURES / "definitely_does_not_exist.yaml")


def test_root_libraries_name_wins() -> None:
    """The merged library's ``name`` is the root file's, not an included one's."""
    lib = load_library(_FIXTURES / "include_simple" / "a.yaml")
    assert lib.name == "a"
