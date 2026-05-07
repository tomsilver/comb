"""Reusable example modes, backed by YAML library files.

Most 2D examples now live as YAML libraries in :mod:`comb.examples.yaml`;
the Python classes in this package are thin wrappers that load + instantiate
the YAML and surface named handles for tests (``ex.base``, ``ex.joint``,
``ex.mode``, ``ex.system``). Two examples remain Python-only: the 3D ones
(the instantiator is SE(2)-only) and ``dual_arm_handover_2d`` (uses a
callable ``remove`` that the spec language doesn't yet express — tracked
in https://github.com/tomsilver/comb/issues/30).
"""

from __future__ import annotations

from pathlib import Path

from comb.spec import (
    InitialModeSpec,
    InstantiatedLibrary,
    InstantiatedTask,
    TaskSpec,
    instantiate_library,
    instantiate_task,
    load_library,
)

_YAML_DIR = Path(__file__).parent / "yaml"


def load_example_library(name: str) -> InstantiatedLibrary:
    """Load and instantiate a bundled example library by name.

    ``name`` is the stem of the YAML file in ``comb/examples/yaml/`` (e.g.
    ``"two_link_arm"`` for ``two_link_arm.lib.yaml``).
    """
    return instantiate_library(load_library(_YAML_DIR / f"{name}.lib.yaml"))


def load_example_default_task(library: InstantiatedLibrary) -> InstantiatedTask:
    """Instantiate a default task that activates every constraint in ``library``.

    Convenience for the bundled examples — they ship as libraries (no task
    file), but their Python wrappers still want a ``System`` whose mode has
    every constraint live.
    """
    task_spec = TaskSpec(
        name=f"{library.spec.name}_default",
        library=f"{library.spec.name}.lib.yaml",
        initial_mode=InitialModeSpec(),
    )
    return instantiate_task(task_spec, library)
