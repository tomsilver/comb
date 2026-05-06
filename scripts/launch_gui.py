"""Launch the matplotlib 2D parameter GUI for a named example system.

Usage:
    python scripts/launch_gui.py <example_name>
    python scripts/launch_gui.py --list

Only 2D examples are launchable today (the GUI uses ``MatplotlibGUI2D`` and
there is no 3D GUI yet). The script identifies an example's dimension from
its module-name suffix (``_2d`` / ``_3d``) and refuses to open 3D ones.
"""

import argparse
import importlib
import inspect
import pkgutil
import sys

from spatialmath import SE2

import comb.examples
from comb.gui.matplotlib_2d import MatplotlibGUI2D


def _examples_by_dimension() -> dict[int, list[str]]:
    """Group available example module names by dimension (2 or 3)."""
    groups: dict[int, list[str]] = {2: [], 3: []}
    for _, name, _ in pkgutil.iter_modules(comb.examples.__path__):
        dim = _dimension_from_name(name)
        if dim in groups:
            groups[dim].append(name)
    for names in groups.values():
        names.sort()
    return groups


def _dimension_from_name(module_name: str) -> int | None:
    if module_name.endswith("_2d"):
        return 2
    if module_name.endswith("_3d"):
        return 3
    return None


def _example_class(module_name: str) -> type:
    """Find the example class defined inside ``comb.examples.<module_name>``."""
    module = importlib.import_module(f"comb.examples.{module_name}")
    candidates = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    ]
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one example class in comb.examples.{module_name}, "
            f"found {[c.__name__ for c in candidates]}"
        )
    return candidates[0]


def _print_examples_listing() -> None:
    groups = _examples_by_dimension()
    print("Available 2D examples (launchable):")
    for name in groups[2] or ["  (none)"]:
        print(f"  {name}")
    print()
    print("Available 3D examples (no GUI yet):")
    for name in groups[3] or ["  (none)"]:
        print(f"  {name}")


def main() -> None:
    """Parse args and launch the GUI for the requested example."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "example",
        nargs="?",
        help="Example module under comb.examples (e.g. single_revolute_2d)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available examples and exit",
    )
    args = parser.parse_args()

    if args.list or not args.example:
        _print_examples_listing()
        sys.exit(0 if args.list else 2)

    dim = _dimension_from_name(args.example)
    if dim == 3:
        raise SystemExit(
            f"{args.example!r} is a 3D example; only 2D examples can be launched "
            f"in the GUI today. Run with --list to see what's available."
        )
    if dim != 2:
        raise SystemExit(
            f"Cannot infer dimension of {args.example!r} from its name; "
            f"expected a module ending in '_2d' or '_3d'."
        )

    instance = _example_class(args.example)()
    sample_pose = instance.system.body_poses[instance.system.bodies[0]]
    if not isinstance(sample_pose, SE2):
        raise SystemExit(
            f"{args.example!r} declared 2D by name but its body poses are "
            f"{type(sample_pose).__name__}; cannot launch the 2D GUI."
        )

    gui = MatplotlibGUI2D(instance.system)
    gui.show()


if __name__ == "__main__":
    main()
