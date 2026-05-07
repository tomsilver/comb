"""YAML library / task files for the bundled examples.

Loaded by the Python ``comb.examples.*`` wrapper classes (which are now thin
shims around :func:`comb.spec.load_library` /
:func:`comb.spec.instantiate_library`) and used directly by the
``comb plan`` / ``comb render`` / ``comb validate`` CLI for end-to-end
demonstration. New examples should be added here as YAML; the Python
wrappers exist only to keep ergonomic test access (named body / constraint
attributes) working.
"""
