"""blockpose — pure-Python tooling layer (dataset, synth, analysis, rerun logging).

The compiled core (`blockpose._core`, nanobind) is built by CMake and installed
into the venv separately; this package must import without it so that dataset
and analysis tools work before/without a compiled build.
"""

__version__ = "0.0.1"
