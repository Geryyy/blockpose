"""Shared fixtures for the python test suite."""

from __future__ import annotations

from pathlib import Path

from blockpose.io import load_ground_truth

REPO_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"


def snapshot_with_ground_truth(snapshots_dir: Path = SNAPSHOTS_DIR) -> Path:
    """First snapshot directory (by name) with at least one registered block.

    A snapshot's ground truth can be empty: the upstream RGB-based
    registration pipeline sometimes detects a candidate but fails to register
    it (e.g. `registration_ok: false, reason: "result code 6"`), so it's not a
    given that any block was actually placed there -- see
    `tools/eval_real_snapshots.py`'s module docstring. Picking
    `sorted(snapshots_dir.iterdir())[0]` blindly broke as soon as a
    zero-ground-truth snapshot sorted first, so tests that need a snapshot
    with real ground truth should go through this helper instead.
    """
    for candidate in sorted(snapshots_dir.iterdir()):
        if not candidate.is_dir():
            continue
        registrations = candidate / "registrations.yaml"
        if registrations.is_file() and load_ground_truth(registrations):
            return candidate
    raise RuntimeError(f"no snapshot with ground truth found under {snapshots_dir}")
