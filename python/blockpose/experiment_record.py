"""Provenance and strict scoring for recorded detector experiments.

An audit invalidated a whole experiment report because a detector default
changed underneath it: the report's own documented command no longer
reproduced its own tables, and nothing in the output said which parameters had
produced them. Two habits prevent a repeat, and both live here so every
experiment gets them the same way:

* `experiment_stamp` records the git commit and the exact parameters a result
  was produced with, so a number can always be re-derived or shown to be stale.
* `strict_match` scores a detection on orientation and overlap, not centre
  distance alone. Centre-only scoring counts a box 0.29 m away with 88 degrees
  of orientation error as a hit; measured over the annotated set it inflated
  recall by two blocks at every setting.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockpose.evaluate import rotation_error_deg
from blockpose.pose import BlockPose

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults chosen to be strict enough to catch a wrong-orientation match while
# still admitting a genuine fit: measured recoveries sit at 1-3 degrees and
# IoU 0.87-0.97, well inside these, and the two spurious centre-only matches
# they exclude are at 86-88 degrees.
DEFAULT_MAX_CENTRE_M = 0.30
DEFAULT_MAX_ROTATION_DEG = 30.0
DEFAULT_MIN_IOU = 0.25


def git_commit(repo_root: Path | None = None) -> str:
    """Short commit of the tree a result was produced from, or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root or _REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def experiment_stamp(**parameters: Any) -> dict[str, Any]:
    """A provenance header to embed in every experiment artifact.

    Pass the actual parameter objects used, not a description of them --
    dataclasses are expanded so a reader can diff a stale result against the
    current defaults instead of guessing which ones were in force.
    """
    recorded: dict[str, Any] = {}
    for name, value in parameters.items():
        if is_dataclass(value) and not isinstance(value, type):
            recorded[name] = {k: _plain(v) for k, v in asdict(value).items()}
        else:
            recorded[name] = _plain(value)
    return {"git_commit": git_commit(), "parameters": recorded}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def oriented_box_iou(
    a_centre: np.ndarray,
    a_rotation: np.ndarray,
    a_dims: tuple[float, float, float],
    b_centre: np.ndarray,
    b_rotation: np.ndarray,
    b_dims: tuple[float, float, float],
    samples: int = 20000,
    seed: int = 0,
) -> float:
    """Monte-Carlo IoU of two oriented boxes.

    Sampled rather than solved exactly: an exact oriented-box intersection is a
    convex-polyhedron clip, and this only ever gates a match at a coarse
    threshold. 20k samples give roughly +/-0.01 at IoU 0.9, far finer than the
    0.25 gate, and the estimate is deterministic for a fixed seed so a recorded
    result stays reproducible.
    """
    a_centre = np.asarray(a_centre, dtype=float)
    b_centre = np.asarray(b_centre, dtype=float)
    a_half = np.asarray(a_dims, dtype=float) / 2.0
    b_half = np.asarray(b_dims, dtype=float) / 2.0
    # Sample inside A only, then correct: |A n B| / |A u B| where
    # |A u B| = |A| + |B| - |A n B|, and |A n B| = P(in B | in A) * |A|.
    rng = np.random.default_rng(seed)
    local = rng.uniform(-a_half, a_half, size=(samples, 3))
    world = local @ np.asarray(a_rotation, dtype=float).T + a_centre
    local_b = (world - b_centre) @ np.asarray(b_rotation, dtype=float)
    in_b = np.all(np.abs(local_b) <= b_half, axis=1)
    vol_a = float(np.prod(np.asarray(a_dims, dtype=float)))
    vol_b = float(np.prod(np.asarray(b_dims, dtype=float)))
    intersection = float(in_b.mean()) * vol_a
    union = vol_a + vol_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def strict_match(
    detected: BlockPose,
    truth: BlockPose,
    *,
    max_centre_m: float = DEFAULT_MAX_CENTRE_M,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
    min_iou: float = DEFAULT_MIN_IOU,
) -> dict[str, Any]:
    """Score one detection against one ground-truth block on all three terms.

    Returns the individual errors alongside the verdict so a report can show
    centre-only and strict numbers side by side -- the difference between them
    is itself the evidence that a result is real rather than a near-miss.
    Rotation is symmetry-aware via `evaluate.rotation_error_deg`, so relabelling
    a symmetric block's faces is not counted as an error.
    """
    centre = float(np.linalg.norm(np.asarray(detected.position) - np.asarray(truth.position)))
    rotation = float(rotation_error_deg(detected, truth))
    iou = oriented_box_iou(
        detected.position,
        detected.rotation,
        tuple(getattr(detected, "dims", None) or truth.dims),
        truth.position,
        truth.rotation,
        tuple(truth.dims),
    )
    # NaN rotation means the two poses describe differently shaped boxes, which
    # is a shape mismatch rather than a near-miss -- never a match.
    ok = (
        centre <= max_centre_m
        and np.isfinite(rotation)
        and rotation <= max_rotation_deg
        and iou >= min_iou
    )
    return {
        "centre_m": centre,
        "rotation_deg": rotation,
        "iou": iou,
        "centre_only_match": centre <= max_centre_m,
        "strict_match": bool(ok),
    }
