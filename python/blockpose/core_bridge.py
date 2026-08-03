"""Thin typed bridge onto the C++ detector core (`blockpose._core`).

Stage 2 of the architecture-consolidation plan makes the C++ detector core
(``cpp/include/blockpose/core/``, exposed to Python via the nanobind
extension module ``blockpose._core``) the single source of truth, with the
pure-Python reimplementation in ``blockpose.detection`` eventually retired
once callers migrate.

This module does not reimplement any detection logic. It calls
``blockpose._core.discover(points)`` -- which returns a plain ``dict`` -- and
reshapes that dict into the exact same dataclasses
``blockpose.detection.detect_blocks_detailed`` already returns
(``DetectionResult``, ``InferenceTrace``, ``TraceHypothesis``, and
``blockpose.pose.BlockPose``), so existing callers can switch to the
C++-backed path with a minimal code change (call this function instead of
``detect_blocks_detailed``) rather than a rewrite.

The compiled extension is imported lazily, inside
``detect_blocks_detailed_cpp``, not at module import time -- mirroring the
lazy torch import style used by ``rgb_verifier.py``/``rgb_region_proposer.py``
-- so importing ``blockpose`` or this module never requires the extension to
be built.

Known, deliberate gap (not papered over here): the C++ core does not yet
compute the `Observability` classification or per-stage timings that the
pure-Python detector produces, so `DetectionResult.diagnostics` is always
`[]` and `stage_timings_ms` is always `{}` for the C++-backed path.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from blockpose.detection import DetectionResult, InferenceTrace, TraceHypothesis
from blockpose.pose import BlockPose


def _pose_from_dict(pose: dict[str, Any], *, source: str) -> BlockPose:
    """Build a `BlockPose` from one of the core's ``{position, rotation, dims,
    confidence}`` pose dicts (used both for `result["poses"]` entries and for
    the `"pose"` sub-dict of each `result["refined_candidate_trace"]` entry).
    """
    position = np.asarray(pose["position"], dtype=np.float64)
    rotation = np.asarray(pose["rotation"], dtype=np.float64).reshape(3, 3)
    dims = tuple(float(value) for value in pose["dims"])
    confidence = float(pose["confidence"])
    return BlockPose(
        position=position,
        rotation=rotation,
        confidence=confidence,
        source=source,
        dims=dims,
    )


def detect_blocks_detailed_cpp(
    points: np.ndarray, *, source_label: str = "cpp_core"
) -> DetectionResult:
    """Run the C++ detector core and reshape its result into `DetectionResult`.

    Args:
        points: (N, 3) world-frame point cloud. Converted to a contiguous
            float64 array before being passed to `blockpose._core.discover`.
        source_label: Value stamped onto each returned `BlockPose.source`.

    Returns:
        A `DetectionResult` with `poses` and `trace` populated from the C++
        core's output. `diagnostics` is always `[]` and `stage_timings_ms` is
        always `{}` -- the C++ core does not compute per-hypothesis
        `Observability`/feature diagnostics or per-stage timings yet.

    Raises:
        ImportError: if the compiled `blockpose._core` extension is not
            built/importable.
        ValueError: if `points` is not a finite (N, 3) real-numeric array,
            matching `blockpose.detection`'s own input validation.
    """
    try:
        # `importlib.import_module` always resolves through `sys.modules`;
        # `import blockpose._core as _core` would instead resolve via
        # `getattr(blockpose, "_core")` once any prior successful import has
        # cached that attribute on the `blockpose` package object, silently
        # bypassing a test's `sys.modules["blockpose._core"]` monkeypatch.
        _core = importlib.import_module("blockpose._core")
    except ImportError as error:
        raise ImportError(
            "blockpose._core is not built/importable. This is the compiled "
            "nanobind extension for the C++ detector core "
            "(cpp/bindings/blockpose_core_module.cpp); build/install it "
            "(e.g. via the project's CMake build, or `pip install -e .` from "
            "the repo root) so that python/blockpose/_core*.so exists, then "
            "retry detect_blocks_detailed_cpp()."
        ) from error

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if not np.issubdtype(pts.dtype, np.number) or np.iscomplexobj(pts):
        raise ValueError("points must contain real numeric coordinates")
    pts = pts.astype(np.float64, copy=False)
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite coordinates")

    result = _core.discover(np.ascontiguousarray(pts, dtype=np.float64))

    poses = [_pose_from_dict(entry, source=source_label) for entry in result["poses"]]

    # Tolerate an older/mid-rebuild extension that does not yet return this
    # key -- a graceful empty trace, not a crash, since the C++ side of this
    # contract is being widened concurrently.
    raw_trace_entries = result.get("refined_candidate_trace", ())
    raw_hypotheses = tuple(
        TraceHypothesis(
            id=entry["id"],
            pose=_pose_from_dict(entry["pose"], source=source_label),
            score=float(entry["score"]),
            stage=entry["stage"],
            fate=entry["fate"],
        )
        for entry in raw_trace_entries
    )
    final_hypotheses = tuple(h for h in raw_hypotheses if h.fate == "final")

    trace = InferenceTrace(
        downsampled_points=np.empty((0, 3)),
        above_support_points=np.empty((0, 3)),
        proposal_points=(),
        raw_hypotheses=raw_hypotheses,
        threshold_hypotheses=(),
        final_hypotheses=final_hypotheses,
    )

    return DetectionResult(
        poses=poses,
        diagnostics=[],
        stage_counts=dict(result["counts"]),
        stage_timings_ms={},
        trace=trace,
    )
