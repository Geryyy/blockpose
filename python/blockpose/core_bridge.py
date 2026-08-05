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

The C++ core also owns evidence-derived observability classification. Native
calibrated confidence is explicitly unavailable until calibration artifacts
are ported.
"""

from __future__ import annotations

import importlib
from dataclasses import fields
from typing import Any

import numpy as np

from blockpose.core_types import (
    DetectionDiagnostics,
    DetectionParameters,
    DetectionResult,
    HypothesisEvidence,
    InferenceTrace,
    Observability,
    RawHypothesisLineage,
    SensorContext,
    TraceHypothesis,
)
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


_PYTHON_TO_CORE_PARAMETER = {
    "ground_ransac_dist": "ground_ransac_distance",
    "ground_ransac_iters": "ground_ransac_iterations",
    "ransac_dist_thresh": "ransac_distance",
    "ransac_num_iterations": "ransac_iterations",
}
_SHARED_PARAMETER_NAMES = frozenset(
    {
        "voxel_size",
        "dbscan_eps",
        "ground_thickness",
        "ground_normal_min_z",
        "ground_ransac_dist",
        "ground_ransac_iters",
        "local_ground_cell_size",
        "local_ground_clearance",
        "dbscan_min_points",
        "cluster_min_size",
        "cluster_max_size",
        "cluster_min_extent_xy",
        "cluster_max_extent_xy",
        "region_max_extent_xy",
        "cluster_min_extent_z",
        "cluster_max_extent_z",
        "cluster_max_center_z",
        "ransac_dist_thresh",
        "ransac_num_iterations",
        "ransac_search_max_points",
        "ransac_seed",
        "max_planes",
        "min_inliers",
        "top_plane_angle_deg",
        "side_plane_angle_deg",
        "block_dims",
        "candidate_dims",
        "refine_band",
        "refine_iterations",
        "refine_min_points",
        "refine_max_translation",
        "refine_max_rotation_deg",
        "min_score",
        "proposal_max_components",
        "conflict_alternatives",
    }
)
_CORE_PARAMETER_NAMES = frozenset(
    {
        "voxel_size",
        "dbscan_eps",
        "ground_thickness",
        "ground_ransac_distance",
        "ground_ransac_iterations",
        "ground_normal_min_z",
        "local_ground_cell_size",
        "local_ground_clearance",
        "dbscan_min_points",
        "cluster_min_size",
        "cluster_max_size",
        "cluster_min_extent_xy",
        "cluster_max_extent_xy",
        "region_max_extent_xy",
        "cluster_min_extent_z",
        "cluster_max_extent_z",
        "cluster_max_center_z",
        "ransac_distance",
        "ransac_iterations",
        "ransac_search_max_points",
        "ransac_seed",
        "max_planes",
        "min_inliers",
        "top_plane_angle_deg",
        "side_plane_angle_deg",
        "block_dims",
        "candidate_dims",
        "refine_band",
        "refine_iterations",
        "refine_min_points",
        "refine_huber_scale",
        "refine_max_translation",
        "refine_max_rotation_deg",
        "refine_preserve_top_axis_if_gravity_worsens",
        "min_score",
        "fk_seed_min_evidence_gain",
        "fit_wide_proposals",
        "conflict_alternatives",
        "proposal_max_components",
    }
)


def _as_core_parameters(core: Any, params: DetectionParameters | dict[str, Any] | None) -> Any:
    """Translate only the documented shared Python parameters to C++.

    Python's prototype-only switches must not silently change or be ignored by
    the production path. Dict callers may pass native C++ field names; a
    ``DetectionParams`` instance is compared with its defaults so merely
    carrying its complete dataclass surface remains compatible.
    """
    if params is None:
        return core.DetectionParameters()
    if isinstance(params, core.DetectionParameters):
        return params
    defaults = DetectionParameters()
    values = (
        dict(params)
        if isinstance(params, dict)
        else {field.name: getattr(params, field.name) for field in fields(DetectionParameters)}
    )
    unsupported = []
    for name, value in values.items():
        if name in _SHARED_PARAMETER_NAMES or name in _CORE_PARAMETER_NAMES:
            continue
        if isinstance(params, dict) or value != getattr(defaults, name):
            unsupported.append(name)
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValueError(
            f"C++ discovery does not support Python-only parameters: {joined}. "
            "Use blockpose._core.DetectionParameters for native options, or remove "
            "prototype-only settings before calling the C++ bridge."
        )
    native = core.DetectionParameters()
    for name, value in values.items():
        if name not in _SHARED_PARAMETER_NAMES and name not in _CORE_PARAMETER_NAMES:
            continue
        native_name = _PYTHON_TO_CORE_PARAMETER.get(name, name)
        if native_name == "ransac_search_max_points" and value is None:
            raise ValueError(
                "C++ discovery requires ransac_search_max_points to be a positive integer; "
                "the Python-only None/unbounded mode is unsupported. Set an explicit limit."
            )
        setattr(native, native_name, value)
    return native


def _evidence_from_dict(evidence: dict[str, Any]) -> HypothesisEvidence:
    values = {name: evidence[name] for name in HypothesisEvidence.__dataclass_fields__}
    return HypothesisEvidence(**values)


def _diagnostics_from_dict(entry: dict[str, Any]) -> DetectionDiagnostics:
    """Reshape native diagnostics without recalculating their classification."""
    return DetectionDiagnostics(
        evidence=_evidence_from_dict(entry["evidence"]),
        observability=Observability(entry["observability"]),
        features={name: float(value) for name, value in entry["features"].items()},
        calibrated_confidence=(
            None
            if entry["calibrated_confidence"] is None
            else float(entry["calibrated_confidence"])
        ),
        proposal_scale_m=(
            None if entry["proposal_scale_m"] is None else float(entry["proposal_scale_m"])
        ),
        support_height_m=(
            None if entry["support_height_m"] is None else float(entry["support_height_m"])
        ),
    )


def _lineage_from_dict(entry: dict[str, Any], *, source: str) -> RawHypothesisLineage:
    def optional_point(name: str) -> np.ndarray | None:
        return None if entry[name] is None else np.asarray(entry[name], dtype=np.float64)

    return RawHypothesisLineage(
        id=entry["id"],
        proposal_component=int(entry["proposal_component"]),
        region=int(entry["region"]),
        top_plane=int(entry["top_plane"]),
        side_plane=None if entry["side_plane"] is None else int(entry["side_plane"]),
        top_normal=np.asarray(entry["top_normal"], dtype=np.float64),
        top_centroid=np.asarray(entry["top_centroid"], dtype=np.float64),
        side_normal=optional_point("side_normal"),
        side_centroid=optional_point("side_centroid"),
        candidate_dims=tuple(float(v) for v in entry["candidate_dims"]),
        top_only=bool(entry["top_only"]),
        synthesized_pose=_pose_from_dict(entry["synthesized_pose"], source=source),
        evidence=_evidence_from_dict(entry["evidence"]),
        top_support_height_m=float(entry["top_support_height_m"]),
        accepted_to_raw=bool(entry["accepted_to_raw"]),
        passed_score_threshold=bool(entry["passed_score_threshold"]),
        fate=entry["fate"],
    )


def _as_core_sensor_context(core: Any, context: SensorContext) -> Any:
    """Create the native validated ray context; no visibility logic lives here."""
    origin = np.ascontiguousarray(np.asarray(context.origin, dtype=np.float64))
    directions = np.ascontiguousarray(np.asarray(context.ray_directions, dtype=np.float64))
    ranges = np.ascontiguousarray(np.asarray(context.ranges, dtype=np.float64))
    misses = np.ascontiguousarray(np.asarray(context.miss_directions, dtype=np.float64))
    ray_origins = (
        None
        if context.ray_origins is None
        else np.ascontiguousarray(np.asarray(context.ray_origins, dtype=np.float64))
    )
    miss_origins = (
        None
        if context.miss_origins is None
        else np.ascontiguousarray(np.asarray(context.miss_origins, dtype=np.float64))
    )
    return core.SensorContext.from_arrays(
        origin, directions, ranges, misses, ray_origins, miss_origins
    )


def detect_blocks_detailed_cpp(
    points: np.ndarray,
    *,
    source_label: str = "cpp_core",
    params: DetectionParameters | dict[str, Any] | None = None,
    sensor_context: SensorContext | None = None,
) -> DetectionResult:
    """Run the C++ detector core and reshape its result into `DetectionResult`.

    Args:
        points: (N, 3) world-frame point cloud. Converted to a contiguous
            float64 array before being passed to `blockpose._core.discover`.
        source_label: Value stamped onto each returned `BlockPose.source`.

    Returns:
        A `DetectionResult` populated from the C++ core's poses, diagnostics,
        and trace. Stage timings are native steady-clock measurements keyed
        like the Python detector's timings.

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

    # Preserve the one-argument call for an extension built before typed
    # parameters landed; supplying params deliberately requires the new API.
    native_params = None if params is None else _as_core_parameters(_core, params)
    native_context = (
        None if sensor_context is None else _as_core_sensor_context(_core, sensor_context)
    )
    native_points = np.ascontiguousarray(pts, dtype=np.float64)
    if native_context is not None:
        result = _core.discover(
            native_points,
            _core.DetectionParameters() if native_params is None else native_params,
            native_context,
        )
    elif native_params is None:
        result = _core.discover(native_points)
    else:
        result = _core.discover(native_points, native_params)

    poses = [_pose_from_dict(entry, source=source_label) for entry in result["poses"]]
    diagnostics = [_diagnostics_from_dict(entry) for entry in result.get("diagnostics", ())]
    if len(diagnostics) != len(poses):
        raise ValueError("C++ discovery returned diagnostics not aligned with final poses")

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

    raw_lineage = tuple(
        _lineage_from_dict(entry, source=source_label) for entry in result.get("raw_lineage", ())
    )
    trace = InferenceTrace(
        downsampled_points=np.empty((0, 3)),
        above_support_points=np.empty((0, 3)),
        proposal_points=(),
        raw_hypotheses=raw_hypotheses,
        threshold_hypotheses=(),
        final_hypotheses=final_hypotheses,
        raw_lineage=raw_lineage,
        proposal_diagnostics=tuple(result.get("proposal_diagnostics", ())),
        wide_proposal_skips=tuple(result.get("wide_proposal_skips", ())),
        plane_region_diagnostics=tuple(result.get("plane_region_diagnostics", ())),
        plane_patch_diagnostics=tuple(result.get("plane_patch_diagnostics", ())),
        pairing_diagnostics=tuple(result.get("pairing_diagnostics", ())),
    )

    return DetectionResult(
        poses=poses,
        diagnostics=diagnostics,
        stage_counts=dict(result["counts"]),
        stage_timings_ms={
            name: float(value) for name, value in result.get("timings_ms", {}).items()
        },
        trace=trace,
    )


def refine_pose_cpp(
    pose: BlockPose, points: np.ndarray, dims: tuple[float, float, float], **kwargs: Any
) -> tuple[BlockPose, bool]:
    """Call the native C++ cuboid refiner for annotation-only local fitting."""
    core = importlib.import_module("blockpose._core")
    result = core.refine_pose(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(pose.position, dtype=np.float64),
        np.ascontiguousarray(pose.rotation, dtype=np.float64),
        np.ascontiguousarray(dims, dtype=np.float64),
        **kwargs,
    )
    refined = BlockPose(
        np.asarray(result["position"], dtype=np.float64),
        np.asarray(result["rotation"], dtype=np.float64),
        pose.confidence,
        pose.source,
        tuple(float(value) for value in dims),
    )
    return refined, bool(result["refined"])
