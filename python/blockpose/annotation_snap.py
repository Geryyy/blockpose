"""Thin UI-facing adapter for native annotation snap fitting."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, fields

import numpy as np

from blockpose.pose import BLOCK_DIMS


@dataclass(frozen=True)
class SnapParams:
    crop_radius_m: float = 1.05
    voxel_m: float = 0.018
    plane_tolerance_m: float = 0.035
    surface_tolerance_m: float = 0.045
    min_points: int = 45
    min_inliers: int = 30
    min_score: float = 0.30
    ambiguity_score_ratio: float = 0.96
    ambiguity_distance_m: float = 0.08
    reject_ambiguous: bool = True


@dataclass(frozen=True)
class SnapResult:
    success: bool
    position: np.ndarray | None
    rotation: np.ndarray | None
    dimensions: tuple[float, float, float] | None
    score: float
    inlier_count: int
    reason: str


def snap_block_to_cloud(
    points: np.ndarray, seed: np.ndarray, params: SnapParams | None = None
) -> SnapResult:
    """Call the C++ annotation fitter; this module deliberately contains no geometry."""
    core = importlib.import_module("blockpose._core")
    native = core.AnnotationSnapParameters()
    for field in fields(SnapParams()):
        setattr(native, field.name, getattr(params or SnapParams(), field.name))
    try:
        result = core.annotation_snap(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(seed, dtype=np.float64), native,
        )
    except (TypeError, ValueError) as error:
        return SnapResult(False, None, None, None, 0.0, 0, str(error))
    if not result["success"]:
        return SnapResult(
            False,
            None,
            None,
            None,
            float(result["score"]),
            int(result["inlier_count"]),
            result["reason"],
        )
    return SnapResult(
        True,
        np.asarray(result["position"]),
        np.asarray(result["rotation"]),
        BLOCK_DIMS,
        float(result["score"]),
        int(result["inlier_count"]),
        "",
    )
