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


FACE_NORMALS = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

# The block cross-section is a 0.6 x 0.6 square, giving it 4-fold rotational
# symmetry about its long (local +X) axis -- see `symmetry_equivalents` in
# pose.py. That leaves exactly two physically distinct face classes: the
# +X/-X square ends, and the +Y/-Y/+Z/-Z quad of congruent 0.9 x 0.6
# rectangles (all equivalent under the symmetry -- "top" and "side" are the
# SAME class here). "Auto" is not a third physical class; it just widens the
# candidate search to all six signed axes so the caller need not know which
# pair is near the support plane.
FACE_CLASSES = {
    "Auto": tuple(FACE_NORMALS),
    "End": ("+X", "-X"),
    "Long face": ("+Y", "-Y", "+Z", "-Z"),
}


def snap_block_face_to_plane(
    points: np.ndarray,
    click: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    dimensions: tuple[float, float, float],
    face: str,
    *,
    radius_m: float = 0.35,
    min_points: int = 20,
) -> SnapResult:
    """Translate a selected block face class onto the local support plane.

    `face` is a class from `FACE_CLASSES` ("Auto", "End" or "Long face"),
    resolved here to the concrete signed axis that currently points most
    toward the fitted support plane. Resolution happens AFTER the plane fit
    below, so the plane is fit exactly once and callers never need to touch
    geometry themselves to pick a face. Signed axes are not accepted: with a
    0.6 x 0.6 square cross-section the four long faces are symmetry-equivalent,
    so naming one is a distinction without a difference.

    Orientation is deliberately preserved: this is a precise contact operation,
    not an implicit reorientation.  The neighborhood PCA normal is oriented
    toward the block so either side of a plane works consistently.
    """
    cloud = np.asarray(points, dtype=float)
    click = np.asarray(click, dtype=float)
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    dims = tuple(float(v) for v in dimensions)
    if face not in FACE_CLASSES:
        return SnapResult(False, None, None, None, 0.0, 0, f"unknown face class {face!r}")
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.all(np.isfinite(cloud)):
        return SnapResult(False, None, None, None, 0.0, 0, "point cloud must be finite (N,3)")
    if click.shape != (3,) or position.shape != (3,) or rotation.shape != (3, 3):
        return SnapResult(False, None, None, None, 0.0, 0, "invalid click or pose")
    if not np.isfinite(radius_m) or radius_m <= 0 or min_points < 3:
        return SnapResult(False, None, None, None, 0.0, 0, "invalid plane neighborhood parameters")
    neighborhood = cloud[np.linalg.norm(cloud - click, axis=1) <= radius_m]
    if len(neighborhood) < min_points:
        return SnapResult(
            False, None, None, None, 0.0, len(neighborhood), "insufficient support-plane points"
        )
    center = neighborhood.mean(axis=0)
    _, _, vh = np.linalg.svd(neighborhood - center, full_matrices=False)
    normal = vh[-1]
    if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-8:
        return SnapResult(
            False, None, None, None, 0.0, len(neighborhood), "degenerate support plane"
        )
    normal /= np.linalg.norm(normal)
    residuals = np.abs((neighborhood - center) @ normal)
    # Reject mixed neighborhoods (edges/clutter) instead of silently fitting a
    # compromise plane through unrelated returns.
    inliers = residuals <= max(0.02, radius_m * 0.08)
    if int(inliers.sum()) < max(min_points, int(0.7 * len(neighborhood))):
        return SnapResult(
            False,
            None,
            None,
            None,
            0.0,
            int(inliers.sum()),
            "neighborhood is not a sufficiently planar support",
        )
    if float(np.median(residuals[inliers])) > 0.01:
        return SnapResult(
            False, None, None, None, 0.0, int(inliers.sum()), "support plane residual is too high"
        )
    # `normal`'s sign is an arbitrary SVD artifact at this point, so orient it
    # toward the block (using `position`, not any candidate face) first --
    # otherwise the winning candidate below would flip with that arbitrary sign
    # instead of tracking which face is physically closest to the plane.
    oriented = normal if np.dot(normal, position - center) >= 0 else -normal
    face = max(
        FACE_CLASSES[face],
        key=lambda axis: -float(np.dot(rotation @ FACE_NORMALS[axis], oriented)),
    )
    local_normal = FACE_NORMALS[face]
    face_normal = rotation @ local_normal
    # A support plane must be reasonably parallel to the chosen face.
    if abs(float(np.dot(face_normal, normal))) < 0.85:
        return SnapResult(
            False,
            None,
            None,
            None,
            0.0,
            len(neighborhood),
            "support plane is not parallel to selected face",
        )
    if np.dot(normal, face_normal) < 0:
        normal = -normal
    half_extent = np.asarray(dims) / 2.0
    axis = "XYZ".index(face[1])
    face_center = position + rotation @ (local_normal * half_extent[axis])
    signed_distance = float(np.dot(normal, center - face_center))
    new_position = position + normal * signed_distance
    return SnapResult(True, new_position, rotation.copy(), dims, 1.0, len(neighborhood), "")


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
            np.ascontiguousarray(seed, dtype=np.float64),
            native,
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
