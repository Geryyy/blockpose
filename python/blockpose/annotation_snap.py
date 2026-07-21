"""Conservative local cuboid fitting for the interactive annotator.

The fitter is deliberately independent of Open3D and annotation state.  It
returns a proposal, never an approval decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from blockpose.pose import BLOCK_DIMS, BLOCK_DIMS_UPRIGHT, BlockPose
from blockpose.ransac import segment_plane
from blockpose.refine import refine_pose


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


def _failure(reason: str) -> SnapResult:
    return SnapResult(False, None, None, None, 0.0, 0, reason)


def _voxel_centroids(points: np.ndarray, size: float) -> np.ndarray:
    """Deterministic voxel reduction, independent of input ordering."""
    keys = np.floor(points / size).astype(np.int64)
    order = np.lexsort(
        (points[:, 2], points[:, 1], points[:, 0], keys[:, 2], keys[:, 1], keys[:, 0])
    )
    keys, points = keys[order], points[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(points)])
    return np.add.reduceat(points, starts, axis=0) / counts[:, None]


@dataclass(frozen=True)
class _Plane:
    normal: np.ndarray
    offset: float
    points: np.ndarray


def _connected_components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Return deterministic Euclidean components of a finite plane consensus."""
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    parent = np.arange(len(points))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in pairs:
        first_root, second_root = root(int(first)), root(int(second))
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(root(index), []).append(index)
    return [points[indices] for _, indices in sorted(groups.items())]


def _extract_planes(points: np.ndarray, tolerance: float) -> list[_Plane]:
    """Extract deterministic finite plane patches, largest first."""
    remaining = points.copy()
    planes: list[_Plane] = []
    for index in range(10):
        if len(remaining) < 18:
            break
        try:
            model, inliers = segment_plane(
                remaining,
                tolerance,
                num_iterations=500,
                seed=index,
            )
        except ValueError:
            break
        if len(inliers) < 18:
            break
        normal = model[:3].copy()
        offset = float(model[3])
        # Stable signs simplify candidate generation and permutation tests.
        pivot = int(np.argmax(np.abs(normal)))
        if normal[pivot] < 0:
            normal, offset = -normal, -offset
        consensus = remaining[inliers].copy()
        for component in _connected_components(consensus, max(0.065, tolerance * 2.5)):
            if len(component) >= 16:
                planes.append(_Plane(normal, offset, component))
        keep = np.ones(len(remaining), dtype=bool)
        keep[inliers] = False
        remaining = remaining[keep]
    return planes


def _candidate_score(
    top_points: np.ndarray,
    side_points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    dims: np.ndarray,
    side_axis: int,
    side_sign: float,
    tolerance: float,
) -> tuple[float, int, np.ndarray]:
    top_local = (top_points - center) @ rotation
    side_local = (side_points - center) @ rotation
    half = dims / 2
    top_residual = np.abs(top_local[:, 2] - half[2])
    top_finite = (
        (np.abs(top_local[:, 0]) <= half[0] + tolerance)
        & (np.abs(top_local[:, 1]) <= half[1] + tolerance)
        & (top_residual <= tolerance)
    )
    other = 1 - side_axis
    side_residual = np.abs(side_local[:, side_axis] - side_sign * half[side_axis])
    side_finite = (
        (np.abs(side_local[:, other]) <= half[other] + tolerance)
        & (np.abs(side_local[:, 2]) <= half[2] + tolerance)
        & (side_residual <= tolerance)
    )
    top_count = int(np.count_nonzero(top_finite))
    side_count = int(np.count_nonzero(side_finite))
    if top_count < 16 or side_count < 16:
        return 0.0, top_count + side_count, np.empty((0, 3))
    assigned = np.vstack([top_points[top_finite], side_points[side_finite]])
    residuals = np.r_[top_residual[top_finite], side_residual[side_finite]]
    quality = float(np.mean(np.clip(1.0 - residuals / tolerance, 0.0, 1.0)))
    balance = 2.0 * min(top_count, side_count) / (top_count + side_count)
    support = min((top_count + side_count) / 180.0, 1.0)
    top_coverage = top_count / len(top_points)
    score = quality * (0.4 + 0.35 * support + 0.25 * balance)
    score *= 0.35 + 0.65 * top_coverage
    return score, len(assigned), assigned


def _spatially_overlap(top: _Plane, side: _Plane, tangent: np.ndarray, top_z: float) -> bool:
    if np.max(side.points[:, 2]) < top_z - 0.16:
        return False
    if np.ptp(side.points[:, 2]) < 0.16:
        return False
    top_range = np.quantile(top.points @ tangent, [0.03, 0.97])
    side_range = np.quantile(side.points @ tangent, [0.03, 0.97])
    ranges_overlap = min(top_range[1], side_range[1]) >= max(
        top_range[0], side_range[0]
    ) - 0.05
    side_normal = side.normal.copy()
    side_normal[2] = 0.0
    side_normal /= np.linalg.norm(side_normal)
    side_coordinate = float(np.median(side.points @ side_normal))
    top_boundary = np.abs(top.points @ side_normal - side_coordinate) <= 0.065
    # Top-first RANSAC consumes junction points from the side consensus; allow
    # that extraction band while still requiring a populated upper boundary.
    side_boundary = np.abs(side.points[:, 2] - top_z) <= 0.16
    return bool(
        ranges_overlap
        and np.count_nonzero(top_boundary) >= 3
        and np.count_nonzero(side_boundary) >= 6
    )


def _validate_params(params: SnapParams) -> str | None:
    positive = {
        "crop_radius_m": params.crop_radius_m,
        "voxel_m": params.voxel_m,
        "plane_tolerance_m": params.plane_tolerance_m,
        "surface_tolerance_m": params.surface_tolerance_m,
        "min_points": params.min_points,
        "min_inliers": params.min_inliers,
        "ambiguity_distance_m": params.ambiguity_distance_m,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            return f"{name} must be positive and finite"
    if not np.isfinite(params.min_score) or not 0 <= params.min_score <= 1:
        return "min_score must lie in [0, 1]"
    if not np.isfinite(params.ambiguity_score_ratio) or not (
        0 < params.ambiguity_score_ratio <= 1
    ):
        return "ambiguity_score_ratio must lie in (0, 1]"
    if int(params.min_points) != params.min_points or params.min_points < 3:
        return "min_points must be an integer of at least 3"
    if int(params.min_inliers) != params.min_inliers or params.min_inliers < 3:
        return "min_inliers must be an integer of at least 3"
    if params.voxel_m >= params.crop_radius_m:
        return "voxel_m must be smaller than crop_radius_m"
    if max(params.plane_tolerance_m, params.surface_tolerance_m) >= params.crop_radius_m:
        return "fit tolerances must be smaller than crop_radius_m"
    return None


def snap_block_to_cloud(
    points: np.ndarray,
    seed: np.ndarray,
    params: SnapParams | None = None,
) -> SnapResult:
    """Fit a known-size laid or upright block around a clicked cloud point."""
    params = params or SnapParams()
    invalid_params = _validate_params(params)
    if invalid_params is not None:
        return _failure(invalid_params)
    points = np.asarray(points, dtype=float)
    seed = np.asarray(seed, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or seed.shape != (3,):
        return _failure("points must be Nx3 and seed must be XYZ")
    if not np.all(np.isfinite(seed)):
        return _failure("seed is not finite")
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.linalg.norm(points - seed, axis=1) <= params.crop_radius_m]
    if len(points) < params.min_points:
        return _failure(f"only {len(points)} local points; need {params.min_points}")
    points = _voxel_centroids(points, params.voxel_m)
    if len(points) < params.min_points:
        return _failure(f"only {len(points)} local voxels; need {params.min_points}")
    planes = _extract_planes(points, params.plane_tolerance_m)
    tops = [plane for plane in planes if abs(plane.normal[2]) >= 0.94]
    sides = [plane for plane in planes if abs(plane.normal[2]) <= 0.18]
    if not tops:
        return _failure("no finite horizontal top patch near the click")
    if not sides:
        return _failure("no compatible vertical front/side patch near the click")

    # score, count, center, rotation, dims, side axis/sign, top points, side points
    candidates: list[tuple] = []
    world_z = np.array([0.0, 0.0, 1.0])
    for top in tops:
        top_z = float(np.median(top.points[:, 2]))
        if top_z < seed[2] - 0.10:
            continue
        for side in sides:
            normal = side.normal.copy()
            normal[2] = 0.0
            normal /= np.linalg.norm(normal)
            tangent = np.cross(world_z, normal)
            side_tangent_extent = float(np.ptp(np.quantile(side.points @ tangent, [0.02, 0.98])))
            side_height_extent = float(
                np.ptp(np.quantile(side.points[:, 2], [0.02, 0.98]))
            )
            if side_tangent_extent > 1.20 or side_height_extent > 1.20:
                continue
            if not _spatially_overlap(top, side, tangent, top_z):
                continue
            plane_coordinate = float(np.median(side.points @ normal))
            for dims_tuple in (BLOCK_DIMS, BLOCK_DIMS_UPRIGHT):
                dims = np.asarray(dims_tuple)
                for side_axis in (0, 1):
                    other = 1 - side_axis
                    for side_sign in (-1.0, 1.0):
                        if side_axis == 0:
                            axis_x = normal
                            axis_y = np.cross(world_z, axis_x)
                        else:
                            axis_y = normal
                            axis_x = np.cross(axis_y, world_z)
                        rotation = np.column_stack([axis_x, axis_y, world_z])
                        top_local_patch = top.points @ rotation
                        top_extents = np.ptp(
                            np.quantile(top_local_patch[:, :2], [0.03, 0.97], axis=0),
                            axis=0,
                        )
                        if np.any(top_extents > dims[:2] + 0.22) or np.any(
                            top_extents < 0.16
                        ):
                            continue
                        face_axis = rotation[:, side_axis]
                        other_axis = rotation[:, other]
                        center_face = plane_coordinate * normal - side_sign * dims[
                            side_axis
                        ] / 2 * face_axis
                        values = top.points @ other_axis
                        low, middle, high = np.quantile(values, [0.04, 0.5, 0.96])
                        half_other = dims[other] / 2
                        center_values = np.unique(
                            np.round(
                                [
                                    (low + high) / 2,
                                    low + half_other,
                                    high - half_other,
                                    middle,
                                    seed @ other_axis,
                                ],
                                5,
                            )
                        )
                        for other_value in center_values:
                            center = (
                                center_face
                                + (other_value - center_face @ other_axis) * other_axis
                            )
                            center[2] = top_z - dims[2] / 2
                            seed_local = (seed - center) @ rotation
                            if np.any(np.abs(seed_local) > dims / 2 + 0.10):
                                continue
                            score, count, assigned = _candidate_score(
                                top.points,
                                side.points,
                                center,
                                rotation,
                                dims,
                                side_axis,
                                side_sign,
                                params.surface_tolerance_m,
                            )
                            if count < params.min_inliers:
                                continue
                            initial = BlockPose(
                                center,
                                rotation,
                                score,
                                "annotation_snap",
                                dims_tuple,
                            )
                            refined, _ = refine_pose(
                                initial,
                                assigned,
                                dims_tuple,
                                max_translation=0.06,
                                max_rotation_deg=6.0,
                                min_points=params.min_inliers,
                            )
                            score, count, _ = _candidate_score(
                                top.points,
                                side.points,
                                refined.position,
                                refined.rotation,
                                dims,
                                side_axis,
                                side_sign,
                                params.surface_tolerance_m,
                            )
                            candidates.append(
                                (
                                    score,
                                    count,
                                    refined.position,
                                    refined.rotation,
                                    dims_tuple,
                                )
                            )
    if not candidates:
        return _failure("no top/side intersection supports a canonical block")
    candidates.sort(key=lambda item: (-item[0], -item[1], *item[2]))
    best = candidates[0]
    if best[1] < params.min_inliers:
        count = best[1]
        return _failure(f"insufficient cuboid surface support ({count} inliers)")
    if best[0] < params.min_score:
        return _failure(f"cuboid fit score {best[0]:.2f} is below {params.min_score:.2f}")
    if params.reject_ambiguous:
        for competitor in candidates[1:]:
            if competitor[0] < best[0] * params.ambiguity_score_ratio:
                break
            distance = float(np.linalg.norm(competitor[2] - best[2]))
            if distance >= params.ambiguity_distance_m:
                return _failure(
                    f"ambiguous fit: near-equal candidates are {distance:.2f} m apart"
                )
    return SnapResult(
        True,
        best[2].copy(),
        best[3].copy(),
        best[4],
        best[0],
        best[1],
        "fit accepted as pending proposal",
    )
