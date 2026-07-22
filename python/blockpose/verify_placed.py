"""Plan-conditioned verification of placed concrete blocks.

Unlike blind detection this module starts from stable plan identities.  Returns
are assigned exclusively to the nearest expected cuboid, then each cuboid is
fitted inside a small, explicit trust region.  An absent block is only called
``Missing`` when rays demonstrate free space; raw XYZ silence is
``NotObservable``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from blockpose.detection import (
    SensorContext,
    _ray_box_interval,
    _validate_sensor_context,
)
from blockpose.pose import BLOCK_DIMS, BlockPose, frame_alignments
from blockpose.refine import _exp_so3


class PlacedStatus(str, Enum):
    CONFIRMED = "Confirmed"
    DEVIATED = "Deviated"
    MISSING = "Missing"
    NOT_OBSERVABLE = "NotObservable"


class PlacedObservability(str, Enum):
    WELL_CONSTRAINED = "WellConstrained"
    PARTIALLY_CONSTRAINED = "PartiallyConstrained"
    SYMMETRY_AMBIGUOUS = "SymmetryAmbiguous"
    LAYOUT_INFERRED = "LayoutInferred"
    UNOBSERVABLE = "Unobservable"


@dataclass(frozen=True)
class VerifyPlacedParams:
    assign_band_m: float = 0.08
    max_translation_m: float = 0.10
    max_rotation_deg: float = 10.0
    huber_scale_m: float = 0.025
    min_support_points: int = 20
    surface_tolerance_m: float = 0.04
    confirmed_translation_m: float = 0.03
    confirmed_rotation_deg: float = 2.0
    missing_min_rays: int = 3
    neighbor_margin_m: float = 0.015
    max_nfev: int = 80
    # A two-face observation has one translation null direction, but can still
    # measure the other five pose components.  The global residual-improvement
    # gate below is intentionally strict for a full 6-DoF correction.  This
    # narrower path permits a small, data-supported partial correction only
    # when both independently observed faces have substantial support.
    partial_correction_min_face_points: int = 100
    partial_correction_min_improvement_m: float = 0.002
    partial_correction_min_relative_improvement: float = 0.15
    partial_correction_max_face_imbalance: float = 5.0
    # Shared-course yaw is deliberately opt-in while its real-data acceptance
    # gates are being collected.  It never changes the default verifier path.
    enable_course_yaw_refinement: bool = False
    course_yaw_max_input_points: int = 250_000


@dataclass(frozen=True)
class PlacedBlockPrior:
    block_id: str
    pose: BlockPose
    covariance: np.ndarray | None = None
    course_id: str | None = None
    neighbor_ids: tuple[str, ...] = ()
    source_snapshot_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CourseConstraint:
    course_id: str
    block_ids: tuple[str, ...]
    nominal_offsets: dict[tuple[str, str], np.ndarray]
    shared_yaw_sigma_deg: float = 1.0
    joint_gap_m: float | None = None
    translation_sigma_m: float = 0.05
    relative_offset_covariance: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    plan_name: str | None = None
    source_snapshot_ids: tuple[str, ...] = ()
    evaluated_snapshot_id: str | None = None


@dataclass(frozen=True)
class FaceBoundaryObservation:
    face_axis: int
    face_sign: int
    boundary_axis: int
    coordinate_m: float
    span_m: tuple[float, float]
    support_points: int
    source: str
    quality: float


def _observed_face_boundaries(
    prior: BlockPose,
    points: np.ndarray,
    face_axes: np.ndarray,
    face_signs: np.ndarray,
) -> tuple[FaceBoundaryObservation, ...]:
    """Extract occupancy falloffs only; planned joints are never observations."""
    if not len(points):
        return ()
    local = (points - prior.position) @ prior.rotation
    observations = []
    for face_axis, face_sign in sorted(
        {(int(axis), int(sign)) for axis, sign in zip(face_axes, face_signs, strict=True)}
    ):
        selected = (face_axes == face_axis) & (face_signs == face_sign)
        if np.count_nonzero(selected) < 20:
            continue
        face_points = local[selected]
        for boundary_axis in range(3):
            if boundary_axis == face_axis:
                continue
            values = face_points[:, boundary_axis]
            low, high = np.percentile(values, [2.0, 98.0])
            if high - low < 0.08:
                continue
            half_extent = float(np.asarray(prior.dims)[boundary_axis] / 2.0)
            counts, _edges = np.histogram(
                values, bins=20, range=(-half_extent - 0.05, half_extent + 0.05)
            )
            interior = counts[3:-3]
            if len(interior) == 0 or np.median(interior) < 2:
                continue
            interior_level = float(np.median(interior))
            for side, edge_count, coordinate in (
                ("low", float(np.mean(counts[:2])), low),
                ("high", float(np.mean(counts[-2:])), high),
            ):
                quality = float(np.clip(1.0 - edge_count / max(interior_level, 1.0), 0.0, 1.0))
                # Ownership partitions can end at a planned joint and create a
                # false falloff. Only physical exterior silhouette coordinates
                # near the canonical face rectangle are boundary evidence.
                if quality < 0.35 or abs(abs(coordinate) - half_extent) > 0.08:
                    continue
                other_axis = next(
                    axis for axis in range(3) if axis not in {face_axis, boundary_axis}
                )
                span = tuple(
                    float(value) for value in np.percentile(face_points[:, other_axis], [5, 95])
                )
                observations.append(
                    FaceBoundaryObservation(
                        face_axis,
                        face_sign,
                        boundary_axis,
                        float(coordinate),
                        span,
                        int(np.count_nonzero(selected)),
                        f"occupancy_falloff_{side}",
                        quality,
                    )
                )
    return tuple(observations)


@dataclass(frozen=True)
class PlacedVerification:
    expected: BlockPose
    pose: BlockPose
    status: PlacedStatus
    observability: PlacedObservability
    covariance: np.ndarray | None
    unconstrained_directions: tuple[str, ...]
    support_points: int
    residual_rms_m: float | None
    translation_from_plan_m: float
    rotation_from_plan_deg: float
    visible_faces: tuple[str, ...]
    occluded_faces: tuple[str, ...]
    free_space_rays: int
    diagnostics: dict[str, object] = field(default_factory=dict)


_DIRECTION_NAMES = ("roll", "pitch", "yaw", "x", "y", "z")


def _validate_params(params: VerifyPlacedParams) -> None:
    positive = (
        params.assign_band_m,
        params.max_translation_m,
        params.max_rotation_deg,
        params.huber_scale_m,
        params.surface_tolerance_m,
        params.confirmed_translation_m,
        params.confirmed_rotation_deg,
        params.neighbor_margin_m,
    )
    if not all(np.isfinite(value) and value > 0 for value in positive):
        raise ValueError("verification distances and angles must be positive and finite")
    if (
        params.min_support_points < 3
        or params.missing_min_rays < 1
        or params.max_nfev < 1
        or params.course_yaw_max_input_points < 1
        or params.partial_correction_min_face_points < 3
    ):
        raise ValueError("verification count parameters are invalid")
    if (
        not np.isfinite(params.partial_correction_min_improvement_m)
        or params.partial_correction_min_improvement_m <= 0.0
        or not np.isfinite(params.partial_correction_min_relative_improvement)
        or not 0.0 < params.partial_correction_min_relative_improvement <= 1.0
        or not np.isfinite(params.partial_correction_max_face_imbalance)
        or params.partial_correction_max_face_imbalance < 1.0
    ):
        raise ValueError("partial correction thresholds are invalid")


def _ray_entry(
    pose: BlockPose, origins: np.ndarray, directions: np.ndarray, margin: float
) -> np.ndarray:
    origins_local = (origins - pose.position) @ pose.rotation
    dirs_local = directions @ pose.rotation
    half = np.asarray(pose.dims) / 2.0 + margin
    entries = np.full(len(origins), np.inf)
    enter, _, hit = _ray_box_interval(origins_local, dirs_local, half)
    entries[hit] = np.maximum(enter[hit], 0.0)
    return entries


def expected_visibility(
    expected: list[BlockPose], sensor_origin: np.ndarray | None, *, margin_m: float = 0.015
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Classify sensor-facing faces, accounting for planned-neighbor occlusion."""
    if sensor_origin is None:
        return [((), ()) for _ in expected]
    origin = np.asarray(sensor_origin, dtype=np.float64)
    output = []
    axis_names = ("x", "y", "z")
    for block_index, pose in enumerate(expected):
        half = np.asarray(pose.dims) / 2.0
        visible: list[str] = []
        occluded: list[str] = []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                center = pose.position + pose.rotation[:, axis] * sign * half[axis]
                normal = pose.rotation[:, axis] * sign
                if np.dot(origin - center, normal) <= 0:
                    continue
                other_axes = [other for other in range(3) if other != axis]
                samples = []
                for first in (-0.7, 0.0, 0.7):
                    for second in (-0.7, 0.0, 0.7):
                        samples.append(
                            center
                            + pose.rotation[:, other_axes[0]] * first * half[other_axes[0]]
                            + pose.rotation[:, other_axes[1]] * second * half[other_axes[1]]
                        )
                samples_array = np.asarray(samples)
                vectors = samples_array - origin
                distances = np.linalg.norm(vectors, axis=1)
                directions = vectors / distances[:, None]
                blocked = np.zeros(len(samples_array), dtype=bool)
                origins = np.broadcast_to(origin, samples_array.shape)
                for neighbor_index, neighbor in enumerate(expected):
                    if neighbor_index == block_index:
                        continue
                    entries = _ray_entry(neighbor, origins, directions, margin_m)
                    blocked |= entries < distances - margin_m
                name = f"{axis_names[axis]}{'+' if sign > 0 else '-'}"
                # One exposed sample makes the face potentially observable;
                # only a fully covered 3x3 grid is classified as occluded.
                (occluded if np.all(blocked) else visible).append(name)
        output.append((tuple(visible), tuple(occluded)))
    return output


def _assign_expected_faces(
    points: np.ndarray,
    expected: list[BlockPose],
    band_m: float,
    visible_faces: list[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    *,
    deadline_s: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not len(points):
        return [
            (np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=float)) for _ in expected
        ]
    distances = []
    block_axes = []
    block_signs = []
    for index in range(len(expected)):
        if deadline_s is not None and time.perf_counter() >= deadline_s:
            raise TimeoutError("course yaw assignment deadline exceeded")
        pose = expected[index]
        local = (points - pose.position) @ pose.rotation
        half = np.asarray(pose.dims, dtype=np.float64) / 2.0
        face_distances = []
        face_labels = []
        bottom_z = pose.position[2] - float(np.sum(np.abs(pose.rotation[2]) * half))
        above_support = points[:, 2] > bottom_z + min(band_m, 0.05)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if deadline_s is not None and time.perf_counter() >= deadline_s:
                    raise TimeoutError("course yaw assignment deadline exceeded")
                normal = pose.rotation[:, axis] * sign
                # The downward support rectangle is not block-presence evidence.
                if normal[2] <= -0.7:
                    continue
                name = f"{'xyz'[axis]}{'+' if sign > 0 else '-'}"
                if visible_faces is not None and visible_faces[index][0]:
                    if name not in visible_faces[index][0]:
                        continue
                plane_distance = np.abs(local[:, axis] - sign * half[axis])
                inside = above_support.copy()
                for other in range(3):
                    if other != axis:
                        inside &= np.abs(local[:, other]) <= half[other] + band_m
                face_distances.append(np.where(inside, plane_distance, np.inf))
                face_labels.append((axis, sign))
        if face_distances:
            matrix = np.stack(face_distances)
            nearest = np.argmin(matrix, axis=0)
            distances.append(matrix[nearest, np.arange(len(points))])
            block_axes.append(np.asarray([face_labels[item][0] for item in nearest]))
            block_signs.append(np.asarray([face_labels[item][1] for item in nearest]))
        else:
            distances.append(np.full(len(points), np.inf))
            block_axes.append(np.zeros(len(points), dtype=int))
            block_signs.append(np.ones(len(points)))
    distances = np.stack(distances, axis=1)
    owner = np.argmin(distances, axis=1)
    within = distances[np.arange(len(points)), owner] <= band_m
    result = []
    for index in range(len(expected)):
        selected = within & (owner == index)
        owned = points[selected]
        result.append((owned, block_axes[index][selected], block_signs[index][selected]))
    return result


def _bounded_fit(
    prior: BlockPose,
    points: np.ndarray,
    face_axes: np.ndarray,
    face_signs: np.ndarray,
    params: VerifyPlacedParams,
    *,
    suppress_ownership_boundaries: bool = False,
) -> tuple[BlockPose, bool, np.ndarray | None, tuple[str, ...], float | None, dict[str, object]]:
    if len(points) < params.min_support_points:
        return prior, False, None, _DIRECTION_NAMES, None, {"fit_reason": "insufficient_points"}
    half = np.asarray(prior.dims, dtype=np.float64) / 2.0
    rotation_bound = np.deg2rad(params.max_rotation_deg)

    def residuals(delta: np.ndarray) -> np.ndarray:
        rotation = prior.rotation @ _exp_so3(delta[:3])
        local = (points - (prior.position + delta[3:])) @ rotation
        surface = local[np.arange(len(local)), face_axes] - face_signs * half[face_axes]
        # A deliberately weak plan tie-breaker prevents a one-face fit from
        # drifting freely along that face. Hundreds of geometric residuals
        # dominate it whenever the motion is actually observable.
        prior_tie_break = np.concatenate((0.05 * delta[:3], 0.05 * delta[3:]))
        return np.concatenate((surface, prior_tie_break))

    bounds = (
        np.array([-rotation_bound] * 3 + [-params.max_translation_m] * 3),
        np.array([rotation_bound] * 3 + [params.max_translation_m] * 3),
    )
    try:
        fit = least_squares(
            residuals,
            np.zeros(6),
            bounds=bounds,
            loss="huber",
            f_scale=params.huber_scale_m,
            max_nfev=params.max_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):
        return prior, False, None, _DIRECTION_NAMES, None, {"fit_reason": "solver_error"}
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return prior, False, None, _DIRECTION_NAMES, None, {"fit_reason": "solver_failed"}
    pose = BlockPose(
        prior.position + fit.x[3:],
        prior.rotation @ _exp_so3(fit.x[:3]),
        source="verify_placed_v1",
        dims=prior.dims,
    )
    prior_local = (points - prior.position) @ prior.rotation
    prior_residual = (
        prior_local[np.arange(len(prior_local)), face_axes] - face_signs * half[face_axes]
    )
    prior_rms = float(np.sqrt(np.mean(prior_residual**2)))
    rms = float(np.sqrt(np.mean(residuals(fit.x)[:-6] ** 2)))
    # The final six rows are only the weak plan tie-breaker. They must not make
    # an actually unobservable geometric direction appear constrained.
    jacobian = np.asarray(fit.jac[:-6], dtype=np.float64)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(float(singular[0]) * 1e-3, 1e-8) if len(singular) else np.inf
    rank = int(np.count_nonzero(singular > tolerance))
    unconstrained: tuple[str, ...] = ()
    covariance = None
    if jacobian.shape[0] > rank and rank:
        information = jacobian.T @ jacobian
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        weak_threshold = max(float(eigenvalues[-1]) * 1e-3, 1e-8)
        weak_mask = eigenvalues <= weak_threshold
        # A pseudoinverse incorrectly reports zero variance in null directions.
        # Give those directions an explicitly large variance instead.
        inverse = np.where(
            weak_mask,
            1e3 / max(rms**2, 1e-8),
            1.0 / np.maximum(eigenvalues, 1e-12),
        )
        covariance = eigenvectors @ np.diag(inverse) @ eigenvectors.T * max(rms**2, 1e-8)
        weak = eigenvectors[:, weak_mask]
        names = {
            _DIRECTION_NAMES[int(np.argmax(np.abs(vector)))]
            for vector in weak.T
            if np.max(np.abs(vector)) > 0.5
        }
        unconstrained = tuple(name for name in _DIRECTION_NAMES if name in names)
    else:
        unconstrained = _DIRECTION_NAMES
    at_bound = bool(np.any(np.isclose(np.abs(fit.x), bounds[1], atol=1e-4)))
    translation_norm = float(np.linalg.norm(fit.x[3:]))
    rotation_norm = float(np.linalg.norm(fit.x[:3]))
    outside_translation = translation_norm > params.max_translation_m + 1e-9
    outside_rotation = rotation_norm > rotation_bound + 1e-9
    face_labels = sorted(
        {(int(axis), int(sign)) for axis, sign in zip(face_axes, face_signs, strict=True)}
    )
    face_support = {
        (axis, sign): int(np.count_nonzero((face_axes == axis) & (face_signs == sign)))
        for axis, sign in face_labels
        if sign != 0
    }
    supported_faces = set(face_support)
    geometry_sufficient = (
        rank >= 5
        and len(supported_faces) >= 2
        and not at_bound
        and not outside_translation
        and not outside_rotation
    )
    improvement = prior_rms - rms
    # Small objective gains are easily caused by clutter or mixed pixels and
    # have moved strong real priors by centimetres. Require a material gain.
    correction_supported = improvement >= max(0.002, prior_rms * 0.30)
    # Do not relax the gate merely because two planes happen to be present:
    # both must carry enough independent return support and the candidate must
    # improve the raw geometric residual by an absolute and relative margin.
    # The Jacobian rank remains five, so the result stays PartiallyConstrained
    # and its null direction is explicitly retained in the covariance/result.
    reliable_partial_faces = sum(
        count >= params.partial_correction_min_face_points for count in face_support.values()
    )
    reliable_face_support = sorted(
        count
        for count in face_support.values()
        if count >= params.partial_correction_min_face_points
    )
    reliable_face_imbalance = (
        reliable_face_support[-1] / reliable_face_support[0]
        if len(reliable_face_support) >= 2
        else np.inf
    )
    partial_correction_supported = (
        rank == 5
        and reliable_partial_faces >= 2
        and reliable_face_imbalance <= params.partial_correction_max_face_imbalance
        and improvement >= params.partial_correction_min_improvement_m
        and improvement >= prior_rms * params.partial_correction_min_relative_improvement
    )
    correction_reason = (
        "strong_residual_gain"
        if correction_supported
        else "two_face_partial_geometry"
        if partial_correction_supported
        else "insufficient_residual_gain"
    )
    correction_supported |= partial_correction_supported
    retained_pose = pose if geometry_sufficient and correction_supported else prior
    boundaries = (
        ()
        if suppress_ownership_boundaries
        else _observed_face_boundaries(prior, points, face_axes, face_signs)
    )
    return (
        retained_pose,
        geometry_sufficient,
        covariance if geometry_sufficient else None,
        unconstrained if geometry_sufficient else _DIRECTION_NAMES,
        rms,
        {
            "fit_reason": (
                "translation_search_exceeded"
                if outside_translation
                else "rotation_search_exceeded"
                if outside_rotation
                else "search_bound"
                if at_bound
                else "insufficient_geometry"
                if not geometry_sufficient
                else "ok"
            ),
            "jacobian_rank": rank,
            "jacobian_singular_values": singular.tolist(),
            "at_search_bound": at_bound,
            "translation_correction_norm_m": translation_norm,
            "rotation_correction_norm_deg": float(np.degrees(rotation_norm)),
            "supported_face_count": len(supported_faces),
            "face_support_points": {
                f"{'xyz'[axis]}{'+' if sign > 0 else '-'}": count
                for (axis, sign), count in sorted(face_support.items())
            },
            "reliable_partial_face_count": reliable_partial_faces,
            "reliable_partial_face_imbalance": reliable_face_imbalance,
            "prior_residual_rms_m": prior_rms,
            "residual_improvement_m": improvement,
            "correction_supported": correction_supported,
            "correction_acceptance": correction_reason,
            "observed_boundaries": [
                {
                    "face_axis": item.face_axis,
                    "face_sign": item.face_sign,
                    "boundary_axis": item.boundary_axis,
                    "coordinate_m": item.coordinate_m,
                    "span_m": item.span_m,
                    "support_points": item.support_points,
                    "source": item.source,
                    "quality": item.quality,
                }
                for item in boundaries
            ],
        },
    )


def _rotation_deviation_deg(prior: BlockPose, pose: BlockPose) -> float:
    best = np.inf
    for alignment in frame_alignments(pose.dims, prior.dims):
        relative = (pose.rotation @ alignment).T @ prior.rotation
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        best = min(best, float(angle))
    return best


def _validate_priors(expected: list[BlockPose]) -> None:
    for pose in expected:
        rotation = np.asarray(pose.rotation, dtype=np.float64)
        if tuple(pose.dims) != BLOCK_DIMS:
            raise ValueError(f"VerifyPlaced requires canonical dimensions {BLOCK_DIMS}")
        if (
            np.asarray(pose.position).shape != (3,)
            or not np.all(np.isfinite(pose.position))
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("VerifyPlaced priors require finite centers and SO(3) rotations")


def _normalize_priors(
    expected: list[BlockPose] | list[PlacedBlockPrior],
) -> tuple[list[PlacedBlockPrior], list[BlockPose]]:
    normalized = [
        item if isinstance(item, PlacedBlockPrior) else PlacedBlockPrior(f"block_{index}", item)
        for index, item in enumerate(expected)
        if isinstance(item, (BlockPose, PlacedBlockPrior))
    ]
    if len(normalized) != len(expected):
        raise ValueError("expected entries must be BlockPose or PlacedBlockPrior")
    identifiers = [item.block_id for item in normalized]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("placed block IDs must be non-empty and unique")
    known = set(identifiers)
    by_id = {item.block_id: item for item in normalized}
    for item in normalized:
        if set(item.neighbor_ids) - known or item.block_id in item.neighbor_ids:
            raise ValueError(f"invalid neighbors for placed block {item.block_id!r}")
        if item.covariance is not None:
            covariance = np.asarray(item.covariance, dtype=np.float64)
            if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"covariance for {item.block_id!r} must be finite 6x6")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
                raise ValueError(f"covariance for {item.block_id!r} must be positive semidefinite")
        for neighbor in item.neighbor_ids:
            if item.block_id not in by_id[neighbor].neighbor_ids:
                raise ValueError(f"non-reciprocal neighbors {item.block_id!r}, {neighbor!r}")
    return normalized, [item.pose for item in normalized]


def _validate_courses(priors: list[PlacedBlockPrior], constraints: list[CourseConstraint]) -> None:
    identifiers = {item.block_id for item in priors}
    prior_by_id = {item.block_id: item for item in priors}
    from blockpose.wall_plans import load_wall_plans

    wall_plans = load_wall_plans()
    allowed_plans = set(wall_plans)
    course_ids = [item.course_id for item in constraints]
    if len(set(course_ids)) != len(course_ids):
        raise ValueError("course IDs must be unique")
    for constraint in constraints:
        if not constraint.course_id or not constraint.block_ids:
            raise ValueError("course constraints require an ID and blocks")
        if len(set(constraint.block_ids)) != len(constraint.block_ids):
            raise ValueError(f"duplicate block in course {constraint.course_id!r}")
        if set(constraint.block_ids) - identifiers:
            raise ValueError(f"unknown block in course {constraint.course_id!r}")
        prior_courses = {
            item.block_id: item.course_id
            for item in priors
            if item.block_id in constraint.block_ids
        }
        if any(value not in (None, constraint.course_id) for value in prior_courses.values()):
            raise ValueError(f"inconsistent prior course for {constraint.course_id!r}")
        if not np.isfinite(constraint.shared_yaw_sigma_deg) or constraint.shared_yaw_sigma_deg <= 0:
            raise ValueError("shared_yaw_sigma_deg must be positive and finite")
        if not np.isfinite(constraint.translation_sigma_m) or constraint.translation_sigma_m <= 0:
            raise ValueError("translation_sigma_m must be positive and finite")
        has_plan = constraint.plan_name is not None
        has_sources = bool(constraint.source_snapshot_ids)
        if has_plan == has_sources:
            raise ValueError("course requires exactly one plan name or source snapshot provenance")
        if has_plan and constraint.plan_name not in allowed_plans:
            raise ValueError(f"unknown course plan {constraint.plan_name!r}")
        if has_plan:
            plan = wall_plans[constraint.plan_name]  # type: ignore[index]
            plan_positions = {
                identifier: np.array([spec.x, spec.y, spec.z], dtype=np.float64)
                for identifier, spec in zip(plan.ids, plan.specs, strict=True)
            }
            if set(constraint.block_ids) - set(plan_positions):
                raise ValueError("course block IDs do not map to the named wall plan")
        if (
            has_sources
            and constraint.evaluated_snapshot_id is not None
            and constraint.evaluated_snapshot_id in constraint.source_snapshot_ids
        ):
            raise ValueError("course topology leaks the evaluated snapshot")
        if has_sources:
            from blockpose.snapshot_plan_fixtures import load_snapshot_plan_fixtures

            fixtures = load_snapshot_plan_fixtures()
            known_snapshots = {target_id.split("/", 1)[0] for target_id in fixtures} | {
                source for fixture in fixtures.values() for source in fixture.source_snapshot_ids
            }
            contributing = {source for prior in priors for source in prior.source_snapshot_ids}
            unknown = set(constraint.source_snapshot_ids) - known_snapshots
            noncontributing = set(constraint.source_snapshot_ids) - contributing
            if unknown:
                raise ValueError(f"unknown source snapshots: {sorted(unknown)}")
            if noncontributing:
                raise ValueError(f"noncontributing source snapshots: {sorted(noncontributing)}")
        for (first, second), raw_offset in constraint.nominal_offsets.items():
            if first not in constraint.block_ids or second not in constraint.block_ids:
                raise ValueError(f"offset references unknown course block {(first, second)!r}")
            offset = np.asarray(raw_offset, dtype=np.float64)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError("nominal course offsets must be finite 3-vectors")
            if np.linalg.norm(offset) > 2.0:
                raise ValueError("nominal course offset is implausibly large")
            if has_plan:
                plan_offset = plan_positions[second] - plan_positions[first]
                if not np.allclose(offset, plan_offset, atol=1e-6):
                    raise ValueError("nominal offset does not match named wall plan geometry")
            prior_delta = prior_by_id[second].pose.position - prior_by_id[first].pose.position
            if has_sources:
                common_sources = (
                    set(constraint.source_snapshot_ids)
                    & set(prior_by_id[first].source_snapshot_ids)
                    & set(prior_by_id[second].source_snapshot_ids)
                )
                if not common_sources:
                    raise ValueError("course offset has no common contributing source snapshot")
                if not np.allclose(offset, prior_delta, atol=1e-6, rtol=0.0):
                    raise ValueError("nominal offset differs from leakage-safe derived geometry")
            if np.linalg.norm(prior_delta - offset) > 0.20 + 1e-9:
                raise ValueError("nominal course offset is inconsistent with leakage-safe priors")
            reciprocal = constraint.nominal_offsets.get((second, first))
            if reciprocal is not None and not np.allclose(
                offset, -np.asarray(reciprocal), atol=1e-9
            ):
                raise ValueError(f"inconsistent reciprocal offset for {(first, second)!r}")
        for pair, raw_covariance in constraint.relative_offset_covariance.items():
            if pair not in constraint.nominal_offsets:
                raise ValueError(f"offset covariance has no nominal offset for {pair!r}")
            offset_covariance = np.asarray(raw_covariance, dtype=np.float64)
            if (
                offset_covariance.shape != (3, 3)
                or not np.all(np.isfinite(offset_covariance))
                or np.min(np.linalg.eigvalsh(offset_covariance)) < -1e-10
            ):
                raise ValueError("relative offset covariance must be finite PSD 3x3")


def _validate_yaw_courses(constraints: list[CourseConstraint]) -> None:
    """Validate the narrower provenance and geometry contract for pooled yaw.

    Translation layout propagation can use LOSO-derived snapshot topology.  A
    pooled geometric yaw fit is intentionally stricter: its membership and
    nominal geometry must be an explicit, named robot wall plan.  In
    particular, this prevents a snapshot-derived grouping from becoming a
    hidden source of evaluated-scene information.
    """
    if not constraints:
        return
    from blockpose.wall_plans import load_wall_plans

    wall_plans = load_wall_plans()
    memberships: dict[str, str] = {}
    for constraint in constraints:
        if constraint.source_snapshot_ids:
            raise ValueError("course yaw refinement does not accept source snapshot topology")
        if constraint.plan_name is None:
            raise ValueError("course yaw refinement requires a named wall plan")
        if len(constraint.block_ids) < 3:
            raise ValueError("course yaw refinement requires at least three blocks")
        for block_id in constraint.block_ids:
            previous = memberships.setdefault(block_id, constraint.course_id)
            if previous != constraint.course_id:
                raise ValueError("course yaw refinement rejects overlapping course membership")
        plan = wall_plans[constraint.plan_name]
        plan_specs = {
            identifier: spec for identifier, spec in zip(plan.ids, plan.specs, strict=True)
        }
        members = [plan_specs[identifier] for identifier in constraint.block_ids]
        if max(spec.z for spec in members) - min(spec.z for spec in members) > 1e-6:
            raise ValueError("course yaw refinement requires a level course")
        # Cuboids have a pi yaw symmetry.  A common nominal heading is required
        # even when a plan happened to serialize an equivalent 180-degree frame.
        nominal = np.deg2rad(members[0].yaw_deg)
        if any(abs(np.sin(np.deg2rad(spec.yaw_deg) - nominal)) > 1e-6 for spec in members[1:]):
            raise ValueError("course yaw refinement requires a common nominal yaw")
        # Offsets need not be redundantly listed for every pair, but they must
        # form a connected, exact plan-derived graph (exactness itself is
        # checked by _validate_courses).
        connected = {constraint.block_ids[0]}
        while True:
            expanded = set(connected)
            for first, second in constraint.nominal_offsets:
                if first in connected:
                    expanded.add(second)
                if second in connected:
                    expanded.add(first)
            if expanded == connected:
                break
            connected = expanded
        if connected != set(constraint.block_ids):
            raise ValueError("course yaw refinement requires connected plan offsets")


def _huber_mean_squared(residuals: np.ndarray, scale: float) -> float:
    """A deterministic Huber point-plane objective in squared metres."""
    absolute = np.abs(residuals)
    loss = np.where(
        absolute <= scale,
        0.5 * residuals**2,
        scale * (absolute - 0.5 * scale),
    )
    return float(np.mean(loss))


def _course_crop(
    points: np.ndarray,
    poses: list[BlockPose],
    params: VerifyPlacedParams,
) -> np.ndarray:
    """Take one conservative raw crop for every trial in a course solve."""
    centers = np.asarray([pose.position for pose in poses], dtype=np.float64)
    half = np.asarray(poses[0].dims, dtype=np.float64) / 2.0
    # A horizontal corner radius covers every orientation in the fixed 10 deg
    # search interval.  Translation and assignment padding make the crop fixed
    # across trials, avoiding a membership-dependent objective.
    horizontal_corner = float(np.hypot(half[0], half[1]))
    extent = np.array([horizontal_corner, horizontal_corner, half[2]])
    padding = params.max_translation_m + params.assign_band_m
    lower = np.min(centers, axis=0) - extent - padding
    upper = np.max(centers, axis=0) + extent + padding
    return points[np.all((points >= lower) & (points <= upper), axis=1)]


def _balanced_course_point_cap(
    crop: np.ndarray,
    base_poses: list[BlockPose],
    params: VerifyPlacedParams,
    deadline_s: float,
    *,
    max_points_per_block: int = 2000,
) -> np.ndarray:
    """Deterministically cap a raw crop without letting a dense block dominate."""
    if len(crop):
        crop = crop[np.lexsort((crop[:, 2], crop[:, 1], crop[:, 0]))]
    half = np.asarray(base_poses[0].dims, dtype=np.float64) / 2.0
    yaw_corner_motion = float(np.hypot(half[0], half[1]) * np.sin(np.deg2rad(10.0)))
    assigned = _assign_expected_faces(
        crop,
        base_poses,
        params.assign_band_m + yaw_corner_motion,
        deadline_s=deadline_s,
    )
    balanced: list[np.ndarray] = []
    for owned, _axes, _signs in assigned:
        if time.perf_counter() >= deadline_s:
            raise TimeoutError("course yaw point-cap deadline exceeded")
        if len(owned) > max_points_per_block:
            indices = np.linspace(0, len(owned) - 1, max_points_per_block, dtype=int)
            owned = owned[indices]
        balanced.append(owned)
    if not balanced:
        return np.empty((0, 3), dtype=np.float64)
    capped = np.vstack(balanced)
    return capped[np.lexsort((capped[:, 2], capped[:, 1], capped[:, 0]))]


def _course_yaw_trial(
    crop: np.ndarray,
    base_poses: list[BlockPose],
    yaw_deg: float,
    params: VerifyPlacedParams,
    deadline_s: float,
) -> tuple[float, tuple[float, ...]] | None:
    """Score one yaw with exclusive ownership and equal block influence."""
    angle = np.deg2rad(yaw_deg)
    world_yaw = _exp_so3(np.array([0.0, 0.0, angle]))
    trial_poses = [
        BlockPose(pose.position, world_yaw @ pose.rotation, dims=pose.dims) for pose in base_poses
    ]
    if time.perf_counter() >= deadline_s:
        raise TimeoutError("course yaw trial deadline exceeded")
    assigned = _assign_expected_faces(
        crop, trial_poses, params.assign_band_m, deadline_s=deadline_s
    )
    scores: list[float] = []
    half = np.asarray(trial_poses[0].dims, dtype=np.float64) / 2.0
    interior_margin = min(0.05, float(np.min(half[:2])) * 0.25)
    for pose, (owned, axes, signs) in zip(trial_poses, assigned, strict=True):
        if time.perf_counter() >= deadline_s:
            raise TimeoutError("course yaw trial deadline exceeded")
        if len(owned) < params.min_support_points:
            return None
        local = (owned - pose.position) @ pose.rotation
        residuals = local[np.arange(len(local)), axes] - signs * half[axes]
        vertical_face = (axes == 0) | (axes == 1)
        interior = vertical_face.copy()
        for other_axis in range(3):
            # A point has to be away from both edges of its assigned face.  This
            # deliberately excludes a planned touching joint from yaw evidence.
            not_face = axes != other_axis
            interior &= ~not_face | (
                np.abs(local[:, other_axis]) <= half[other_axis] - interior_margin
            )
        interior_count = int(np.count_nonzero(interior))
        if interior_count < params.min_support_points:
            return None
        # Only physical vertical-face interiors are yaw evidence. Top faces,
        # silhouette edges, and planned/ownership joints never enter the score.
        scores.append(_huber_mean_squared(residuals[interior], params.huber_scale_m))
    if len(scores) < 3 or not np.all(np.isfinite(scores)):
        return None
    return float(np.mean(scores)), tuple(scores)


def _signed_world_yaw_deg(reference: BlockPose, pose: BlockPose) -> float:
    reference_heading = np.asarray(reference.rotation[:2, 0], dtype=np.float64)
    pose_heading = np.asarray(pose.rotation[:2, 0], dtype=np.float64)
    reference_heading /= max(float(np.linalg.norm(reference_heading)), 1e-12)
    pose_heading /= max(float(np.linalg.norm(pose_heading)), 1e-12)
    angle = np.degrees(
        np.arctan2(
            reference_heading[0] * pose_heading[1] - reference_heading[1] * pose_heading[0],
            float(np.dot(reference_heading, pose_heading)),
        )
    )
    # The block frame is pi-symmetric. Select the equivalent delta nearest zero.
    return float((angle + 90.0) % 180.0 - 90.0)


def _yaw_changed_pose(result: PlacedVerification, prior: BlockPose, yaw_deg: float) -> BlockPose:
    independent_yaw = _signed_world_yaw_deg(prior, result.pose)
    correction = np.deg2rad(yaw_deg - independent_yaw)
    # Apply only a world-Z correction to the independent data pose. Its measured
    # roll and pitch are retained rather than reconstructed from the plan frame.
    rotation = _exp_so3(np.array([0.0, 0.0, correction])) @ result.pose.rotation
    return BlockPose(
        result.pose.position,
        rotation,
        confidence=result.pose.confidence,
        source=f"{result.pose.source}+course_yaw",
        dims=result.pose.dims,
    )


def _with_course_yaw_diagnostic(
    result: PlacedVerification, course_id: str, **detail: object
) -> PlacedVerification:
    return replace(
        result,
        diagnostics={
            **result.diagnostics,
            "course_yaw_refinement": {"course_id": course_id, **detail},
        },
    )


def _pose_diagnostic(pose: BlockPose) -> dict[str, object]:
    return {
        "position_m": np.asarray(pose.position, dtype=np.float64).tolist(),
        "rotation": np.asarray(pose.rotation, dtype=np.float64).tolist(),
        "source": pose.source,
    }


def _apply_course_yaw_refinement(
    results: list[PlacedVerification],
    priors: list[PlacedBlockPrior],
    constraints: list[CourseConstraint],
    points: np.ndarray,
    params: VerifyPlacedParams,
) -> list[PlacedVerification]:
    """Refine a verified, named course's common world-Z yaw conservatively.

    This phase only pools raw geometry from frozen independently verified
    anchors.  It does not create blocks, infer a hidden member, or chain one
    corrected result into another.
    """
    if not params.enable_course_yaw_refinement or not constraints:
        return results
    _validate_yaw_courses(constraints)
    by_id = {prior.block_id: index for index, prior in enumerate(priors)}
    updated = list(results)
    supported = {PlacedStatus.CONFIRMED, PlacedStatus.DEVIATED}
    limit_deg = 10.0
    coarse = np.arange(-limit_deg, limit_deg + 1e-9, 0.5)
    for constraint in constraints:
        member_indices = [by_id[block_id] for block_id in constraint.block_ids]
        eligible_indices = sorted(
            (
                index
                for index in member_indices
                if results[index].status in supported
                and results[index].support_points >= params.min_support_points
                and results[index].diagnostics.get("fit_reason") == "ok"
                and not bool(results[index].diagnostics.get("at_search_bound"))
                and results[index].observability is not PlacedObservability.LAYOUT_INFERRED
            ),
            key=lambda index: priors[index].block_id,
        )
        if len(eligible_indices) < 3:
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index],
                    constraint.course_id,
                    accepted=False,
                    reason="insufficient_independent_anchors",
                )
            continue
        base_poses = [priors[index].pose for index in eligible_indices]
        started = time.perf_counter()
        deadline_s = started + 0.20
        if len(points) > params.course_yaw_max_input_points:
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index],
                    constraint.course_id,
                    accepted=False,
                    reason="input_point_limit",
                    input_points=len(points),
                    max_input_points=params.course_yaw_max_input_points,
                )
            continue
        crop = _course_crop(points, base_poses, params)
        if len(crop) < len(eligible_indices) * params.min_support_points:
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index],
                    constraint.course_id,
                    accepted=False,
                    reason="insufficient_raw_course_crop",
                )
            continue
        try:
            crop = _balanced_course_point_cap(crop, base_poses, params, deadline_s)
        except TimeoutError:
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index], constraint.course_id, accepted=False, reason="timeout"
                )
            continue
        scored: dict[float, float] = {}
        per_anchor_scores: dict[float, tuple[float, ...]] = {}
        timed_out = False
        for candidate in coarse:
            if time.perf_counter() >= deadline_s:
                timed_out = True
                break
            try:
                trial = _course_yaw_trial(
                    crop, base_poses, float(candidate), params, deadline_s
                )
            except TimeoutError:
                timed_out = True
                break
            if trial is not None:
                scored[float(candidate)] = trial[0]
                per_anchor_scores[float(candidate)] = trial[1]
        if timed_out or 0.0 not in scored or not scored:
            reason = "timeout" if timed_out else "unsupported_trial_geometry"
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index], constraint.course_id, accepted=False, reason=reason
                )
            continue
        best_coarse = min(scored, key=lambda value: (scored[value], abs(value), value))
        fine = np.arange(
            max(-limit_deg, best_coarse - 0.5), min(limit_deg, best_coarse + 0.5) + 1e-9, 0.1
        )
        for candidate in fine:
            candidate = float(np.round(candidate, 10))
            if candidate in scored:
                continue
            if time.perf_counter() >= deadline_s:
                timed_out = True
                break
            try:
                trial = _course_yaw_trial(crop, base_poses, candidate, params, deadline_s)
            except TimeoutError:
                timed_out = True
                break
            if trial is not None:
                scored[candidate] = trial[0]
                per_anchor_scores[candidate] = trial[1]
        if timed_out:
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index], constraint.course_id, accepted=False, reason="timeout"
                )
            continue
        best_yaw = min(scored, key=lambda value: (scored[value], abs(value), value))
        valid_angles = sorted(per_anchor_scores)
        preferred_yaws = [
            min(
                valid_angles,
                key=lambda angle: (per_anchor_scores[angle][anchor], abs(angle), angle),
            )
            for anchor in range(len(eligible_indices))
        ]
        leave_one_out_yaws = [
            min(
                valid_angles,
                key=lambda angle: (
                    float(
                        np.mean(
                            [
                                score
                                for anchor, score in enumerate(per_anchor_scores[angle])
                                if anchor != omitted
                            ]
                        )
                    ),
                    abs(angle),
                    angle,
                ),
            )
            for omitted in range(len(eligible_indices))
        ]
        consensus_ok = (
            max(preferred_yaws) - min(preferred_yaws) <= 2.0
            and max(leave_one_out_yaws) - min(leave_one_out_yaws) <= 1.0
            and max(abs(value - best_yaw) for value in leave_one_out_yaws) <= 1.0
        )
        baseline = scored[0.0]
        objective = scored[best_yaw]
        improvement = baseline - objective
        absolute_improvement_m = float(np.sqrt(baseline) - np.sqrt(objective))
        relative_improvement = improvement / max(baseline, 1e-12)
        step = 0.1
        left = scored.get(float(np.round(best_yaw - step, 10)))
        right = scored.get(float(np.round(best_yaw + step, 10)))
        curvature = (
            (left + right - 2.0 * objective) / np.deg2rad(step) ** 2
            if left is not None and right is not None
            else np.nan
        )
        interior = abs(best_yaw) < limit_deg - step
        gates_ok = (
            np.isfinite(objective)
            and interior
            and np.isfinite(curvature)
            and curvature > 0.0
            and absolute_improvement_m >= 0.002
            and relative_improvement >= 0.20
            and consensus_ok
        )
        candidate_poses = [
            _yaw_changed_pose(results[index], priors[index].pose, best_yaw)
            for index in eligible_indices
        ]
        in_trust_region = all(
            _rotation_deviation_deg(results[index].expected, pose) <= params.max_rotation_deg + 1e-9
            and _rotation_deviation_deg(results[index].expected, pose) <= limit_deg + 1e-9
            for index, pose in zip(eligible_indices, candidate_poses, strict=True)
        )
        if not gates_ok or not in_trust_region:
            reason = "trust_region" if not in_trust_region else "confidence_gates"
            for index in eligible_indices:
                updated[index] = _with_course_yaw_diagnostic(
                    results[index],
                    constraint.course_id,
                    accepted=False,
                    reason=reason,
                    yaw_deg=best_yaw,
                    baseline_objective_m2=baseline,
                    objective_m2=objective,
                    improvement_m2=improvement,
                    absolute_improvement_m=absolute_improvement_m,
                    relative_improvement=relative_improvement,
                    curvature_m2_per_rad2=curvature,
                    anchor_preferred_yaw_deg=tuple(preferred_yaws),
                    leave_one_out_yaw_deg=tuple(leave_one_out_yaws),
                )
            continue
        variance = max(
            np.deg2rad(constraint.shared_yaw_sigma_deg) ** 2,
            2.0 * params.huber_scale_m**2 / curvature,
        )
        for index, pose in zip(eligible_indices, candidate_poses, strict=True):
            result = results[index]
            covariance = (
                result.covariance.copy()
                if result.covariance is not None
                else np.eye(6) * 1e3
            )
            covariance[2, 2] = max(float(covariance[2, 2]), float(variance))
            updated[index] = replace(
                result,
                pose=pose,
                covariance=covariance,
                observability=PlacedObservability.LAYOUT_INFERRED,
                diagnostics={
                    **result.diagnostics,
                    "layout_filled_directions": ("yaw",),
                    "data_pose": _pose_diagnostic(result.pose),
                    "layout_pose": _pose_diagnostic(pose),
                    "course_yaw_refinement": {
                        "course_id": constraint.course_id,
                        "accepted": True,
                        "yaw_deg": best_yaw,
                        "baseline_objective_m2": baseline,
                        "objective_m2": objective,
                        "improvement_m2": improvement,
                        "absolute_improvement_m": absolute_improvement_m,
                        "relative_improvement": relative_improvement,
                        "curvature_m2_per_rad2": curvature,
                        "anchor_count": len(eligible_indices),
                        "anchor_preferred_yaw_deg": tuple(preferred_yaws),
                        "leave_one_out_yaw_deg": tuple(leave_one_out_yaws),
                        "covariance_source": "pooled_raw_geometry_conservative_curvature",
                    },
                },
            )
    return updated


def _apply_course_constraints(
    results: list[PlacedVerification],
    priors: list[PlacedBlockPrior],
    constraints: list[CourseConstraint],
    *,
    max_translation_m: float = 0.10,
    max_rotation_deg: float = 10.0,
) -> list[PlacedVerification]:
    """Fill translation null directions only; never decide block presence/status."""
    if not constraints:
        return results
    by_id = {prior.block_id: index for index, prior in enumerate(priors)}
    updated = list(results)
    supported = {PlacedStatus.CONFIRMED, PlacedStatus.DEVIATED}
    axis_names = ("x", "y", "z")
    for constraint in constraints:
        for block_id in constraint.block_ids:
            index = by_id[block_id]
            result = updated[index]
            if result.status not in supported:
                continue
            translation_nulls = [
                axis
                for axis, name in enumerate(axis_names)
                if name in result.unconstrained_directions
            ]
            if not translation_nulls:
                continue
            for reference_id in constraint.block_ids:
                if reference_id == block_id:
                    continue
                reference = updated[by_id[reference_id]]
                if reference.status not in supported:
                    continue
                offset = constraint.nominal_offsets.get((reference_id, block_id))
                offset_pair = (reference_id, block_id)
                if offset is None:
                    reverse = constraint.nominal_offsets.get((block_id, reference_id))
                    offset = -np.asarray(reverse) if reverse is not None else None
                    offset_pair = (block_id, reference_id)
                if offset is None:
                    continue
                position = result.pose.position.copy()
                inferred = reference.pose.position + np.asarray(offset, dtype=np.float64)
                for axis in translation_nulls:
                    position[axis] = inferred[axis]
                pose = BlockPose(
                    position,
                    result.pose.rotation,
                    confidence=result.pose.confidence,
                    source=f"{result.pose.source}+course_layout",
                    dims=result.pose.dims,
                )
                final_translation = float(np.linalg.norm(pose.position - result.expected.position))
                final_rotation = _rotation_deviation_deg(result.expected, pose)
                if (
                    final_translation > max_translation_m + 1e-9
                    or final_rotation > max_rotation_deg + 1e-9
                ):
                    updated[index] = replace(
                        result,
                        diagnostics={
                            **result.diagnostics,
                            "layout_rejected": "reacquisition_needed",
                            "layout_candidate_translation_m": final_translation,
                            "layout_candidate_rotation_deg": final_rotation,
                            "layout_course_id": constraint.course_id,
                        },
                    )
                    break
                covariance = (
                    result.covariance.copy() if result.covariance is not None else np.eye(6) * 1e3
                )
                offset_covariance = np.asarray(
                    constraint.relative_offset_covariance.get(
                        offset_pair, np.eye(3) * constraint.translation_sigma_m**2
                    ),
                    dtype=np.float64,
                )
                for axis in translation_nulls:
                    reference_variance = (
                        float(reference.covariance[axis + 3, axis + 3])
                        if reference.covariance is not None
                        else 1e3
                    )
                    # Cross-correlation is not represented by BlockResult's
                    # marginal 6x6 covariance; conservatively inflate it.
                    propagated = 2.0 * (reference_variance + float(offset_covariance[axis, axis]))
                    covariance[axis + 3, axis + 3] = max(covariance[axis + 3, axis + 3], propagated)
                remaining = tuple(
                    name for name in result.unconstrained_directions if name not in axis_names
                )
                prior_layout_directions = tuple(
                    str(name) for name in result.diagnostics.get("layout_filled_directions", ())
                )
                filled_directions = tuple(
                    dict.fromkeys(
                        prior_layout_directions
                        + tuple(axis_names[axis] for axis in translation_nulls)
                    )
                )
                updated[index] = replace(
                    result,
                    pose=pose,
                    covariance=covariance,
                    observability=PlacedObservability.LAYOUT_INFERRED,
                    unconstrained_directions=remaining,
                    diagnostics={
                        **result.diagnostics,
                        "layout_inferred_from": reference_id,
                        "layout_course_id": constraint.course_id,
                        "layout_filled_directions": filled_directions,
                        "layout_covariance_source": "reference_plus_relative_offset",
                        "layout_cross_correlation": "unrepresented_conservative_marginal",
                        "layout_translation_from_plan_m": final_translation,
                        "layout_rotation_from_plan_deg": final_rotation,
                    },
                )
                break
    return updated


def _unoccluded_free_space_rays(
    pose: BlockPose,
    neighbors: list[BlockPose],
    origins: np.ndarray,
    directions: np.ndarray,
    tolerance: float,
    neighbor_margin: float,
    ranges: np.ndarray | None = None,
) -> tuple[int, int]:
    """Count target-through rays for which the target is the first planned volume."""
    if not len(directions):
        return 0, 0
    origin_local = (origins - pose.position) @ pose.rotation
    directions_local = directions @ pose.rotation
    half = np.maximum(np.asarray(pose.dims) / 2.0 - tolerance, 1e-3)
    entry_distance, exit_distance, hit = _ray_box_interval(
        origin_local, directions_local, half
    )
    candidates = hit.copy()
    if ranges is not None:
        candidates &= ranges > exit_distance + tolerance
    target_entry = np.maximum(entry_distance, 0.0)
    target_first = candidates.copy()
    for neighbor in neighbors:
        neighbor_entry = _ray_entry(neighbor, origins, directions, neighbor_margin)
        # Near-equal entries at a planned joint are ambiguous, so suppress them
        # rather than turning uncertain visibility into a Missing claim.
        target_first &= neighbor_entry > target_entry + neighbor_margin
    return (
        int(np.count_nonzero(target_first)),
        int(np.count_nonzero(candidates & ~target_first)),
    )


def verify_placed_blocks(
    points_world: np.ndarray,
    expected: list[BlockPose] | list[PlacedBlockPrior],
    *,
    sensor_context: SensorContext | None = None,
    params: VerifyPlacedParams | None = None,
    course_constraints: list[CourseConstraint] | None = None,
) -> list[PlacedVerification]:
    """Verify expected blocks without converting an unsupported prior into a measurement."""
    settings = params or VerifyPlacedParams()
    _validate_params(settings)
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world must have shape (N, 3) and be finite")
    if not expected:
        return []
    structured, expected_poses = _normalize_priors(expected)
    _validate_priors(expected_poses)
    _validate_courses(structured, course_constraints or [])
    context = _validate_sensor_context(sensor_context) if sensor_context is not None else None
    # Deterministic ordering makes ownership stable under input permutation.
    if len(points):
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    visibility = expected_visibility(
        expected_poses,
        context.origin if context is not None else None,
        margin_m=settings.neighbor_margin_m,
    )
    assigned = _assign_expected_faces(points, expected_poses, settings.assign_band_m, visibility)
    results = []
    for block_index, (prior, owned_data, (visible, occluded)) in enumerate(
        zip(expected_poses, assigned, visibility, strict=True)
    ):
        owned, face_axes, face_signs = owned_data
        pose, fitted, covariance, unconstrained, rms, diagnostics = _bounded_fit(
            prior,
            owned,
            face_axes,
            face_signs,
            settings,
            suppress_ownership_boundaries=(len(expected_poses) > 1 or bool(course_constraints)),
        )
        translation = float(np.linalg.norm(pose.position - prior.position))
        rotation = _rotation_deviation_deg(prior, pose)
        returned_free = 0
        missed_free = 0
        neighbor_occluded_free = 0
        if context is not None:
            neighbors = [
                pose for index, pose in enumerate(expected_poses) if index != block_index
            ]
            returned_free, returned_blocked = _unoccluded_free_space_rays(
                prior,
                neighbors,
                context.ray_origins,
                context.ray_directions,
                settings.surface_tolerance_m,
                settings.neighbor_margin_m,
                context.ranges,
            )
            missed_free, missed_blocked = _unoccluded_free_space_rays(
                prior,
                neighbors,
                context.miss_origins,
                context.miss_directions,
                settings.surface_tolerance_m,
                settings.neighbor_margin_m,
            )
            neighbor_occluded_free = returned_blocked + missed_blocked
        free_rays = returned_free + missed_free
        fit_rejected_nearby = diagnostics.get("fit_reason") in {
            "search_bound",
            "translation_search_exceeded",
            "rotation_search_exceeded",
        }
        if fitted and rms is not None and rms <= settings.surface_tolerance_m:
            status = (
                PlacedStatus.CONFIRMED
                if translation <= settings.confirmed_translation_m
                and rotation <= settings.confirmed_rotation_deg
                else PlacedStatus.DEVIATED
            )
            if not unconstrained:
                observability = PlacedObservability.WELL_CONSTRAINED
            elif len(unconstrained) <= 2:
                observability = PlacedObservability.PARTIALLY_CONSTRAINED
            else:
                observability = PlacedObservability.SYMMETRY_AMBIGUOUS
        elif (
            free_rays >= settings.missing_min_rays
            and not fit_rejected_nearby
            and len(owned) < settings.min_support_points
        ):
            # Positive through-volume evidence is required. Raw XYZ absence can
            # never take this branch because it has no sensor context.
            status = PlacedStatus.MISSING
            observability = PlacedObservability.UNOBSERVABLE
            pose = prior
            covariance = None
            unconstrained = _DIRECTION_NAMES
        else:
            status = PlacedStatus.NOT_OBSERVABLE
            observability = PlacedObservability.UNOBSERVABLE
            pose = prior
            covariance = None
            unconstrained = _DIRECTION_NAMES
        results.append(
            PlacedVerification(
                prior,
                pose,
                status,
                observability,
                covariance,
                unconstrained,
                len(owned),
                rms,
                translation,
                rotation,
                visible,
                occluded,
                free_rays,
                {
                    **diagnostics,
                    "returned_free_space_rays": returned_free,
                    "no_return_free_space_rays": missed_free,
                    "neighbor_occluded_free_space_rays": neighbor_occluded_free,
                },
            )
        )
    yaw_refined = _apply_course_yaw_refinement(
        results,
        structured,
        course_constraints or [],
        points,
        settings,
    )
    return _apply_course_constraints(
        yaw_refined,
        structured,
        course_constraints or [],
        max_translation_m=settings.max_translation_m,
        max_rotation_deg=settings.max_rotation_deg,
    )
