"""Plan-conditioned verification of placed concrete blocks.

Unlike blind detection this module starts from stable plan identities.  Returns
are assigned exclusively to the nearest expected cuboid, then each cuboid is
fitted inside a small, explicit trust region.  An absent block is only called
``Missing`` when rays demonstrate free space; raw XYZ silence is
``NotObservable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from blockpose.detection import (
    SensorContext,
    _ray_box_interval,
    _validate_sensor_context,
    free_space_violations_from_misses,
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
    if params.min_support_points < 3 or params.missing_min_rays < 1 or params.max_nfev < 1:
        raise ValueError("verification count parameters are invalid")


def _ray_entry(
    pose: BlockPose, origins: np.ndarray, directions: np.ndarray, margin: float
) -> np.ndarray:
    origins_local = (origins - pose.position) @ pose.rotation
    dirs_local = directions @ pose.rotation
    half = np.asarray(pose.dims) / 2.0 + margin
    entries = np.full(len(origins), np.inf)
    for index, (origin, direction) in enumerate(zip(origins_local, dirs_local, strict=True)):
        enter, _, hit = _ray_box_interval(origin, direction[None], half)
        if hit[0]:
            entries[index] = max(float(enter[0]), 0.0)
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
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not len(points):
        return [
            (np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0, dtype=float)) for _ in expected
        ]
    distances = []
    block_axes = []
    block_signs = []
    for index in range(len(expected)):
        pose = expected[index]
        local = (points - pose.position) @ pose.rotation
        half = np.asarray(pose.dims, dtype=np.float64) / 2.0
        face_distances = []
        face_labels = []
        bottom_z = pose.position[2] - float(np.sum(np.abs(pose.rotation[2]) * half))
        above_support = points[:, 2] > bottom_z + min(band_m, 0.05)
        for axis in range(3):
            for sign in (-1.0, 1.0):
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
    supported_faces = {
        (int(axis), int(sign))
        for axis, sign in zip(face_axes, face_signs, strict=True)
        if sign != 0
    }
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
    retained_pose = pose if geometry_sufficient and correction_supported else prior
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
            "prior_residual_rms_m": prior_rms,
            "residual_improvement_m": improvement,
            "correction_supported": correction_supported,
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


def _returned_free_space_rays(pose: BlockPose, context: SensorContext, tolerance: float) -> int:
    origin_local = pose.rotation.T @ (context.origin - pose.position)
    directions_local = context.ray_directions @ pose.rotation
    half = np.maximum(np.asarray(pose.dims) / 2.0 - tolerance, 1e-3)
    _, exit_distance, hit = _ray_box_interval(origin_local, directions_local, half)
    return int(np.count_nonzero(hit & (context.ranges > exit_distance + tolerance)))


def verify_placed_blocks(
    points_world: np.ndarray,
    expected: list[BlockPose],
    *,
    sensor_context: SensorContext | None = None,
    params: VerifyPlacedParams | None = None,
) -> list[PlacedVerification]:
    """Verify expected blocks without converting an unsupported prior into a measurement."""
    settings = params or VerifyPlacedParams()
    _validate_params(settings)
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world must have shape (N, 3) and be finite")
    if not expected:
        return []
    _validate_priors(expected)
    context = _validate_sensor_context(sensor_context) if sensor_context is not None else None
    # Deterministic ordering makes ownership stable under input permutation.
    if len(points):
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    visibility = expected_visibility(
        expected,
        context.origin if context is not None else None,
        margin_m=settings.neighbor_margin_m,
    )
    assigned = _assign_expected_faces(points, expected, settings.assign_band_m, visibility)
    results = []
    for prior, owned_data, (visible, occluded) in zip(expected, assigned, visibility, strict=True):
        owned, face_axes, face_signs = owned_data
        pose, fitted, covariance, unconstrained, rms, diagnostics = _bounded_fit(
            prior, owned, face_axes, face_signs, settings
        )
        translation = float(np.linalg.norm(pose.position - prior.position))
        rotation = _rotation_deviation_deg(prior, pose)
        returned_free = 0
        missed_free = 0
        if context is not None:
            returned_free = _returned_free_space_rays(prior, context, settings.surface_tolerance_m)
            missed_free = free_space_violations_from_misses(
                prior,
                context.origin,
                context.miss_directions,
                prior.dims,
                settings.surface_tolerance_m,
            )
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
            and visible
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
                },
            )
        )
    return results
