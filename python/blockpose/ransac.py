"""Deterministic RANSAC plane fitting.

Open3D 0.19's `PointCloud.segment_plane` takes no seed, and
`o3d.utility.random.seed` does not reach it, so repeated calls on identical
input return different planes. Everything else in the pipeline is careful about
reproducibility -- points are sorted before voxelization, hypothesis selection
breaks ties on position, scenario seeds come from a SeedSequence -- but that one
call made detection non-reproducible end to end, which undermines a regression
gate and made one calibration test flaky about one run in three.

This is a plain seeded RANSAC with adaptive early termination. The adaptive stop
matters for speed: a dominant plane with an 80% inlier ratio needs about 13
trials for four-nines confidence rather than the nominal 1000, so a pure-numpy
implementation stays competitive with the parallel C++ one it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

# Four-nines confidence that at least one trial drew an all-inlier sample.
DEFAULT_CONFIDENCE = 0.9999


@dataclass(frozen=True)
class PlaneFitDiagnostics:
    """Stable work counters and timings for one plane fit."""

    input_points: int
    search_points: int
    trials_evaluated: int
    valid_trials: int
    search_ms: float
    full_score_refit_ms: float


def _plane_from_triplet(triplet: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Plane through three points as (unit normal, offset), or None if collinear."""
    origin, second, third = triplet
    normal = np.cross(second - origin, third - origin)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return None
    normal = normal / length
    return normal, -float(normal @ origin)


def segment_plane(
    points: np.ndarray,
    distance_threshold: float,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    *,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
    max_search_points: int | None = None,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, PlaneFitDiagnostics]:
    """Fit the dominant plane. Returns ``(plane_model, inlier_indices)``.

    `plane_model` is ``[a, b, c, d]`` with a unit normal, matching Open3D's
    convention so call sites need no change beyond the import. Ties on inlier
    count are broken by the earlier trial. Canonical point ordering makes the
    result depend only on point values and `seed`, not input ordering.

    `ransac_n` is accepted for signature compatibility; the minimal sample for a
    plane is three points and larger samples are not used.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    count = len(points)
    if count < 3:
        raise ValueError(f"plane fitting needs at least 3 points, got {count}")
    if not np.isfinite(distance_threshold) or distance_threshold <= 0.0:
        raise ValueError(f"distance_threshold must be positive, got {distance_threshold}")
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")
    if max_search_points is not None and max_search_points < 3:
        raise ValueError(f"max_search_points must be at least 3, got {max_search_points}")

    # Search a canonical, uniformly distributed subset when a connected region
    # is oversized. Sorting makes both the subset and the sampled triplets
    # independent of input point order. Only hypothesis search is bounded: the
    # winning plane is always scored and least-squares-refitted on all points.
    canonical = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    if max_search_points is not None and count > max_search_points:
        subset_indices = np.linspace(0, count - 1, max_search_points, dtype=np.int64)
        search_points = canonical[subset_indices]
    else:
        search_points = canonical
    search_count = len(search_points)

    rng = np.random.default_rng(seed)
    # Draw every candidate sample up front: one vectorized call is far cheaper
    # than num_iterations separate rng.choice calls, and unused rows cost
    # nothing once the adaptive stop fires.
    samples = rng.integers(0, search_count, size=(num_iterations, 3))

    best_inlier_count = -1
    best_plane = np.array([0.0, 0.0, 1.0, 0.0])
    best_mask = np.zeros(search_count, dtype=bool)
    required = num_iterations
    trials_evaluated = 0
    valid_trials = 0
    search_started = perf_counter()

    # Batch the expensive point-to-plane products. Candidate decisions remain
    # in trial order, including adaptive termination and first-wins tie breaks.
    # This removes Python/matmul dispatch per trial without changing RANSAC's
    # deterministic selection semantics.
    batch_size = 64
    iteration = 0
    while iteration < min(required, num_iterations):
        stop = min(iteration + batch_size, required, num_iterations)
        batch_samples = samples[iteration:stop]
        triplets = search_points[batch_samples]
        raw_normals = np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0])
        lengths = np.linalg.norm(raw_normals, axis=1)
        distinct = (
            (batch_samples[:, 0] != batch_samples[:, 1])
            & (batch_samples[:, 1] != batch_samples[:, 2])
            & (batch_samples[:, 0] != batch_samples[:, 2])
        )
        valid = distinct & (lengths >= 1e-12)
        normals = np.zeros_like(raw_normals)
        normals[valid] = raw_normals[valid] / lengths[valid, None]
        offsets = -np.einsum("ij,ij->i", normals, triplets[:, 0])
        inlier_counts = np.zeros(len(batch_samples), dtype=np.int64)
        if np.any(valid):
            residuals = search_points @ normals[valid].T + offsets[valid]
            inlier_counts[valid] = np.count_nonzero(
                np.abs(residuals) <= distance_threshold, axis=0
            )

        for local_index in range(len(batch_samples)):
            absolute_index = iteration + local_index
            if absolute_index >= required:
                break
            trials_evaluated += 1
            if not valid[local_index]:
                continue
            valid_trials += 1
            inliers = int(inlier_counts[local_index])
            if inliers <= best_inlier_count:
                continue
            normal = normals[local_index]
            offset = float(offsets[local_index])
            best_inlier_count = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], offset])
            best_mask = np.abs(search_points @ normal + offset) <= distance_threshold
            ratio = inliers / search_count
            if ratio >= 1.0:
                required = absolute_index + 1
            elif ratio > 0.0:
                denominator = np.log1p(-(ratio**3))
                if denominator < 0.0:
                    needed = int(np.ceil(np.log1p(-confidence) / denominator))
                    required = min(num_iterations, absolute_index + 1 + needed)
        iteration = stop

    if best_inlier_count < 3:
        raise ValueError("RANSAC found no plane with at least 3 inliers")

    search_ms = (perf_counter() - search_started) * 1000.0
    full_started = perf_counter()

    # Restore full-data evidence semantics. The subset chooses a candidate only;
    # consensus indices and the returned least-squares plane use the full input.
    best_mask = np.abs(points @ best_plane[:3] + best_plane[3]) <= distance_threshold
    best_inlier_count = int(np.count_nonzero(best_mask))
    if best_inlier_count < 3:
        raise ValueError("RANSAC found no plane with at least 3 full-data inliers")

    # Least-squares refit on the consensus set; the minimal sample only seeds it.
    inlier_points = points[best_mask]
    inlier_points = inlier_points[
        np.lexsort((inlier_points[:, 2], inlier_points[:, 1], inlier_points[:, 0]))
    ]
    centroid = inlier_points.mean(axis=0)
    _, _, basis = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = basis[2]
    # Keep the sign stable so repeated fits of the same surface agree.
    if float(normal @ best_plane[:3]) < 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    refined = np.abs(points @ normal + offset) <= distance_threshold
    if int(np.count_nonzero(refined)) >= best_inlier_count:
        best_plane = np.array([normal[0], normal[1], normal[2], offset])
        best_mask = refined

    diagnostics = PlaneFitDiagnostics(
        input_points=count,
        search_points=search_count,
        trials_evaluated=trials_evaluated,
        valid_trials=valid_trials,
        search_ms=search_ms,
        full_score_refit_ms=(perf_counter() - full_started) * 1000.0,
    )
    result = (best_plane, np.flatnonzero(best_mask))
    if return_diagnostics:
        return (*result, diagnostics)
    return result
