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

import numpy as np

# Four-nines confidence that at least one trial drew an all-inlier sample.
DEFAULT_CONFIDENCE = 0.9999


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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the dominant plane. Returns ``(plane_model, inlier_indices)``.

    `plane_model` is ``[a, b, c, d]`` with a unit normal, matching Open3D's
    convention so call sites need no change beyond the import. Ties on inlier
    count are broken by the earlier trial, so the result depends only on `seed`
    and the input ordering.

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

    rng = np.random.default_rng(seed)
    # Draw every candidate sample up front: one vectorized call is far cheaper
    # than num_iterations separate rng.choice calls, and unused rows cost
    # nothing once the adaptive stop fires.
    samples = rng.integers(0, count, size=(num_iterations, 3))

    best_inlier_count = -1
    best_plane = np.array([0.0, 0.0, 1.0, 0.0])
    best_mask = np.zeros(count, dtype=bool)
    required = num_iterations

    for iteration in range(num_iterations):
        if iteration >= required:
            break
        index = samples[iteration]
        if index[0] == index[1] or index[1] == index[2] or index[0] == index[2]:
            continue
        plane = _plane_from_triplet(points[index])
        if plane is None:
            continue
        normal, offset = plane
        mask = np.abs(points @ normal + offset) <= distance_threshold
        inliers = int(np.count_nonzero(mask))
        if inliers <= best_inlier_count:
            continue
        best_inlier_count = inliers
        best_plane = np.array([normal[0], normal[1], normal[2], offset])
        best_mask = mask
        # Adaptive stop: with this inlier ratio, how many trials are needed for
        # `confidence` that one sample was clean?
        ratio = inliers / count
        if ratio >= 1.0:
            required = iteration + 1
        elif ratio > 0.0:
            denominator = np.log1p(-(ratio**3))
            if denominator < 0.0:
                needed = int(np.ceil(np.log1p(-confidence) / denominator))
                required = min(num_iterations, iteration + 1 + needed)

    if best_inlier_count < 3:
        raise ValueError("RANSAC found no plane with at least 3 inliers")

    # Least-squares refit on the consensus set; the minimal sample only seeds it.
    inlier_points = points[best_mask]
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

    return best_plane, np.flatnonzero(best_mask)
