"""Evaluation utilities: matching and error metrics for block detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from blockpose.pose import BlockPose, symmetry_equivalents


def translation_error(det: BlockPose, gt: BlockPose) -> float:
    """Euclidean distance between detected and GT block centers (meters)."""
    return float(np.linalg.norm(det.position - gt.position))


def rotation_error_deg(det: BlockPose, gt: BlockPose) -> float:
    """Symmetry-aware minimum rotation error in degrees.

    Returns the minimum rotation angle (in degrees) over all 4 symmetry-equivalent
    orientations of the detected block relative to the GT orientation.
    """
    equivalents = symmetry_equivalents(det.rotation)
    min_angle = np.inf
    for R_eq in equivalents:
        dR = R_eq.T @ gt.rotation
        # Clamp trace to valid range for arccos
        trace_val = float(np.trace(dR))
        cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = float(np.arccos(cos_angle))
        if angle_rad < min_angle:
            min_angle = angle_rad
    return float(np.degrees(min_angle))


def match_detections(
    detected: list[BlockPose],
    ground_truth: list[BlockPose],
    max_dist: float = 0.5,
) -> list[tuple[int, int]]:
    """Match detected poses to ground truth using Hungarian algorithm.

    Args:
        detected: List of detected BlockPose objects.
        ground_truth: List of ground-truth BlockPose objects.
        max_dist: Maximum distance (m) for a valid match.

    Returns:
        List of (det_idx, gt_idx) pairs for matched blocks.
    """
    if not np.isfinite(max_dist) or max_dist < 0.0:
        raise ValueError(f"max_dist must be non-negative and finite, got {max_dist}")
    if not detected or not ground_truth:
        return []

    n_det = len(detected)
    n_gt = len(ground_truth)
    distances = np.zeros((n_det, n_gt), dtype=np.float64)
    for i, d in enumerate(detected):
        for j, g in enumerate(ground_truth):
            distances[i, j] = translation_error(d, g)

    # Include explicit unmatched rows/columns in the assignment.  Filtering a
    # plain Hungarian assignment after the fact can discard an over-threshold
    # pair even when a different assignment contains more valid matches.
    unmatched_cost = max(float(max_dist), 0.0) + 1.0
    invalid_cost = 3.0 * unmatched_cost
    size = n_det + n_gt
    cost = np.full((size, size), invalid_cost, dtype=np.float64)
    cost[:n_det, :n_gt] = np.where(distances <= max_dist, distances, invalid_cost)
    cost[:n_det, n_gt:] = unmatched_cost
    cost[n_det:, :n_gt] = unmatched_cost
    cost[n_det:, n_gt:] = 0.0

    det_idx, gt_idx = linear_sum_assignment(cost)
    pairs = []
    for di, gi in zip(det_idx, gt_idx, strict=True):
        if di < n_det and gi < n_gt and distances[di, gi] <= max_dist:
            pairs.append((int(di), int(gi)))
    return pairs


@dataclass
class EvalResult:
    matched: int
    unmatched_det: int
    unmatched_gt: int
    translation_errors_m: list[float] = field(default_factory=list)
    rotation_errors_deg: list[float] = field(default_factory=list)
    mean_translation_m: float | None = None
    mean_rotation_deg: float | None = None
    translation_p50_m: float | None = None
    translation_p95_m: float | None = None
    rotation_p50_deg: float | None = None
    rotation_p95_deg: float | None = None
    precision: float = 0.0
    recall: float = 0.0
    exact_count: bool = False


def evaluate(
    detected: list[BlockPose],
    ground_truth: list[BlockPose],
    max_match_dist: float = 0.5,
) -> EvalResult:
    """Evaluate detections against ground truth.

    Args:
        detected: List of detected BlockPose objects.
        ground_truth: List of ground-truth BlockPose objects.
        max_match_dist: Maximum translation distance for a valid match (m).

    Returns:
        EvalResult with match counts and error statistics.
    """
    pairs = match_detections(detected, ground_truth, max_dist=max_match_dist)
    matched_det = {di for di, _ in pairs}
    matched_gt = {gi for _, gi in pairs}

    t_errors = [translation_error(detected[di], ground_truth[gi]) for di, gi in pairs]
    r_errors = [rotation_error_deg(detected[di], ground_truth[gi]) for di, gi in pairs]

    return EvalResult(
        matched=len(pairs),
        unmatched_det=len(detected) - len(matched_det),
        unmatched_gt=len(ground_truth) - len(matched_gt),
        translation_errors_m=t_errors,
        rotation_errors_deg=r_errors,
        mean_translation_m=float(np.mean(t_errors)) if t_errors else None,
        mean_rotation_deg=float(np.mean(r_errors)) if r_errors else None,
        translation_p50_m=float(np.percentile(t_errors, 50)) if t_errors else None,
        translation_p95_m=float(np.percentile(t_errors, 95)) if t_errors else None,
        rotation_p50_deg=float(np.percentile(r_errors, 50)) if r_errors else None,
        rotation_p95_deg=float(np.percentile(r_errors, 95)) if r_errors else None,
        precision=len(pairs) / len(detected) if detected else 0.0,
        recall=len(pairs) / len(ground_truth) if ground_truth else 0.0,
        exact_count=len(detected) == len(ground_truth),
    )
