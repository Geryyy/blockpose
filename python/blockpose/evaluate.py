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
    if not detected or not ground_truth:
        return []

    n_det = len(detected)
    n_gt = len(ground_truth)
    cost = np.zeros((n_det, n_gt), dtype=np.float64)
    for i, d in enumerate(detected):
        for j, g in enumerate(ground_truth):
            cost[i, j] = translation_error(d, g)

    det_idx, gt_idx = linear_sum_assignment(cost)
    pairs = []
    for di, gi in zip(det_idx, gt_idx):
        if cost[di, gi] <= max_dist:
            pairs.append((int(di), int(gi)))
    return pairs


@dataclass
class EvalResult:
    matched: int
    unmatched_det: int
    unmatched_gt: int
    translation_errors_m: list[float] = field(default_factory=list)
    rotation_errors_deg: list[float] = field(default_factory=list)
    mean_translation_m: float = 0.0
    mean_rotation_deg: float = 0.0


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
        mean_translation_m=float(np.mean(t_errors)) if t_errors else 0.0,
        mean_rotation_deg=float(np.mean(r_errors)) if r_errors else 0.0,
    )
