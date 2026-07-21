"""Tests for blockpose.evaluate metric functions."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.evaluate import (
    evaluate,
    match_detections,
    rotation_error_deg,
    translation_error,
)
from blockpose.pose import BlockPose


def _make_pose(position=(0, 0, 0), yaw_deg=0.0) -> BlockPose:
    """Helper: create a BlockPose at given position with yaw about world Z."""
    angle = np.radians(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return BlockPose(position=np.array(position, dtype=np.float64), rotation=R)


def test_translation_error_zero_for_same() -> None:
    pose = _make_pose([1.0, 2.0, 3.0])
    assert translation_error(pose, pose) == pytest.approx(0.0)


def test_translation_error_known_distance() -> None:
    a = _make_pose([0.0, 0.0, 0.0])
    b = _make_pose([3.0, 4.0, 0.0])
    assert translation_error(a, b) == pytest.approx(5.0)


_UPRIGHT_DIMS = (0.6, 0.6, 0.9)  # square footprint about Z: yaw is a free symmetry


def test_rotation_error_zero_for_same() -> None:
    pose = _make_pose(yaw_deg=37.0)
    assert rotation_error_deg(pose, pose) == pytest.approx(0.0, abs=1e-5)


def test_rotation_error_90deg_is_zero_by_symmetry() -> None:
    """90-degree rotation about world Z is a symmetry equivalence — error = 0.

    Pinned to an explicit upright (square-about-Z) dims tuple rather than
    relying on blockpose.pose.BLOCK_DIMS, since which axis pair is equal (and
    thus which rotation is "free") depends entirely on that shape.
    """
    det = _make_pose(yaw_deg=90.0)
    gt = _make_pose(yaw_deg=0.0)
    assert rotation_error_deg(det, gt, _UPRIGHT_DIMS) == pytest.approx(0.0, abs=1e-5)


def test_rotation_error_180deg_is_zero_by_symmetry() -> None:
    det = _make_pose(yaw_deg=180.0)
    gt = _make_pose(yaw_deg=0.0)
    assert rotation_error_deg(det, gt, _UPRIGHT_DIMS) == pytest.approx(0.0, abs=1e-5)


def test_rotation_error_45deg_is_maximum_ambiguity() -> None:
    """45° is halfway between two symmetry equivalents — error = 45°."""
    det = _make_pose(yaw_deg=45.0)
    gt = _make_pose(yaw_deg=0.0)
    assert rotation_error_deg(det, gt, _UPRIGHT_DIMS) == pytest.approx(45.0, abs=0.1)


def test_match_detections_empty_inputs() -> None:
    assert match_detections([], []) == []
    assert match_detections([_make_pose()], []) == []
    assert match_detections([], [_make_pose()]) == []


@pytest.mark.parametrize("max_dist", [-0.1, np.nan, np.inf])
def test_match_detections_rejects_invalid_cutoff(max_dist: float) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        match_detections([], [], max_dist=max_dist)


def test_match_detections_exact_match() -> None:
    pose = _make_pose([1.0, 0.0, 0.5])
    pairs = match_detections([pose], [pose])
    assert pairs == [(0, 0)]


def test_match_detections_far_detection_unmatched() -> None:
    det = _make_pose([10.0, 0.0, 0.0])
    gt = _make_pose([0.0, 0.0, 0.0])
    pairs = match_detections([det], [gt], max_dist=0.5)
    assert pairs == []


def test_match_detections_hungarian_beats_greedy() -> None:
    """Construct a case where greedy nearest-neighbor is suboptimal."""
    # GT at A=(0,0,0) and B=(1,0,0)
    # Det at C=(0.4,0,0) and D=(0.0,0,0)
    # Greedy: C→A (dist 0.4), D→B (dist 1.0), total 1.4
    # Hungarian: C→B (dist 0.6), D→A (dist 0.0), total 0.6  ← optimal
    gt = [_make_pose([0.0, 0.0, 0.0]), _make_pose([1.0, 0.0, 0.0])]
    det = [_make_pose([0.4, 0.0, 0.0]), _make_pose([0.0, 0.0, 0.0])]
    pairs = match_detections(det, gt, max_dist=2.0)
    # Hungarian should give: det[1]→gt[0] (dist=0) and det[0]→gt[1] (dist=0.6)
    pair_dict = dict(pairs)
    assert pair_dict.get(1) == 0  # det[1] (at 0,0,0) matches gt[0] (at 0,0,0)
    assert pair_dict.get(0) == 1  # det[0] (at 0.4,0,0) matches gt[1] (at 1,0,0)


def test_match_maximizes_valid_cardinality_before_distance() -> None:
    """An invalid cheap global assignment must not hide two valid matches."""
    gt = [_make_pose([0.0, 0.0, 0.0]), _make_pose([0.3, 0.0, 0.0])]
    det = [_make_pose([0.1, 0.0, 0.0]), _make_pose([-0.2, 0.0, 0.0])]
    pairs = match_detections(det, gt, max_dist=0.25)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_evaluate_all_matched() -> None:
    gt = [_make_pose([0.0, 0.0, 0.5]), _make_pose([1.0, 0.0, 0.5])]
    det = [_make_pose([0.05, 0.0, 0.5]), _make_pose([1.05, 0.0, 0.5])]
    result = evaluate(det, gt)
    assert result.matched == 2
    assert result.unmatched_det == 0
    assert result.unmatched_gt == 0
    assert result.mean_translation_m == pytest.approx(0.05, abs=1e-6)


def test_evaluate_unmatched_detection() -> None:
    gt = [_make_pose([0.0, 0.0, 0.5])]
    det = [_make_pose([0.0, 0.0, 0.5]), _make_pose([5.0, 0.0, 0.5])]
    result = evaluate(det, gt, max_match_dist=0.5)
    assert result.matched == 1
    assert result.unmatched_det == 1
    assert result.unmatched_gt == 0


def test_evaluate_no_matches_has_undefined_pose_errors() -> None:
    result = evaluate([_make_pose([10, 0, 0])], [_make_pose()], max_match_dist=0.5)
    assert result.mean_translation_m is None
    assert result.translation_p95_m is None
    assert result.mean_rotation_deg is None
    assert result.precision == 0.0
    assert result.recall == 0.0
