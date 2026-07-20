"""Tests for multi-orientation hypotheses and overlap-based selection.

The same 0.6 x 0.6 x 0.9 block stands upright as (0.6, 0.6, 0.9) or lies on a
long face as (0.9, 0.6, 0.6). Every wall course in config/wall_plans.yaml is
laid, so a detector locked to one orientation is blind to the target task.
"""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.detection import (
    DetectionParams,
    _boxes_overlap,
    _candidate_dims,
    detect_scene_sample,
)
from blockpose.evaluate import evaluate
from blockpose.pose import BlockPose
from blockpose.synth import BlockSpec, SynthParams, generate_scene

UPRIGHT = (0.6, 0.6, 0.9)
LAID = (0.9, 0.6, 0.6)


def _pose(position, dims, yaw_deg=0.0) -> BlockPose:
    angle = np.radians(yaw_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return BlockPose(np.asarray(position, dtype=np.float64), rotation, dims=dims)


def _scene(dims, seed=0):
    return generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=23.0, dims=dims)],
        SynthParams(dims=dims),
        np.random.default_rng(seed),
    )


def test_pose_carries_its_own_extents() -> None:
    sample = _scene(LAID)
    assert sample.ground_truth[0].dims == LAID
    assert _scene(UPRIGHT).ground_truth[0].dims == UPRIGHT


def test_candidate_dims_defaults_to_block_dims_only() -> None:
    params = DetectionParams()
    assert params.candidate_dims == ()
    assert _candidate_dims(params) == (UPRIGHT,)


def test_candidate_dims_expands_non_square_footprints_and_deduplicates() -> None:
    """A laid block's visible side may be the long or the short face."""
    params = DetectionParams(candidate_dims=(UPRIGHT, UPRIGHT, LAID))
    assert _candidate_dims(params) == (UPRIGHT, LAID, (0.6, 0.9, 0.6))


def test_detector_locked_to_upright_cannot_see_a_laid_block() -> None:
    """The blindness that motivates multi-orientation. Pins current behaviour."""
    sample = _scene(LAID)
    detections = detect_scene_sample(sample, DetectionParams(block_dims=UPRIGHT))
    assert detections == []


def test_enabling_the_laid_candidate_recovers_it() -> None:
    sample = _scene(LAID)
    params = DetectionParams(block_dims=UPRIGHT, candidate_dims=(LAID,))
    detections = detect_scene_sample(sample, params)
    result = evaluate(detections, sample.ground_truth)
    assert result.matched == 1
    assert result.unmatched_det == 0
    assert max(result.translation_errors_m) < 0.02
    assert max(result.rotation_errors_deg) < 2.0
    # Either axis ordering describes the same laid block; only the shape is fixed.
    assert sorted(detections[0].dims) == sorted(LAID)


def test_both_candidates_recover_a_laid_block() -> None:
    sample = _scene(LAID)
    params = DetectionParams(candidate_dims=(UPRIGHT, LAID))
    detections = detect_scene_sample(sample, params)
    result = evaluate(detections, sample.ground_truth)
    assert result.matched == 1
    assert max(result.rotation_errors_deg) < 2.0
    assert sorted(detections[0].dims) == sorted(LAID)


def test_both_candidates_still_recover_an_upright_block() -> None:
    sample = _scene(UPRIGHT)
    params = DetectionParams(candidate_dims=(UPRIGHT, LAID))
    detections = detect_scene_sample(sample, params)
    result = evaluate(detections, sample.ground_truth)
    assert result.matched == 1
    assert detections[0].dims == UPRIGHT


def test_refinement_preserves_the_chosen_orientation() -> None:
    sample = _scene(LAID)
    params = DetectionParams(block_dims=UPRIGHT, candidate_dims=(LAID,))
    (detection,) = detect_scene_sample(sample, params)
    assert sorted(detection.dims) == sorted(LAID)
    assert detection.source.endswith("+sdf")


# --- overlap-based selection -------------------------------------------------


def test_identical_boxes_conflict() -> None:
    box = _pose((0.0, 0.0, 0.45), UPRIGHT)
    assert _boxes_overlap(box, box, 0.08)


def test_competing_orientations_of_one_block_conflict() -> None:
    """They sit ~0.21 m apart, outside the old fixed 0.2 m NMS radius."""
    upright = _pose((0.0, 0.0, 0.45), UPRIGHT)
    laid = _pose((0.15, 0.0, 0.6), LAID)
    assert np.linalg.norm(upright.position - laid.position) > 0.2
    assert _boxes_overlap(upright, laid, 0.08)


@pytest.mark.parametrize("spacing", [0.90, 0.92, 1.2])
def test_neighbouring_wall_blocks_do_not_conflict(spacing) -> None:
    """Touching and 2 cm-gap courses must both survive selection."""
    first = _pose((0.0, 0.0, 0.3), LAID)
    second = _pose((spacing, 0.0, 0.3), LAID)
    assert not _boxes_overlap(first, second, 0.08)


def test_stacked_blocks_do_not_conflict() -> None:
    lower = _pose((0.0, 0.0, 0.3), LAID)
    upper = _pose((0.0, 0.0, 0.9), LAID)
    assert not _boxes_overlap(lower, upper, 0.08)


def test_distant_boxes_do_not_conflict() -> None:
    assert not _boxes_overlap(
        _pose((0.0, 0.0, 0.45), UPRIGHT), _pose((3.0, 0.0, 0.45), UPRIGHT), 0.08
    )


def test_half_overlapping_boxes_conflict() -> None:
    assert _boxes_overlap(
        _pose((0.0, 0.0, 0.45), UPRIGHT), _pose((0.25, 0.0, 0.45), UPRIGHT), 0.08
    )


def test_rotated_overlapping_boxes_conflict() -> None:
    """Separating-axis must handle the edge-edge cases a radius test misses."""
    assert _boxes_overlap(
        _pose((0.0, 0.0, 0.45), UPRIGHT), _pose((0.2, 0.2, 0.45), UPRIGHT, yaw_deg=45.0), 0.08
    )


def test_overlap_is_symmetric() -> None:
    first = _pose((0.0, 0.0, 0.45), UPRIGHT)
    second = _pose((0.2, 0.1, 0.5), LAID, yaw_deg=30.0)
    assert _boxes_overlap(first, second, 0.08) == _boxes_overlap(second, first, 0.08)
