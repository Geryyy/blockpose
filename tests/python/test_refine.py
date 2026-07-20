"""Tests for model-based cuboid pose refinement."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.detection import DetectionParams, detect_scene_sample
from blockpose.evaluate import evaluate, rotation_error_deg, translation_error
from blockpose.pose import BlockPose
from blockpose.refine import (
    assign_points,
    box_sdf,
    refine_pose,
    refine_poses,
    surface_distance,
)
from blockpose.synth import BlockSpec, SynthParams, generate_scene

DIMS = (0.6, 0.6, 0.9)
HALF = np.array([0.3, 0.3, 0.45])


def _pose(position, yaw_deg=0.0) -> BlockPose:
    angle = np.radians(yaw_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return BlockPose(np.asarray(position, dtype=np.float64), rotation)


def test_box_sdf_sign_and_magnitude() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # centre: -min half extent
            [0.3, 0.0, 0.0],  # on the +x face
            [0.5, 0.0, 0.0],  # 0.2 outside the +x face
            [0.3, 0.3, 0.45],  # exactly on a corner
        ]
    )
    distances = box_sdf(points, HALF)
    assert distances[0] == pytest.approx(-0.3)
    assert distances[1] == pytest.approx(0.0)
    assert distances[2] == pytest.approx(0.2)
    assert distances[3] == pytest.approx(0.0)


def test_box_sdf_diagonal_outside_is_euclidean() -> None:
    point = np.array([[0.3 + 0.3, 0.3 + 0.4, 0.0]])
    assert box_sdf(point, HALF)[0] == pytest.approx(0.5)


def test_surface_distance_respects_pose() -> None:
    pose = _pose((1.0, 2.0, 0.45), yaw_deg=90.0)
    # Under a 90 deg yaw the block's local +x face points along world +y.
    on_face = np.array([[1.0, 2.3, 0.45]])
    assert surface_distance(pose, on_face, DIMS)[0] == pytest.approx(0.0, abs=1e-9)


def test_refine_recovers_a_perturbed_pose() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=27.0)],
        SynthParams(range_noise_std=0.005),
        np.random.default_rng(3),
    )
    truth = sample.ground_truth[0]
    on_block = sample.points[sample.instance_ids >= 0]

    disturbed = BlockPose(truth.position + np.array([0.05, -0.04, 0.02]), truth.rotation)
    refined, did_refine = refine_pose(disturbed, on_block, DIMS)

    assert did_refine
    assert translation_error(refined, truth) < translation_error(disturbed, truth)
    assert translation_error(refined, truth) < 0.01
    assert rotation_error_deg(refined, truth) < 1.0


def test_refine_is_deterministic() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=12.0)],
        SynthParams(range_noise_std=0.01),
        np.random.default_rng(5),
    )
    points = sample.points[sample.instance_ids >= 0]
    start = BlockPose(sample.ground_truth[0].position + 0.03, sample.ground_truth[0].rotation)
    first, _ = refine_pose(start, points, DIMS)
    second, _ = refine_pose(start, points, DIMS)
    np.testing.assert_allclose(first.position, second.position)
    np.testing.assert_allclose(first.rotation, second.rotation)


def test_refine_declines_when_evidence_is_insufficient() -> None:
    pose = _pose((0.0, 0.0, 0.45))
    sparse = np.array([[0.3, 0.0, 0.4], [0.3, 0.1, 0.5]])
    refined, did_refine = refine_pose(pose, sparse, DIMS)
    assert not did_refine
    assert refined is pose


def test_refine_declines_when_correction_exceeds_guard() -> None:
    """A hypothesis fitted to a distant block must not be dragged onto it."""
    sample = generate_scene(
        [BlockSpec(0.0, 0.0)], SynthParams(range_noise_std=0.0), np.random.default_rng(11)
    )
    on_block = sample.points[sample.instance_ids >= 0]
    far_off = BlockPose(sample.ground_truth[0].position + np.array([1.2, 0.0, 0.0]),
                        sample.ground_truth[0].rotation)
    refined, did_refine = refine_pose(far_off, on_block, DIMS, max_translation=0.15)
    assert not did_refine
    assert refined is far_off


def test_refine_keeps_source_and_confidence_provenance() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0)], SynthParams(range_noise_std=0.005), np.random.default_rng(2)
    )
    truth = sample.ground_truth[0]
    start = BlockPose(truth.position + 0.02, truth.rotation, confidence=0.7, source="plane_fit")
    refined, did_refine = refine_pose(start, sample.points[sample.instance_ids >= 0], DIMS)
    assert did_refine
    assert refined.confidence == 0.7
    assert refined.source == "plane_fit+sdf"


def test_assign_points_splits_a_stack_between_hypotheses() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, z=0.45), BlockSpec(0.0, 0.0, z=1.35)],
        SynthParams(range_noise_std=0.005, look_at=(0.0, 0.0, 0.9)),
        np.random.default_rng(4),
    )
    poses = [BlockPose(g.position, g.rotation) for g in sample.ground_truth]
    on_blocks = sample.points[sample.instance_ids >= 0]
    labels = sample.instance_ids[sample.instance_ids >= 0]

    lower, upper = assign_points(poses, on_blocks, DIMS, band=0.10)
    assert len(lower) > 50 and len(upper) > 50
    # Each hypothesis should predominantly own the returns from its own block.
    for index, owned in enumerate((lower, upper)):
        matched = np.array([np.any(np.all(np.isclose(on_blocks[labels == index], p), axis=1))
                            for p in owned])
        assert matched.mean() > 0.9


def test_assign_points_drops_returns_far_from_every_hypothesis() -> None:
    poses = [_pose((0.0, 0.0, 0.45))]
    points = np.array([[0.3, 0.0, 0.45], [5.0, 5.0, 0.45]])
    assigned = assign_points(poses, points, DIMS, band=0.10)
    assert len(assigned[0]) == 1
    np.testing.assert_allclose(assigned[0][0], [0.3, 0.0, 0.45])


def test_assign_points_handles_empty_inputs() -> None:
    assert assign_points([], np.zeros((5, 3)), DIMS, band=0.1) == []
    assigned = assign_points([_pose((0.0, 0.0, 0.45))], np.zeros((0, 3)), DIMS, band=0.1)
    assert len(assigned) == 1 and len(assigned[0]) == 0


def test_refine_poses_improves_a_stack_without_collapsing_it() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, z=0.45), BlockSpec(0.25, 0.0, z=1.35)],
        SynthParams(range_noise_std=0.01, look_at=(0.1, 0.0, 0.9)),
        np.random.default_rng(8),
    )
    truth = sample.ground_truth
    start = [BlockPose(g.position + np.array([0.03, 0.02, 0.0]), g.rotation) for g in truth]
    refined = refine_poses(start, sample.points, DIMS)

    assert len(refined) == 2
    for before, after, gt in zip(start, refined, truth, strict=True):
        assert translation_error(after, gt) <= translation_error(before, gt)
    # The two hypotheses must stay distinct rather than converging on one block.
    separation = np.linalg.norm(refined[0].position - refined[1].position)
    assert separation > 0.5


def test_refinement_improves_detector_output_end_to_end() -> None:
    errors = {}
    for enabled in (False, True):
        translations, rotations = [], []
        for seed in range(4):
            rng = np.random.default_rng(seed)
            spec = BlockSpec(
                x=float(rng.uniform(-0.3, 0.3)),
                y=float(rng.uniform(-0.3, 0.3)),
                yaw_deg=float(rng.uniform(0.0, 360.0)),
                pitch_deg=float(rng.uniform(-3.0, 3.0)),
            )
            sample = generate_scene([spec], SynthParams(range_noise_std=0.02), rng)
            detections = detect_scene_sample(sample, DetectionParams(refine_enabled=enabled))
            result = evaluate(detections, sample.ground_truth, max_match_dist=0.5)
            translations += result.translation_errors_m
            rotations += result.rotation_errors_deg
        errors[enabled] = (max(translations), max(rotations))

    assert errors[True][0] < errors[False][0]
    assert errors[True][1] < errors[False][1]


def test_refinement_can_be_disabled() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=15.0)],
        SynthParams(range_noise_std=0.01),
        np.random.default_rng(6),
    )
    plain = detect_scene_sample(sample, DetectionParams(refine_enabled=False))
    refined = detect_scene_sample(sample, DetectionParams(refine_enabled=True))
    assert plain and refined
    assert plain[0].source == "plane_fit"
    assert refined[0].source.endswith("+sdf")
