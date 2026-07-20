"""Tests for plan-prior detection: verify and re-estimate blocks at known poses."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.detection import DetectionParams, SensorContext
from blockpose.evaluate import rotation_error_deg, translation_error
from blockpose.plan_prior import detect_with_plan
from blockpose.pose import BlockPose
from blockpose.synth import SynthParams, generate_scene
from blockpose.wall_plans import load_wall_plans

LAID = (0.9, 0.6, 0.6)
PLANS = load_wall_plans()


def _scene(name, standoff=5.0, present=None, seed=0):
    plan = PLANS[name]
    specs = list(plan.specs) if present is None else present
    params = SynthParams(
        az_fov_deg=85.0, el_fov_deg=60.0, **plan.sensor_params_kwargs(standoff=standoff)
    )
    return plan, generate_scene(specs, params, np.random.default_rng(seed))


def _context(sample) -> SensorContext:
    return SensorContext(
        sample.sensor_origin, sample.ray_directions, sample.ranges, sample.miss_directions
    )


def _priors(sample, jitter=0.0, seed=1):
    rng = np.random.default_rng(seed)
    priors = []
    for gt in sample.ground_truth:
        offset = rng.uniform(-jitter, jitter, 3) if jitter else np.zeros(3)
        priors.append(BlockPose(gt.position + offset, gt.rotation, dims=gt.dims))
    return priors


def test_recovers_a_wall_row_blind_detection_cannot_separate() -> None:
    """row3_middle_last is 0/3 blind; the plan prior recovers all three."""
    _, sample = _scene("row3_middle_last")
    results = detect_with_plan(
        sample.points, _priors(sample, jitter=0.04), DetectionParams(),
        sensor_context=_context(sample),
    )
    assert len(results) == 3
    for result, gt in zip(results, sample.ground_truth, strict=True):
        assert result.score > 0.6
        assert translation_error(result.pose, gt) < 0.03
        assert rotation_error_deg(result.pose, gt, dims=gt.dims) < 2.0


def test_refinement_pulls_a_jittered_prior_onto_the_block() -> None:
    _, sample = _scene("row3_middle_last")
    priors = _priors(sample, jitter=0.04)
    results = detect_with_plan(
        sample.points, priors, DetectionParams(), sensor_context=_context(sample)
    )
    for prior, result, gt in zip(priors, results, sample.ground_truth, strict=True):
        assert translation_error(result.pose, gt) < translation_error(prior, gt)
        assert result.pose.source == "plan_prior"
        assert result.pose.dims == gt.dims


def test_absent_block_scores_below_present_blocks() -> None:
    """A block in the plan but not yet placed must be distinguishable by score."""
    plan = PLANS["example_wall"]
    present = list(plan.specs)[:3]
    _, sample = _scene("example_wall", present=present)
    priors = [BlockPose(np.array([s.x, s.y, s.z]), np.eye(3), dims=s.dims) for s in plan.specs]
    results = detect_with_plan(
        sample.points, priors, DetectionParams(), sensor_context=_context(sample)
    )
    present_scores = [results[i].score for i in range(3)]
    absent_scores = [results[i].score for i in (3, 4)]
    assert min(present_scores) > max(absent_scores)


def test_results_follow_expected_order() -> None:
    _, sample = _scene("row3_middle_last")
    priors = _priors(sample)
    results = detect_with_plan(
        sample.points, priors, DetectionParams(), sensor_context=_context(sample)
    )
    for prior, result in zip(priors, results, strict=True):
        assert result.expected is prior


def test_empty_expected_list_returns_nothing() -> None:
    _, sample = _scene("row3_middle_last")
    assert detect_with_plan(sample.points, [], DetectionParams()) == []


def test_translation_from_plan_reports_the_correction() -> None:
    _, sample = _scene("row3_middle_last")
    priors = _priors(sample, jitter=0.05)
    results = detect_with_plan(
        sample.points, priors, DetectionParams(), sensor_context=_context(sample)
    )
    for prior, result in zip(priors, results, strict=True):
        expected = np.linalg.norm(result.pose.position - prior.position)
        assert result.translation_from_plan == pytest.approx(expected)


def test_rejects_bad_point_shape() -> None:
    priors = [BlockPose(np.zeros(3), np.eye(3), dims=LAID)]
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        detect_with_plan(np.zeros((5, 2)), priors)


def test_works_without_sensor_context() -> None:
    """The prior path still functions on a raw cloud, just without ray evidence."""
    _, sample = _scene("row3_middle_last")
    results = detect_with_plan(sample.points, _priors(sample, jitter=0.03), DetectionParams())
    assert len(results) == 3
    for result, gt in zip(results, sample.ground_truth, strict=True):
        assert translation_error(result.pose, gt) < 0.03
