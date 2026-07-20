"""Tests for loading the robot's real wall plans as detector geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from blockpose.detection import DetectionParams, detect_scene_sample
from blockpose.evaluate import evaluate
from blockpose.synth import SynthParams, generate_scene
from blockpose.wall_plans import DEFAULT_PLANS, load_wall_plans, neighbour_gaps

PLANS = load_wall_plans()


def test_every_plan_in_the_file_is_loaded() -> None:
    raw = yaml.safe_load(Path(DEFAULT_PLANS).read_text())
    assert set(PLANS) == set(raw["wall_plans"])


def test_blocks_are_laid_down_not_stood_up() -> None:
    """The plans use (0.9, 0.6, 0.6): the long axis is horizontal."""
    for plan in PLANS.values():
        assert plan.block_size == (0.9, 0.6, 0.6)
        assert plan.block_size[2] < plan.block_size[0]


def test_course_heights_match_a_0_6_m_vertical_extent() -> None:
    """A laid block resting on the ground has its centre at 0.3, not 0.45."""
    wall = PLANS["example_wall"]
    heights = sorted({spec.z for spec in wall.specs})
    assert heights == [0.3, 0.9]  # first course, then a 0.6 m rise


def test_basic_pick_and_place_is_the_upright_exception() -> None:
    """z = 0.45 only makes sense for a block standing on its 0.6 x 0.6 end.

    Documented rather than normalised: the detector has to cope with both
    orientations, and they have different symmetry groups.
    """
    (spec,) = PLANS["basic_pick_and_place"].specs
    assert spec.z == pytest.approx(0.45)


def test_relative_placement_resolves_against_the_referenced_block() -> None:
    stack = PLANS["stack_block_1_on_block_2"]
    lower, upper = stack.specs
    assert upper.z - lower.z == pytest.approx(0.6)
    assert (upper.x, upper.y) == (lower.x, lower.y)


def test_relative_step_falls_back_when_reference_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "plans.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": {"block_size": [0.9, 0.6, 0.6]},
                "wall_plans": {
                    "orphan": {
                        "sequence": [
                            {
                                "id": "a",
                                "relative_to_world_model": "does_not_exist",
                                "offset": [0.0, 0.0, 0.6],
                                "fallback_absolute_position": [1.0, 2.0, 3.0],
                                "yaw_deg": 0.0,
                            }
                        ]
                    }
                },
            }
        )
    )
    (spec,) = load_wall_plans(path)["orphan"].specs
    assert (spec.x, spec.y, spec.z) == (1.0, 2.0, 3.0)


def test_unresolvable_step_raises(tmp_path: Path) -> None:
    path = tmp_path / "plans.yaml"
    path.write_text(
        yaml.safe_dump(
            {"wall_plans": {"broken": {"sequence": [{"id": "a", "yaw_deg": 0.0}]}}}
        )
    )
    with pytest.raises(ValueError, match="no resolvable position"):
        load_wall_plans(path)


def test_course_gaps_are_two_centimetres() -> None:
    """The number that decides whether a wall is separable from one scan."""
    for name in ("example_wall", "row3_middle_last"):
        gaps = neighbour_gaps(PLANS[name])
        assert gaps, f"{name} has no same-course neighbours"
        assert all(gap == pytest.approx(0.02, abs=1e-6) for gap in gaps)


def test_single_isolated_plan_block_is_detected() -> None:
    plan = PLANS["basic_pick_and_place"]
    params = SynthParams(az_fov_deg=70.0, el_fov_deg=50.0, **plan.sensor_params_kwargs())
    sample = generate_scene(list(plan.specs), params, np.random.default_rng(0))
    detections = detect_scene_sample(sample, DetectionParams(block_dims=plan.block_size))
    result = evaluate(detections, sample.ground_truth, dims=plan.block_size)
    assert result.matched == 1
    assert result.unmatched_det == 0
    assert max(result.translation_errors_m) < 0.02


def test_wall_plans_are_currently_not_detected() -> None:
    """Regression pin for the largest open gap, so a fix cannot pass unnoticed.

    Blocks laid with the 2 cm gaps these plans specify are not recovered. This
    asserts the *current* behaviour deliberately: when the gap closes this test
    must be updated, which is the point.
    """
    for name in ("example_wall", "row3_middle_last"):
        plan = PLANS[name]
        params = SynthParams(az_fov_deg=70.0, el_fov_deg=50.0, **plan.sensor_params_kwargs())
        sample = generate_scene(list(plan.specs), params, np.random.default_rng(0))
        detections = detect_scene_sample(sample, DetectionParams(block_dims=plan.block_size))
        result = evaluate(detections, sample.ground_truth, dims=plan.block_size)
        # No false positives is the property that must hold regardless.
        assert result.unmatched_det == 0
        assert result.matched == 0, f"{name} now detects blocks -- update this pin"
