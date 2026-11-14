"""Tests for the deterministic detector diagnostic scenario matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from blockpose.pose import BlockPose
from blockpose.scenario_matrix import (
    ROBIN_W_PATTERN,
    generate_scenario_matrix,
    request_families,
    scenario_registry,
    world_model_prior,
)


def test_registry_has_unique_names_and_required_families() -> None:
    registry = scenario_registry()
    assert len({scenario.name for scenario in registry}) == len(registry)
    assert {scenario.family for scenario in registry} >= {
        "isolated_range",
        "tilt",
        "noise",
        "dropout",
        "multi_block",
        "touching",
        "stack",
        "clutter",
        "hard_negative",
    }


def test_matrix_is_deterministic_and_has_reproducible_names() -> None:
    first = generate_scenario_matrix(17)
    second = generate_scenario_matrix(17)
    assert [case.name for case, _ in first] == [case.name for case, _ in second]
    for (_, a), (_, b) in zip(first, second, strict=True):
        assert np.array_equal(a.points, b.points)
        assert a.manifest == b.manifest


def test_registry_contains_full_height_elongated_hard_negative() -> None:
    scenario = next(
        case for case in scenario_registry() if case.name == "elongated_full_height_negative"
    )
    assert scenario.clutter is not None
    assert scenario.clutter.near_field_dims == (0.6, 1.8, 0.9)


def test_request_families_have_stable_names_and_a_visible_target() -> None:
    families = request_families()
    assert set(families) == {
        "pickup_touching_row", "placed_beside_neighbour", "occluded_by_gripper"
    }
    names = [case.scenario.name for cases in families.values() for case in cases]
    assert len(names) == len(set(names))
    if not Path(ROBIN_W_PATTERN).exists():
        pytest.skip("seyond_simulation scan pattern not in this checkout")
    case = families["pickup_touching_row"][0]
    sample = case.scenario.generate(0)
    assert (sample.instance_ids == case.target).sum() > 50


def test_world_model_prior_is_centimetres_off_unless_stale() -> None:
    truth = BlockPose(position=np.zeros(3), rotation=np.eye(3))
    rng = np.random.default_rng(3)
    draws = [world_model_prior(truth, rng) for _ in range(400)]
    fresh = [np.linalg.norm(p[:2]) for p, _, stale in draws if not stale]
    assert np.percentile(fresh, 95) < 0.1
    assert 0.05 < np.mean([stale for _, _, stale in draws]) < 0.15
    assert all(abs(np.linalg.norm(p[:2]) - 0.15) < 1e-9 for p, _, stale in draws if stale)
