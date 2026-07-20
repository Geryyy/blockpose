"""Tests for the deterministic detector diagnostic scenario matrix."""

from __future__ import annotations

import numpy as np
from blockpose.scenario_matrix import generate_scenario_matrix, scenario_registry


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
