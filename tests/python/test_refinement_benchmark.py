"""Fast contract checks for the full Stage-2 refinement characterization tool."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_refinement import SCENARIOS, run_benchmark, scenario_points, summarize


def test_benchmark_scenarios_are_repeatable_and_cover_visibility_failures() -> None:
    assert {scenario.name for scenario in SCENARIOS} == {
        "isolated_full",
        "isolated_missing_endcaps",
        "wall_row_occluded",
    }
    first, _ = scenario_points(SCENARIOS[1], 0.02, 7)
    second, _ = scenario_points(SCENARIOS[1], 0.02, 7)
    assert first.shape == second.shape
    assert (first == second).all()


def test_benchmark_exercises_the_native_refiner_for_each_trial() -> None:
    pytest.importorskip("blockpose._core")
    records = run_benchmark(seeds=1, noise_levels=(0.0, 0.02))
    assert len(records) == len(SCENARIOS) * 2
    assert {record["solver"] for record in records} == {"cpp_lm"}
    summary = summarize(records)
    assert len(summary) == len(SCENARIOS) * 2
    assert all(entry["trials"] == 1 for entry in summary)
