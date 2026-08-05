"""Fast, dataset-free tests for the pure helpers in `tools/sweep_detector_params.py`.

Mirrors `tests/python/test_rgb_ablation_splits.py`'s pattern of importing a
`tools/` script directly (tools has no `__init__.py`, so it is not an
importable package -- REPO_ROOT/"tools" is added to sys.path instead of
duplicating the logic here).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from sweep_detector_params import (  # noqa: E402
    nearest_component_label,
    score_matches,
    voxel_downsample_mean,
    xy_extent,
)


def test_voxel_downsample_mean_averages_within_cell_and_is_order_independent() -> None:
    # Two points share a 0.1 m voxel (both floor(./0.1) == (0, 0, 0)), one sits alone in
    # another -- mirrors cpp/include/blockpose/core/pipeline.hpp's own unit test fixture.
    points = np.array([[0.01, 0.01, 0.01], [0.03, 0.03, 0.03], [0.12, 0.01, 0.01]])
    down = voxel_downsample_mean(points, 0.1)
    assert down.shape == (2, 3)
    reversed_down = voxel_downsample_mean(points[::-1], 0.1)
    # Order-independent: same set of voxel means regardless of input order.
    assert sorted(down[:, 0].tolist()) == sorted(reversed_down[:, 0].tolist())
    merged = down[np.isclose(down[:, 0], 0.02, atol=1e-8)]
    assert merged.shape == (1, 3)
    np.testing.assert_allclose(merged[0], [0.02, 0.02, 0.02], atol=1e-8)


def test_voxel_downsample_mean_rejects_nonpositive_voxel_size() -> None:
    with pytest.raises(ValueError):
        voxel_downsample_mean(np.zeros((1, 3)), 0.0)


def test_voxel_downsample_mean_empty_input() -> None:
    out = voxel_downsample_mean(np.zeros((0, 3)), 0.05)
    assert out.shape == (0, 3)


def test_xy_extent_is_per_axis_bounding_box() -> None:
    # A 2 m x 0.5 m x 3 m box: extent must report (2.0, 0.5), not a diagonal --
    # the core's cluster_max_extent_xy gate checks each axis independently.
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 3.0], [1.0, 0.25, 1.5]])
    ex, ey = xy_extent(points)
    assert ex == pytest.approx(2.0)
    assert ey == pytest.approx(0.5)


def test_xy_extent_empty_is_zero() -> None:
    assert xy_extent(np.zeros((0, 3))) == (0.0, 0.0)


def test_nearest_component_label_picks_closest_points_label() -> None:
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    labels = np.array([5, 9])
    label, distance = nearest_component_label(points, labels, np.array([0.1, 0.0, 0.0]))
    assert label == 5
    assert distance == pytest.approx(0.1)


def test_nearest_component_label_empty_points_is_noise() -> None:
    label, distance = nearest_component_label(
        np.zeros((0, 3)), np.zeros((0,), dtype=np.int64), np.zeros(3)
    )
    assert label == -1
    assert distance == float("inf")


def test_score_matches_many_to_many_recall_and_precision() -> None:
    # Two GT blocks: one has a nearby accepted candidate (true positive), the
    # other has none (missed). One extra accepted candidate matches nothing
    # (false positive) -- exercises both the recall and precision arms of the
    # baseline snippet's many-to-many match rule in one synthetic case.
    gt = [np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0])]
    candidates = [np.array([0.05, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])]
    gt_hit, cand_hit = score_matches(gt, candidates, radius=0.30)
    assert gt_hit == [True, False]
    assert cand_hit == [True, False]


def test_score_matches_respects_radius_boundary() -> None:
    gt = [np.array([0.0, 0.0, 0.0])]
    just_inside = [np.array([0.30, 0.0, 0.0])]
    just_outside = [np.array([0.31, 0.0, 0.0])]
    inside_hit, _ = score_matches(gt, just_inside, radius=0.30)
    outside_hit, _ = score_matches(gt, just_outside, radius=0.30)
    assert inside_hit == [True]
    assert outside_hit == [False]


def test_score_matches_empty_inputs() -> None:
    assert score_matches([], []) == ([], [])
    gt_hit, cand_hit = score_matches([np.zeros(3)], [])
    assert gt_hit == [False]
    assert cand_hit == []


def test_cluster_gate_ships_the_measured_value_in_both_cores() -> None:
    """Guard the gate: at 1.5 it silently costs 11 points of recall.

    `pipeline.hpp` drops any unsplit component wider than this BEFORE fitting a
    plane to it, so blocks touching in a wall were discarded wholesale. 2.25 is
    the knee of a plateau running to 3.5, not a tuned peak -- see
    experiments/detector_cluster_failures.md. The Python dataclass and the C++
    header must not drift apart, or a Python-side sweep would measure something
    the native detector never runs.
    """
    from pathlib import Path

    from blockpose.core_types import DetectionParameters

    assert DetectionParameters().cluster_max_extent_xy == pytest.approx(2.25)
    header = Path("cpp/include/blockpose/core/proposals.hpp").read_text(encoding="utf-8")
    assert "cluster_max_extent_xy{2.25}" in header, "C++ default drifted from the Python one"
