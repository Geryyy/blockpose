"""Tests for the deterministic in-repo RANSAC plane fitter."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.ransac import segment_plane


def _plane_cloud(normal, offset, count=600, noise=0.002, seed=0):
    """Points on `normal . x + offset = 0`, plus Gaussian thickness."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    basis = np.linalg.svd(normal.reshape(1, 3), full_matrices=True)[2][1:]
    coords = rng.uniform(-2.0, 2.0, size=(count, 2))
    points = coords @ basis - offset * normal
    return points + rng.normal(0.0, noise, size=points.shape) * normal


def test_recovers_a_known_plane() -> None:
    truth = np.array([0.0, 0.0, 1.0])
    points = _plane_cloud(truth, 0.0)
    model, inliers = segment_plane(points, distance_threshold=0.01)
    assert abs(abs(float(model[:3] @ truth)) - 1.0) < 1e-3
    assert abs(float(model[3])) < 0.01
    assert len(inliers) > 0.95 * len(points)


def test_recovers_an_offset_tilted_plane() -> None:
    truth = np.array([0.3, -0.2, 0.9])
    truth = truth / np.linalg.norm(truth)
    points = _plane_cloud(truth, -1.5)
    model, _ = segment_plane(points, distance_threshold=0.01)
    assert abs(abs(float(model[:3] @ truth)) - 1.0) < 1e-3
    # Sign of (normal, offset) may flip together; compare the plane, not the pair.
    residual = np.abs(points @ model[:3] + model[3])
    assert float(np.median(residual)) < 0.005


def test_is_deterministic_for_a_fixed_seed() -> None:
    points = _plane_cloud([0.1, 0.2, 0.97], 0.4, seed=3)
    runs = [segment_plane(points, distance_threshold=0.01, seed=7) for _ in range(6)]
    for model, inliers in runs[1:]:
        np.testing.assert_array_equal(model, runs[0][0])
        np.testing.assert_array_equal(inliers, runs[0][1])


def test_different_seeds_still_agree_on_a_dominant_plane() -> None:
    points = _plane_cloud([0.0, 0.0, 1.0], 0.0, seed=5)
    normals = [segment_plane(points, 0.01, seed=seed)[0][:3] for seed in range(5)]
    for normal in normals[1:]:
        assert abs(abs(float(normal @ normals[0])) - 1.0) < 1e-3


def test_selects_the_dominant_plane_when_two_are_present() -> None:
    dominant = _plane_cloud([0.0, 0.0, 1.0], 0.0, count=900, seed=1)
    minor = _plane_cloud([1.0, 0.0, 0.0], 0.0, count=200, seed=2)
    points = np.vstack([dominant, minor])
    model, inliers = segment_plane(points, distance_threshold=0.01, seed=0)
    assert abs(abs(float(model[2])) - 1.0) < 1e-2
    assert len(inliers) > 800


def test_ignores_outliers() -> None:
    points = _plane_cloud([0.0, 0.0, 1.0], 0.0, count=800, seed=4)
    outliers = np.random.default_rng(9).uniform(-2.0, 2.0, size=(400, 3))
    model, inliers = segment_plane(np.vstack([points, outliers]), 0.01, seed=0)
    assert abs(abs(float(model[2])) - 1.0) < 1e-2
    assert 700 < len(inliers) < 1000


def test_returned_normal_is_unit_length() -> None:
    model, _ = segment_plane(_plane_cloud([1.0, 1.0, 1.0], 0.2), 0.01)
    assert float(np.linalg.norm(model[:3])) == pytest.approx(1.0)


def test_inlier_indices_index_the_input() -> None:
    points = _plane_cloud([0.0, 0.0, 1.0], 0.0, count=300, seed=6)
    model, inliers = segment_plane(points, distance_threshold=0.01)
    assert inliers.dtype.kind == "i"
    assert inliers.max() < len(points)
    residual = np.abs(points[inliers] @ model[:3] + model[3])
    assert float(residual.max()) <= 0.01 + 1e-9


def test_adaptive_stop_does_not_degrade_a_clean_fit() -> None:
    """A high inlier ratio should terminate early yet still fit correctly."""
    points = _plane_cloud([0.0, 0.0, 1.0], 0.0, count=1000, noise=0.001, seed=8)
    few = segment_plane(points, 0.01, num_iterations=20, seed=0)[0]
    many = segment_plane(points, 0.01, num_iterations=2000, seed=0)[0]
    assert abs(abs(float(few[:3] @ many[:3])) - 1.0) < 1e-4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distance_threshold": 0.0}, "distance_threshold must be positive"),
        ({"distance_threshold": -1.0}, "distance_threshold must be positive"),
        ({"distance_threshold": np.nan}, "distance_threshold must be positive"),
        ({"distance_threshold": 0.01, "num_iterations": 0}, "num_iterations must be positive"),
    ],
)
def test_rejects_invalid_parameters(kwargs, message) -> None:
    points = _plane_cloud([0.0, 0.0, 1.0], 0.0, count=50)
    with pytest.raises(ValueError, match=message):
        segment_plane(points, **kwargs)


def test_rejects_bad_shapes_and_too_few_points() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        segment_plane(np.zeros((10, 2)), 0.01)
    with pytest.raises(ValueError, match="at least 3 points"):
        segment_plane(np.zeros((2, 3)), 0.01)


def test_collinear_points_raise_rather_than_return_a_bogus_plane() -> None:
    line = np.column_stack([np.linspace(0, 1, 40), np.zeros(40), np.zeros(40)])
    with pytest.raises(ValueError, match="no plane"):
        segment_plane(line, distance_threshold=1e-6)
