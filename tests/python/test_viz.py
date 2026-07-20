"""Tests for the interactive detection-vs-ground-truth visualization."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.detection import DetectionParams, detect_scene_sample
from blockpose.pose import BlockPose
from blockpose.synth import BlockSpec, SynthParams, generate_scene
from blockpose.viz import box_corners, error_table, sample_figure, scene_figure

DIMS = (0.6, 0.6, 0.9)


def _pose(position, yaw_deg=0.0) -> BlockPose:
    angle = np.radians(yaw_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    return BlockPose(
        np.asarray(position, dtype=np.float64),
        np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
    )


def test_box_corners_match_dimensions_and_centre() -> None:
    pose = _pose((1.0, 2.0, 0.45))
    corners = box_corners(pose, DIMS)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.mean(axis=0), pose.position)
    np.testing.assert_allclose(np.ptp(corners, axis=0), DIMS)


def test_box_corners_rotate_with_pose() -> None:
    upright = box_corners(_pose((0.0, 0.0, 0.45)), DIMS)
    turned = box_corners(_pose((0.0, 0.0, 0.45), yaw_deg=45.0), DIMS)
    # A 45 deg yaw widens the axis-aligned footprint by sqrt(2).
    assert np.ptp(turned, axis=0)[0] == pytest.approx(0.6 * np.sqrt(2))
    assert np.ptp(upright, axis=0)[0] == pytest.approx(0.6)


def test_scene_figure_builds_expected_traces() -> None:
    points = np.random.default_rng(0).normal(size=(500, 3))
    figure = scene_figure(
        points,
        [_pose((0.0, 0.0, 0.45))],
        [_pose((0.02, 0.0, 0.45))],
        sensor_origin=np.array([-5.0, 0.0, 4.0]),
        title="test",
    )
    names = [trace.name for trace in figure.data]
    assert any("cloud" in name for name in names)
    assert any("ground truth" in name for name in names)
    assert any("detections" in name for name in names)
    assert "sensor" in names
    assert figure.layout.scene.aspectmode == "cube"


def test_scene_figure_subsamples_large_clouds() -> None:
    points = np.random.default_rng(1).normal(size=(50000, 3))
    figure = scene_figure(points, max_points=5000)
    assert len(figure.data[0].x) <= 5000
    assert "50,000" in figure.data[0].name


def test_scene_figure_without_optional_layers() -> None:
    figure = scene_figure(np.zeros((10, 3)))
    assert len(figure.data) == 1


def test_scene_figure_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        scene_figure(np.zeros((10, 2)))


def test_equal_aspect_ranges_are_cubic() -> None:
    # A cloud far wider in x than z must still produce equal-length axis ranges.
    points = np.array([[0.0, 0.0, 0.0], [10.0, 1.0, 0.5]])
    scene = scene_figure(points).layout.scene
    spans = [
        axis.range[1] - axis.range[0] for axis in (scene.xaxis, scene.yaxis, scene.zaxis)
    ]
    assert spans[0] == pytest.approx(spans[1]) == pytest.approx(spans[2])


def test_sample_figure_from_scene_sample() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=20.0)],
        SynthParams(range_noise_std=0.01),
        np.random.default_rng(2),
    )
    detections = detect_scene_sample(sample, DetectionParams())
    figure = sample_figure(sample, detections, title="scene")
    assert "sensor" in [trace.name for trace in figure.data]


def test_error_table_reports_matches_and_misses() -> None:
    ground_truth = [_pose((0.0, 0.0, 0.45)), _pose((3.0, 0.0, 0.45))]
    detections = [_pose((0.005, 0.0, 0.45))]
    rows = error_table(detections, ground_truth)
    assert len(rows) == 2
    matched = [row for row in rows if row["detection"] is not None]
    missed = [row for row in rows if row["detection"] is None]
    assert len(matched) == 1 and len(missed) == 1
    assert matched[0]["translation_mm"] == pytest.approx(5.0)
    assert missed[0]["gt"] == 1


def test_error_table_is_empty_without_ground_truth() -> None:
    assert error_table([_pose((0.0, 0.0, 0.45))], []) == []
