"""Regression tests: `HybridBlockDetector.detect_free` must build its
`SensorContext` from the full, pre-ROI point cloud, never from the
ROI-cropped subset handed to clustering.

Background (see docs/production-readiness-plan.md and
experiments/dino_proposer_pooled_ablation.md): 60% of the hypothesis score
(evidence.hpp features f[2]/f[3]/f[4]) is ray-based evidence -- covered vs.
expected faces, supported vs. free-space-violating rays -- computed against
whatever SensorContext is supplied. A ray from a point OUTSIDE the ROI can
still pass through a hypothesis box INSIDE the ROI and must be charged as a
free-space violation. Building the context from the ROI-cropped points
instead of the full cloud deletes exactly those rays and inflates the score
(measured 0.7714 vs. 0.7205 for an identical crop/fit/pose), which matters
because scores are compared against a fixed `min_score` acceptance
threshold.

These tests monkeypatch `blockpose.engine.detect_blocks_detailed_cpp` to
capture the arguments `detect_free` hands to the C++ detector -- an honest
substitute for driving the full clustering/fitting pipeline, since what is
under test is purely which points end up in `points` vs. in
`sensor_context`.
"""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.core_types import DetectionResult, SensorContext
from blockpose.engine import HybridBlockDetector, QueryErrorCode
from blockpose.frames import PointFrame, PoseTrajectory, TimedPose

_IDENTITY = np.eye(4)


def _static_frame(points: np.ndarray) -> PointFrame:
    """A `PointFrame` with a static sensor trajectory at the world origin.

    A static trajectory is required to get `point_origins_in_world()` to
    return non-`None` per-point origins -- without it `detect_free` cannot
    build a `SensorContext` at all (origin is `None`), which would make this
    test vacuous rather than a regression check.
    """
    trajectory = PoseTrajectory((TimedPose(0.0, _IDENTITY),))
    return PointFrame(points, timestamp=0.0, frame_id="world", sensor_trajectory=trajectory)


class _Capture:
    """Records the arguments the engine hands to the (stubbed) C++ detector."""

    def __init__(self) -> None:
        self.points: np.ndarray | None = None
        self.sensor_context: SensorContext | None = None
        self.calls = 0

    def __call__(self, points, *, params=None, sensor_context=None, source_label="cpp_core"):
        self.calls += 1
        self.points = np.asarray(points)
        self.sensor_context = sensor_context
        return DetectionResult(poses=[], diagnostics=[])


def test_sensor_context_keeps_the_full_cloud_while_clustering_gets_the_roi_crop(
    monkeypatch,
) -> None:
    # Sensor at the world origin. `in_roi` sits inside the workspace box.
    # `outside_roi` sits further out along the SAME ray direction, so the ray
    # to it passes straight through (and beyond) `in_roi` -- exactly the
    # "ray from an out-of-ROI point crosses an in-ROI hypothesis" scenario
    # the bug drops evidence for.
    in_roi = np.array([0.5, 0.0, 0.0])
    outside_roi = np.array([2.0, 0.0, 0.0])
    frame = _static_frame(np.stack([in_roi, outside_roi]))

    capture = _Capture()
    monkeypatch.setattr("blockpose.engine.detect_blocks_detailed_cpp", capture)

    detector = HybridBlockDetector()
    low = np.array([0.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    result = detector.detect_free(frame, workspace_roi=(low, high))

    assert not result.failures
    assert capture.calls == 1

    # Clustering/fitting only sees the ROI-cropped point.
    assert capture.points.shape == (1, 3)
    np.testing.assert_allclose(capture.points[0], in_roi)

    # The SensorContext must still carry BOTH rays: dropping the ray to
    # `outside_roi` is exactly the bug this test guards against.
    context = capture.sensor_context
    assert context is not None
    assert context.ranges.shape == (2,)
    assert context.ray_origins.shape == (2, 3)
    ranges_sorted = np.sort(context.ranges)
    np.testing.assert_allclose(
        ranges_sorted, np.sort([np.linalg.norm(in_roi), np.linalg.norm(outside_roi)])
    )
    # Both rays share the same direction (collinear along +x): the
    # out-of-ROI ray is the one that would be charged as a free-space
    # violation against any hypothesis placed between 0.5 m and 2.0 m out.
    for direction in context.ray_directions:
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-9)


def test_context_built_from_roi_crop_would_have_dropped_the_through_box_ray() -> None:
    """Directly contrasts the fixed behaviour with the old buggy one.

    Builds a `SensorContext` the way the (fixed) engine now does -- from the
    full cloud -- and the way the buggy engine used to -- from the
    ROI-cropped cloud -- and shows the buggy version is missing the ray to
    the out-of-ROI point entirely.
    """
    in_roi = np.array([0.5, 0.0, 0.0])
    outside_roi = np.array([2.0, 0.0, 0.0])
    points = np.stack([in_roi, outside_roi])
    origin = np.zeros(3)

    full_context = HybridBlockDetector._sensor_context(points, origin, None)
    assert full_context.ranges.shape == (2,)

    roi_mask = np.all(
        (points >= np.array([0.0, -1.0, -1.0])) & (points <= np.array([1.0, 1.0, 1.0])), axis=1
    )
    cropped_context = HybridBlockDetector._sensor_context(points[roi_mask], origin, None)
    assert cropped_context.ranges.shape == (1,)
    assert cropped_context.ranges.shape != full_context.ranges.shape


def test_empty_roi_still_fails_after_the_fix(monkeypatch) -> None:
    """The EMPTY_ROI failure path (checked against the ROI crop, not the
    full cloud) must be unaffected by building the context from the full
    cloud."""
    points = np.array([[5.0, 5.0, 5.0]])  # nowhere near the ROI below.
    frame = _static_frame(points)

    capture = _Capture()
    monkeypatch.setattr("blockpose.engine.detect_blocks_detailed_cpp", capture)

    detector = HybridBlockDetector()
    low = np.array([0.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])
    result = detector.detect_free(frame, workspace_roi=(low, high))

    assert capture.calls == 0
    assert len(result.failures) == 1
    assert result.failures[0].code == QueryErrorCode.EMPTY_ROI


def test_no_roi_still_uses_the_full_cloud_as_both_context_and_cluster_input(monkeypatch) -> None:
    points = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    frame = _static_frame(points)

    capture = _Capture()
    monkeypatch.setattr("blockpose.engine.detect_blocks_detailed_cpp", capture)

    detector = HybridBlockDetector()
    result = detector.detect_free(frame)

    assert not result.failures
    assert capture.points.shape == (2, 3)
    assert capture.sensor_context.ranges.shape == (2,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
