import time

import blockpose.engine as engine_module
import numpy as np
from blockpose.detection import (
    DetectionDiagnostics,
    DetectionResult,
    HypothesisEvidence,
    Observability,
)
from blockpose.engine import HybridBlockDetector, QueryErrorCode, QueryStatus
from blockpose.frames import FrameBurst, PointFrame, PoseTrajectory, TimedPose
from blockpose.plan_prior import PlanPriorResult
from blockpose.pose import BlockPose


def test_detect_free_returns_structured_result(monkeypatch) -> None:
    pose = BlockPose(np.array([0, 0, 0.3]), np.eye(3), source="test")
    evidence = HypothesisEvidence(100, 0.0, 0.8)
    diagnostics = DetectionDiagnostics(evidence, Observability.FULL, {"support": 1.0})
    monkeypatch.setattr(
        engine_module,
        "detect_blocks_detailed",
        lambda *_args, **_kwargs: DetectionResult([pose], [diagnostics]),
    )
    result = HybridBlockDetector().detect_free(PointFrame(np.zeros((4, 3)), frame_id="world"))
    assert result.blocks[0].status == QueryStatus.DETECTED
    assert result.blocks[0].covariance is None
    assert "intensity" in result.unavailable_evidence


def test_detect_free_reports_empty_roi() -> None:
    frame = PointFrame(np.ones((2, 3)), frame_id="world")
    result = HybridBlockDetector().detect_free(frame, (np.zeros(3), np.full(3, 0.5)))
    assert result.failures[0].code == QueryErrorCode.EMPTY_ROI


def test_verify_dense_low_score_support_is_not_observable(monkeypatch) -> None:
    prior = BlockPose(np.array([0, 0, 0.3]), np.eye(3))
    low_score = PlanPriorResult(prior, prior, 0.1, {"support": 1.0}, 1000, True)
    monkeypatch.setattr(engine_module, "detect_with_plan", lambda *_args, **_kwargs: [low_score])
    points = np.random.default_rng(0).normal(size=(1000, 3))
    result = HybridBlockDetector().verify_placed(points, [prior])
    assert result.blocks[0].status == QueryStatus.NOT_OBSERVABLE


def test_verify_without_ray_evidence_never_calls_zero_support_missing(monkeypatch) -> None:
    prior = BlockPose(np.array([0, 0, 0.3]), np.eye(3))
    unsupported = PlanPriorResult(prior, prior, 0.0, {}, 0, False)
    monkeypatch.setattr(engine_module, "detect_with_plan", lambda *_args, **_kwargs: [unsupported])
    result = HybridBlockDetector().verify_placed(np.zeros((1, 3)), [prior])
    assert result.blocks[0].status == QueryStatus.NOT_OBSERVABLE


def test_soft_timeout_discards_blocks(monkeypatch) -> None:
    pose = BlockPose(np.array([0, 0, 0.3]), np.eye(3))
    evidence = HypothesisEvidence(100, 0.0, 0.8)
    diagnostics = DetectionDiagnostics(evidence, Observability.FULL, {})

    def slow_detection(*_args, **_kwargs):
        time.sleep(0.002)
        return DetectionResult([pose], [diagnostics])

    monkeypatch.setattr(engine_module, "detect_blocks_detailed", slow_detection)
    result = HybridBlockDetector(timeout_ms=0.1).detect_free(np.zeros((3, 3)))
    assert result.blocks == ()
    assert result.failures[0].code == QueryErrorCode.TIMEOUT


def test_raw_nonfinite_input_and_roi_are_rejected() -> None:
    detector = HybridBlockDetector()
    invalid_points = detector.detect_free(np.array([[np.nan, 0.0, 0.0]]))
    assert invalid_points.failures[0].code == QueryErrorCode.INVALID_INPUT
    invalid_roi = detector.detect_free(np.zeros((1, 3)), (np.zeros(3), np.array([1, 1, np.inf])))
    assert invalid_roi.failures[0].code == QueryErrorCode.INVALID_INPUT


def test_point_frame_trajectory_builds_sensor_context(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        engine_module,
        "detect_blocks_detailed",
        lambda *_args, **kwargs: captured.setdefault("context", kwargs["sensor_context"])
        and DetectionResult([], []),
    )
    trajectory = PoseTrajectory((TimedPose(0.0, np.eye(4)),))
    frame = PointFrame(
        np.array([[1.0, 0.0, 0.0]]),
        frame_id="sensor",
        sensor_trajectory=trajectory,
        no_return_rays=np.array([[0.0, 1.0, 0.0]]),
    )
    HybridBlockDetector().detect_free(frame)
    assert captured["context"] is not None
    np.testing.assert_allclose(captured["context"].miss_directions, [[0.0, 1.0, 0.0]])


def test_burst_uses_per_return_and_no_return_origins_in_sensor_context(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        engine_module,
        "detect_blocks_detailed",
        lambda *_args, **kwargs: captured.setdefault("context", kwargs["sensor_context"])
        and DetectionResult([], []),
    )

    def translated(time_s: float, x: float) -> PoseTrajectory:
        transform = np.eye(4)
        transform[0, 3] = x
        return PoseTrajectory((TimedPose(time_s, transform),))

    burst = FrameBurst(
        (
            PointFrame(
                np.array([[1.0, 0.0, 0.0]]),
                timestamp=0.0,
                sensor_trajectory=translated(0.0, 0.0),
                no_return_rays=np.array([[0.0, 1.0, 0.0]]),
            ),
            PointFrame(
                np.array([[1.0, 0.0, 0.0]]),
                timestamp=1.0,
                sensor_trajectory=translated(1.0, 2.0),
                no_return_rays=np.array([[0.0, -1.0, 0.0]]),
            ),
        )
    )
    HybridBlockDetector().detect_free(burst)
    np.testing.assert_allclose(captured["context"].ray_origins, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(captured["context"].miss_origins, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_zero_return_frame_keeps_valid_miss_ray_context(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        engine_module,
        "verify_placed_blocks",
        lambda _points, _expected, **kwargs: captured.setdefault(
            "context", kwargs["sensor_context"]
        )
        and [],
    )
    frame = PointFrame(
        np.empty((0, 3)),
        sensor_trajectory=PoseTrajectory((TimedPose(0.0, np.eye(4)),)),
        no_return_rays=np.array([[1.0, 0.0, 0.0]]),
    )
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    HybridBlockDetector(verify_placed_v1=True).verify_placed(frame, [prior])
    assert captured["context"] is not None
    assert captured["context"].ray_origins.shape == (0, 3)
    np.testing.assert_allclose(captured["context"].miss_origins, [[0.0, 0.0, 0.0]])
