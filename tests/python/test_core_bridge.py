"""Tests for `blockpose.core_bridge.detect_blocks_detailed_cpp`.

The primary tests are hermetic: they monkeypatch `sys.modules["blockpose._core"]`
with a fake module whose `discover()` returns a literal fixture dict matching
the contract documented on `detect_blocks_detailed_cpp`, so they pass whether
or not the compiled `blockpose._core` nanobind extension happens to be built
in this environment. One additional test does a live round-trip against the
real extension and is skipped gracefully if it isn't importable.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest
from blockpose.core_bridge import detect_blocks_detailed_cpp
from blockpose.detection import DetectionResult, InferenceTrace, TraceHypothesis
from blockpose.pose import BlockPose

_FIXTURE_POSE: dict[str, Any] = {
    "position": (1.0, 2.0, 0.3),
    "rotation": [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    "dims": [0.9, 0.6, 0.6],
    "confidence": 0.75,
}

_FIXTURE_TRACE_POSE: dict[str, Any] = {
    "position": (1.05, 2.05, 0.31),
    "rotation": [
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    "dims": [0.9, 0.6, 0.6],
    "confidence": 0.9,
}

_FIXTURE_RESULT: dict[str, Any] = {
    "poses": [_FIXTURE_POSE],
    "count": 1,
    "counts": {
        "input_points": 100,
        "downsampled_points": 80,
        "above_support_points": 50,
        "proposal_components": 1,
        "raw_hypotheses": 2,
        "selected_hypotheses": 1,
        "plane_regions": 1,
        "plane_fit_calls": 2,
        "plane_search_points": 40,
        "plane_full_points_scored": 40,
        "plane_trials_evaluated": 20,
        "plane_valid_trials": 18,
        "refinement_candidates": 1,
    },
    "refined_candidate_trace": [
        {
            "id": "component-0/region-0/dims-0/top-0/side-0/refined",
            "source": "plane_fit",
            "lineage_id": "component-0/region-0/dims-0/top-0/side-0",
            "pose": _FIXTURE_TRACE_POSE,
            "score": 0.42,
            "support_points": 55,
            "selection_score": 0.42,
            "stage": "post_refinement_pre_nms",
            "fate": "final",
        },
        {
            "id": "component-0/region-0/dims-1/top-0/side-0/refined",
            "source": "plane_fit",
            "lineage_id": "component-0/region-0/dims-1/top-0/side-0",
            "pose": _FIXTURE_TRACE_POSE,
            "score": 0.10,
            "support_points": 12,
            "selection_score": 0.10,
            "stage": "post_refinement_pre_nms",
            "fate": "post_refinement_nms_rejected",
        },
    ],
}


class _FakeCore:
    """Stand-in for `blockpose._core` with a scripted `discover()`."""

    def __init__(self, result: dict[str, Any]):
        self._result = result
        self.last_call_points: np.ndarray | None = None

    def discover(self, points: np.ndarray) -> dict[str, Any]:
        self.last_call_points = points
        return self._result


@pytest.fixture
def fake_core(monkeypatch: pytest.MonkeyPatch) -> _FakeCore:
    fake = _FakeCore(_FIXTURE_RESULT)
    module = types.ModuleType("blockpose._core")
    module.discover = fake.discover  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "blockpose._core", module)
    return fake


def _points() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(100, 3))


def test_detect_blocks_detailed_cpp_reshapes_poses(fake_core: _FakeCore) -> None:
    result = detect_blocks_detailed_cpp(_points(), source_label="cpp_core")

    assert isinstance(result, DetectionResult)
    assert len(result.poses) == 1
    pose = result.poses[0]
    assert isinstance(pose, BlockPose)
    assert isinstance(pose.position, np.ndarray)
    np.testing.assert_allclose(pose.position, [1.0, 2.0, 0.3])
    assert isinstance(pose.rotation, np.ndarray)
    assert pose.rotation.shape == (3, 3)
    np.testing.assert_allclose(pose.rotation, np.eye(3))
    assert pose.dims == (0.9, 0.6, 0.6)
    assert isinstance(pose.dims, tuple)
    assert pose.confidence == pytest.approx(0.75)
    assert pose.source == "cpp_core"


def test_detect_blocks_detailed_cpp_default_source_label(fake_core: _FakeCore) -> None:
    result = detect_blocks_detailed_cpp(_points())
    assert result.poses[0].source == "cpp_core"


def test_detect_blocks_detailed_cpp_stage_counts(fake_core: _FakeCore) -> None:
    result = detect_blocks_detailed_cpp(_points())
    assert result.stage_counts == _FIXTURE_RESULT["counts"]
    # Deliberately empty gaps, not fabricated data.
    assert result.diagnostics == []
    assert result.stage_timings_ms == {}


def test_detect_blocks_detailed_cpp_trace_hypotheses(fake_core: _FakeCore) -> None:
    result = detect_blocks_detailed_cpp(_points(), source_label="cpp_core")

    assert isinstance(result.trace, InferenceTrace)
    assert len(result.trace.raw_hypotheses) == 2
    first, second = result.trace.raw_hypotheses
    assert isinstance(first, TraceHypothesis)
    assert first.id == "component-0/region-0/dims-0/top-0/side-0/refined"
    assert first.score == pytest.approx(0.42)
    assert first.stage == "post_refinement_pre_nms"
    assert first.fate == "final"
    assert isinstance(first.pose, BlockPose)
    np.testing.assert_allclose(first.pose.position, [1.05, 2.05, 0.31])
    assert first.pose.source == "cpp_core"

    assert second.fate == "post_refinement_nms_rejected"

    # final_hypotheses is filtered to fate == "final" only.
    assert len(result.trace.final_hypotheses) == 1
    assert result.trace.final_hypotheses[0].id == first.id

    # threshold_hypotheses/proposal_points are not derivable from this
    # contract and are left empty rather than faked.
    assert result.trace.threshold_hypotheses == ()
    assert result.trace.proposal_points == ()
    assert result.trace.downsampled_points.shape == (0, 3)
    assert result.trace.above_support_points.shape == (0, 3)


def test_detect_blocks_detailed_cpp_tolerates_missing_trace_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older/mid-rebuild extension without `refined_candidate_trace` yet
    should degrade to an empty trace, not crash."""
    result_without_trace = {
        key: value for key, value in _FIXTURE_RESULT.items() if key != "refined_candidate_trace"
    }
    fake = _FakeCore(result_without_trace)
    module = types.ModuleType("blockpose._core")
    module.discover = fake.discover  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "blockpose._core", module)

    result = detect_blocks_detailed_cpp(_points())
    assert result.trace is not None
    assert result.trace.raw_hypotheses == ()
    assert result.trace.final_hypotheses == ()


def test_detect_blocks_detailed_cpp_calls_discover_with_contiguous_float64(
    fake_core: _FakeCore,
) -> None:
    points = _points().astype(np.float32)
    detect_blocks_detailed_cpp(points)
    called = fake_core.last_call_points
    assert called is not None
    assert called.dtype == np.float64
    assert called.flags["C_CONTIGUOUS"]


def test_detect_blocks_detailed_cpp_missing_extension_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "blockpose._core", None)
    with pytest.raises(ImportError, match="blockpose._core is not built"):
        detect_blocks_detailed_cpp(_points())


@pytest.mark.parametrize(
    "bad_points",
    [
        np.zeros((5, 2)),  # wrong shape
        np.zeros((5,)),  # wrong ndim
        np.array([[1.0, 2.0, np.nan]]),  # non-finite
        np.array([[1.0, 2.0, np.inf]]),  # non-finite
    ],
)
def test_detect_blocks_detailed_cpp_rejects_invalid_points(
    fake_core: _FakeCore, bad_points: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        detect_blocks_detailed_cpp(bad_points)


def test_detect_blocks_detailed_cpp_rejects_complex_points(fake_core: _FakeCore) -> None:
    bad_points = np.zeros((5, 3), dtype=np.complex128)
    with pytest.raises(ValueError):
        detect_blocks_detailed_cpp(bad_points)


def test_detect_blocks_detailed_cpp_live_round_trip() -> None:
    """Live round-trip against the real compiled extension, if available.

    Skips gracefully when `blockpose._core` is not built in this environment.
    Also tolerant of a mid-rebuild extension that doesn't yet return
    `refined_candidate_trace` (the other half of this contract, being widened
    concurrently in cpp/bindings/blockpose_core_module.cpp).
    """
    pytest.importorskip("blockpose._core")
    from blockpose import synth

    scene = synth.generate_scene([synth.BlockSpec(x=0.0, y=1.5, z=None)])
    result = detect_blocks_detailed_cpp(scene.points, source_label="cpp_core_live")

    assert isinstance(result, DetectionResult)
    assert result.diagnostics == []
    assert result.stage_timings_ms == {}
    assert isinstance(result.stage_counts, dict)
    for pose in result.poses:
        assert isinstance(pose, BlockPose)
        assert pose.position.shape == (3,)
        assert pose.rotation.shape == (3, 3)
        assert pose.source == "cpp_core_live"

    assert result.trace is not None
    if result.trace.raw_hypotheses:
        for hypothesis in result.trace.raw_hypotheses:
            assert isinstance(hypothesis, TraceHypothesis)
            assert isinstance(hypothesis.pose, BlockPose)
