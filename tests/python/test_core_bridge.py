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
from blockpose.core_types import (
    DetectionParameters,
    DetectionResult,
    InferenceTrace,
    Observability,
    SensorContext,
    TraceHypothesis,
)
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
    "timings_ms": {"total": 12.5, "refinement": 3.0},
    "diagnostics": [{
        "evidence": {
            "support_points": 55, "top_height_error_m": 0.02, "score": 0.42,
            "expected_visible_faces": 2, "covered_visible_faces": 2,
            "free_space_violations": 0, "supported_rays": 4,
            "observed_geometry_faces": 2, "incident_rays": 5,
        },
        "features": {
            "support": 0.1375, "height_consistency": 0.9,
            "visible_face_coverage": 1.0, "free_space_consistency": 1.0,
            "surface_explained": 0.8,
        },
        "observability": "partial", "calibrated_confidence": None,
        "proposal_scale_m": None, "support_height_m": 0.02,
    }],
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
    # C++ timings and diagnostics pass through without bridge-side calculation.
    assert result.diagnostics[0].observability is Observability.PARTIAL
    assert result.diagnostics[0].features["surface_explained"] == pytest.approx(0.8)
    assert result.diagnostics[0].calibrated_confidence is None
    assert result.stage_timings_ms == _FIXTURE_RESULT["timings_ms"]


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


def test_detect_blocks_detailed_cpp_maps_shared_params_and_rejects_python_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeParameters:
        pass

    calls: list[tuple[object, ...]] = []
    module = types.ModuleType("blockpose._core")
    module.DetectionParameters = NativeParameters  # type: ignore[attr-defined]
    module.discover = lambda *args: (calls.append(args) or _FIXTURE_RESULT)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "blockpose._core", module)

    detect_blocks_detailed_cpp(_points(), params=DetectionParameters(min_score=0.25))
    assert len(calls[0]) == 2
    assert calls[0][1].min_score == pytest.approx(0.25)

    with pytest.raises(ValueError, match="multiscale_proposals"):
        detect_blocks_detailed_cpp(_points(), params={"multiscale_proposals": True})
    with pytest.raises(ValueError, match="ransac_search_max_points"):
        detect_blocks_detailed_cpp(_points(), params={"ransac_search_max_points": None})


def test_detect_blocks_detailed_cpp_converts_sensor_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeParameters:
        pass

    class NativeContext:
        @staticmethod
        def from_arrays(*arrays: object) -> tuple[str, tuple[object, ...]]:
            return ("native_context", arrays)

    calls: list[tuple[object, ...]] = []
    module = types.ModuleType("blockpose._core")
    module.DetectionParameters = NativeParameters  # type: ignore[attr-defined]
    module.SensorContext = NativeContext  # type: ignore[attr-defined]
    module.discover = lambda *args: (calls.append(args) or _FIXTURE_RESULT)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "blockpose._core", module)
    context = SensorContext(
        origin=np.zeros(3), ray_directions=np.array([[0.0, 0.0, 1.0]]), ranges=np.array([1.0])
    )

    detect_blocks_detailed_cpp(_points(), sensor_context=context)
    assert len(calls[0]) == 3
    assert calls[0][2][0] == "native_context"


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
    assert len(result.diagnostics) == len(result.poses)
    assert all(diagnostic.calibrated_confidence is None for diagnostic in result.diagnostics)
    assert set(result.stage_timings_ms) == {
        "voxel_downsample",
        "ground_model",
        "proposals",
        "region_split",
        "plane_search",
        "plane_full_score_refit",
        "hypothesis_generation_evidence",
        "plane_hypotheses",
        "selection",
        "refinement",
        "total",
    }
    assert all(value >= 0.0 for value in result.stage_timings_ms.values())
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ground_normal_min_z", 1.01, "ground_normal_min_z"),
        ("local_ground_clearance", -0.01, "local_ground_clearance"),
        ("cluster_max_size", 1, "invalid zero or non-finite threshold"),
        ("cluster_max_extent_xy", 0.1, "invalid zero or non-finite threshold"),
        ("top_plane_angle_deg", 91.0, "plane angles"),
        ("fk_seed_min_evidence_gain", -0.01, "invalid zero or non-finite threshold"),
        ("proposal_max_components", 0, "invalid zero or non-finite threshold"),
    ],
)
def test_native_detection_parameters_reject_invalid_scalar_domains(
    field: str, value: object, message: str,
) -> None:
    core = pytest.importorskip("blockpose._core")
    params = core.DetectionParameters()
    setattr(params, field, value)
    with pytest.raises(ValueError, match=message):
        core.discover(np.empty((0, 3), dtype=np.float64), params)
