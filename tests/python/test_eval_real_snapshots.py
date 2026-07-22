"""Smoke test for the real-snapshot eval harness (tools/eval_real_snapshots.py).

This is the Phase D (real-validation) counterpart to test_eval_detector.py, but
unlike the synthetic Phase-A gate it does NOT assert fixed thresholds: with only
a handful of real snapshots (and ground truth that is itself sometimes
incomplete -- see the harness's module docstring), thresholds would be fit to
noise. This only checks the harness runs end-to-end against the checked-in
snapshots and produces well-formed output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_real_snapshots  # noqa: E402
from blockpose.annotations import AnnotationTarget, SnapshotAnnotations  # noqa: E402
from blockpose.detection import (  # noqa: E402
    DetectionDiagnostics,
    DetectionResult,
    HypothesisEvidence,
    Observability,
)
from blockpose.pose import BlockPose  # noqa: E402

SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"


def test_discovers_every_checked_in_snapshot_as_tf_enabled() -> None:
    with_tf, without_tf = eval_real_snapshots.discover_snapshots(SNAPSHOTS_DIR)
    assert len(with_tf) >= 1
    assert without_tf == []
    assert all(path.name != "1783428861_719600128_seq3" for path in with_tf)
    assert all(path.name != "1783428498_520086272_seq4" for path in with_tf)
    assert all(path.name != "1783428995_118905856_seq6" for path in with_tf)


def test_run_suite_reports_exclusions_and_survives_bad_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    (tmp_path / "excluded").mkdir()
    for name in ("good", "bad", "excluded"):
        (tmp_path / name / "tf.yaml").touch()
    (tmp_path / "snapshot_exclusions.yaml").write_text(
        "schema_version: 1\nexcluded_snapshots:\n  - {snapshot: excluded, reason: unusable}\n"
    )

    def evaluate(path, *_args):
        return {
            "snapshot": path.name,
            "annotation_complete": False,
            "reference_targets": 0,
            "detected": 0,
            "matched": 0,
            "unknown_detections": 0,
            "false_positives": None,
            "translation_errors_m": [],
            "rotation_errors_deg": [],
            "match_curves": {key: {"matched": 0} for key in ("5cm", "10cm", "25cm", "50cm")},
            "runtime_ms": 1.0,
        }

    monkeypatch.setattr(eval_real_snapshots, "evaluate_snapshot", evaluate)
    monkeypatch.setattr(
        eval_real_snapshots,
        "load_tf_snapshot",
        lambda path: (
            (_ for _ in ()).throw(ValueError("invalid TF"))
            if path.parent.name == "bad"
            else SimpleNamespace(T_world_cloud=np.eye(4))
        ),
    )
    metrics = eval_real_snapshots.run_suite(tmp_path, TRANSFORM_CONFIG)
    assert metrics["snapshot_ids"] == ["good"]
    assert metrics["n_snapshots_failed_integrity"] == 1
    assert metrics["integrity_failures"][0]["snapshot"] == "bad"
    assert metrics["n_snapshots_excluded"] == 1
    assert metrics["excluded_snapshots"][0]["snapshot"] == "excluded"


def test_run_suite_does_not_hide_detector_or_evaluator_errors(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "scene"
    snapshot.mkdir()
    (snapshot / "tf.yaml").touch()
    monkeypatch.setattr(
        eval_real_snapshots,
        "load_tf_snapshot",
        lambda _path: SimpleNamespace(T_world_cloud=np.eye(4)),
    )
    monkeypatch.setattr(
        eval_real_snapshots,
        "evaluate_snapshot",
        lambda *_args: (_ for _ in ()).throw(ValueError("detector regression")),
    )
    with np.testing.assert_raises_regex(ValueError, "detector regression"):
        eval_real_snapshots.run_suite(tmp_path, TRANSFORM_CONFIG)


def test_run_suite_produces_well_formed_metrics() -> None:
    metrics = eval_real_snapshots.run_suite(SNAPSHOTS_DIR, TRANSFORM_CONFIG)
    assert metrics["n_snapshots_scored"] >= 1
    assert metrics["n_snapshots_scored"] == len(metrics["per_snapshot"])
    assert metrics["total_reference_targets"] >= 0
    assert metrics["total_detected"] >= 0
    assert (
        0
        <= metrics["total_matched"]
        <= min(metrics["total_reference_targets"], metrics["total_detected"])
    )
    if metrics["n_complete_annotations"]:
        assert metrics["precision"] is not None
        assert 0.0 <= metrics["precision"] <= 1.0
        assert metrics["false_positives"] is not None
        assert metrics["false_positives"] >= 0
    else:
        assert metrics["precision"] is None
        assert metrics["false_positives"] is None
    assert metrics["n_matched_pose_errors"] == metrics["total_matched"]
    for entry in metrics["per_snapshot"]:
        assert entry["reference_targets"] >= 0
        assert entry["detected"] >= 0
        assert entry["matched"] <= entry["reference_targets"]
        assert entry["matched"] <= entry["detected"]
        # precision/recall must be caveated to None, not a misleading 0.0,
        # whenever there is no registered ground truth to compare against.
        if not entry["annotation_complete"]:
            assert entry["precision"] is None
            assert entry["recall"] is None
            assert entry["false_positives"] is None
            assert entry["unknown_detections"] >= 0


def test_complete_roi_suppresses_ignored_detection_before_fp_metrics(
    monkeypatch, tmp_path: Path
) -> None:
    active = BlockPose(np.array([0.25, 0.25, 0.25]), np.eye(3))
    ignored = BlockPose(np.array([0.75, 0.75, 0.75]), np.eye(3))
    outside = BlockPose(np.array([2.0, 2.0, 2.0]), np.eye(3))
    annotations = SnapshotAnnotations(
        1,
        tmp_path.name,
        True,
        np.zeros(3),
        np.ones(3),
        (
            AnnotationTarget("active", active, "manual"),
            AnnotationTarget("ignored", ignored, "manual", ignore_reason="occluded"),
        ),
    )
    diagnostic = DetectionDiagnostics(HypothesisEvidence(100, 0.0, 0.9), Observability.FULL, {})
    monkeypatch.setattr(eval_real_snapshots, "load_annotations", lambda _path: annotations)
    monkeypatch.setattr(
        eval_real_snapshots,
        "load_tf_snapshot",
        lambda _path: SimpleNamespace(T_world_cloud=np.eye(4)),
    )
    monkeypatch.setattr(
        eval_real_snapshots,
        "load_snapshot",
        lambda *_args, **_kwargs: (np.array([[0.5, 0.5, 0.5]]), []),
    )
    monkeypatch.setattr(
        eval_real_snapshots,
        "detect_blocks_detailed",
        lambda *_args, **_kwargs: DetectionResult(
            [active, ignored, outside], [diagnostic, diagnostic, diagnostic]
        ),
    )
    (tmp_path / "annotations.yaml").touch()
    result = eval_real_snapshots.evaluate_snapshot(
        tmp_path, TRANSFORM_CONFIG, eval_real_snapshots.DetectionParams(min_score=0.4)
    )
    assert result["detected"] == 1
    assert result["matched"] == 1
    assert result["suppressed_ignored_detections"] == 1
    assert result["false_positives"] == 0
    assert result["exact_count"] is True
    assert result["target_diagnostics"] == [
        {
            "index": 0,
            "id": "active",
            "pose": eval_real_snapshots._pose_dict(active),
            "nearest_final_detection_index": 0,
            "nearest_final_distance_m": 0.0,
            "matched_final_detection_index": 0,
            "matched_translation_error_m": 0.0,
            "matched_rotation_error_deg": 0.0,
            "matched_at_5cm": True,
            "matched_at_10cm": True,
            "matched_at_25cm": True,
            "matched_at_50cm": True,
        }
    ]
