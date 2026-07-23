from __future__ import annotations

from pathlib import Path

import numpy as np
from blockpose.annotations import AnnotationTarget, SnapshotAnnotations
from blockpose.engine import (
    BlockResult,
    QueryResult,
    QueryStatus,
    ResultObservability,
)
from blockpose.mode_benchmark import run_gripped_audit, run_scene_discovery, run_verify_placed
from blockpose.pose import BlockPose
from blockpose.verify_placed import PlacedBlockPrior


def test_gripped_audit_withholds_oracle_pose_priors(monkeypatch, tmp_path: Path) -> None:
    class Scenario:
        query = "estimate_gripped"
        classification_status = "reviewed"
        pose_quality = "precise"

    pose = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_snapshot_scenarios", lambda: {"scene/block": Scenario()}
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_annotations",
        lambda _path: SnapshotAnnotations(
            2, "scene", True, np.zeros(3), np.ones(3), (AnnotationTarget("block", pose, "manual"),)
        ),
    )
    report = run_gripped_audit(tmp_path)
    assert report["status"] == "unavailable"
    assert report["records"][0]["status"] == "NotRun"
    assert "independent FK/grasp prior" in report["records"][0]["reason"]


def test_verify_placed_loads_labels_after_query_and_records_overlay(
    monkeypatch, tmp_path: Path
) -> None:
    pose = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    prior = PlacedBlockPrior("block", pose, source_snapshot_ids=("other",))

    class Fixture:
        prior = pose
        search_window_m = 0.1

    class Topology:
        priors_by_snapshot = {"scene": (prior,)}

    events: list[str] = []
    monkeypatch.setattr("blockpose.mode_benchmark.load_snapshot_plan_topology", lambda: Topology())
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_snapshot_plan_fixtures", lambda: {"scene/block": Fixture()}
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark._load_v1_baseline",
        lambda: {
            "identity": "legacy",
            "git_revision": "old",
            "source_kind": "recorded_legacy_evidence",
            "reproducible_with_current_code": False,
            "target_keys": ["scene/block"],
            "metrics": {
                "all_targets": {
                    "mean_translation_m": 0.01,
                    "mean_rotation_deg": 1.0,
                    "p95_translation_m": 0.01,
                    "p95_rotation_deg": 1.0,
                },
                "precise_targets": {
                    "mean_translation_m": 0.01,
                    "mean_rotation_deg": 1.0,
                    "p95_translation_m": 0.01,
                    "p95_rotation_deg": 1.0,
                },
            },
        },
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark._load_frame",
        lambda *_args: np.array([[0.0, 0.0, 0.0]]),
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_snapshot_scenarios",
        lambda: {
            "scene/block": type(
                "S",
                (),
                {
                    "scenario": "free",
                    "query": "detect_free",
                    "classification_status": "reviewed",
                    "pose_quality": "precise",
                },
            )()
        },
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_annotations",
        lambda _path: (
            events.append("annotations")
            or SnapshotAnnotations(
                2,
                "scene",
                True,
                np.zeros(3),
                np.ones(3),
                (AnnotationTarget("block", pose, "manual"),),
            )
        ),
    )

    def query(_self, _points, _poses):
        events.append("query")
        return QueryResult(
            (
                BlockResult(
                    pose,
                    pose.dims,
                    None,
                    None,
                    ResultObservability.WELL_CONSTRAINED,
                    (),
                    "test",
                    {},
                    QueryStatus.CONFIRMED,
                ),
            ),
            (),
            1.0,
            (),
        )

    monkeypatch.setattr("blockpose.mode_benchmark.HybridBlockDetector.verify_placed", query)
    report = run_verify_placed(tmp_path)
    assert events == ["query", "annotations"]
    assert report["records"][0]["translation_error_m"] == 0.0
    assert report["records"][0]["overlay"]["ground_truth"]["position_m"] == [0.0, 0.0, 0.3]
    assert report["baseline_comparison"]["same_target_keys"] is True
    assert (
        report["baseline_comparison"]["candidate_minus_baseline"]["all_targets"][
            "mean_translation_m"
        ]
        == -0.01
    )


def test_scene_discovery_reads_annotation_and_roi_only_after_query(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "scene"
    snapshot.mkdir()
    (snapshot / "tf.yaml").touch()
    pose = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    events: list[str] = []
    monkeypatch.setattr("blockpose.mode_benchmark.load_snapshot_exclusions", lambda _path: {})
    monkeypatch.setattr("blockpose.mode_benchmark._load_frame", lambda *_args: object())
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_snapshot_scenarios",
        lambda: {
            "scene/block": type(
                "S",
                (),
                {
                    "scenario": "free",
                    "query": "detect_free",
                    "classification_status": "reviewed",
                    "pose_quality": "precise",
                },
            )()
        },
    )
    monkeypatch.setattr(
        "blockpose.mode_benchmark.load_annotations",
        lambda _path: (
            events.append("annotations")
            or SnapshotAnnotations(
                2,
                "scene",
                True,
                np.array([-1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0]),
                (AnnotationTarget("block", pose, "manual"),),
            )
        ),
    )

    def query(_self, _frame):
        events.append("query")
        return QueryResult(
            (
                BlockResult(
                    pose,
                    pose.dims,
                    None,
                    None,
                    ResultObservability.WELL_CONSTRAINED,
                    (),
                    "test",
                    {},
                    QueryStatus.DETECTED,
                ),
            ),
            (),
            1.0,
            (),
        )

    monkeypatch.setattr("blockpose.mode_benchmark.HybridBlockDetector.detect_free", query)
    report = run_scene_discovery(tmp_path)
    assert events == ["query", "annotations"]
    assert report["precision"] == report["recall"] == 1.0
    assert report["records"][0]["prediction_count_scored_roi"] == 1
    assert report["exact_count_rate"] == 1.0
    assert report["target_cohorts"]["scenario"]["free"]["recall"] == 1.0
