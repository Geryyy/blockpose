"""Reproducible, leakage-aware benchmark for the frozen real snapshots.

This is deliberately a *development diagnostic*, not a production acceptance
test.  The annotation poses are loaded only after the corresponding query has
finished.  The only placed-block priors are the existing leave-one-snapshot-out
fixtures, and gripped queries are withheld unless an independently recorded
FK/grasp prior is available.
"""

from __future__ import annotations

import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from blockpose.annotations import load_annotations
from blockpose.detection import DetectionParams
from blockpose.engine import HybridBlockDetector, QueryStatus
from blockpose.evaluate import rotation_error_deg
from blockpose.frames import PointFrame, PoseTrajectory, TimedPose
from blockpose.io import load_pcd, load_tf_snapshot
from blockpose.pose import rotmat_to_quat_xyzw
from blockpose.snapshot_exclusions import load_snapshot_exclusions
from blockpose.snapshot_plan_fixtures import load_snapshot_plan_fixtures
from blockpose.snapshot_plan_topology import load_snapshot_plan_topology
from blockpose.snapshot_scenarios import audit_manifest, load_snapshot_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_TRANSFORM = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
DEFAULT_V1_BASELINE = REPO_ROOT / "config" / "verify_placed_v1_baseline_c9f069d.yaml"


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _worktree_provenance() -> dict[str, Any]:
    """Make an artifact from a dirty tree explicitly non-revision-identical."""
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"source_worktree_dirty": None, "source_worktree_status": None}
    return {
        "source_worktree_dirty": bool(status),
        "source_worktree_status": status,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _load_v1_baseline(path: Path = DEFAULT_V1_BASELINE) -> dict[str, Any]:
    baseline = yaml.safe_load(path.read_text()) or {}
    if baseline.get("schema_version") != 1 or not baseline.get("target_keys"):
        raise ValueError("invalid VerifyPlaced v1 baseline artifact")
    return baseline


def _pose_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    translation = [
        record["translation_error_m"]
        for record in records
        if record["translation_error_m"] is not None
    ]
    rotation = [
        record["rotation_error_deg"]
        for record in records
        if record["rotation_error_deg"] is not None
    ]
    return {
        "target_count": len(records),
        "pose_error_sample_count": len(translation),
        "mean_translation_m": float(np.mean(translation)) if translation else None,
        "mean_rotation_deg": float(np.mean(rotation)) if rotation else None,
        "p95_translation_m": _percentile(translation, 95),
        "p95_rotation_deg": _percentile(rotation, 95),
    }


def _pose_record(pose) -> dict[str, Any]:
    return {
        "position_m": np.asarray(pose.position, dtype=float).tolist(),
        "orientation_xyzw": rotmat_to_quat_xyzw(pose.rotation).tolist(),
        "dimensions_m": [float(value) for value in pose.dims],
        "source": pose.source,
    }


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["status"] for record in records).items()))


def _compatible_match(detections, targets, max_distance: float) -> list[tuple[int, int]]:
    """One-to-one scoring association which never changes predictions."""
    if not detections or not targets:
        return []
    invalid = 1e6
    cost = np.full((len(detections) + len(targets),) * 2, invalid)
    for detection_index, detection in enumerate(detections):
        for target_index, target in enumerate(targets):
            distance = float(np.linalg.norm(detection.position - target.position))
            if (
                np.allclose(sorted(detection.dims), sorted(target.dims), atol=0.03)
                and distance <= max_distance
                and rotation_error_deg(detection, target) <= 30.0
            ):
                cost[detection_index, target_index] = distance
    unmatched = max_distance + 1.0
    cost[: len(detections), len(targets) :] = unmatched
    cost[len(detections) :, : len(targets)] = unmatched
    cost[len(detections) :, len(targets) :] = 0.0
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if row < len(detections) and column < len(targets) and cost[row, column] < invalid
    ]


def _load_frame(snapshot: Path, config_path: Path) -> PointFrame:
    """Load only LiDAR and TF, without opening registrations or annotations."""
    tf = load_tf_snapshot(snapshot / "tf.yaml")
    # ``PointFrame`` preserves the known static sensor origin so the facade can
    # use actual return rays.  Existing snapshots cannot provide no-return
    # rays, intensity, time, or gripper evidence; the resulting response says
    # so explicitly in ``unavailable_evidence``.
    T_world_sensor = tf.T_world_cloud
    del config_path  # TF is mandatory for this benchmark; no static fallback.
    points_sensor = load_pcd(snapshot / "cloud.pcd")
    return PointFrame(
        points_sensor,
        frame_id="seyond",
        sensor_trajectory=PoseTrajectory((TimedPose(0.0, T_world_sensor),)),
    )


def run_scene_discovery(
    snapshots_dir: Path = DEFAULT_SNAPSHOTS,
    config_path: Path = DEFAULT_TRANSFORM,
    *,
    min_score: float = 0.4,
    timeout_ms: float = 500.0,
) -> dict[str, Any]:
    """Prediction-first free discovery; annotation ROI is scoring-only."""
    detector = HybridBlockDetector(DetectionParams(min_score=min_score), timeout_ms=timeout_ms)
    excluded = load_snapshot_exclusions(snapshots_dir)
    records: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for snapshot in sorted(snapshots_dir.iterdir()):
        if (
            not snapshot.is_dir()
            or snapshot.name in excluded
            or not (snapshot / "tf.yaml").is_file()
        ):
            continue
        try:
            frame = _load_frame(snapshot, config_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"snapshot": snapshot.name, "reason": str(error)})
            continue
        # Deliberately before any annotation/ROI read.
        started = time.perf_counter()
        query = detector.detect_free(frame)
        invocation_ms = (time.perf_counter() - started) * 1000.0
        invocation = {
            "snapshot": snapshot.name,
            "core_runtime_ms": query.runtime_ms,
            "end_to_end_runtime_ms": invocation_ms,
            "timeout": any(failure.code.value == "timeout" for failure in query.failures),
            "failures": [
                {"code": failure.code.value, "message": failure.message}
                for failure in query.failures
            ],
        }
        invocations.append(invocation)

        # Evaluation-only data loaded after inference.  ROI filters output for
        # metrics; it never crops the cloud or affects detector selection.
        annotations = load_annotations(snapshot / "annotations.yaml")
        scenarios = load_snapshot_scenarios()
        detections = [block.pose for block in query.blocks]
        targets = [target for target in annotations.targets if target.is_evaluation_target]
        roi = (annotations.roi_min, annotations.roi_max)
        if annotations.has_complete_roi:
            assert roi[0] is not None and roi[1] is not None
            low, high = roi
            detections = [
                pose
                for pose in detections
                if np.all(pose.position >= low) and np.all(pose.position <= high)
            ]
            targets = [
                target
                for target in targets
                if np.all(target.pose.position >= low) and np.all(target.pose.position <= high)
            ]
        pairs = _compatible_match(detections, [target.pose for target in targets], 0.50)
        records.append(
            {
                "snapshot": snapshot.name,
                "annotation_complete": annotations.has_complete_roi,
                "scoring_roi": (
                    {"min_m": roi[0].tolist(), "max_m": roi[1].tolist()}
                    if annotations.has_complete_roi
                    else None
                ),
                "status": "QueryFailure" if query.failures else "Completed",
                "query_failures": invocation["failures"],
                "prediction_count_scored_roi": len(detections),
                "target_count_scored_roi": len(targets),
                "matched": len(pairs),
                "false_positives": len(detections) - len(pairs)
                if annotations.has_complete_roi
                else None,
                "missed_targets": len(targets) - len(pairs),
                "detections": [
                    {
                        "index": index,
                        "pose": _pose_record(pose),
                        "matched_target_index": next(
                            (
                                target_index
                                for detection_index, target_index in pairs
                                if detection_index == index
                            ),
                            None,
                        ),
                    }
                    for index, pose in enumerate(detections)
                ],
                "targets": [
                    {
                        "index": index,
                        "id": target.id,
                        "pose": _pose_record(target.pose),
                        "pose_quality": scenarios[f"{snapshot.name}/{target.id}"].pose_quality,
                        "matched_detection_index": next(
                            (
                                detection_index
                                for detection_index, target_index in pairs
                                if target_index == index
                            ),
                            None,
                        ),
                    }
                    for index, target in enumerate(targets)
                ],
            }
        )
    complete = [record for record in records if record["annotation_complete"]]
    total_detections = sum(record["prediction_count_scored_roi"] for record in complete)
    total_targets = sum(record["target_count_scored_roi"] for record in complete)
    total_matches = sum(record["matched"] for record in complete)
    core = [item["core_runtime_ms"] for item in invocations]
    end_to_end = [item["end_to_end_runtime_ms"] for item in invocations]
    return {
        "mode": "SCENE_DISCOVERY",
        "method": "detect_free",
        "development_only": True,
        "acceptance_eligible": False,
        "inference_inputs": "full LiDAR XYZ and TF; annotation ROI applies only to scoring",
        "n_snapshots_scored": len(records),
        "total_targets": total_targets,
        "total_predictions_scored_roi": total_detections,
        "total_matched": total_matches,
        "precision": total_matches / total_detections if total_detections else None,
        "recall": total_matches / total_targets if total_targets else None,
        "empty_scene_count": sum(record["target_count_scored_roi"] == 0 for record in complete),
        "matching_policy": {
            "max_translation_m": 0.50,
            "max_rotation_deg": 30.0,
            "dimension_sorted_atol_m": 0.03,
        },
        "query_invocation_count": len(invocations),
        "timeout_count": sum(item["timeout"] for item in invocations),
        "core_runtime_p50_ms": _percentile(core, 50),
        "core_runtime_p95_ms": _percentile(core, 95),
        "core_runtime_max_ms": max(core, default=None),
        "end_to_end_runtime_p50_ms": _percentile(end_to_end, 50),
        "end_to_end_runtime_p95_ms": _percentile(end_to_end, 95),
        "end_to_end_runtime_max_ms": max(end_to_end, default=None),
        "invocations": invocations,
        "snapshot_failures": failures,
        "records": records,
    }


def run_verify_placed(
    snapshots_dir: Path = DEFAULT_SNAPSHOTS,
    config_path: Path = DEFAULT_TRANSFORM,
    *,
    min_score: float = 0.4,
    timeout_ms: float = 500.0,
) -> dict[str, Any]:
    """Run only plan priors that are auditable as leave-one-snapshot-out.

    Annotation labels are intentionally read after ``verify_placed`` returns.
    The result makes no Missing-status accuracy claim: every evaluated target
    is an annotated visible block, and these snapshots contain no independently
    labelled absent planned blocks.
    """
    topology = load_snapshot_plan_topology()
    fixtures = load_snapshot_plan_fixtures()
    detector = HybridBlockDetector(
        DetectionParams(min_score=min_score), timeout_ms=timeout_ms, verify_placed_v1=True
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    invocations: list[dict[str, Any]] = []
    for snapshot_id, priors in sorted(topology.priors_by_snapshot.items()):
        snapshot = snapshots_dir / snapshot_id
        try:
            frame = _load_frame(snapshot, config_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"snapshot": snapshot_id, "reason": str(error)})
            continue

        # Inference uses only point data and LOSO priors/topology above.
        started = time.perf_counter()
        query = detector.verify_placed(frame, [prior.pose for prior in priors])
        invocation_ms = (time.perf_counter() - started) * 1000.0
        invocations.append(
            {
                "snapshot": snapshot_id,
                "core_runtime_ms": query.runtime_ms,
                "end_to_end_runtime_ms": invocation_ms,
                "timeout": any(failure.code.value == "timeout" for failure in query.failures),
                "failures": [
                    {"code": failure.code.value, "message": failure.message}
                    for failure in query.failures
                ],
            }
        )

        # Scoring-only join.  Keep this after inference to make leakage review
        # evident in both code order and the result artifact.
        annotations = load_annotations(snapshot / "annotations.yaml")
        targets = {
            target.id: target for target in annotations.targets if target.is_evaluation_target
        }
        scenarios = load_snapshot_scenarios()
        for index, prior in enumerate(priors):
            result = query.blocks[index] if index < len(query.blocks) else None
            target = targets.get(prior.block_id)
            fixture = fixtures[f"{snapshot_id}/{prior.block_id}"]
            record: dict[str, Any] = {
                "snapshot": snapshot_id,
                "target_id": prior.block_id,
                "target_key": f"{snapshot_id}/{prior.block_id}",
                "prior": _pose_record(prior.pose),
                "prior_source_snapshots": list(prior.source_snapshot_ids),
                "search_window_m": fixture.search_window_m,
                "result": _pose_record(result.pose) if result is not None else None,
                "status": result.status.value if result is not None else "QueryFailure",
                "observability": result.observability.value if result is not None else None,
                "unconstrained_directions": (
                    list(result.unconstrained_directions) if result is not None else []
                ),
                "prior_deviation_m": result.prior_deviation_m if result is not None else None,
                "score_features": result.score_features if result is not None else {},
                "diagnostics": result.diagnostics if result is not None else {},
                "core_runtime_ms": query.runtime_ms,
                "end_to_end_runtime_ms": invocation_ms,
                "query_failures": [
                    {"code": failure.code.value, "message": failure.message}
                    for failure in query.failures
                ],
                "overlay": (
                    {"prediction": _pose_record(result.pose)} if result is not None else {}
                ),
            }
            if target is not None:
                record["annotation"] = {
                    "pose": _pose_record(target.pose),
                    "pose_quality": scenarios[record["target_key"]].pose_quality,
                    "visibility": target.visibility,
                    "provenance": target.provenance,
                }
                record["translation_error_m"] = (
                    float(np.linalg.norm(result.pose.position - target.pose.position))
                    if result is not None
                    else None
                )
                record["rotation_error_deg"] = (
                    rotation_error_deg(result.pose, target.pose) if result is not None else None
                )
                record["overlay"]["ground_truth"] = _pose_record(target.pose)
            else:
                record["annotation"] = None
                record["translation_error_m"] = None
                record["rotation_error_deg"] = None
            records.append(record)

    precise = [
        record
        for record in records
        if record["annotation"] is not None and record["annotation"]["pose_quality"] == "precise"
    ]
    approximate = [
        record
        for record in records
        if record["annotation"] is not None
        and record["annotation"]["pose_quality"] == "approximate"
    ]
    errors = [
        record["translation_error_m"]
        for record in precise
        if record["translation_error_m"] is not None
    ]
    rotations = [
        record["rotation_error_deg"]
        for record in precise
        if record["rotation_error_deg"] is not None
    ]
    approximate_errors = [
        record["translation_error_m"]
        for record in approximate
        if record["translation_error_m"] is not None
    ]
    baseline = _load_v1_baseline()
    candidate_keys = sorted(record["target_key"] for record in records)
    baseline_keys = sorted(str(key) for key in baseline["target_keys"])
    if candidate_keys != baseline_keys:
        raise ValueError("VerifyPlaced candidate target set differs from frozen v1 baseline")
    candidate_all = _pose_summary(records)
    candidate_precise = _pose_summary(precise)
    baseline_metrics = baseline["metrics"]

    def delta(group: str, candidate: dict[str, Any]) -> dict[str, float | None]:
        reference = baseline_metrics[group]
        return {
            "mean_translation_m": (
                candidate["mean_translation_m"] - float(reference["mean_translation_m"])
                if candidate["mean_translation_m"] is not None
                else None
            ),
            "mean_rotation_deg": (
                candidate["mean_rotation_deg"] - float(reference["mean_rotation_deg"])
                if candidate["mean_rotation_deg"] is not None
                else None
            ),
            "p95_translation_m": (
                candidate["p95_translation_m"] - float(reference["p95_translation_m"])
                if candidate["p95_translation_m"] is not None
                else None
            ),
            "p95_rotation_deg": (
                candidate["p95_rotation_deg"] - float(reference["p95_rotation_deg"])
                if candidate["p95_rotation_deg"] is not None
                else None
            ),
        }

    present_statuses = {QueryStatus.CONFIRMED.value, QueryStatus.DEVIATED.value}
    return {
        "mode": "REFINE_BLOCK",
        "method": "verify_placed_v1",
        "development_only": True,
        "acceptance_eligible": False,
        "inference_inputs": "LiDAR XYZ plus LOSO plan priors; annotations loaded after query",
        "limitations": [
            "LOSO priors/topology are development fixtures, not an independent held-out plan test.",
            "No labelled absent planned blocks: Missing-status accuracy is unavailable.",
            "Two annotated poses are marked approximate and must not be treated "
            "as precise pose labels.",
        ],
        "n_planned_targets": len(records),
        "candidate_identity": "verify_placed_v1_candidate_v2",
        "baseline_comparison": {
            "same_target_keys": True,
            "baseline": {
                "identity": baseline["identity"],
                "git_revision": baseline["git_revision"],
                "evidence_kind": baseline["source_kind"],
                "reproducible_with_current_code": baseline["reproducible_with_current_code"],
                "metrics": baseline_metrics,
            },
            "candidate": {
                "identity": "verify_placed_v1_candidate_v2",
                "all_targets": candidate_all,
                "precise_targets": candidate_precise,
            },
            "candidate_minus_baseline": {
                "all_targets": delta("all_targets", candidate_all),
                "precise_targets": delta("precise_targets", candidate_precise),
            },
        },
        "n_fixture_targets_without_prior": sum(
            fixture.prior is None for fixture in fixtures.values()
        ),
        "observed_status_counts_on_present_targets": _status_counts(records),
        "present_target_response_rate": (
            sum(record["status"] in present_statuses for record in records) / len(records)
            if records
            else None
        ),
        "precise_pose_error_sample_count": len(errors),
        "precise_target_count": len(precise),
        "translation_error_p50_m": _percentile(errors, 50),
        "translation_error_p95_m": _percentile(errors, 95),
        "rotation_error_p50_deg": _percentile(rotations, 50),
        "rotation_error_p95_deg": _percentile(rotations, 95),
        "approximate_pose_error_sample_count": len(approximate_errors),
        "approximate_target_count": len(approximate),
        "approximate_translation_error_p50_m": _percentile(approximate_errors, 50),
        "approximate_translation_error_p95_m": _percentile(approximate_errors, 95),
        "query_invocation_count": len(invocations),
        "timeout_count": sum(item["timeout"] for item in invocations),
        "core_runtime_p50_ms": _percentile([item["core_runtime_ms"] for item in invocations], 50),
        "core_runtime_p95_ms": _percentile([item["core_runtime_ms"] for item in invocations], 95),
        "core_runtime_max_ms": max((item["core_runtime_ms"] for item in invocations), default=None),
        "end_to_end_runtime_p50_ms": _percentile(
            [item["end_to_end_runtime_ms"] for item in invocations], 50
        ),
        "end_to_end_runtime_p95_ms": _percentile(
            [item["end_to_end_runtime_ms"] for item in invocations], 95
        ),
        "end_to_end_runtime_max_ms": max(
            (item["end_to_end_runtime_ms"] for item in invocations), default=None
        ),
        "invocations": invocations,
        "snapshot_failures": failures,
        "records": records,
    }


def run_gripped_audit(snapshots_dir: Path = DEFAULT_SNAPSHOTS) -> dict[str, Any]:
    """Report why no real gripped estimate is manufactured from annotations.

    The repository has reviewed gripped taxonomy labels but no per-snapshot
    grasp pose/FK covariance or gripper trajectory.  A target-derived prior
    would make the apparent estimator evaluation circular, so the correct
    baseline result is an explicit unavailable mode rather than an oracle run.
    """
    scenarios = load_snapshot_scenarios()
    records = []
    for key, scenario in sorted(scenarios.items()):
        if scenario.query != "estimate_gripped":
            continue
        snapshot_id, target_id = key.split("/", 1)
        annotation = load_annotations(snapshots_dir / snapshot_id / "annotations.yaml")
        target = next((item for item in annotation.targets if item.id == target_id), None)
        records.append(
            {
                "snapshot": snapshot_id,
                "target_id": target_id,
                "target_key": key,
                "taxonomy_status": scenario.classification_status,
                "pose_quality": scenario.pose_quality,
                "annotation_visible": target.visibility if target is not None else None,
                "status": "NotRun",
                "reason": (
                    "no independent FK/grasp prior, covariance, or gripper trajectory in snapshot"
                ),
            }
        )
    return {
        "mode": "REFINE_GRASPED",
        "method": "withheld",
        "development_only": True,
        "acceptance_eligible": False,
        "status": "unavailable",
        "limitations": [
            "Do not derive a grasp prior from the annotated target pose.",
            "No gripper filtering or FK covariance can be validated from XYZ snapshots.",
        ],
        "n_reviewed_gripped_targets": len(records),
        "records": records,
    }


def run_mode_benchmark(
    snapshots_dir: Path = DEFAULT_SNAPSHOTS,
    config_path: Path = DEFAULT_TRANSFORM,
    *,
    min_score: float = 0.4,
    timeout_ms: float = 500.0,
) -> dict[str, Any]:
    """Run the frozen snapshot diagnostic with GT kept out of inference."""
    discovery = run_scene_discovery(
        snapshots_dir,
        config_path,
        min_score=min_score,
        timeout_ms=timeout_ms,
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "snapshot_root": str(snapshots_dir),
        "config": {
            "min_score": min_score,
            "timeout_ms": timeout_ms,
            "snapshot_tf_policy": "required_per_snapshot_tf.yaml",
            "legacy_static_transform_argument": str(config_path),
            "legacy_static_transform_used": False,
        },
        **_worktree_provenance(),
        "taxonomy_audit": audit_manifest(snapshots_dir),
        "exclusions": {
            name: {"reason": entry.reason, "retained_snapshot": entry.retained_snapshot}
            for name, entry in load_snapshot_exclusions(snapshots_dir).items()
        },
        "modes": {
            "SCENE_DISCOVERY": discovery,
            "REFINE_BLOCK": run_verify_placed(
                snapshots_dir, config_path, min_score=min_score, timeout_ms=timeout_ms
            ),
            "REFINE_GRASPED": run_gripped_audit(snapshots_dir),
        },
        "interpretation": {
            "purpose": "frozen real-snapshot development diagnostic and notebook input",
            "not_a_production_gate": True,
            "ground_truth_policy": (
                "annotations are scoring/overlay labels only and are loaded after each mode query"
            ),
            "annotation_quality_policy": (
                "approximate annotations remain separately identifiable in per-target records"
            ),
        },
    }
