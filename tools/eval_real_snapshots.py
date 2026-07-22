#!/usr/bin/env python3
"""Run the block detector over real registration snapshots and report metrics.

This is the Phase D (real-validation) counterpart to `tools/eval_detector.py`,
which gates on synthetic data only. Real ground truth was untrustworthy until
a snapshot carried its own `tf.yaml` (the crane's actual joint state at
capture time; see `blockpose.io.load_tf_snapshot`) -- the static reference
config can misplace the cloud by several metres relative to the recorded
ground truth. Snapshots without a `tf.yaml` are skipped and reported
separately rather than silently misdetected on.

There is no fixed pass/fail gate here yet: with only a handful of real
snapshots, thresholds would be fit to noise rather than signal. This prints
(and can save) the same shape of metrics `eval_detector.py` reports, so a
regression is visible by comparing runs, and re-running against a growing set
of real snapshots is the whole workflow for checking that a fix generalizes
rather than overfitting to one scene.

CAVEAT -- the "ground truth" here is not exact. `registrations.yaml` is itself
the output of the upstream RGB-based registration pipeline, which can fail to
register a real block it saw (e.g. `registration_ok: false, reason: "result
code 6"`) -- observed on 2 of the 4 snapshots available at time of writing,
both reporting zero ground-truth blocks despite having candidate detections.
An "unmatched" LiDAR detection in a snapshot with few/zero registered blocks
may therefore be a real, un-registered block rather than a false positive.
Treat `unmatched_detections`/`precision` as provisional wherever
`gt_blocks` is small, and prefer snapshots with more registered blocks when
judging precision.

Examples:
    python tools/eval_real_snapshots.py
    python tools/eval_real_snapshots.py --metrics-out real_metrics.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.detection import DetectionParams, SensorContext, detect_blocks_detailed  # noqa: E402
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.io import load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.pose import rotmat_to_quat_xyzw  # noqa: E402
from blockpose.snapshot_exclusions import load_snapshot_exclusions  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

_DEFAULT_TRANSFORM = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
_DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"


def discover_snapshots(snapshots_dir: Path) -> tuple[list[Path], list[Path]]:
    """Split snapshot directories into (with_tf, without_tf).

    Only `with_tf` snapshots have ground truth trustworthy enough to score
    against (see module docstring); `without_tf` is reported for visibility,
    not scored.
    """
    if not snapshots_dir.is_dir():
        return [], []
    excluded = load_snapshot_exclusions(snapshots_dir)
    with_tf, without_tf = [], []
    for entry in sorted(snapshots_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in excluded:
            continue
        if (entry / "tf.yaml").is_file():
            with_tf.append(entry)
        elif (entry / "registrations.yaml").is_file():
            without_tf.append(entry)
    return with_tf, without_tf


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _compatible_match(
    detections, targets, max_distance: float, max_rotation_deg: float = 30.0
) -> list[tuple[int, int]]:
    """Maximum-cardinality assignment constrained by shape, distance and rotation."""
    if not detections or not targets:
        return []
    invalid = 1e6
    cost = np.full((len(detections) + len(targets),) * 2, invalid)
    for i, detection in enumerate(detections):
        for j, target in enumerate(targets):
            same_shape = np.allclose(sorted(detection.dims), sorted(target.dims), atol=0.03)
            distance = translation_error(detection, target)
            rotation = rotation_error_deg(detection, target)
            if same_shape and distance <= max_distance and rotation <= max_rotation_deg:
                cost[i, j] = distance + rotation * 1e-5
    unmatched = max_distance + 1.0
    cost[: len(detections), len(targets) :] = unmatched
    cost[len(detections) :, : len(targets)] = unmatched
    cost[len(detections) :, len(targets) :] = 0.0
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
        if i < len(detections) and j < len(targets) and cost[i, j] < invalid
    ]


def _pose_dict(pose) -> dict:
    return {
        "position": pose.position.tolist(),
        "orientation_xyzw": rotmat_to_quat_xyzw(pose.rotation).tolist(),
        "dimensions": list(pose.dims),
        "source": pose.source,
    }


def evaluate_snapshot(snap_dir: Path, config_path: Path, params: DetectionParams) -> dict:
    """Run detection on one real snapshot and return its metrics.

    `unmatched_detections` is NOT the same as a confirmed false positive: the
    registered ground truth can itself be incomplete (see module docstring),
    so an unmatched detection may be a real block the upstream RGB pipeline
    failed to register. `precision`/`recall` are reported as None when there
    is no ground truth to compare against, rather than the misleading 0.0
    `evaluate()` returns by default for that case.
    """
    tf_snapshot = load_tf_snapshot(snap_dir / "tf.yaml")
    pts_world, registrations = load_snapshot(
        snap_dir, config_path, T_world_cloud=tf_snapshot.T_world_cloud
    )
    annotation_path = snap_dir / "annotations.yaml"
    annotations = load_annotations(annotation_path) if annotation_path.is_file() else None
    targets = (
        [target.pose for target in annotations.targets if target.is_evaluation_target]
        if annotations
        else registrations
    )
    ignored_targets = (
        [
            target.pose
            for target in annotations.targets
            if target.review_status == "approved" and target.ignore_reason is not None
        ]
        if annotations
        else []
    )
    if annotations and annotations.roi_min is not None and annotations.roi_max is not None:
        assert annotations.roi_min is not None and annotations.roi_max is not None
        in_roi = np.all(
            (pts_world >= annotations.roi_min) & (pts_world <= annotations.roi_max), axis=1
        )
        pts_world = pts_world[in_roi]
    sensor_origin = tf_snapshot.T_world_cloud[:3, 3]
    context = SensorContext.from_points(sensor_origin, pts_world)

    started = time.perf_counter()
    result = detect_blocks_detailed(pts_world, params, sensor_context=context)
    runtime_ms = (time.perf_counter() - started) * 1000.0
    detections = result.poses
    diagnostics = result.diagnostics
    suppressed_ignored = 0
    if annotations and annotations.roi_min is not None and annotations.roi_max is not None:
        # ROI membership is center-based and inclusive for both targets and detections.
        inside = [
            bool(
                np.all(pose.position >= annotations.roi_min)
                and np.all(pose.position <= annotations.roi_max)
            )
            for pose in detections
        ]
        detections = [pose for pose, keep in zip(detections, inside, strict=True) if keep]
        diagnostics = [diag for diag, keep in zip(diagnostics, inside, strict=True) if keep]
        targets = [
            pose
            for pose in targets
            if np.all(pose.position >= annotations.roi_min)
            and np.all(pose.position <= annotations.roi_max)
        ]
        ignored_targets = [
            pose
            for pose in ignored_targets
            if np.all(pose.position >= annotations.roi_min)
            and np.all(pose.position <= annotations.roi_max)
        ]
        if annotations.has_complete_roi:
            # Active targets take precedence. Only otherwise-unmatched detections
            # matching an ignored target are removed before FP/count metrics.
            active_detection_indices = {
                index for index, _ in _compatible_match(detections, targets, 0.5)
            }
            eligible = [
                (index, pose)
                for index, pose in enumerate(detections)
                if index not in active_detection_indices
            ]
            ignored_pairs = _compatible_match(
                [pose for _, pose in eligible], ignored_targets, 0.5
            )
            ignored_detection_indices = {eligible[index][0] for index, _ in ignored_pairs}
            suppressed_ignored = len(ignored_detection_indices)
            detections = [
                pose
                for index, pose in enumerate(detections)
                if index not in ignored_detection_indices
            ]
            diagnostics = [
                diag
                for index, diag in enumerate(diagnostics)
                if index not in ignored_detection_indices
            ]
    pairs = _compatible_match(detections, targets, 0.5)
    t_errors = [translation_error(detections[i], targets[j]) for i, j in pairs]
    r_errors = [rotation_error_deg(detections[i], targets[j]) for i, j in pairs]
    complete = bool(annotations and annotations.has_complete_roi)
    match_curves = {}
    for threshold in (0.05, 0.10, 0.25, 0.50):
        curve_pairs = _compatible_match(detections, targets, threshold)
        match_curves[f"{int(threshold * 100)}cm"] = {
            "matched": len(curve_pairs),
            "target_recall": len(curve_pairs) / len(targets) if targets else None,
        }

    detection_records = []
    for index, (pose, diag) in enumerate(zip(detections, diagnostics, strict=True)):
        detection_records.append({
            "index": index,
            "pose": _pose_dict(pose),
            "score": diag.evidence.score,
            "observability": diag.observability.value,
            "features": diag.features,
            "proposal_scale_m": diag.proposal_scale_m,
            "support_height_m": diag.support_height_m,
            "matched_target_index": next((j for i, j in pairs if i == index), None),
        })

    return {
        "snapshot": snap_dir.name,
        "annotation_complete": complete,
        "reference_targets": len(targets),
        "gt_blocks": len(targets) if complete else None,
        "detected": len(detections),
        "suppressed_ignored_detections": suppressed_ignored,
        "matched": len(pairs),
        "unknown_detections": None if complete else len(detections) - len(pairs),
        "false_positives": len(detections) - len(pairs) if complete else None,
        "missed_targets": len(targets) - len(pairs),
        "translation_errors_m": t_errors,
        "rotation_errors_deg": r_errors,
        "precision": len(pairs) / len(detections) if complete and detections else None,
        "recall": len(pairs) / len(targets) if complete and targets else None,
        "f1": (
            2 * len(pairs) / (len(detections) + len(targets))
            if complete and (detections or targets)
            else None
        ),
        "exact_count": len(detections) == len(targets) if complete else None,
        "match_curves": match_curves,
        "runtime_ms": runtime_ms,
        "detector_stage_counts": result.stage_counts,
        "detector_stage_timings_ms": result.stage_timings_ms,
        "detections": detection_records,
    }


def run_suite(
    snapshots_dir: Path, config_path: Path, params: DetectionParams | None = None
) -> dict:
    if params is None:
        params = DetectionParams(min_score=0.4)
    with_tf, without_tf = discover_snapshots(snapshots_dir)
    per_snapshot = []
    integrity_failures = []
    for snapshot in with_tf:
        try:
            # Keep this catch strictly around TF input preflight. Detector and
            # metric exceptions must remain visible as suite failures.
            tf_snapshot = load_tf_snapshot(snapshot / "tf.yaml")
            _ = tf_snapshot.T_world_cloud
        except (KeyError, OSError, TypeError, ValueError) as error:
            integrity_failures.append({"snapshot": snapshot.name, "reason": str(error)})
            continue
        per_snapshot.append(evaluate_snapshot(snapshot, config_path, params))
    exclusions = load_snapshot_exclusions(snapshots_dir)

    t_errors = [value for metric in per_snapshot for value in metric["translation_errors_m"]]
    r_errors = [value for metric in per_snapshot for value in metric["rotation_errors_deg"]]
    total_targets = sum(m["reference_targets"] for m in per_snapshot)
    total_det = sum(m["detected"] for m in per_snapshot)
    total_matched = sum(m["matched"] for m in per_snapshot)
    complete = [m for m in per_snapshot if m["annotation_complete"]]
    complete_det = sum(m["detected"] for m in complete)
    complete_targets = sum(m["reference_targets"] for m in complete)
    complete_matches = sum(m["matched"] for m in complete)
    match_curves = {}
    for label in ("5cm", "10cm", "25cm", "50cm"):
        matched = sum(metric["match_curves"][label]["matched"] for metric in per_snapshot)
        match_curves[label] = {
            "matched": matched,
        "known_target_recovery": matched / total_targets if total_targets else None,
        }
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None

    return {
        "n_snapshots_scored": len(per_snapshot),
        "n_snapshots_skipped_no_tf": len(without_tf),
        "skipped_snapshots": [s.name for s in without_tf],
        "n_snapshots_excluded": len(exclusions),
        "excluded_snapshots": [
            {
                "snapshot": entry.snapshot,
                "reason": entry.reason,
                "retained_snapshot": entry.retained_snapshot,
            }
            for entry in exclusions.values()
        ],
        "n_snapshots_failed_integrity": len(integrity_failures),
        "integrity_failures": integrity_failures,
        "n_snapshots_with_zero_reference_targets": sum(
            metric["reference_targets"] == 0 for metric in per_snapshot
        ),
        "schema_version": 1,
        "git_revision": git_revision,
        "detector_params": asdict(params),
        "snapshot_ids": [m["snapshot"] for m in per_snapshot],
        "n_complete_annotations": len(complete),
        "total_reference_targets": total_targets,
        "total_detected": total_det,
        "total_matched": total_matched,
        "known_target_recovery": total_matched / total_targets if total_targets else None,
        # Deprecated compatibility alias; remove after downstream artifact readers migrate.
        "registered_target_recovery": total_matched / total_targets if total_targets else None,
        "match_curves": match_curves,
        "precision": complete_matches / complete_det if complete_det else None,
        "recall": complete_matches / complete_targets if complete_targets else None,
        "false_positives": sum(m["false_positives"] or 0 for m in complete) if complete else None,
        "unknown_detections": sum(
            m["unknown_detections"] or 0
            for m in per_snapshot
            if not m["annotation_complete"]
        ),
        "n_matched_pose_errors": len(t_errors),
        "translation_p50_m": _percentile(t_errors, 50),
        "translation_p95_m": _percentile(t_errors, 95),
        "translation_max_m": max(t_errors) if t_errors else None,
        "rotation_p50_deg": _percentile(r_errors, 50),
        "rotation_p95_deg": _percentile(r_errors, 95),
        "rotation_max_deg": max(r_errors) if r_errors else None,
        "runtime_p50_ms": _percentile([m["runtime_ms"] for m in per_snapshot], 50),
        "runtime_p95_ms": _percentile([m["runtime_ms"] for m in per_snapshot], 95),
        "per_snapshot": per_snapshot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshots-dir", type=Path, default=_DEFAULT_SNAPSHOTS, help="Snapshots root directory."
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_TRANSFORM, help="Static transforms YAML fallback."
    )
    parser.add_argument("--metrics-out", type=Path, default=None, help="Write metrics JSON here.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.4,
        help="Minimum evidence score to keep a hypothesis (default: 0.4; see "
        "tools/detect_blocks.py --help for the measurement behind this default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = DetectionParams(min_score=args.min_score)
    metrics = run_suite(args.snapshots_dir, args.config, params)
    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
    if metrics["n_snapshots_scored"] == 0:
        print("\nNo tf.yaml-enabled snapshots found -- nothing scored.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
