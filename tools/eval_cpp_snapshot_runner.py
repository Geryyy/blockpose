#!/usr/bin/env python3
"""Score C++ snapshot-runner JSON against reviewed Blockpose annotations.

The C++ runner owns inference on full clouds. This script only reads its
already-produced poses, then opens annotations to apply the complete ROI and
one-to-one scoring association. It therefore cannot crop or otherwise leak
annotation data into C++ scene discovery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.mode_benchmark import _compatible_match, _percentile  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.snapshot_scenarios import load_snapshot_scenarios  # noqa: E402


def _pose(value: dict) -> BlockPose:
    return BlockPose(
        position=np.asarray(value["position"], dtype=np.float64),
        rotation=np.asarray(value["rotation"], dtype=np.float64),
        dims=tuple(float(item) for item in value["dims"]),
        confidence=float(value.get("confidence", 0.0)),
        source="concrete_block_detector_snapshot_runner",
    )


def _inside(pose: BlockPose, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(np.all(pose.position >= low) and np.all(pose.position <= high))


def _cohorts(records: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        for target in record["targets"]:
            buckets.setdefault(target[field], []).append(target)
    result = {}
    for label, targets in sorted(buckets.items()):
        matched = [target for target in targets if target["matched_detection_index"] is not None]
        translation = [target["translation_error_m"] for target in matched]
        rotation = [target["rotation_error_deg"] for target in matched]
        result[label] = {
            "targets": len(targets),
            "matched": len(matched),
            "recall": len(matched) / len(targets) if targets else None,
            "translation_error_p95_m": _percentile(translation, 95),
            "rotation_error_p95_deg": _percentile(rotation, 95),
        }
    return result


def score(runner_json: dict, snapshots_dir: Path) -> dict:
    scenarios = load_snapshot_scenarios()
    records = []
    for raw in runner_json["snapshots"]:
        snapshot = snapshots_dir / raw["snapshot"]
        annotations = load_annotations(snapshot / "annotations.yaml")
        if not annotations.has_complete_roi:
            raise ValueError(f"complete annotation ROI required: {snapshot}")
        assert annotations.roi_min is not None and annotations.roi_max is not None
        low, high = annotations.roi_min, annotations.roi_max
        detections = [_pose(item) for item in raw["poses"]]
        detections = [pose for pose in detections if _inside(pose, low, high)]
        targets = [target for target in annotations.targets if target.is_evaluation_target]
        targets = [target for target in targets if _inside(target.pose, low, high)]
        pairs = _compatible_match(detections, [target.pose for target in targets], 0.50)
        by_target = {target_index: detection_index for detection_index, target_index in pairs}
        records.append(
            {
                "snapshot": raw["snapshot"],
                "runtime_ms": raw["runtime_ms"],
                "prediction_count_scored_roi": len(detections),
                "target_count_scored_roi": len(targets),
                "matched": len(pairs),
                "false_positives": len(detections) - len(pairs),
                "missed_targets": len(targets) - len(pairs),
                "exact_count": len(detections) == len(targets),
                "targets": [
                    {
                        "id": target.id,
                        "visibility": target.visibility,
                        "scenario": scenarios[f"{raw['snapshot']}/{target.id}"].scenario,
                        "pose_quality": scenarios[f"{raw['snapshot']}/{target.id}"].pose_quality,
                        "matched_detection_index": by_target.get(index),
                        "translation_error_m": (
                            translation_error(detections[by_target[index]], target.pose)
                            if index in by_target
                            else None
                        ),
                        "rotation_error_deg": (
                            rotation_error_deg(detections[by_target[index]], target.pose)
                            if index in by_target
                            else None
                        ),
                    }
                    for index, target in enumerate(targets)
                ],
            }
        )
    translation = [
        target["translation_error_m"]
        for record in records
        for target in record["targets"]
        if target["translation_error_m"] is not None
    ]
    rotation = [
        target["rotation_error_deg"]
        for record in records
        for target in record["targets"]
        if target["rotation_error_deg"] is not None
    ]
    total_predictions = sum(record["prediction_count_scored_roi"] for record in records)
    total_targets = sum(record["target_count_scored_roi"] for record in records)
    total_matched = sum(record["matched"] for record in records)
    return {
        "schema_version": 1,
        "runner": runner_json.get("runner"),
        "runner_parameter_files": runner_json.get("parameter_files", []),
        "inference_inputs": "full C++ cloud; reviewed annotation ROI applies only to scoring",
        "n_snapshots_scored": len(records),
        "total_targets": total_targets,
        "total_predictions_scored_roi": total_predictions,
        "total_matched": total_matched,
        "precision": total_matched / total_predictions if total_predictions else None,
        "recall": total_matched / total_targets if total_targets else None,
        "exact_count_rate": sum(record["exact_count"] for record in records) / len(records),
        "translation_error_p50_m": _percentile(translation, 50),
        "translation_error_p95_m": _percentile(translation, 95),
        "rotation_error_p50_deg": _percentile(rotation, 50),
        "rotation_error_p95_deg": _percentile(rotation, 95),
        "runtime_p50_ms": _percentile([record["runtime_ms"] for record in records], 50),
        "runtime_p95_ms": _percentile([record["runtime_ms"] for record in records], 95),
        "target_cohorts": {
            "scenario": _cohorts(records, "scenario"),
            "visibility": _cohorts(records, "visibility"),
            "pose_quality": _cohorts(records, "pose_quality"),
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-json", type=Path, required=True)
    parser.add_argument(
        "--snapshots-dir", type=Path, default=REPO_ROOT / "data" / "registration_snapshots"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(json.loads(args.runner_json.read_text()), args.snapshots_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "n_snapshots_scored",
                    "precision",
                    "recall",
                    "exact_count_rate",
                    "translation_error_p95_m",
                    "rotation_error_p95_deg",
                    "runtime_p95_ms",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
