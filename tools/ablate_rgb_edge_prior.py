#!/usr/bin/env python3
"""Compare raw geometry ranking with conservative classical RGB edge reranking.

This is an offline diagnostic only.  It joins ground truth after inference and
does not claim that the currently timestamp-correlated snapshots are a held-out
evaluation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.detection import (  # noqa: E402
    DetectionParams,
    SensorContext,
    detect_blocks_detailed,
)
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.io import load_camera_model, load_rgb, load_snapshot, load_tf_snapshot  # noqa: E402
from blockpose.rgb_edge_prior import build_edge_map, score_projected_cuboid  # noqa: E402

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"


def _rank(
    candidates: list[dict], *, score_band: float, edge_bands: set[int] | None = None
) -> list[dict]:
    """Keep geometry order across score bands; RGB can order only within a band."""
    best = max(item["geometry_score"] for item in candidates)
    for item in candidates:
        item["geometry_band"] = int(np.floor((best - item["geometry_score"]) / score_band))
    return sorted(
        candidates,
        key=lambda item: (
            item["geometry_band"],
            -(
                item["edge_score"]
                if edge_bands is not None
                and item["geometry_band"] in edge_bands
                and item["edge_evidence"]
                else 0.0
            ),
            -item["geometry_score"],
            item["id"],
        ),
    )


def _first_correct_rank(candidates: list[dict]) -> int | None:
    for index, candidate in enumerate(candidates):
        if candidate["correct"]:
            return index
    return None


def _load_targets(snapshot: Path):
    annotation_path = snapshot / "annotations.yaml"
    if not annotation_path.is_file():
        return []
    annotations = load_annotations(annotation_path)
    if not annotations.annotation_complete:
        return []
    return [target for target in annotations.targets if target.is_evaluation_target]


def evaluate_snapshot(
    snapshot: Path,
    *,
    calibration_path: Path,
    candidate_radius_m: float,
    score_band: float,
    edge_margin: float,
) -> dict:
    targets = _load_targets(snapshot)
    if not targets:
        return {"snapshot": snapshot.name, "status": "skipped_no_complete_annotations"}
    tf = load_tf_snapshot(snapshot / "tf.yaml")
    points, _ = load_snapshot(snapshot, T_world_cloud=tf.T_world_cloud)
    result = detect_blocks_detailed(
        points,
        DetectionParams(),
        sensor_context=SensorContext.from_points(tf.T_world_cloud[:3, 3], points),
        collect_trace=True,
    )
    assert result.trace is not None
    rgb = load_rgb(snapshot / "rgb.png")
    camera = load_camera_model(calibration_path)
    edge_started = perf_counter()
    edge_map = build_edge_map(rgb)
    edge_ms = (perf_counter() - edge_started) * 1000.0
    scored: list[dict] = []
    score_cache: dict[str, dict] = {}
    for hypothesis in result.trace.raw_hypotheses:
        score = score_projected_cuboid(edge_map, hypothesis.pose, tf.T_world_camera, camera)
        score_cache[hypothesis.id] = {
            "id": hypothesis.id,
            "geometry_score": float(hypothesis.score),
            "edge_score": score.score,
            "edge_evidence": score.evidence,
            "edge_samples": score.used_samples,
            "edge_support_fraction": score.support_fraction,
            "edge_reason": score.reason,
        }
    for target in targets:
        nearby: list[dict] = []
        for hypothesis in result.trace.raw_hypotheses:
            if translation_error(hypothesis.pose, target.pose) > candidate_radius_m:
                continue
            record = dict(score_cache[hypothesis.id])
            record["correct"] = (
                translation_error(hypothesis.pose, target.pose) <= 0.25
                and rotation_error_deg(hypothesis.pose, target.pose) <= 30.0
            )
            nearby.append(record)
        if not nearby:
            scored.append({"target": target.id, "status": "no_nearby_raw_candidate"})
            continue
        geometry = _rank([dict(item) for item in nearby], score_band=score_band)
        by_band: dict[int, list[dict]] = {}
        for candidate in geometry:
            by_band.setdefault(candidate["geometry_band"], []).append(candidate)
        edge_bands = {
            band
            for band, members in by_band.items()
            if len(
                scores := sorted(
                    (item["edge_score"] for item in members if item["edge_evidence"]), reverse=True
                )
            )
            >= 2
            and scores[0] - scores[1] >= edge_margin
        }
        edge_accepted = bool(edge_bands)
        edge = _rank([dict(item) for item in nearby], score_band=score_band, edge_bands=edge_bands)
        scored.append(
            {
                "target": target.id,
                "status": "scored",
                "candidate_count": len(nearby),
                "edge_evidence_candidates": sum(item["edge_evidence"] for item in nearby),
                "edge_accepted": edge_accepted,
                "edge_accepted_geometry_bands": sorted(edge_bands),
                "geometry_first_correct_rank": _first_correct_rank(geometry),
                "edge_first_correct_rank": _first_correct_rank(edge),
                "geometry_top_id": geometry[0]["id"],
                "edge_top_id": edge[0]["id"],
                "candidates": [
                    {
                        key: candidate[key]
                        for key in (
                            "id",
                            "geometry_score",
                            "geometry_band",
                            "edge_score",
                            "edge_evidence",
                            "edge_samples",
                            "edge_support_fraction",
                            "edge_reason",
                            "correct",
                        )
                    }
                    for candidate in geometry
                ],
            }
        )
    return {
        "snapshot": snapshot.name,
        "status": "scored",
        "raw_hypotheses": len(result.trace.raw_hypotheses),
        "edge_map_runtime_ms": edge_ms,
        "targets": scored,
    }


def _summary(records: list[dict]) -> dict:
    targets = [
        target
        for record in records
        for target in record.get("targets", [])
        if target["status"] == "scored"
    ]
    return {
        "selection_eligible": False,
        "reason": "snapshot groups are not validated independent recording runs",
        "scored_targets": len(targets),
        "geometry_top1_correct": sum(item["geometry_first_correct_rank"] == 0 for item in targets),
        "edge_top1_correct": sum(item["edge_first_correct_rank"] == 0 for item in targets),
        "edge_evidence_targets": sum(item["edge_evidence_candidates"] > 0 for item in targets),
        "edge_accepted_targets": sum(item["edge_accepted"] for item in targets),
        "edge_map_runtime_ms_mean": float(
            np.mean(
                [
                    record["edge_map_runtime_ms"]
                    for record in records
                    if record["status"] == "scored"
                ]
            )
        )
        if any(record["status"] == "scored" for record in records)
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-radius-m", type=float, default=0.50)
    parser.add_argument("--geometry-score-band", type=float, default=0.05)
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.08,
        help="Minimum best-vs-runner-up RGB edge-score separation to rerank (default: 0.08).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_radius_m <= 0.0 or args.geometry_score_band <= 0.0 or args.edge_margin < 0.0:
        raise ValueError(
            "candidate radius/score band must be positive and edge margin non-negative"
        )
    snapshots = sorted(
        path for path in args.snapshots_dir.iterdir() if (path / "tf.yaml").is_file()
    )
    records = [
        evaluate_snapshot(
            snapshot,
            calibration_path=args.calibration,
            candidate_radius_m=args.candidate_radius_m,
            score_band=args.geometry_score_band,
            edge_margin=args.edge_margin,
        )
        for snapshot in snapshots
    ]
    report = {"schema_version": 1, "records": records, "summary": _summary(records)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
