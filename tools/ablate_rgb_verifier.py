#!/usr/bin/env python3
"""Compare raw geometry ranking with bounded DINOv2 `dino_verifier` reranking.

This is an offline diagnostic only, mirroring `ablate_rgb_edge_prior.py`'s
shape and its band-bounded reranking rule (RGB may only reorder candidates
within a geometry score band, never override geometry across bands) for the
learned `dino_verifier` head instead of the classical edge prior.  It joins
ground truth after inference and does not claim that the currently
timestamp-correlated snapshots are a held-out evaluation set.
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
sys.path.insert(0, str(REPO_ROOT))  # so `experiments.rgb_candidate_prior.render` resolves too
sys.path.insert(0, str(REPO_ROOT / "tools"))  # so `ablate_rgb_edge_prior` resolves too

from ablate_rgb_edge_prior import _first_correct_rank, _load_targets  # noqa: E402
from blockpose.detection import (  # noqa: E402
    DetectionParams,
    SensorContext,
    detect_blocks_detailed,
)
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.io import (  # noqa: E402
    CameraModel,
    TFSnapshot,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.rgb_verifier import (  # noqa: E402
    DinoVerifierConfig,
    DinoVerifierHead,
    load_dino_verifier,
    score_candidate_with_dino,
)

from experiments.rgb_candidate_prior.render import (  # noqa: E402
    crop_bbox,
    project_corners,
    render_conditioning,
    render_crop,
    union_crop_bbox,
)

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
CROP_MARGIN_FRAC = 0.2
CROP_OUT_SIZE = 224


def _rank(
    candidates: list[dict], *, score_band: float, dino_bands: set[int] | None = None
) -> list[dict]:
    """Keep geometry order across score bands; DINO can order only within a band."""
    best = max(item["geometry_score"] for item in candidates)
    for item in candidates:
        item["geometry_band"] = int(np.floor((best - item["geometry_score"]) / score_band))
    return sorted(
        candidates,
        key=lambda item: (
            item["geometry_band"],
            -(
                item["dino_contribution"]
                if dino_bands is not None
                and item["geometry_band"] in dino_bands
                and not item["dino_abstained"]
                else 0.0
            ),
            -item["geometry_score"],
            item["id"],
        ),
    )


def _candidate_bbox(
    pose: BlockPose, tf: TFSnapshot, camera: CameraModel
) -> tuple[int, int, int, int] | None:
    _, uv, in_front = project_corners(pose, tf.T_world_camera, camera)
    return crop_bbox(uv, in_front, camera, margin_frac=CROP_MARGIN_FRAC)


def evaluate_snapshot(
    snapshot: Path,
    *,
    head: DinoVerifierHead,
    config: DinoVerifierConfig,
    calibration_temperature: float,
    camera: CameraModel,
    candidate_radius_m: float,
    score_band: float,
    dino_margin: float,
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
    dino_ms_total = 0.0
    scored: list[dict] = []
    for target in targets:
        nearby_hypotheses = [
            hypothesis
            for hypothesis in result.trace.raw_hypotheses
            if translation_error(hypothesis.pose, target.pose) <= candidate_radius_m
        ]
        if not nearby_hypotheses:
            scored.append({"target": target.id, "status": "no_nearby_raw_candidate"})
            continue

        # Every candidate competing for this target must see identical RGB
        # pixels (mirrors experiments/rgb_candidate_prior/export.py's shared
        # crop): otherwise a pose-dependent crop framing becomes a shortcut
        # the trained head never saw during training, tainting this ablation.
        candidate_bboxes = [
            bbox
            for hypothesis in nearby_hypotheses
            if (bbox := _candidate_bbox(hypothesis.pose, tf, camera)) is not None
        ]
        shared_bbox = union_crop_bbox(candidate_bboxes, camera)
        if shared_bbox is None:
            scored.append({"target": target.id, "status": "no_projectable_candidate"})
            continue
        crop = render_crop(rgb, shared_bbox, out_size=CROP_OUT_SIZE)

        dino_started = perf_counter()
        nearby: list[dict] = []
        for hypothesis in nearby_hypotheses:
            conditioning = render_conditioning(
                hypothesis.pose, tf.T_world_camera, camera, shared_bbox, out_size=CROP_OUT_SIZE
            )
            score = score_candidate_with_dino(
                head,
                config,
                crop,
                conditioning,
                calibration_temperature=calibration_temperature,
            )
            nearby.append(
                {
                    "id": hypothesis.id,
                    "geometry_score": float(hypothesis.score),
                    "dino_contribution": score.contribution,
                    "dino_raw_score": score.raw_score,
                    "dino_calibrated_score": score.calibrated_score,
                    "dino_uncertainty": score.uncertainty,
                    "dino_abstained": score.abstained,
                    "dino_reason": score.reason,
                    "correct": (
                        translation_error(hypothesis.pose, target.pose) <= 0.25
                        and rotation_error_deg(hypothesis.pose, target.pose) <= 30.0
                    ),
                }
            )
        dino_ms_total += (perf_counter() - dino_started) * 1000.0
        geometry = _rank([dict(item) for item in nearby], score_band=score_band)
        by_band: dict[int, list[dict]] = {}
        for candidate in geometry:
            by_band.setdefault(candidate["geometry_band"], []).append(candidate)
        dino_bands = {
            band
            for band, members in by_band.items()
            if len(
                scores := sorted(
                    (item["dino_contribution"] for item in members if not item["dino_abstained"]),
                    reverse=True,
                )
            )
            >= 2
            and scores[0] - scores[1] >= dino_margin
        }
        dino_accepted = bool(dino_bands)
        dino = _rank([dict(item) for item in nearby], score_band=score_band, dino_bands=dino_bands)
        scored.append(
            {
                "target": target.id,
                "status": "scored",
                "candidate_count": len(nearby),
                "dino_evidence_candidates": sum(not item["dino_abstained"] for item in nearby),
                "dino_accepted": dino_accepted,
                "dino_accepted_geometry_bands": sorted(dino_bands),
                "geometry_first_correct_rank": _first_correct_rank(geometry),
                "dino_first_correct_rank": _first_correct_rank(dino),
                "geometry_top_id": geometry[0]["id"],
                "dino_top_id": dino[0]["id"],
                "candidates": [
                    {
                        key: candidate[key]
                        for key in (
                            "id",
                            "geometry_score",
                            "geometry_band",
                            "dino_contribution",
                            "dino_abstained",
                            "dino_calibrated_score",
                            "dino_uncertainty",
                            "dino_reason",
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
        "dino_runtime_ms": dino_ms_total,
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
        "dino_top1_correct": sum(item["dino_first_correct_rank"] == 0 for item in targets),
        "dino_evidence_targets": sum(item["dino_evidence_candidates"] > 0 for item in targets),
        "dino_accepted_targets": sum(item["dino_accepted"] for item in targets),
        "dino_runtime_ms_mean": float(
            np.mean(
                [record["dino_runtime_ms"] for record in records if record["status"] == "scored"]
            )
        )
        if any(record["status"] == "scored" for record in records)
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="dino_verifier checkpoint directory (see blockpose.rgb_verifier.save_checkpoint).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-radius-m", type=float, default=0.50)
    parser.add_argument("--geometry-score-band", type=float, default=0.05)
    parser.add_argument(
        "--dino-margin",
        type=float,
        default=0.08,
        help="Minimum best-vs-runner-up DINO contribution separation to rerank (default: 0.08).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_radius_m <= 0.0 or args.geometry_score_band <= 0.0 or args.dino_margin < 0.0:
        raise ValueError(
            "candidate radius/score band must be positive and dino margin non-negative"
        )
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(
            f"--checkpoint {args.checkpoint} does not exist. A dino_verifier checkpoint "
            "(a directory written by blockpose.rgb_verifier.save_checkpoint) is required -- "
            "this tool refuses to silently fall back to a geometry-only no-op."
        )
    head, config, calibration_temperature = load_dino_verifier(args.checkpoint)
    camera = load_camera_model(args.calibration)
    snapshots = sorted(
        path for path in args.snapshots_dir.iterdir() if (path / "tf.yaml").is_file()
    )
    records = [
        evaluate_snapshot(
            snapshot,
            head=head,
            config=config,
            calibration_temperature=calibration_temperature,
            camera=camera,
            candidate_radius_m=args.candidate_radius_m,
            score_band=args.geometry_score_band,
            dino_margin=args.dino_margin,
        )
        for snapshot in snapshots
    ]
    report = {"schema_version": 1, "records": records, "summary": _summary(records)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
