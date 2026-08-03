#!/usr/bin/env python3
"""Export one JSONL record per RGB candidate for the offline candidate-prior baseline.

Captured snapshots and annotations are external, read-only inputs.  For each
annotated block this generates a labelled candidate set (real geometry siblings
plus controlled perturbations), renders an RGB crop, a conditioning image, and an
audit overlay, and writes a JSONL record with full provenance.  Samples that fail
an input gate are dropped and counted, never guessed.

Run inside the core devcontainer (no torch needed):

    python -m experiments.rgb_candidate_prior.export --export-root out/rgb_candidate_prior
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.core_bridge import detect_blocks_detailed_cpp  # noqa: E402
from blockpose.core_types import DetectionParameters, SensorContext  # noqa: E402
from blockpose.io import (  # noqa: E402
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import rotmat_to_quat_xyzw  # noqa: E402

from experiments.rgb_candidate_prior import gates, render  # noqa: E402
from experiments.rgb_candidate_prior.candidates import (  # noqa: E402
    CandidateSetParams,
    build_candidate_set,
)

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"


def _evaluation_targets(snapshot: Path):
    annotation_path = snapshot / "annotations.yaml"
    if not annotation_path.is_file():
        return []
    annotations = load_annotations(annotation_path)
    if not annotations.annotation_complete:
        return []
    return [target for target in annotations.targets if target.is_evaluation_target]


def _raw_candidates(snapshot: Path, T_world_cloud: np.ndarray) -> list[tuple[str, object]]:
    points, _ = load_snapshot(snapshot, T_world_cloud=T_world_cloud)
    result = detect_blocks_detailed_cpp(
        points,
        params=DetectionParameters(),
        sensor_context=SensorContext.from_points(T_world_cloud[:3, 3], points),
    )
    assert result.trace is not None
    return [(h.id, h.pose) for h in result.trace.raw_hypotheses]


def export_snapshot(
    snapshot: Path,
    *,
    export_root: Path,
    calibration_path: Path,
    calibration_checksum: str,
    split_group: str,
    candidate_params: CandidateSetParams,
    crop_margin_frac: float,
    out_size: int,
    sync_tolerance_s: float,
    seed: int,
    records: list[dict],
) -> dict:
    """Export every gated candidate for one snapshot; append records in place."""
    targets = _evaluation_targets(snapshot)
    if not targets:
        return {"snapshot": snapshot.name, "status": "skipped_no_complete_annotations"}

    camera = load_camera_model(calibration_path)
    rgb = load_rgb(snapshot / "rgb.png")

    calib = gates.calibration_gate(camera, rgb.shape)
    if not calib.passed:
        return {
            "snapshot": snapshot.name,
            "status": "gate_failed",
            "gate": "calibration",
            "reason": calib.reason,
            "details": calib.details,
        }

    tf = load_tf_snapshot(snapshot / "tf.yaml")
    tf_check = gates.tf_gate(tf.T_world_camera)
    if not tf_check.passed:
        return {
            "snapshot": snapshot.name,
            "status": "gate_failed",
            "gate": "tf",
            "reason": tf_check.reason,
            "details": tf_check.details,
        }

    image_stamp_ns, cloud_stamp_ns = gates.read_stamps(snapshot / "metadata.yaml")
    sync = gates.sync_gate(image_stamp_ns, cloud_stamp_ns, tolerance_s=sync_tolerance_s)
    if not sync.passed:
        return {
            "snapshot": snapshot.name,
            "status": "gate_failed",
            "gate": "sync",
            "reason": sync.reason,
            "details": sync.details,
        }

    rgb_checksum = gates.sha256_file(snapshot / "rgb.png")
    tf_checksum = gates.sha256_file(snapshot / "tf.yaml")
    raw = _raw_candidates(snapshot, tf.T_world_cloud)
    rng = np.random.default_rng(seed)

    counts = {"written": 0, "gate_dropped": 0, "labels": {}}
    for target in targets:
        candidate_set = build_candidate_set(
            f"{snapshot.name}/{target.id}",
            target.pose,
            target.visibility,
            raw,
            candidate_params,
            rng,
        )

        # One target is a ranking set.  Every candidate must see identical RGB
        # pixels; otherwise its pose-dependent crop can become a shortcut for
        # the classifier.  Build the shared ROI from the target GT plus every
        # candidate that cleared projection.  Candidate-specific conditioning
        # and audit overlay remain rendered in this common ROI.
        target_corners, target_uv, target_in_front = render.project_corners(
            target.pose, tf.T_world_camera, camera
        )
        target_projection = gates.projection_gate(target_corners, tf.T_world_camera, camera)
        target_bbox = (
            render.crop_bbox(target_uv, target_in_front, camera, margin_frac=crop_margin_frac)
            if target_projection.passed
            else None
        )
        prepared: list[tuple[object, object, tuple[int, int, int, int]]] = []
        crop_boxes = [target_bbox] if target_bbox is not None else []
        for candidate in candidate_set:
            _, uv, in_front = render.project_corners(candidate.pose, tf.T_world_camera, camera)
            corners_world = render.box_corners(candidate.pose, candidate.pose.dims)
            projection = gates.projection_gate(corners_world, tf.T_world_camera, camera)
            if not projection.passed:
                counts["gate_dropped"] += 1
                continue
            bbox = render.crop_bbox(uv, in_front, camera, margin_frac=crop_margin_frac)
            if bbox is None:
                counts["gate_dropped"] += 1
                continue

            prepared.append((candidate, projection, bbox))
            crop_boxes.append(bbox)
        shared_bbox = render.union_crop_bbox(crop_boxes, camera)
        if shared_bbox is None:
            counts["gate_dropped"] += len(prepared)
            continue

        for candidate, projection, _ in prepared:
            sample_id = candidate.candidate_id
            rel_rgb = Path("images") / snapshot.name / f"{sample_id.replace('/', '__')}.png"
            rel_cond = Path("renders") / snapshot.name / f"{sample_id.replace('/', '__')}.png"
            rel_overlay = Path("overlays") / snapshot.name / f"{sample_id.replace('/', '__')}.png"
            render.save_png(
                render.render_crop(rgb, shared_bbox, out_size=out_size), export_root / rel_rgb
            )
            render.save_png(
                render.render_conditioning(
                    candidate.pose, tf.T_world_camera, camera, shared_bbox, out_size=out_size
                ),
                export_root / rel_cond,
            )
            render.save_png(
                render.render_overlay(rgb, candidate.pose, tf.T_world_camera, camera, shared_bbox),
                export_root / rel_overlay,
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "split_group": split_group,
                    "rgb": rel_rgb.as_posix(),
                    "conditioning": rel_cond.as_posix(),
                    "overlay": rel_overlay.as_posix(),
                    "candidate_pose_world": {
                        "translation_m": candidate.pose.position.tolist(),
                        "quaternion_xyzw": rotmat_to_quat_xyzw(candidate.pose.rotation).tolist(),
                    },
                    "label": candidate.label,
                    "provenance": candidate.provenance,
                    "gt_provenance": target.provenance,
                    "gt_visibility": target.visibility,
                    "label_details": candidate.details,
                    "crop_bbox_xyxy": list(shared_bbox),
                    "projection": projection.details,
                    "source": {
                        "snapshot": snapshot.name,
                        "image_stamp_ns": image_stamp_ns,
                        "cloud_stamp_ns": cloud_stamp_ns,
                        "sync_delta_s": sync.details["sync_delta_s"],
                        "rgb_checksum": rgb_checksum,
                        "calibration_checksum": calibration_checksum,
                        "tf_checksum": tf_checksum,
                    },
                }
            )
            counts["written"] += 1
            counts["labels"][candidate.label] = counts["labels"].get(candidate.label, 0) + 1

    return {
        "snapshot": snapshot.name,
        "status": "exported",
        "split_group": split_group,
        "raw_hypotheses": len(raw),
        "targets": len(targets),
        **counts,
    }


def _health_report(records: list[dict], per_snapshot: list[dict]) -> dict:
    """Summaries that make problems in newly-added data visible, not silent."""
    from collections import Counter

    status = Counter(entry["status"] for entry in per_snapshot)
    gate_failures = Counter(
        f"{entry['gate']}:{entry['reason']}"
        for entry in per_snapshot
        if entry["status"] == "gate_failed"
    )
    errors = [
        {"snapshot": entry["snapshot"], "error": entry["error"]}
        for entry in per_snapshot
        if entry["status"] == "error"
    ]
    per_session: dict[str, dict] = {}
    for record in records:
        session = per_session.setdefault(
            record["split_group"], {"blocks": set(), "labels": Counter()}
        )
        session["blocks"].add(record["sample_id"].rsplit("/", 1)[0])
        session["labels"][record["label"]] += 1
    sessions = {
        name: {"blocks": len(info["blocks"]), "labels": dict(info["labels"])}
        for name, info in sorted(per_session.items())
    }

    warnings: list[str] = []
    if len(sessions) < 3:
        warnings.append(f"only {len(sessions)} sessions: held-out AUROC will be high-variance")
    for name, info in sessions.items():
        if info["labels"].get("positive", 0) == 0 or info["labels"].get("negative", 0) == 0:
            warnings.append(f"{name} has a single-class label set: not usable as a held-out fold")
    return {
        "snapshot_status": dict(status),
        "gate_failures": dict(gate_failures),
        "errors": errors,
        "label_totals": dict(Counter(record["label"] for record in records)),
        "gt_provenance_totals": dict(Counter(record["gt_provenance"] for record in records)),
        "sessions": sessions,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="YAML mapping snapshot name -> session id for honest held-out splits. "
        "Without it, each snapshot is its own placeholder group (NOT a valid split).",
    )
    parser.add_argument("--crop-margin-frac", type=float, default=0.25)
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument("--sync-tolerance-s", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="Export only the first N snapshots."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_root = args.export_root
    export_root.mkdir(parents=True, exist_ok=True)
    sessions: dict[str, str] = {}
    if args.sessions is not None:
        sessions = yaml.safe_load(args.sessions.read_text()) or {}

    calibration_checksum = gates.sha256_file(args.calibration)
    candidate_params = CandidateSetParams()
    snapshots = sorted(p for p in args.snapshots_dir.iterdir() if (p / "tf.yaml").is_file())
    if args.limit is not None:
        snapshots = snapshots[: args.limit]

    records: list[dict] = []
    per_snapshot: list[dict] = []
    for snapshot in snapshots:
        try:
            per_snapshot.append(
                export_snapshot(
                    snapshot,
                    export_root=export_root,
                    calibration_path=args.calibration,
                    calibration_checksum=calibration_checksum,
                    split_group=sessions.get(snapshot.name, snapshot.name),
                    candidate_params=candidate_params,
                    crop_margin_frac=args.crop_margin_frac,
                    out_size=args.out_size,
                    sync_tolerance_s=args.sync_tolerance_s,
                    seed=args.seed,
                    records=records,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad capture must not abort the run
            per_snapshot.append(
                {"snapshot": snapshot.name, "status": "error", "error": repr(error)}
            )

    (export_root / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    health = _health_report(records, per_snapshot)
    manifest = {
        "schema_version": 2,
        "session_grouping": "sessions_yaml" if sessions else "per_snapshot_placeholder",
        "selection_eligible": False,
        "reason": "held-out split requires independent recording sessions",
        "candidate_params": asdict(candidate_params),
        "sync_tolerance_s": args.sync_tolerance_s,
        "crop_margin_frac": args.crop_margin_frac,
        "out_size": args.out_size,
        "total_records": len(records),
        "health": health,
        "snapshots": per_snapshot,
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "total_records": len(records),
                "session_grouping": manifest["session_grouping"],
                **health,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
