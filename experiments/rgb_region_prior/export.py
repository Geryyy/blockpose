#!/usr/bin/env python3
"""Export one JSONL record per snapshot for the ``dino_proposer`` region-target baseline.

Unlike ``experiments.rgb_candidate_prior.export`` (one record per *candidate*,
a per-crop classification/ranking shape), this exports one record per
*snapshot*: a single full-image UNION foreground mask over every approved
block, since the proposer's job is "is there a block anywhere in this
region", not "which specific block is this".

**Why the projected 3-D box footprint, not the stored per-candidate
segmentation masks.** ``metadata.yaml``'s ``candidate_masks`` entries are
keyed by ``detection_id``, and ``blockpose.annotations.AnnotationTarget`` has
no ``detection_id`` field -- there is no honest, non-guessed way to associate
an approved block back to one specific stored mask (see
``blockpose.rgb_annotation.load_candidate_masks``). Using the projected
3-D box's own convex-hull silhouette (``render.project_corners`` +
``render._hull_polygon``, the same construction ``render.render_conditioning``
uses per-candidate) instead sidesteps that guess entirely: every approved
block's own registered pose already gives an honest image footprint. This is
a deliberate v1 choice, not an oversight, and it means the region target is a
convex approximation of each block's silhouette (a loose box, not a tight
segmentation) -- consistent with ``dino_proposer`` being a high-recall *region*
proposer, never a precise mask.

**Occlusion filtering, resolved.** ``--occluded-visibilities`` controls
whether blocks annotated ``heavy_occlusion``/``unknown`` enter the union
target; ``--occluded-visibilities none`` includes every approved block, and
that is now the default. The two arms were measured against each other on
19-fold leave-one-scenario-out cross-validation over
``data/seibersdorf_dataset`` (see
``experiments/pose_quality_and_partial_observation.md``): excluding them held
overall recall flat (46/62 either way) but halved recall on exactly the class
this head exists to rescue (``heavy_occlusion`` 4/10 including them vs. 2/10
excluding), and excluding them roughly quadrupled false positives (8 vs. 2).
Pass ``--occluded-visibilities heavy_occlusion,unknown`` to reproduce the old
default.

Run inside the core devcontainer (no torch needed):

    python -m experiments.rgb_region_prior.export --export-root out/rgb_region_prior
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotations import (  # noqa: E402
    AnnotationTarget,
    SnapshotAnnotations,
    load_annotations,
)
from blockpose.io import CameraModel, load_camera_model, load_rgb, load_tf_snapshot  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402

from experiments.rgb_candidate_prior import gates, render  # noqa: E402

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"

# See module docstring "Occlusion filtering". Mirrors the real annotation
# vocabulary (full / partial / heavy_occlusion / unknown); "partial" stays
# judgeable.
OCCLUDED_VISIBILITIES = ("heavy_occlusion", "unknown")


def approved_targets(
    annotations: SnapshotAnnotations, occluded_visibilities: tuple[str, ...] = OCCLUDED_VISIBILITIES
) -> list[AnnotationTarget]:
    """Human-approved, non-ignored, non-occluded targets eligible for the region union."""
    return [
        target
        for target in annotations.targets
        if target.is_evaluation_target and target.visibility not in occluded_visibilities
    ]


def block_silhouette_mask(
    pose: BlockPose, T_world_camera: np.ndarray, camera: CameraModel
) -> np.ndarray | None:
    """``(H, W)`` bool convex-hull silhouette of one posed cuboid, or ``None`` if ungated.

    Mirrors ``render.render_conditioning``'s G-channel silhouette (convex hull
    of the in-front projected corners), rendered at full image resolution
    instead of inside a per-candidate crop. Returns ``None`` when the
    projection gate fails (e.g. every corner behind the camera) or fewer than
    3 corners are in front to form a hull.
    """
    corners_world, uv, in_front = render.project_corners(pose, T_world_camera, camera)
    projection = gates.projection_gate(corners_world, T_world_camera, camera)
    if not projection.passed:
        return None
    polygon = render._hull_polygon(uv[in_front])
    if polygon is None:
        return None
    canvas = Image.new("L", (camera.width, camera.height), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    return np.asarray(canvas, dtype=bool)


def union_mask(
    targets: list[AnnotationTarget], T_world_camera: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, list[str], list[str]]:
    """OR-combine every approved block's silhouette into one full-image foreground mask.

    Returns ``(mask, contributing_block_ids, dropped_block_ids)``. A block is
    "dropped" only when its own projection gate fails -- never merely because
    it is small or near an edge -- so a bad projection is visible in the
    exported record instead of silently shrinking the union.
    """
    combined = np.zeros((camera.height, camera.width), dtype=bool)
    contributing, dropped = [], []
    for target in targets:
        mask = block_silhouette_mask(target.pose, T_world_camera, camera)
        if mask is None:
            dropped.append(target.id)
            continue
        combined |= mask
        contributing.append(target.id)
    return combined, contributing, dropped


def _render_union_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full-resolution RGB with the union mask alpha-blended in green, for audit."""
    overlay = np.asarray(rgb, dtype=np.float64).copy()
    alpha = 0.35
    green = np.zeros_like(overlay)
    green[..., 1] = 255.0
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * green[mask]
    return overlay.astype(np.uint8)


def export_snapshot(
    snapshot: Path,
    *,
    export_root: Path,
    camera: CameraModel,
    calibration_checksum: str,
    split_group: str,
    sync_tolerance_s: float,
    records: list[dict],
    occluded_visibilities: tuple[str, ...] = OCCLUDED_VISIBILITIES,
) -> dict:
    """Export one full-image union-mask record for one snapshot, or a status entry."""
    annotation_path = snapshot / "annotations.yaml"
    if not annotation_path.is_file():
        return {"snapshot": snapshot.name, "status": "skipped_no_annotations"}
    annotations = load_annotations(annotation_path)
    if not annotations.annotation_complete:
        return {"snapshot": snapshot.name, "status": "skipped_annotation_incomplete"}

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

    targets = approved_targets(annotations, occluded_visibilities)
    mask, contributing, dropped = union_mask(targets, tf.T_world_camera, camera)
    excluded_occluded = [
        target.id
        for target in annotations.targets
        if target.is_evaluation_target and target.visibility in occluded_visibilities
    ]

    rel_image = Path("images") / f"{snapshot.name}.png"
    rel_mask = Path("masks") / f"{snapshot.name}.png"
    rel_overlay = Path("overlays") / f"{snapshot.name}.png"
    render.save_png(rgb, export_root / rel_image)
    render.save_png((mask.astype(np.uint8) * 255), export_root / rel_mask)
    render.save_png(_render_union_overlay(rgb, mask), export_root / rel_overlay)

    rgb_checksum = gates.sha256_file(snapshot / "rgb.png")
    tf_checksum = gates.sha256_file(snapshot / "tf.yaml")
    annotations_checksum = gates.sha256_file(annotation_path)

    records.append(
        {
            "sample_id": snapshot.name,
            "split_group": split_group,
            "image": rel_image.as_posix(),
            "mask": rel_mask.as_posix(),
            "overlay": rel_overlay.as_posix(),
            "image_shape": [camera.height, camera.width],
            "block_count": len(contributing),
            "block_ids": contributing,
            "blocks_dropped_projection": dropped,
            "blocks_excluded_occluded": excluded_occluded,
            "source": {
                "snapshot": snapshot.name,
                "image_stamp_ns": image_stamp_ns,
                "cloud_stamp_ns": cloud_stamp_ns,
                "sync_delta_s": sync.details["sync_delta_s"],
                "rgb_checksum": rgb_checksum,
                "calibration_checksum": calibration_checksum,
                "tf_checksum": tf_checksum,
                "annotations_checksum": annotations_checksum,
            },
        }
    )
    return {
        "snapshot": snapshot.name,
        "status": "exported",
        "split_group": split_group,
        "targets_annotated": len(annotations.targets),
        "targets_approved": len(targets),
        "block_count": len(contributing),
        "blocks_dropped_projection": len(dropped),
        "blocks_excluded_occluded": len(excluded_occluded),
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
    zero_block_snapshots = [record["sample_id"] for record in records if record["block_count"] == 0]

    per_session: dict[str, dict] = {}
    for record in records:
        info = per_session.setdefault(record["split_group"], {"snapshots": 0, "blocks": 0})
        info["snapshots"] += 1
        info["blocks"] += record["block_count"]
    sessions = {name: info for name, info in sorted(per_session.items())}

    warnings: list[str] = []
    if len(sessions) < 3:
        warnings.append(f"only {len(sessions)} sessions: held-out recall will be high-variance")
    for name, info in sessions.items():
        if info["blocks"] == 0:
            warnings.append(
                f"{name} has zero approved blocks across all its snapshots: unusable for recall"
            )
    if zero_block_snapshots:
        warnings.append(
            f"{len(zero_block_snapshots)} snapshot(s) exported with zero approved blocks "
            f"(flagged, not dropped -- still valid all-background training data): "
            f"{zero_block_snapshots}"
        )
    return {
        "snapshot_status": dict(status),
        "gate_failures": dict(gate_failures),
        "errors": errors,
        "zero_block_snapshots": zero_block_snapshots,
        "total_blocks": sum(record["block_count"] for record in records),
        "sessions": sessions,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument(
        "--occluded-visibilities",
        default="none",
        help="Comma-separated visibility labels to EXCLUDE from the union target, or "
        "'none' (the default) to include every approved block -- measured better in "
        "experiments/pose_quality_and_partial_observation.md. Pass "
        f"'{','.join(OCCLUDED_VISIBILITIES)}' to reproduce the old (v1) default.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="YAML mapping snapshot name -> session id for honest held-out splits. "
        "Without it, each snapshot is its own placeholder group (NOT a valid split).",
    )
    parser.add_argument("--sync-tolerance-s", type=float, default=0.10)
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

    # "none" means keep every approved block; the default preserves v1 behaviour.
    occluded = (
        ()
        if args.occluded_visibilities.strip().lower() == "none"
        else tuple(v.strip() for v in args.occluded_visibilities.split(",") if v.strip())
    )
    camera = load_camera_model(args.calibration)
    calibration_checksum = gates.sha256_file(args.calibration)
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
                    camera=camera,
                    calibration_checksum=calibration_checksum,
                    split_group=sessions.get(snapshot.name, snapshot.name),
                    sync_tolerance_s=args.sync_tolerance_s,
                    records=records,
                    occluded_visibilities=occluded,
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
        "schema_version": 1,
        "session_grouping": "sessions_yaml" if sessions else "per_snapshot_placeholder",
        "selection_eligible": False,
        "reason": "held-out split requires independent recording sessions",
        "occluded_visibilities": list(occluded),
        "sync_tolerance_s": args.sync_tolerance_s,
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
