#!/usr/bin/env python3
"""Export one JSONL record per weak-anchor-bearing capture, for the ``dino_proposer`` head.

Sibling to ``export.py`` (real 23-snapshot union-silhouette exporter), NOT a
flag on it: the input schema (anchors, not approved annotation targets), the
confidence semantics (a geometry-only PCL score + cross-frame agreement
count, never a human review), and the "zero evidence must not become a
negative label" rule are different enough that one shared script would just
be a pile of conditionals.

Reads ``experiments/rgb_candidate_prior/weak_anchors.py``'s output
(``out/weak_anchors/anchors.jsonl``, one record per weak 3-D block-pose
hypothesis, ``provenance: "weak_pcl_multiframe_anchor"``), groups anchors by
their ``source_capture``, and for every ``data/seibersdorf_dataset`` capture
with >=1 anchor builds one union-silhouette weak-target mask from that
capture's own anchors' poses -- the SAME geometric construction ``export.py``
uses for real snapshots (``export.block_silhouette_mask``, itself
``render.project_corners`` + ``render._hull_polygon``), gated through the
same ``gates.py`` sync/calibration/TF checks (reused, not reimplemented).

**Why zero-anchor captures are never exported, unlike ``export.py``'s
deliberate all-background inclusion for zero-block real snapshots.** A real
snapshot with zero *approved* blocks is a positive fact: a human looked and
confirmed there is nothing there, so "no target" is honest all-background
training signal. A seibersdorf capture with zero surviving anchors is not
that -- it only means the geometry-only detector found nothing strong enough
or multi-frame-agreeing enough *this time*; per the design doc ("Never treat
an absent detector result as a negative label: it is unknown"), that absence
must never be turned into a negative/background training sample. Such
captures are skipped, and counted (not silently dropped) in the health
report.

Each exported record is written in the same field shape
``experiments.rgb_region_prior.dataset.RegionDataset``/``RegionSample``
already expects (``sample_id``, ``split_group``, ``image``, ``mask``,
``block_count``), plus ``weak_supervision: true`` and a ``weak_evidence``
block so a weak record can never be silently mistaken for a real one.
``split_group`` is each anchor's own ``scenario_group``
(``config/rgb_ablation_splits_seibersdorf.yaml``'s grouping) -- a completely
different namespace from real snapshot session ids, so weak samples can never
silently pool into a real development fold and inflate its apparent size.

Run inside the core devcontainer (no torch needed):

    python -m experiments.rgb_region_prior.export_weak \\
        --anchors out/weak_anchors/anchors.jsonl --export-root out/rgb_region_prior_weak
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.io import CameraModel, load_camera_model, load_rgb, load_tf_snapshot  # noqa: E402
from blockpose.pose import BlockPose, quat_xyzw_to_rotmat  # noqa: E402

from experiments.rgb_candidate_prior import gates, render  # noqa: E402
from experiments.rgb_region_prior.export import (  # noqa: E402
    _render_union_overlay,
    block_silhouette_mask,
)

DEFAULT_ANCHORS = REPO_ROOT / "out" / "weak_anchors" / "anchors.jsonl"
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"

_REQUIRED_FILES = ("metadata.yaml", "tf.yaml", "rgb.png")


def load_anchors(path: Path) -> list[dict]:
    """Parse ``anchors.jsonl`` into a plain list of records; empty file -> empty list."""
    anchors = []
    for line in path.read_text().splitlines():
        if line.strip():
            anchors.append(json.loads(line))
    return anchors


def group_anchors_by_capture(anchors: list[dict]) -> dict[str, list[dict]]:
    """Join key for this exporter: each anchor's ``source_capture``."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for anchor in anchors:
        grouped[anchor["source_capture"]].append(anchor)
    return dict(grouped)


def anchor_pose(anchor: dict) -> BlockPose:
    """Reconstruct the anchor's world-frame ``BlockPose`` from its recorded fields."""
    pose = anchor["pose_world"]
    return BlockPose(
        position=np.asarray(pose["translation_m"], dtype=np.float64),
        rotation=quat_xyzw_to_rotmat(np.asarray(pose["quaternion_xyzw"], dtype=np.float64)),
        dims=tuple(float(d) for d in pose["dims_m"]),
    )


def anchor_union_mask(
    anchors: list[dict], T_world_camera: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, list[str], list[str]]:
    """OR-combine every anchor's silhouette into one full-image weak-foreground mask.

    Reuses ``export.block_silhouette_mask`` -- the exact same convex-hull
    construction the real exporter uses -- so a weak union mask and a real
    union mask are geometrically consistent training targets. Returns
    ``(mask, contributing_anchor_ids, dropped_anchor_ids)``; an anchor is
    "dropped" only when its own projection gate fails at export time (a
    second, independent gating pass from the one ``weak_anchors.py`` already
    did when mining it).
    """
    combined = np.zeros((camera.height, camera.width), dtype=bool)
    contributing, dropped = [], []
    for anchor in anchors:
        mask = block_silhouette_mask(anchor_pose(anchor), T_world_camera, camera)
        if mask is None:
            dropped.append(anchor["id"])
            continue
        combined |= mask
        contributing.append(anchor["id"])
    return combined, contributing, dropped


def export_capture(
    capture: Path,
    anchors: list[dict],
    *,
    export_root: Path,
    camera: CameraModel,
    calibration_checksum: str,
    anchors_checksum: str,
    sync_tolerance_s: float,
    records: list[dict],
) -> dict:
    """Export one full-image weak union-mask record for one capture, or a status entry."""
    missing = [name for name in _REQUIRED_FILES if not (capture / name).is_file()]
    if missing:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "files",
            "reason": "missing_required_files",
            "details": {"missing": missing},
        }

    rgb = load_rgb(capture / "rgb.png")

    calib = gates.calibration_gate(camera, rgb.shape)
    if not calib.passed:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "calibration",
            "reason": calib.reason,
            "details": calib.details,
        }

    tf = load_tf_snapshot(capture / "tf.yaml")
    try:
        T_world_camera = tf.T_world_camera
    except KeyError as error:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "tf",
            "reason": "missing_transform",
            "details": {"error": str(error)},
        }
    tf_check = gates.tf_gate(T_world_camera)
    if not tf_check.passed:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "tf",
            "reason": tf_check.reason,
            "details": tf_check.details,
        }

    image_stamp_ns, cloud_stamp_ns = gates.read_stamps(capture / "metadata.yaml")
    sync = gates.sync_gate(image_stamp_ns, cloud_stamp_ns, tolerance_s=sync_tolerance_s)
    if not sync.passed:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "sync",
            "reason": sync.reason,
            "details": sync.details,
        }

    scenario_groups = sorted({anchor["scenario_group"] for anchor in anchors})
    if len(scenario_groups) != 1:
        return {
            "capture": capture.name,
            "status": "gate_failed",
            "gate": "scenario_group_consistency",
            "reason": "anchors_span_multiple_scenario_groups",
            "details": {"scenario_groups": scenario_groups},
        }
    split_group = scenario_groups[0]

    mask, contributing, dropped = anchor_union_mask(anchors, T_world_camera, camera)
    if not contributing:
        return {
            "capture": capture.name,
            "status": "skipped_all_anchors_dropped_projection",
            "split_group": split_group,
            "anchors_considered": len(anchors),
            "anchors_dropped_projection": len(dropped),
        }

    rel_image = Path("images") / f"{capture.name}.png"
    rel_mask = Path("masks") / f"{capture.name}.png"
    rel_overlay = Path("overlays") / f"{capture.name}.png"
    render.save_png(rgb, export_root / rel_image)
    render.save_png((mask.astype(np.uint8) * 255), export_root / rel_mask)
    render.save_png(_render_union_overlay(rgb, mask), export_root / rel_overlay)

    rgb_checksum = gates.sha256_file(capture / "rgb.png")
    tf_checksum = gates.sha256_file(capture / "tf.yaml")

    kept = [anchor for anchor in anchors if anchor["id"] in set(contributing)]
    pcl_scores = [float(anchor["pcl_evidence"]["score"]) for anchor in kept]
    agreement_counts = [
        int(anchor["multiframe_agreement"]["agreement_count"]) for anchor in kept
    ]

    records.append(
        {
            "sample_id": capture.name,
            "split_group": split_group,
            "image": rel_image.as_posix(),
            "mask": rel_mask.as_posix(),
            "overlay": rel_overlay.as_posix(),
            "image_shape": [camera.height, camera.width],
            "block_count": len(contributing),
            "block_ids": contributing,
            "anchors_dropped_projection": dropped,
            "weak_supervision": True,
            "weak_evidence": {
                "anchor_count": len(contributing),
                "anchor_ids": contributing,
                "pcl_scores": pcl_scores,
                "mean_pcl_score": float(np.mean(pcl_scores)) if pcl_scores else None,
                "min_pcl_score": float(np.min(pcl_scores)) if pcl_scores else None,
                "min_agreement_count": min(agreement_counts) if agreement_counts else None,
            },
            "source": {
                "capture": capture.name,
                "scenario_group": split_group,
                "image_stamp_ns": image_stamp_ns,
                "cloud_stamp_ns": cloud_stamp_ns,
                "sync_delta_s": sync.details["sync_delta_s"],
                "rgb_checksum": rgb_checksum,
                "calibration_checksum": calibration_checksum,
                "tf_checksum": tf_checksum,
                "anchors_checksum": anchors_checksum,
            },
        }
    )
    return {
        "capture": capture.name,
        "status": "exported",
        "split_group": split_group,
        "anchors_considered": len(anchors),
        "anchors_used": len(contributing),
        "anchors_dropped_projection": len(dropped),
    }


def _health_report(
    records: list[dict],
    per_capture: list[dict],
    *,
    captures_considered: int,
    zero_anchor_captures: list[str],
) -> dict:
    """Summaries that make problems in the weak-anchor pool visible, not silent."""
    status = Counter(entry["status"] for entry in per_capture)
    gate_failures = Counter(
        f"{entry['gate']}:{entry['reason']}"
        for entry in per_capture
        if entry["status"] == "gate_failed"
    )

    per_group: dict[str, dict] = {}
    for record in records:
        info = per_group.setdefault(
            record["split_group"], {"captures_exported": 0, "anchors_used": 0}
        )
        info["captures_exported"] += 1
        info["anchors_used"] += record["block_count"]
    groups = {name: info for name, info in sorted(per_group.items())}

    warnings: list[str] = []
    if zero_anchor_captures:
        warnings.append(
            f"{len(zero_anchor_captures)} capture(s) had zero mined anchors and were skipped "
            "(NOT exported as all-background: absence is not a negative label), not silently "
            f"dropped: {zero_anchor_captures}"
        )
    all_dropped = [
        entry["capture"]
        for entry in per_capture
        if entry["status"] == "skipped_all_anchors_dropped_projection"
    ]
    if all_dropped:
        warnings.append(
            f"{len(all_dropped)} capture(s) had mined anchors but every one failed the "
            f"export-time projection gate, so nothing was exported for them: {all_dropped}"
        )
    if not records:
        warnings.append("zero weak records exported: nothing to blend into training")

    return {
        "capture_status": dict(status),
        "gate_failures": dict(gate_failures),
        "captures_considered": captures_considered,
        "captures_with_anchors": captures_considered - len(zero_anchor_captures),
        "captures_zero_anchors_skipped": len(zero_anchor_captures),
        "zero_anchor_captures": zero_anchor_captures,
        "total_anchors_used": sum(record["block_count"] for record in records),
        "scenario_groups": groups,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--sync-tolerance-s", type=float, default=0.10)
    parser.add_argument(
        "--limit", type=int, default=None, help="Export only the first N anchor-bearing captures."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_root = args.export_root
    export_root.mkdir(parents=True, exist_ok=True)

    camera = load_camera_model(args.calibration)
    calibration_checksum = gates.sha256_file(args.calibration)
    anchors_checksum = gates.sha256_file(args.anchors)
    anchors = load_anchors(args.anchors)
    by_capture = group_anchors_by_capture(anchors)

    all_captures = sorted(
        p.name for p in args.snapshots_dir.iterdir() if p.is_dir()
    )
    zero_anchor_captures = [name for name in all_captures if name not in by_capture]

    capture_names = sorted(by_capture)
    if args.limit is not None:
        capture_names = capture_names[: args.limit]

    records: list[dict] = []
    per_capture: list[dict] = []
    for name in capture_names:
        capture = args.snapshots_dir / name
        try:
            per_capture.append(
                export_capture(
                    capture,
                    by_capture[name],
                    export_root=export_root,
                    camera=camera,
                    calibration_checksum=calibration_checksum,
                    anchors_checksum=anchors_checksum,
                    sync_tolerance_s=args.sync_tolerance_s,
                    records=records,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad capture must not abort the run
            per_capture.append({"capture": name, "status": "error", "error": repr(error)})

    (export_root / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    health = _health_report(
        records,
        per_capture,
        captures_considered=len(all_captures),
        zero_anchor_captures=zero_anchor_captures,
    )
    manifest = {
        "schema_version": 1,
        "provenance": "weak_pcl_multiframe_anchor",
        "weak_supervision": True,
        "anchors_path": str(args.anchors),
        "anchors_checksum": anchors_checksum,
        "selection_eligible": False,
        "reason": "weak anchors are geometry-only pseudo-labels, never clean ground truth, and "
        "the underlying capture grouping is a scenario-name prefix, not a verified independent "
        "run id (see config/rgb_ablation_splits_seibersdorf.yaml)",
        "sync_tolerance_s": args.sync_tolerance_s,
        "total_records": len(records),
        "health": health,
        "captures": per_capture,
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "total_records": len(records),
                **health,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
