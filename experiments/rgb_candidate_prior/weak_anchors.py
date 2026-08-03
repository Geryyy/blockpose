#!/usr/bin/env python3
"""Mine weak positive anchors from ``data/seibersdorf_dataset`` for dino_proposer/verifier.

``data/seibersdorf_dataset`` has 286 unlabelled captures (no ``registrations.yaml``,
no ``annotations.yaml``) across 24 real scenario bursts. This script turns
high-confidence, cross-frame-agreeing geometry-only detections into weak positive
anchors, per the design doc's "Offline data and weak supervision" section
(``docs/dino-rgb-prior-ros-integration.md``):

    Use high-confidence PCL/FK tracks as weak positive anchors ... Never treat
    an absent detector result as a negative label: it is unknown. Keep only
    anchors with valid sync/TF, multi-frame or multi-view agreement, strong
    PCL/ray evidence, and no ambiguous gripper occlusion.

This dataset is sparse ~1 Hz bursts per scenario, not dense video, so "multi-frame
agreement" here means: a geometry hypothesis in one capture of a scenario burst
also has a spatially-close hypothesis in at least one OTHER capture of the SAME
scenario burst (grouped via
``experiments.rgb_candidate_prior.sessions.assign_scenario_groups``, i.e. the
scenario label encoded in each capture directory name). Both the anchor
hypothesis and the neighbour it agrees with must independently clear the PCL/ray
evidence threshold below -- agreement between one confident and one weak/noisy
hypothesis is not treated as agreement.

PCL/ray evidence: the native discovery bridge's trace exposes exactly one scalar
per hypothesis, ``TraceHypothesis.score`` (``blockpose.core_types``), a
[0, 1] weighted blend of support-point count, height consistency, visible-face
coverage, free-space consistency, and ray-explained-surface fraction
(``_SCORE_WEIGHTS``). Its own docstring records the empirical split measured on
real snapshots: spurious hypotheses (e.g. gripper hardware above a block) score
under ~0.2, true blocks score over ~0.4. This script reuses that ~0.4 cut
(``--min-score``) as "strong PCL/ray evidence" -- there is no other evidence
field to draw on, and no new metric is invented here.

Multi-frame agreement position/orientation tolerances default to
``CandidateSetParams.positive_pos_tol_m`` / ``positive_rot_tol_deg``
(``experiments/rgb_candidate_prior/candidates.py``, 0.25 m / 30 deg): that is
the codebase's existing scale for "close enough to be the same block", reused
here rather than invented from scratch.

No RGB model, SAM, or hand labelling is used to generate an anchor -- every
anchor comes from the geometry-only PCL detector. RGB (``render.project_corners``
/ ``render.crop_bbox``) is used only afterwards, to compute where the anchor
lands in its own capture's image.

Known gap, stated plainly rather than silently no-op'd: this pipeline has no
occlusion signal (the same gap noted for ``render_conditioning``'s reserved B
channel in ``experiments/rgb_candidate_prior/render.py``). There is no gripper
mask, no depth-based occlusion test, nothing that could distinguish an
unoccluded anchor from a gripper-occluded one. This script does NOT filter for
gripper occlusion; it only requires geometry re-detection plus a passing image
projection. A future occlusion signal (e.g. a gripper URDF/TF-based mask) would
need to be added as a real, verifiable filter, not assumed here.

Run inside the core devcontainer (no torch needed):

    python -m experiments.rgb_candidate_prior.weak_anchors --export-root out/weak_anchors
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.core_bridge import detect_blocks_detailed_cpp  # noqa: E402
from blockpose.core_types import DetectionParameters, SensorContext, TraceHypothesis  # noqa: E402
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.io import (  # noqa: E402
    CameraModel,
    apply_transform,
    load_camera_model,
    load_pcd,
    load_rgb,
    load_tf_snapshot,
)
from blockpose.pose import rotmat_to_quat_xyzw  # noqa: E402

from experiments.rgb_candidate_prior import gates, render  # noqa: E402
from experiments.rgb_candidate_prior.sessions import assign_scenario_groups  # noqa: E402

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
DEFAULT_GROUPS_YAML = REPO_ROOT / "config" / "rgb_ablation_splits_seibersdorf.yaml"

# See module docstring: reused, not invented, scales.
DEFAULT_MIN_SCORE = 0.4
DEFAULT_AGREEMENT_POS_TOL_M = 0.25
DEFAULT_AGREEMENT_ROT_TOL_DEG = 30.0

PROVENANCE = "weak_pcl_multiframe_anchor"

_REQUIRED_FILES = ("metadata.yaml", "tf.yaml", "rgb.png", "cloud.pcd")
_EMPTY_COUNTS = {
    "hypotheses_considered": 0,
    "hypotheses_strong": 0,
    "no_agreement": 0,
    "projection_dropped": 0,
    "anchors_kept": 0,
}


@dataclass(frozen=True)
class Agreement:
    """One neighbouring capture's hypothesis that spatially agrees with an anchor."""

    capture: str
    hypothesis_id: str
    translation_error_m: float
    rotation_error_deg: float


def find_agreements(
    hypothesis: TraceHypothesis,
    others: list[tuple[str, TraceHypothesis]],
    *,
    pos_tol_m: float,
    rot_tol_deg: float,
) -> list[Agreement]:
    """Other-capture hypotheses within tolerance of ``hypothesis`` (the agreement proxy).

    ``others`` must already be pre-filtered to the same scenario group and to
    hypotheses that themselves clear the PCL/ray evidence threshold -- agreement
    is only meaningful between two independently confident observations.
    """
    agreements = []
    for capture, other in others:
        translation_m = translation_error(hypothesis.pose, other.pose)
        if translation_m > pos_tol_m:
            continue
        rotation_deg = rotation_error_deg(hypothesis.pose, other.pose)
        if not np.isfinite(rotation_deg) or rotation_deg > rot_tol_deg:
            continue
        agreements.append(Agreement(capture, other.id, translation_m, rotation_deg))
    return agreements


def _gate_and_detect(capture: Path, *, camera: CameraModel, sync_tolerance_s: float) -> dict:
    """Run the input gates and geometry-only detector for one capture.

    Always returns a dict with ``"status"``; on ``"ok"`` it also carries the
    world->camera TF, raw hypotheses, and stamp/checksum bookkeeping needed to
    write an anchor record.
    """
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
        T_world_cloud = tf.T_world_cloud
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

    pts_sensor = load_pcd(capture / "cloud.pcd")
    pts_world = apply_transform(pts_sensor, T_world_cloud)
    result = detect_blocks_detailed_cpp(
        pts_world,
        params=DetectionParameters(),
        sensor_context=SensorContext.from_points(T_world_cloud[:3, 3], pts_world),
    )
    assert result.trace is not None

    return {
        "capture": capture.name,
        "status": "ok",
        "T_world_camera": T_world_camera,
        "hypotheses": result.trace.raw_hypotheses,
        "image_stamp_ns": image_stamp_ns,
        "cloud_stamp_ns": cloud_stamp_ns,
        "sync_delta_s": sync.details["sync_delta_s"],
        "rgb_checksum": gates.sha256_file(capture / "rgb.png"),
        "tf_checksum": gates.sha256_file(capture / "tf.yaml"),
    }


def anchors_for_group(
    group: str,
    capture_outcomes: dict[str, dict[str, Any]],
    *,
    camera: CameraModel,
    calibration_checksum: str,
    min_score: float,
    agreement_pos_tol_m: float,
    agreement_rot_tol_deg: float,
    crop_margin_frac: float,
) -> tuple[list[dict], dict]:
    """Build anchor records for one scenario group's gate-clean captures.

    ``capture_outcomes`` maps capture name -> the ``"ok"`` dict from
    ``_gate_and_detect`` (only gate-clean captures; the caller accounts for
    gate failures separately). Pure geometry: no file I/O here, which is what
    makes this unit-testable with synthetic hypotheses.
    """
    strong: dict[str, list[TraceHypothesis]] = {
        name: [h for h in outcome["hypotheses"] if h.score >= min_score]
        for name, outcome in capture_outcomes.items()
    }
    flat_strong = [(name, h) for name, hyps in strong.items() for h in hyps]

    counts = dict(_EMPTY_COUNTS)
    records: list[dict] = []
    for name, outcome in capture_outcomes.items():
        counts["hypotheses_considered"] += len(outcome["hypotheses"])
        others = [(oname, oh) for oname, oh in flat_strong if oname != name]
        for hypothesis in strong[name]:
            counts["hypotheses_strong"] += 1
            agreements = find_agreements(
                hypothesis, others, pos_tol_m=agreement_pos_tol_m, rot_tol_deg=agreement_rot_tol_deg
            )
            if not agreements:
                counts["no_agreement"] += 1
                continue

            corners_world, uv, in_front = render.project_corners(
                hypothesis.pose, outcome["T_world_camera"], camera
            )
            projection = gates.projection_gate(corners_world, outcome["T_world_camera"], camera)
            bbox = (
                render.crop_bbox(uv, in_front, camera, margin_frac=crop_margin_frac)
                if projection.passed
                else None
            )
            if not projection.passed or bbox is None:
                counts["projection_dropped"] += 1
                continue

            records.append(
                {
                    "id": f"{group}/{name}/{hypothesis.id}",
                    "scenario_group": group,
                    "source_capture": name,
                    "hypothesis_id": hypothesis.id,
                    "pose_world": {
                        "translation_m": hypothesis.pose.position.tolist(),
                        "quaternion_xyzw": rotmat_to_quat_xyzw(hypothesis.pose.rotation).tolist(),
                        "dims_m": list(hypothesis.pose.dims),
                    },
                    "pcl_evidence": {"score": float(hypothesis.score), "min_score": min_score},
                    "multiframe_agreement": {
                        "position_tolerance_m": agreement_pos_tol_m,
                        "rotation_tolerance_deg": agreement_rot_tol_deg,
                        "agreement_count": len(agreements),
                        "agreeing_captures": [
                            {
                                "capture": agreement.capture,
                                "hypothesis_id": agreement.hypothesis_id,
                                "translation_error_m": agreement.translation_error_m,
                                "rotation_error_deg": agreement.rotation_error_deg,
                            }
                            for agreement in agreements
                        ],
                    },
                    "projected_region": {
                        "crop_bbox_xyxy": list(bbox),
                        "projection": projection.details,
                    },
                    "source": {
                        "image_stamp_ns": outcome["image_stamp_ns"],
                        "cloud_stamp_ns": outcome["cloud_stamp_ns"],
                        "sync_delta_s": outcome["sync_delta_s"],
                        "rgb_checksum": outcome["rgb_checksum"],
                        "tf_checksum": outcome["tf_checksum"],
                        "calibration_checksum": calibration_checksum,
                    },
                    "provenance": PROVENANCE,
                }
            )
            counts["anchors_kept"] += 1
    return records, counts


def process_group(
    group: str,
    captures: list[Path],
    *,
    camera: CameraModel,
    calibration_checksum: str,
    sync_tolerance_s: float,
    min_score: float,
    agreement_pos_tol_m: float,
    agreement_rot_tol_deg: float,
    crop_margin_frac: float,
    records: list[dict],
) -> dict:
    """Gate + detect every capture in a scenario group, then mine anchors."""
    per_capture = []
    outcomes: dict[str, dict] = {}
    for capture in captures:
        outcome = _gate_and_detect(capture, camera=camera, sync_tolerance_s=sync_tolerance_s)
        per_capture.append({key: value for key, value in outcome.items() if key != "hypotheses"})
        if outcome["status"] == "ok":
            outcomes[capture.name] = outcome

    group_records, counts = anchors_for_group(
        group,
        outcomes,
        camera=camera,
        calibration_checksum=calibration_checksum,
        min_score=min_score,
        agreement_pos_tol_m=agreement_pos_tol_m,
        agreement_rot_tol_deg=agreement_rot_tol_deg,
        crop_margin_frac=crop_margin_frac,
    )
    records.extend(group_records)
    return {"group": group, "captures": per_capture, "counts": counts}


def _load_groups(snapshots_dir: Path, groups_yaml: Path | None) -> dict[str, list[str]]:
    """Map scenario group -> capture names, preferring the checked-in manifest.

    Falls back to ``assign_scenario_groups`` for any capture the manifest
    doesn't mention (e.g. a newer capture added after the manifest was
    generated), so nothing under ``snapshots_dir`` is silently skipped.
    """
    names = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir())
    by_name: dict[str, str] = {}
    if groups_yaml is not None and groups_yaml.is_file():
        data = yaml.safe_load(groups_yaml.read_text()) or {}
        for group, info in (data.get("groups") or {}).items():
            for name in info.get("snapshots", []):
                by_name[name] = group
    missing = [name for name in names if name not in by_name]
    by_name.update(assign_scenario_groups(missing))

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(by_name[name], []).append(name)
    return groups


def _health_report(per_group: list[dict]) -> dict:
    """Summaries that make problems in a large, slow-to-eyeball dataset visible."""
    capture_status = Counter(
        capture["status"] for group in per_group for capture in group["captures"]
    )
    gate_failures = Counter(
        f"{capture['gate']}:{capture['reason']}"
        for group in per_group
        for capture in group["captures"]
        if capture["status"] == "gate_failed"
    )
    errors = [
        {"group": group["group"], "capture": capture["capture"], "error": capture["error"]}
        for group in per_group
        for capture in group["captures"]
        if capture["status"] == "error"
    ]
    totals: Counter[str] = Counter()
    for group in per_group:
        totals.update(group["counts"])

    per_group_summary = {
        group["group"]: {
            "captures": len(group["captures"]),
            "captures_ok": sum(capture["status"] == "ok" for capture in group["captures"]),
            **group["counts"],
        }
        for group in per_group
    }

    warnings: list[str] = []
    single_capture = sorted(
        name for name, info in per_group_summary.items() if info["captures_ok"] <= 1
    )
    if single_capture:
        warnings.append(
            f"{len(single_capture)} scenario group(s) have <=1 gate-clean capture, so "
            f"multi-frame agreement can never fire there: {', '.join(single_capture)}"
        )
    zero_anchor = sorted(
        name
        for name, info in per_group_summary.items()
        if info["captures_ok"] > 1 and info["anchors_kept"] == 0
    )
    if zero_anchor:
        warnings.append(
            f"{len(zero_anchor)} scenario group(s) had multiple gate-clean captures but "
            f"produced zero anchors: {', '.join(zero_anchor)}"
        )

    return {
        "capture_status": dict(capture_status),
        "gate_failures": dict(gate_failures),
        "errors": errors,
        "totals": dict(totals),
        "per_scenario_group": per_group_summary,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--groups-yaml",
        type=Path,
        default=DEFAULT_GROUPS_YAML,
        help="Precomputed scenario grouping in the rgb_ablation_splits_seibersdorf.yaml "
        "shape. Pass a non-existent path to re-derive every group with "
        "assign_scenario_groups instead.",
    )
    parser.add_argument("--sync-tolerance-s", type=float, default=0.10)
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum TraceHypothesis.score ('strong PCL/ray evidence'); see module docstring.",
    )
    parser.add_argument("--agreement-pos-tol-m", type=float, default=DEFAULT_AGREEMENT_POS_TOL_M)
    parser.add_argument(
        "--agreement-rot-tol-deg", type=float, default=DEFAULT_AGREEMENT_ROT_TOL_DEG
    )
    parser.add_argument("--crop-margin-frac", type=float, default=0.25)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N scenario groups."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_root = args.export_root
    export_root.mkdir(parents=True, exist_ok=True)

    camera = load_camera_model(args.calibration)
    calibration_checksum = gates.sha256_file(args.calibration)
    groups = _load_groups(args.snapshots_dir, args.groups_yaml)
    group_names = sorted(groups)
    if args.limit is not None:
        group_names = group_names[: args.limit]

    records: list[dict] = []
    per_group: list[dict] = []
    for group in group_names:
        captures = [args.snapshots_dir / name for name in sorted(groups[group])]
        try:
            per_group.append(
                process_group(
                    group,
                    captures,
                    camera=camera,
                    calibration_checksum=calibration_checksum,
                    sync_tolerance_s=args.sync_tolerance_s,
                    min_score=args.min_score,
                    agreement_pos_tol_m=args.agreement_pos_tol_m,
                    agreement_rot_tol_deg=args.agreement_rot_tol_deg,
                    crop_margin_frac=args.crop_margin_frac,
                    records=records,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad group must not abort the run
            per_group.append(
                {
                    "group": group,
                    "captures": [{"capture": "*", "status": "error", "error": repr(error)}],
                    "counts": dict(_EMPTY_COUNTS),
                }
            )

    (export_root / "anchors.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    health = _health_report(per_group)
    manifest = {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "snapshots_dir": str(args.snapshots_dir),
        "calibration": str(args.calibration),
        "calibration_checksum": calibration_checksum,
        "sync_tolerance_s": args.sync_tolerance_s,
        "min_score": args.min_score,
        "agreement_pos_tol_m": args.agreement_pos_tol_m,
        "agreement_rot_tol_deg": args.agreement_rot_tol_deg,
        "crop_margin_frac": args.crop_margin_frac,
        "total_anchors": len(records),
        "health": health,
    }
    (export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"total_anchors": len(records), **health}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
