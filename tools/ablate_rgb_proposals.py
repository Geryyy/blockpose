#!/usr/bin/env python3
"""Replay the bounded RGB proposal ablation on one grouped snapshot fold.

This deliberately does not tune parameters or select a winner.  It reports one
held-out fold at a time and labels registration-derived pose metrics as
provisional.  RGB masks are proposal support only: all poses below are produced
by the existing geometric detector and no output is a verification/Missing
status.

Examples:
    uv run python tools/ablate_rgb_proposals.py --fold 0 --metrics-out rgb_fold0.json
    uv run python tools/ablate_rgb_proposals.py --fold 1 --mode geometry --mode union
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, SensorContext, detect_blocks_detailed  # noqa: E402
from blockpose.evaluate import evaluate  # noqa: E402
from blockpose.io import (  # noqa: E402
    CameraModel,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.rgb_proposals import (  # noqa: E402
    DinoRegionProposalProvider,
    ImageProposalProvider,
    MaskAssociationParams,
    StoredMaskProposalProvider,
    associate_mask_to_lidar,
    load_proposal_cache,
    make_cache_provenance,
    proposal_cache_key,
    proposal_point_indices,
    save_proposal_cache,
)
from blockpose.snapshot_exclusions import load_snapshot_exclusions  # noqa: E402

DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits.yaml"
MODES = ("geometry", "hard", "soft", "union")


def load_split_manifest(path: Path) -> dict:
    """Load explicit groups; never infer independent runs from folder names."""
    data = yaml.safe_load(path.read_text()) or {}
    if data.get("schema_version") != 1 or not isinstance(data.get("groups"), dict):
        raise ValueError("RGB ablation split manifest must use schema_version 1")
    for key in ("selection_eligible", "final_enabled"):
        if type(data.get(key)) is not bool:
            raise ValueError(f"{key} must be a YAML boolean")
    final_enabled = data["final_enabled"]
    if final_enabled and (
        not data["selection_eligible"]
        or data.get("grouping_evidence") != "validated_recording_run_ids"
    ):
        raise ValueError(
            "final_enabled requires selection_eligible and validated_recording_run_ids"
        )
    seen: set[str] = set()
    run_ids: set[str] = set()
    for name, group in data["groups"].items():
        if not isinstance(group, dict) or group.get("split") not in {"development", "final"}:
            raise ValueError(f"invalid split group {name!r}")
        if final_enabled:
            run_id = group.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip() or run_id in run_ids:
                raise ValueError(
                    "final-enabled manifests require a unique explicit run_id per group"
                )
            run_ids.add(run_id)
        snapshots = group.get("snapshots")
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError(f"group {name!r} must have snapshots")
        for snapshot in snapshots:
            if not isinstance(snapshot, str) or Path(snapshot).name != snapshot or snapshot in seen:
                raise ValueError("snapshot groups must use unique direct-child names")
            seen.add(snapshot)
        if group["split"] == "development" and not isinstance(group.get("fold"), int):
            raise ValueError(f"development group {name!r} must declare an integer fold")
    return data


def discover_split_snapshots(
    root: Path, manifest: dict, *, split: str, fold: int | None
) -> list[Path]:
    """Resolve only manifest members; missing data is a visible hard error."""
    excluded = load_snapshot_exclusions(root)
    names = []
    for group in manifest["groups"].values():
        if group["split"] != split or (split == "development" and group["fold"] != fold):
            continue
        names.extend(group["snapshots"])
    if not names:
        raise ValueError(f"no {split} snapshots selected by manifest")
    unavailable = [
        name
        for name in names
        if name in excluded
        or not (root / name / "tf.yaml").is_file()
        or not (root / name / "cloud.pcd").is_file()
        or not (root / name / "rgb.png").is_file()
    ]
    if unavailable:
        raise ValueError(f"manifest snapshots unavailable/excluded: {sorted(unavailable)}")
    return [root / name for name in names]


def _dedupe_poses(poses: list, *, minimum_separation_m: float = 0.25) -> list:
    """Deterministic center-distance de-duplication for a proposal union."""
    kept = []
    for pose in poses:
        distinct = all(
            np.linalg.norm(pose.position - existing.position) > minimum_separation_m
            for existing in kept
        )
        if distinct:
            kept.append(pose)
    return kept


def _run_geometry(points: np.ndarray, origin: np.ndarray) -> tuple[list, float, str | None]:
    if len(points) < 30:
        return [], 0.0, "too_few_points"
    started = perf_counter()
    try:
        result = detect_blocks_detailed(
            points, DetectionParams(), sensor_context=SensorContext.from_points(origin, points)
        )
    except (RuntimeError, ValueError) as error:
        return [], (perf_counter() - started) * 1000.0, str(error)
    return result.poses, (perf_counter() - started) * 1000.0, None


def _metrics(detections: list, targets: list) -> dict:
    result = evaluate(detections, targets)
    return {
        "detections": len(detections),
        "registered_targets": len(targets),
        "matched": result.matched,
        "precision": result.precision if targets else None,
        "recall": result.recall if targets else None,
        "translation_p95_m": result.translation_p95_m,
        "rotation_p95_deg": result.rotation_p95_deg,
    }


def _association_stress(
    points: np.ndarray,
    transform: np.ndarray,
    camera: CameraModel,
    proposals: tuple,
    params: MaskAssociationParams,
    *,
    jitter_px: float,
) -> dict:
    """Bounded association stress; legacy snapshots cannot test time offset."""
    shifted_K = camera.K.copy()
    shifted_K[0, 2] += jitter_px
    shifted_K[1, 2] -= jitter_px
    jittered_camera = CameraModel(shifted_K, camera.width, camera.height, camera.D)
    jittered = [
        associate_mask_to_lidar(points, transform, jittered_camera, proposal, params)
        for proposal in proposals
    ]
    overlap_pairs = [
        [first, second]
        for first, left in enumerate(proposals)
        for second, right in enumerate(proposals[first + 1 :], start=first + 1)
        if np.any(left.mask & right.mask)
    ]
    return {
        "calibration_jitter": {
            "jitter_px": jitter_px,
            "hard_points": [int(len(item.hard_indices)) for item in jittered],
            "soft_points": [int(len(item.soft_indices)) for item in jittered],
            "status": "association_only_not_metric_pose",
        },
        "rgb_dropout": {
            "union_behavior": "geometry_candidate_fallback",
            "status": "tested_by_union_construction",
        },
        "overlap_ambiguity": {
            "overlapping_proposal_pairs": overlap_pairs,
            "status": "unresolved_rgb_assignment; geometry remains authoritative",
        },
        "sync_offset": {
            "status": "unavailable_legacy_snapshots_no_sensor_trajectory_or_time_offset",
            "selection_eligible": False,
        },
    }


def evaluate_snapshot(
    snapshot: Path,
    *,
    calibration_path: Path,
    modes: tuple[str, ...],
    association_params: MaskAssociationParams,
    cache_dir: Path | None,
    calibration_jitter_px: float,
    provider: ImageProposalProvider | None = None,
    provider_name: str = "stored_mask_replay",
    provider_config: dict | None = None,
) -> dict:
    """Run an inference-only proposal replay; target poses are loaded afterward.

    ``provider``/``provider_name``/``provider_config`` default to the original
    ``StoredMaskProposalProvider`` oracle/replay baseline; pass a different
    ``ImageProposalProvider`` (e.g. ``DinoRegionProposalProvider``) plus its own
    name/config to ablate a real model the same way.
    """
    tf = load_tf_snapshot(snapshot / "tf.yaml")
    points, targets = load_snapshot(snapshot, T_world_cloud=tf.T_world_cloud)
    rgb = load_rgb(snapshot / "rgb.png")
    camera = load_camera_model(calibration_path)
    if rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(f"RGB/camera shape mismatch in {snapshot.name}")
    if provider is None:
        provider = StoredMaskProposalProvider()
        provider_name = "stored_mask_replay"
        provider_config = {"source": "metadata.candidate_masks"}
    provenance = make_cache_provenance(
        snapshot,
        provider=provider_name,
        provider_config=provider_config,
        calibration={"K": camera.K.tolist(), "D": camera.D.tolist(), "shape": list(rgb.shape[:2])},
        extrinsics=tf.T_world_camera.tolist(),
        association_params=association_params,
    )
    key = proposal_cache_key(snapshot, provider_name, provenance=provenance)
    batch = None
    cache_hit = False
    if cache_dir is not None:
        cache_path = cache_dir / snapshot.name / provider_name
        batch = load_proposal_cache(cache_path, expected_key=key, expected_provenance=provenance)
        cache_hit = batch is not None
    if batch is None:
        batch = provider.propose(rgb, snapshot=snapshot)
        if cache_dir is not None:
            save_proposal_cache(cache_path, batch, cache_key=key, provenance=provenance)

    origin = tf.T_world_cloud[:3, 3]
    geometry_poses, geometry_runtime_ms, geometry_error = _run_geometry(points, origin)
    associations = [
        associate_mask_to_lidar(points, tf.T_world_camera, camera, proposal, association_params)
        for proposal in batch.proposals
    ]
    detail = {
        "snapshot": snapshot.name,
        "cache_hit": cache_hit,
        "provider": {
            "name": batch.provider,
            "runtime_ms": batch.runtime_ms,
            "provenance": batch.provenance,
        },
        "association": {
            "proposals": len(batch.proposals),
            "hard_points": [int(len(item.hard_indices)) for item in associations],
            "soft_points": [int(len(item.soft_indices)) for item in associations],
            "calibration_margin_px": [item.calibration_margin_px for item in associations],
        },
        "stress": _association_stress(
            points,
            tf.T_world_camera,
            camera,
            batch.proposals,
            association_params,
            jitter_px=calibration_jitter_px,
        ),
        "modes": {},
    }
    for mode in modes:
        if mode == "geometry":
            poses, runtime_ms, errors = geometry_poses, geometry_runtime_ms, [geometry_error]
        else:
            rgb_poses: list = []
            runtime_ms = 0.0
            errors: list[str | None] = []
            for association in associations:
                indices = proposal_point_indices(np.empty(0, dtype=int), association, mode)
                proposal_poses, elapsed_ms, error = _run_geometry(points[indices], origin)
                rgb_poses.extend(proposal_poses)
                runtime_ms += elapsed_ms
                errors.append(error)
            # A union is candidate-level, preserving the complete geometry path
            # on image dropout instead of turning absent RGB into an absence claim.
            poses = (
                _dedupe_poses(geometry_poses + rgb_poses)
                if mode == "union"
                else _dedupe_poses(rgb_poses)
            )
            if mode == "union":
                runtime_ms += geometry_runtime_ms
                errors.append(geometry_error)
        detail["modes"][mode] = {
            **_metrics(poses, targets),
            "runtime_ms": runtime_ms,
            "detector_errors": [error for error in errors if error is not None],
            "experimental_note": (
                "Masked point sets are a Python compatibility ablation; "
                "metric pose remains geometric."
            ),
        }
    return detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--split", choices=("development", "final"), default="development")
    parser.add_argument("--fold", type=int, help="required for an explicit development fold")
    parser.add_argument("--mode", choices=MODES, action="append", dest="modes")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--dilation-px", type=int, default=4)
    parser.add_argument("--calibration-sigma-px", type=float, default=2.0)
    parser.add_argument("--soft-margin-px", type=float, default=12.0)
    parser.add_argument("--calibration-jitter-px", type=float, default=3.0)
    parser.add_argument(
        "--provider",
        choices=("stored_mask_replay", "dino_region_proposer"),
        default="stored_mask_replay",
        help="stored_mask_replay is the oracle/replay baseline (default); "
        "dino_region_proposer ablates a trained checkpoint (requires --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="dino_region_proposer checkpoint dir (see train.py)"
    )
    args = parser.parse_args()
    if args.provider == "dino_region_proposer" and args.checkpoint is None:
        parser.error("--provider dino_region_proposer requires --checkpoint")
    manifest = load_split_manifest(args.splits)
    if args.split == "development" and args.fold is None:
        parser.error("--fold is required for the development split")
    if args.split == "final":
        if args.fold is not None:
            parser.error("--fold is only valid for the development split")
        if not manifest.get("final_enabled", False):
            parser.error("final split is disabled until validated recording/run IDs are supplied")
    params = MaskAssociationParams(
        dilation_px=args.dilation_px,
        calibration_sigma_px=args.calibration_sigma_px,
        soft_margin_px=args.soft_margin_px,
    )
    # Validate before touching data, including NaN user input.
    _ = params.effective_dilation_px
    if not np.isfinite(args.calibration_jitter_px) or args.calibration_jitter_px < 0:
        parser.error("--calibration-jitter-px must be finite and non-negative")
    modes = tuple(args.modes or MODES)
    try:
        snapshots = discover_split_snapshots(
            args.snapshots, manifest, split=args.split, fold=args.fold
        )
    except ValueError as error:
        parser.error(str(error))
    if args.provider == "dino_region_proposer":
        provider: ImageProposalProvider | None = DinoRegionProposalProvider(
            checkpoint=args.checkpoint
        )
        provider_name = "dino_region_proposer"
        provider_config = {"checkpoint": str(args.checkpoint)}
    else:
        provider = None  # evaluate_snapshot defaults to StoredMaskProposalProvider
        provider_name = "stored_mask_replay"
        provider_config = None
    details = [
        evaluate_snapshot(
            snapshot,
            calibration_path=args.calibration,
            modes=modes,
            association_params=params,
            cache_dir=args.cache_dir,
            calibration_jitter_px=args.calibration_jitter_px,
            provider=provider,
            provider_name=provider_name,
            provider_config=provider_config,
        )
        for snapshot in snapshots
    ]
    aggregate = {}
    for mode in modes:
        rows = [detail["modes"][mode] for detail in details]
        detections, targets, matched = (
            sum(row["detections"] for row in rows),
            sum(row["registered_targets"] for row in rows),
            sum(row["matched"] for row in rows),
        )
        aggregate[mode] = {
            "detections": detections,
            "registered_targets": targets,
            "matched": matched,
            "precision": matched / detections if detections and targets else None,
            "recall": matched / targets if targets else None,
            "runtime_p95_ms": (
                float(np.percentile([row["runtime_ms"] for row in rows], 95)) if rows else None
            ),
        }
    output = {
        "schema_version": 1,
        "purpose": "diagnostic RGB proposal ablation; not a status/covariance evaluation",
        "split": args.split,
        "fold": args.fold,
        "selection_eligible": bool(manifest.get("selection_eligible", False)),
        "split_manifest": str(args.splits),
        "grouping_evidence": manifest.get("grouping_evidence"),
        "modes": modes,
        "association_params": asdict(params),
        "provenance": {
            "calibration": str(args.calibration),
            "device": "provider-dependent; stored replay uses no accelerator",
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "registered_pose_metric": "provisional; registrations are upstream outputs",
            "inference_uses_ground_truth": False,
            "stored_mask_replay": "oracle/replay baseline, not a deployable RGB model",
            "sync_offset_stress": "unavailable without preserved trajectories",
        },
        "aggregate": aggregate,
        "snapshots": details,
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
