#!/usr/bin/env python3
"""Export candidate-ranking records from the production C++ detector trace.

Unlike :mod:`export`, this command never calls the Python detector and never
adds synthetic GT/perturbation candidates.  It consumes the diagnostic
``candidate_trace`` emitted by ``concrete_block_detector_snapshot_runner`` so
the RGB experiment measures the candidates the ROS detector can actually rank.

The trace is deliberately an input artifact: generate it with the same detector
commit and parameter files that are being evaluated, then retain its checksum in
the JSONL provenance.

This is NOT the ``scene_discovery_capture/*/detector_candidates.yaml`` capture
file: that is a different, schema_version 1 YAML artifact (flat ``candidates``
list, ``dimensions`` alongside ``pose``). ``--detector-trace`` must instead be
the schema_version 2 JSON trace (a ``snapshots`` list, per-candidate
``pose["dims"]``) produced on demand by the C++
``concrete_block_detector_snapshot_runner``.
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

from blockpose.annotations import AnnotationTarget, load_annotations  # noqa: E402
from blockpose.evaluate import rotation_error_deg, translation_error  # noqa: E402
from blockpose.io import load_camera_model, load_rgb, load_tf_snapshot  # noqa: E402
from blockpose.pose import BlockPose, quat_xyzw_to_rotmat, rotmat_to_quat_xyzw  # noqa: E402

from experiments.rgb_candidate_prior import gates, render  # noqa: E402
from experiments.rgb_candidate_prior.candidates import CandidateSetParams  # noqa: E402

DEFAULT_CALIBRATION = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"


@dataclass(frozen=True)
class CppCandidate:
    """One post-refinement, pre-NMS candidate from the C++ runner trace."""

    candidate_id: str
    pose: BlockPose
    origin: str
    details: dict[str, Any]


def _pose_from_json(value: dict[str, Any]) -> BlockPose:
    """Parse the stable pose subset of runner JSON, rejecting guessed formats."""
    pose = value.get("pose", value)
    position = np.asarray(pose["position"], dtype=np.float64)
    dims = tuple(float(v) for v in pose["dims"])
    if position.shape != (3,) or len(dims) != 3 or not np.all(np.isfinite(position)):
        raise ValueError("candidate pose must have finite position and three dimensions")
    if "rotation" in pose:
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("candidate pose.rotation must be 3x3")
    elif "orientation_xyzw" in pose:
        rotation = quat_xyzw_to_rotmat(np.asarray(pose["orientation_xyzw"], dtype=np.float64))
    else:
        raise ValueError("candidate pose needs rotation or orientation_xyzw")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("candidate rotation contains non-finite values")
    return BlockPose(position=position, rotation=rotation, dims=dims)


def _fnv1a64_file(path: Path) -> str:
    """Match the runner's non-cryptographic replay-artifact identity."""
    value = 14_695_981_039_346_656_037
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            for byte in chunk:
                value ^= byte
                value = (value * 1_099_511_628_211) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}"


def _verify_trace_inputs(snapshot: Path, trace_snapshot: dict[str, Any]) -> None:
    expected = trace_snapshot.get("input_fingerprints")
    if not isinstance(expected, dict):
        raise ValueError("schema-v2 trace lacks input_fingerprints")
    actual = {
        "cloud_pcd": _fnv1a64_file(snapshot / "cloud.pcd"),
        "tf_yaml": _fnv1a64_file(snapshot / "tf.yaml"),
    }
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError("detector trace cloud/TF fingerprint does not match capture")


_SCHEMA1_CAPTURE_HINT = (
    "this looks like a schema_version 1 'detector_candidates.yaml' scene-discovery capture file "
    "(YAML, a flat 'candidates' list, 'dimensions' as a sibling of 'pose'), which is NOT "
    "cpp_export.py's input. cpp_export.py requires the schema_version 2 JSON trace with a "
    "'snapshots' list and per-candidate pose['dims'], produced on demand by the C++ "
    "concrete_block_detector_snapshot_runner (schema-v2 candidate-trace mode); point "
    "--detector-trace at that instead."
)


def _looks_like_schema1_capture(data: Any) -> bool:
    """Detect the detector_candidates.yaml shape so the error can name it explicitly."""
    return (
        isinstance(data, dict)
        and data.get("schema_version") == 1
        and isinstance(data.get("candidates"), list)
    )


def load_detector_trace(path: Path) -> tuple[dict[str, list[CppCandidate]], dict[str, Any]]:
    """Load C++ post-refinement candidates keyed by capture name.

    ``raw_lineage`` is intentionally *not* accepted here: it holds synthesized,
    pre-refinement poses.  Refusing it prevents silently training on a different
    candidate distribution than the production selector sees.
    """
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        try:
            yaml_guess = yaml.safe_load(text)
        except yaml.YAMLError:
            yaml_guess = None
        if _looks_like_schema1_capture(yaml_guess):
            raise ValueError(
                f"{path} is not a schema_version 2 detector trace: {_SCHEMA1_CAPTURE_HINT}"
            ) from error
        raise ValueError(f"detector trace at {path} is not valid JSON: {error}") from error
    if _looks_like_schema1_capture(data):
        raise ValueError(
            f"{path} is not a schema_version 2 detector trace: {_SCHEMA1_CAPTURE_HINT}"
        )
    if data.get("schema_version") != 2:
        raise ValueError(
            "detector trace must use schema_version 2 with an explicit candidate stage"
        )
    snapshots = data.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("detector trace must contain a snapshots list")
    result: dict[str, list[CppCandidate]] = {}
    for snapshot in snapshots:
        name = snapshot.get("snapshot")
        trace = snapshot.get("candidate_trace")
        stage = snapshot.get("candidate_trace_stage")
        if (
            not isinstance(name, str)
            or not isinstance(trace, list)
            or stage not in {"post_refinement_pre_nms", "pre_refinement_pre_selection"}
        ):
            raise ValueError(
                "each trace snapshot needs snapshot, candidate_trace, and an explicit supported "
                "candidate_trace_stage; re-run the schema-v2 C++ runner"
            )
        if name in result:
            raise ValueError(f"duplicate snapshot {name!r} in detector trace")
        candidates: list[CppCandidate] = []
        for index, item in enumerate(trace):
            if not isinstance(item, dict):
                raise ValueError(f"candidate_trace[{index}] in {name!r} is not an object")
            candidate_id = str(item.get("id", item.get("candidate_id", f"candidate_{index:03d}")))
            if not candidate_id:
                raise ValueError(f"candidate_trace[{index}] in {name!r} has an empty id")
            origin = str(item.get("source", item.get("origin", "c++_geometry")))
            if item.get("stage") != stage:
                raise ValueError(
                    f"candidate_trace[{index}] in {name!r} has no matching candidate stage"
                )
            candidates.append(CppCandidate(candidate_id, _pose_from_json(item), origin, dict(item)))
        result[name] = candidates
    return result, data


def _targets(snapshot: Path) -> list[AnnotationTarget]:
    path = snapshot / "annotations.yaml"
    if not path.is_file():
        return []
    annotations = load_annotations(path)
    if not annotations.annotation_complete:
        return []
    return [target for target in annotations.targets if target.is_evaluation_target]


def _label(
    candidate: CppCandidate, target: AnnotationTarget, params: CandidateSetParams
) -> tuple[str, dict[str, float]]:
    position_error = translation_error(candidate.pose, target.pose)
    rotation_error = rotation_error_deg(candidate.pose, target.pose)
    positive = (
        position_error <= params.positive_pos_tol_m
        and rotation_error <= params.positive_rot_tol_deg
    )
    return ("positive" if positive else "negative"), {
        "translation_error_m": position_error,
        "rotation_error_deg": rotation_error,
    }


def _record_target(
    *,
    snapshot: Path,
    target: AnnotationTarget,
    candidates: list[CppCandidate],
    export_root: Path,
    rgb: np.ndarray,
    camera: Any,
    T_world_camera: np.ndarray,
    image_stamp_ns: int,
    cloud_stamp_ns: int,
    sync_delta_s: float,
    calibration_checksum: str,
    trace_checksum: str,
    tf_checksum: str,
    split_group: str,
    crop_margin_frac: float,
    out_size: int,
    params: CandidateSetParams,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """Write a target's real C++ candidate set; return (written, dropped)."""
    if target.visibility in params.occluded_visibilities:
        return 0, 0
    nearby = [
        candidate
        for candidate in candidates
        if translation_error(candidate.pose, target.pose) <= params.candidate_radius_m
    ]
    if not nearby:
        return 0, 0

    target_corners, target_uv, target_front = render.project_corners(
        target.pose, T_world_camera, camera
    )
    target_gate = gates.projection_gate(target_corners, T_world_camera, camera)
    crop_boxes = []
    if target_gate.passed:
        bbox = render.crop_bbox(target_uv, target_front, camera, margin_frac=crop_margin_frac)
        if bbox is not None:
            crop_boxes.append(bbox)

    prepared: list[tuple[CppCandidate, dict[str, Any]]] = []
    dropped = 0
    for candidate in nearby:
        corners, uv, in_front = render.project_corners(candidate.pose, T_world_camera, camera)
        projection = gates.projection_gate(corners, T_world_camera, camera)
        bbox = (
            render.crop_bbox(uv, in_front, camera, margin_frac=crop_margin_frac)
            if projection.passed
            else None
        )
        if bbox is None:
            dropped += 1
            continue
        prepared.append((candidate, {"projection": projection.details, "bbox": bbox}))
        crop_boxes.append(bbox)
    shared_bbox = render.union_crop_bbox(crop_boxes, camera)
    if shared_bbox is None:
        return 0, dropped + len(prepared)

    for candidate, prepared_data in prepared:
        label, label_details = _label(candidate, target, params)
        sample_id = f"{snapshot.name}/{target.id}/cpp_{candidate.candidate_id}"
        stem = sample_id.replace("/", "__")
        rel_rgb = Path("images") / snapshot.name / f"{stem}.png"
        rel_cond = Path("renders") / snapshot.name / f"{stem}.png"
        rel_overlay = Path("overlays") / snapshot.name / f"{stem}.png"
        render.save_png(
            render.render_crop(rgb, shared_bbox, out_size=out_size), export_root / rel_rgb
        )
        render.save_png(
            render.render_conditioning(
                candidate.pose, T_world_camera, camera, shared_bbox, out_size=out_size
            ),
            export_root / rel_cond,
        )
        render.save_png(
            render.render_overlay(rgb, candidate.pose, T_world_camera, camera, shared_bbox),
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
                "label": label,
                "provenance": "cpp_candidate_trace",
                "gt_provenance": target.provenance,
                "gt_visibility": target.visibility,
                "label_details": label_details,
                "crop_bbox_xyxy": list(shared_bbox),
                "projection": prepared_data["projection"],
                "candidate_trace": {
                    "id": candidate.candidate_id,
                    "origin": candidate.origin,
                    "details": candidate.details,
                },
                "source": {
                    "snapshot": snapshot.name,
                    "image_stamp_ns": image_stamp_ns,
                    "cloud_stamp_ns": cloud_stamp_ns,
                    "sync_delta_s": sync_delta_s,
                    "rgb_checksum": gates.sha256_file(snapshot / "rgb.png"),
                    "calibration_checksum": calibration_checksum,
                    "tf_checksum": tf_checksum,
                    "detector_trace_checksum": trace_checksum,
                },
            }
        )
    return len(prepared), dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--detector-trace", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--sync-tolerance-s", type=float, default=0.10)
    parser.add_argument("--crop-margin-frac", type=float, default=0.25)
    parser.add_argument("--out-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_by_snapshot, trace_header = load_detector_trace(args.detector_trace)
    trace_snapshots = {
        entry["snapshot"]: entry
        for entry in trace_header["snapshots"]
        if isinstance(entry, dict) and isinstance(entry.get("snapshot"), str)
    }
    sessions = yaml.safe_load(args.sessions.read_text()) or {}
    if not isinstance(sessions, dict):
        raise ValueError("sessions must be a YAML mapping of snapshot name to recording session")
    args.export_root.mkdir(parents=True, exist_ok=True)
    calibration_checksum = gates.sha256_file(args.calibration)
    trace_checksum = gates.sha256_file(args.detector_trace)
    params = CandidateSetParams()
    records: list[dict[str, Any]] = []
    per_snapshot: list[dict[str, Any]] = []
    for snapshot in sorted(path for path in args.snapshots_dir.iterdir() if path.is_dir()):
        try:
            targets = _targets(snapshot)
            if not targets:
                per_snapshot.append(
                    {"snapshot": snapshot.name, "status": "skipped_no_complete_annotations"}
                )
                continue
            if snapshot.name not in trace_by_snapshot:
                per_snapshot.append({"snapshot": snapshot.name, "status": "missing_detector_trace"})
                continue
            _verify_trace_inputs(snapshot, trace_snapshots[snapshot.name])
            camera = load_camera_model(args.calibration)
            rgb = load_rgb(snapshot / "rgb.png")
            calibration = gates.calibration_gate(camera, rgb.shape)
            tf = load_tf_snapshot(snapshot / "tf.yaml")
            tf_check = gates.tf_gate(tf.T_world_camera)
            image_stamp_ns, cloud_stamp_ns = gates.read_stamps(snapshot / "metadata.yaml")
            sync = gates.sync_gate(
                image_stamp_ns, cloud_stamp_ns, tolerance_s=args.sync_tolerance_s
            )
            if not calibration.passed or not tf_check.passed or not sync.passed:
                failed = (
                    calibration
                    if not calibration.passed
                    else tf_check
                    if not tf_check.passed
                    else sync
                )
                per_snapshot.append(
                    {
                        "snapshot": snapshot.name,
                        "status": "gate_failed",
                        "gate": failed.reason,
                        "details": failed.details,
                    }
                )
                continue
            written = dropped = zero_candidate_targets = ignored_targets = 0
            for target in targets:
                if target.visibility in params.occluded_visibilities:
                    ignored_targets += 1
                    continue
                target_written, target_dropped = _record_target(
                    snapshot=snapshot,
                    target=target,
                    candidates=trace_by_snapshot[snapshot.name],
                    export_root=args.export_root,
                    rgb=rgb,
                    camera=camera,
                    T_world_camera=tf.T_world_camera,
                    image_stamp_ns=image_stamp_ns,
                    cloud_stamp_ns=cloud_stamp_ns,
                    sync_delta_s=sync.details["sync_delta_s"],
                    calibration_checksum=calibration_checksum,
                    trace_checksum=trace_checksum,
                    tf_checksum=gates.sha256_file(snapshot / "tf.yaml"),
                    split_group=str(sessions.get(snapshot.name, "session_unassigned")),
                    crop_margin_frac=args.crop_margin_frac,
                    out_size=args.out_size,
                    params=params,
                    records=records,
                )
                written += target_written
                dropped += target_dropped
                zero_candidate_targets += target_written == 0
            per_snapshot.append(
                {
                    "snapshot": snapshot.name,
                    "status": "exported",
                    "written": written,
                    "gate_dropped": dropped,
                    "targets": len(targets),
                    "ignored_targets": ignored_targets,
                    "zero_candidate_targets": zero_candidate_targets,
                }
            )
        except Exception as error:  # noqa: BLE001 - retain other captures when one input is malformed
            per_snapshot.append(
                {"snapshot": snapshot.name, "status": "error", "error": repr(error)}
            )

    (args.export_root / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    manifest = {
        "schema_version": 1,
        "candidate_source": "production_cpp_candidate_trace",
        "selection_eligible": False,
        "reason": (
            "requires independently verified recording-session splits and held-out ranking report"
        ),
        "detector_trace": str(args.detector_trace),
        "detector_trace_checksum": trace_checksum,
        "detector_trace_header": {
            key: value for key, value in trace_header.items() if key != "snapshots"
        },
        "sync_tolerance_s": args.sync_tolerance_s,
        "total_records": len(records),
        "label_totals": dict(Counter(record["label"] for record in records)),
        "snapshots": per_snapshot,
    }
    (args.export_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"total_records": len(records), "snapshots": per_snapshot}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
