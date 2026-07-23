#!/usr/bin/env python3
"""Export deterministic DetectFree inputs and golden outputs for the C++ port.

This is intentionally a one-way exporter: C++ tests consume the NPZ/JSON
artefacts, while this script remains the single place that invokes the Python
prototype.  It never looks at ground truth when producing expected detections.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from blockpose.detection import DetectionParams, detect_blocks_detailed
from blockpose.io import load_snapshot
from blockpose.refine import assign_points, refine_pose
from blockpose.synth import BlockSpec, ClutterParams, SynthParams, generate_scene


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _result_json(result) -> dict[str, Any]:
    return {
        "poses": [
            {
                "position": pose.position.tolist(),
                "rotation": pose.rotation.tolist(),
                "dims": list(pose.dims),
            }
            for pose in result.poses
        ],
        "diagnostics": [
            {
                "evidence": _json_value(asdict(diagnostic.evidence)),
                "observability": diagnostic.observability.value,
                "features": _json_value(diagnostic.features),
                "proposal_scale_m": diagnostic.proposal_scale_m,
                "support_height_m": diagnostic.support_height_m,
            }
            for diagnostic in result.diagnostics
        ],
        "stage_counts": _json_value(result.stage_counts),
    }


def _write_case(output: Path, case_id: str, points: np.ndarray, result) -> None:
    np.savez_compressed(output / f"{case_id}.npz", points=np.asarray(points, dtype=np.float64))
    (output / f"{case_id}.json").write_text(
        json.dumps(_result_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_plain_text_case(output: Path, case_id: str, points: np.ndarray, result) -> None:
    """Portable C++ input/golden files, generated deterministically at test time."""
    np.savetxt(output / f"{case_id}.xyz", np.asarray(points, dtype=np.float64), fmt="%.17g")
    payload = _result_json(result)
    with (output / f"{case_id}.expected").open("w", encoding="utf-8") as stream:
        stream.write(f"poses {len(payload['poses'])}\n")
        for pose, diagnostic in zip(payload["poses"], payload["diagnostics"], strict=True):
            values = pose["position"] + [value for row in pose["rotation"] for value in row]
            values += pose["dims"] + [diagnostic["evidence"]["score"]]
            stream.write("pose " + " ".join(f"{value:.17g}" for value in values) + "\n")
        for key, value in sorted(payload["stage_counts"].items()):
            stream.write(f"count {key} {value}\n")


def _write_refinement_fixture(output: Path, case_id: str, result, params: DetectionParams) -> None:
    """Export the exact solver input/output boundary for a parity case.

    The full detector golden tells us that a pose differs, but not whether the
    divergence began in proposal generation or the optimizer.  This companion
    fixture freezes the Python selected initializations and their assigned
    above-support returns, then records each SciPy refinement pass.
    """
    trace = result.trace
    if trace is None:
        raise RuntimeError("refinement fixture requires collect_trace=True")
    current = [item.pose for item in trace.pre_refinement_hypotheses]
    passes: list[dict[str, Any]] = []
    for _ in range(max(params.refine_iterations, 1)):
        assigned = assign_points(current, trace.above_support_points, None, params.refine_band)
        next_poses = []
        refined_flags = []
        for pose, owned in zip(current, assigned, strict=True):
            refined, success = refine_pose(
                pose,
                owned,
                pose.dims,
                max_translation=params.refine_max_translation,
                max_rotation_deg=params.refine_max_rotation_deg,
                min_points=params.refine_min_points,
            )
            next_poses.append(refined)
            refined_flags.append(success)
        passes.append(
            {
                "assigned_counts": [len(owned) for owned in assigned],
                "refined": refined_flags,
                "poses": [
                    {
                        "position": pose.position.tolist(),
                        "rotation": pose.rotation.tolist(),
                        "dims": list(pose.dims),
                    }
                    for pose in next_poses
                ],
            }
        )
        current = next_poses
    np.savetxt(
        output / f"{case_id}.refine_points.xyz",
        np.asarray(trace.above_support_points, dtype=np.float64),
        fmt="%.17g",
    )
    (output / f"{case_id}.refinement.json").write_text(
        json.dumps(
            {
                "initial": [
                    {
                        "id": item.id,
                        "position": item.pose.position.tolist(),
                        "rotation": item.pose.rotation.tolist(),
                        "dims": list(item.pose.dims),
                    }
                    for item in trace.pre_refinement_hypotheses
                ],
                "passes": passes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _synthetic_case(entry: dict[str, Any]) -> np.ndarray:
    blocks = [BlockSpec(**block) for block in entry["blocks"]]
    synth = SynthParams(**entry.get("synth", {}))
    clutter = ClutterParams(**entry["clutter"]) if "clutter" in entry else None
    sample = generate_scene(blocks, synth, np.random.default_rng(entry["seed"]), clutter=clutter)
    return sample.points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-real", action="store_true")
    parser.add_argument("--plain-text", action="store_true")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "registration_snapshots",
    )
    parser.add_argument(
        "--transform-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "transforms_pzs100_initial_pose_1.yaml",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported parity fixture manifest schema")
    args.output.mkdir(parents=True, exist_ok=True)
    params = DetectionParams()
    produced: list[str] = []
    for entry in manifest["synthetic_cases"]:
        points = _synthetic_case(entry)
        result = detect_blocks_detailed(points, params, collect_trace=True)
        _write_case(args.output, entry["id"], points, result)
        _write_refinement_fixture(args.output, entry["id"], result, params)
        if args.plain_text:
            _write_plain_text_case(args.output, entry["id"], points, result)
        produced.append(entry["id"])

    missing_snapshots: list[str] = []
    if args.include_real:
        for snapshot_id in manifest.get("real_snapshots", []):
            snapshot = args.snapshot_root / snapshot_id
            if not snapshot.is_dir():
                missing_snapshots.append(snapshot_id)
                continue
            points, _ = load_snapshot(snapshot, args.transform_config)
            result = detect_blocks_detailed(points, params, collect_trace=True)
            _write_case(args.output, f"real_{snapshot_id}", points, result)
            _write_refinement_fixture(args.output, f"real_{snapshot_id}", result, params)
            if args.plain_text:
                _write_plain_text_case(args.output, f"real_{snapshot_id}", points, result)
            produced.append(f"real_{snapshot_id}")

    (args.output / "export_manifest.json").write_text(
        json.dumps(
            {
                "fixture_schema_version": manifest["schema_version"],
                "producer": manifest["producer"],
                "detection_params": _json_value(asdict(params)),
                "produced": produced,
                "missing_real_snapshots": missing_snapshots,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if missing_snapshots:
        raise FileNotFoundError(f"missing requested real snapshots: {', '.join(missing_snapshots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
