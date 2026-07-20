#!/usr/bin/env python3
"""Run the block detector over a synthetic suite and gate on regression thresholds.

Generates a deterministic suite of raycast scenes (single, multi-block, and empty),
runs `detect_blocks` on each, aggregates metrics via `blockpose.evaluate`, writes
`metrics.json`, and exits non-zero if any threshold in config/detector_eval.yaml is
missed. This is the machine-checkable Phase-A gate.

Examples:
    python tools/eval_detector.py
    python tools/eval_detector.py --metrics-out metrics.json --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, detect_blocks, detect_scene_sample  # noqa: E402
from blockpose.evaluate import evaluate  # noqa: E402
from blockpose.scenario_matrix import generate_scenario_matrix  # noqa: E402
from blockpose.synth import BlockSpec, SynthParams, generate_scene, random_scene  # noqa: E402

DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "detector_eval.yaml"


def build_suite(seed: int) -> tuple[list, list, list]:
    """Deterministic suite: (gate_scenes, diag_scenes, empty_scenes).

    gate_scenes: isolated, well-observed blocks (varied yaw/tilt/noise) — the
        Phase-A geometry+pose core; these are gated.
    diag_scenes: 2-4 block scenes with mutual occlusion — reported as
        diagnostics (Phase C progress), NOT gated.
    empty_scenes: no blocks — false-positive check (gated).
    """
    rng = np.random.default_rng(seed)
    params = SynthParams(range_noise_std=0.01)

    gate_scenes = []
    for _ in range(8):
        spec = BlockSpec(
            x=float(rng.uniform(-0.3, 0.3)),
            y=float(rng.uniform(-0.3, 0.3)),
            yaw_deg=float(rng.uniform(0.0, 360.0)),
            pitch_deg=float(rng.uniform(-3.0, 3.0)),
            roll_deg=float(rng.uniform(-3.0, 3.0)),
        )
        gate_scenes.append(generate_scene([spec], params, rng))

    diag_scenes = [random_scene(n, params, rng) for n in (2, 3, 4)]
    empty_scenes = [generate_scene([], params, rng) for _ in range(2)]
    return gate_scenes, diag_scenes, empty_scenes


# A block yielding fewer returns than this cannot produce a cluster (the
# detector's cluster_min_size is 30) let alone a plane fit, so it is excluded
# from the recall denominator and reported separately as `unobservable_gt`.
MIN_OBSERVABLE_POINTS = 30


def _aggregate(scenes: list, params: DetectionParams) -> dict:
    total_gt = total_det = total_matched = unobservable = 0
    t_errors: list[float] = []
    r_errors: list[float] = []
    precision_per_scene: list[float] = []
    recall_per_scene: list[float] = []
    exact_counts = 0
    for scene in scenes:
        pts, gt = scene
        # Block extents vary per scene: config/wall_plans.yaml lays blocks down
        # as (0.9, 0.6, 0.6) while staging blocks stand upright as (0.6, 0.6, 0.9).
        # The detector needs the right prior, and the evaluator needs the right
        # symmetry group -- using the upright group on a laid block scores a
        # 90 deg yaw error as zero.
        manifest = getattr(scene, "manifest", {}) or {}
        dims = tuple(manifest.get("params", {}).get("dims", params.block_dims))
        scene_params = (
            params if dims == tuple(params.block_dims) else replace(params, block_dims=dims)
        )
        det = (
            detect_scene_sample(scene, scene_params)
            if hasattr(scene, "sensor_origin")
            else detect_blocks(pts, scene_params)
        )
        res = evaluate(det, gt, max_match_dist=0.5, dims=dims)
        # Blocks that produced (almost) no returns are not detectable and must
        # not be scored as misses; a random dropout stripe can erase a target
        # outright, which otherwise turns the metric into a coin flip.
        if hasattr(scene, "visible_point_counts"):
            counts = scene.visible_point_counts()
            observable = int(np.count_nonzero(counts >= MIN_OBSERVABLE_POINTS))
        else:
            observable = len(gt)
        unobservable += len(gt) - observable
        total_gt += observable
        total_det += len(det)
        total_matched += res.matched
        t_errors.extend(res.translation_errors_m)
        r_errors.extend(res.rotation_errors_deg)
        precision_per_scene.append(res.precision)
        recall_per_scene.append(res.matched / observable if observable else 0.0)
        exact_counts += int(res.exact_count)
    def percentile(values: list[float], q: float) -> float | None:
        return float(np.percentile(values, q)) if values else None

    # `None` where the metric is undefined, rather than 0.0. A hard-negative
    # family has no ground truth and (correctly) no detections; reporting
    # "recall 0.0, precision 0.0" made a clean pass look like a total failure.
    # For those families `false_positives` carries the signal.
    return {
        "n_scenes": len(scenes),
        "total_gt": total_gt,
        "unobservable_gt": unobservable,
        "total_detected": total_det,
        "total_matched": total_matched,
        "recall": total_matched / total_gt if total_gt else None,
        "precision": total_matched / total_det if total_det else None,
        "macro_recall": float(np.mean(recall_per_scene)) if total_gt else None,
        "macro_precision": float(np.mean(precision_per_scene)) if total_det else None,
        "exact_count_accuracy": exact_counts / len(scenes) if scenes else 0.0,
        "translation_mean_m": float(np.mean(t_errors)) if t_errors else None,
        "translation_p50_m": percentile(t_errors, 50),
        "translation_p95_m": percentile(t_errors, 95),
        "translation_max_m": max(t_errors) if t_errors else None,
        "rotation_mean_deg": float(np.mean(r_errors)) if r_errors else None,
        "rotation_p50_deg": percentile(r_errors, 50),
        "rotation_p95_deg": percentile(r_errors, 95),
        "rotation_max_deg": max(r_errors) if r_errors else None,
        "false_positives": total_det - total_matched,
    }


def run_suite(seed: int) -> dict:
    gate_scenes, diag_scenes, empty_scenes = build_suite(seed)
    params = DetectionParams()

    gate = _aggregate(gate_scenes, params)
    diagnostics = _aggregate(diag_scenes, params)
    false_positives_empty = sum(len(detect_scene_sample(scene, params)) for scene in empty_scenes)

    metrics = {"seed": seed, **gate, "false_positives_empty": int(false_positives_empty)}
    metrics["diagnostics_multiblock"] = diagnostics
    stratified: dict[str, list] = {}
    for scenario, sample in generate_scenario_matrix(seed):
        stratified.setdefault(scenario.family, []).append(sample)
    metrics["diagnostics_stratified"] = {
        family: _aggregate(samples, params) for family, samples in stratified.items()
    }
    return metrics


def check_thresholds(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of human-readable failures (empty == all pass)."""
    failures = []

    def fail_if(cond, msg):
        if cond:
            failures.append(msg)

    # The gate suite always contains ground truth, so a `None` here means the
    # detector returned nothing at all -- a failure, not an undefined metric.
    fail_if(
        metrics["recall"] is None or metrics["recall"] < thresholds["recall_min"],
        f"recall {metrics['recall']} < {thresholds['recall_min']}",
    )
    fail_if(
        metrics["precision"] is None or metrics["precision"] < thresholds["precision_min"],
        f"precision {metrics['precision']} < {thresholds['precision_min']}",
    )
    fail_if(
        metrics["translation_p95_m"] is None
        or metrics["translation_p95_m"] > thresholds["translation_p95_m"],
        f"translation_p95_m {metrics['translation_p95_m']} > {thresholds['translation_p95_m']}",
    )
    fail_if(
        metrics["rotation_p95_deg"] is None
        or metrics["rotation_p95_deg"] > thresholds["rotation_p95_deg"],
        f"rotation_p95_deg {metrics['rotation_p95_deg']} > {thresholds['rotation_p95_deg']}",
    )
    fail_if(
        metrics["false_positives_empty"] > thresholds["false_positives_empty_max"],
        f"false_positives_empty {metrics['false_positives_empty']} > "
        f"{thresholds['false_positives_empty_max']}",
    )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Suite generation seed.")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Write metrics JSON here.")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="YAML of regression thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_suite(args.seed)

    with args.thresholds.open() as f:
        thresholds = yaml.safe_load(f)["thresholds"]
    failures = check_thresholds(metrics, thresholds)

    print(json.dumps(metrics, indent=2))
    if args.metrics_out is not None:
        args.metrics_out.write_text(json.dumps(metrics, indent=2))

    if failures:
        print("\nGATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
