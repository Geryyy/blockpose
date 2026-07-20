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
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.detection import DetectionParams, detect_blocks  # noqa: E402
from blockpose.evaluate import evaluate  # noqa: E402
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


def _aggregate(scenes: list, params: DetectionParams) -> dict:
    total_gt = total_det = total_matched = 0
    t_errors: list[float] = []
    r_errors: list[float] = []
    for pts, gt in scenes:
        det = detect_blocks(pts, params)
        res = evaluate(det, gt, max_match_dist=0.5)
        total_gt += len(gt)
        total_det += len(det)
        total_matched += res.matched
        t_errors.extend(res.translation_errors_m)
        r_errors.extend(res.rotation_errors_deg)
    t = np.array(t_errors) if t_errors else np.array([0.0])
    r = np.array(r_errors) if r_errors else np.array([0.0])
    return {
        "n_scenes": len(scenes),
        "total_gt": total_gt,
        "total_detected": total_det,
        "total_matched": total_matched,
        "recall": total_matched / total_gt if total_gt else 0.0,
        "precision": total_matched / total_det if total_det else 0.0,
        "translation_mean_m": float(t.mean()),
        "translation_p95_m": float(np.percentile(t, 95)),
        "rotation_mean_deg": float(r.mean()),
        "rotation_p95_deg": float(np.percentile(r, 95)),
    }


def run_suite(seed: int) -> dict:
    gate_scenes, diag_scenes, empty_scenes = build_suite(seed)
    params = DetectionParams()

    gate = _aggregate(gate_scenes, params)
    diagnostics = _aggregate(diag_scenes, params)
    false_positives_empty = sum(len(detect_blocks(pts, params)) for pts, _ in empty_scenes)

    metrics = {"seed": seed, **gate, "false_positives_empty": int(false_positives_empty)}
    metrics["diagnostics_multiblock"] = diagnostics
    return metrics


def check_thresholds(metrics: dict, thresholds: dict) -> list[str]:
    """Return a list of human-readable failures (empty == all pass)."""
    failures = []

    def fail_if(cond, msg):
        if cond:
            failures.append(msg)

    fail_if(
        metrics["recall"] < thresholds["recall_min"],
        f"recall {metrics['recall']:.3f} < {thresholds['recall_min']}",
    )
    fail_if(
        metrics["precision"] < thresholds["precision_min"],
        f"precision {metrics['precision']:.3f} < {thresholds['precision_min']}",
    )
    fail_if(
        metrics["translation_p95_m"] > thresholds["translation_p95_m"],
        f"translation_p95_m {metrics['translation_p95_m']:.3f} > {thresholds['translation_p95_m']}",
    )
    fail_if(
        metrics["rotation_p95_deg"] > thresholds["rotation_p95_deg"],
        f"rotation_p95_deg {metrics['rotation_p95_deg']:.2f} > {thresholds['rotation_p95_deg']}",
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
