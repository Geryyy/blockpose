#!/usr/bin/env python3
"""Fit the versioned synthetic-only block validity calibrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.confidence import (  # noqa: E402
    DEFAULT_ARTIFACT,
    FEATURE_SCHEMA,
    calibration_metrics,
    feature_vector,
    fit_logistic,
    predict,
    save_artifact,
)
from blockpose.detection import (  # noqa: E402
    DetectionParams,
    SensorContext,
    detect_blocks_detailed,
)
from blockpose.evaluate import match_detections  # noqa: E402
from blockpose.scenario_matrix import generate_scenario_matrix  # noqa: E402

DEVELOPMENT_SEEDS = list(range(10, 18))
GATE_SEEDS = list(range(100, 104))


def build_corpus(seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    default_params = DetectionParams()
    for seed in seeds:
        for _, sample in generate_scenario_matrix(seed):
            context = SensorContext(sample.sensor_origin, sample.ray_directions, sample.ranges)
            # Scenarios mix upright staging blocks and laid wall-plan blocks
            # (see scenario_matrix.py); the detector needs the matching prior,
            # same as tools/eval_detector.py's _aggregate.
            dims = tuple(sample.manifest.get("params", {}).get("dims", default_params.block_dims))
            params = (
                default_params
                if dims == tuple(default_params.block_dims)
                else DetectionParams(block_dims=dims)
            )
            result = detect_blocks_detailed(sample.points, params, sensor_context=context)
            candidates = result.poses
            features = [item.features for item in result.diagnostics]
            matched = {index for index, _ in match_detections(candidates, sample.ground_truth, 0.5)}
            for index, values in enumerate(features):
                rows.append(feature_vector(values))
                labels.append(float(index in matched))
    return np.asarray(rows), np.asarray(labels)


def run(output: Path = DEFAULT_ARTIFACT) -> dict:
    development_x, development_y = build_corpus(DEVELOPMENT_SEEDS)
    gate_x, gate_y = build_corpus(GATE_SEEDS)
    development_counts = {
        "count": int(len(development_y)),
        "positives": int(development_y.sum()),
        "negatives": int(len(development_y) - development_y.sum()),
    }
    gate_counts = {
        "count": int(len(gate_y)),
        "positives": int(gate_y.sum()),
        "negatives": int(len(gate_y) - gate_y.sum()),
    }
    if min(development_counts["positives"], development_counts["negatives"]) < 20:
        return {
            "status": "blocked",
            "reason": "selected-output corpus lacks at least 20 organic examples of each class",
            "development": development_counts,
            "gate": gate_counts,
            "development_seeds": DEVELOPMENT_SEEDS,
            "gate_seeds": GATE_SEEDS,
        }
    artifact = fit_logistic(
        development_x,
        development_y,
        l2=1.0,
        development_seeds=DEVELOPMENT_SEEDS,
        gate_seeds=GATE_SEEDS,
    )
    save_artifact(artifact, output)
    development_p = np.array(
        [predict(dict(zip(FEATURE_SCHEMA, row, strict=True)), artifact) for row in development_x]
    )
    gate_p = np.array(
        [predict(dict(zip(FEATURE_SCHEMA, row, strict=True)), artifact) for row in gate_x]
    )
    return {
        "artifact": str(output),
        "development": calibration_metrics(development_y, development_p),
        "gate": calibration_metrics(gate_y, gate_p),
        "development_seeds": DEVELOPMENT_SEEDS,
        "gate_seeds": GATE_SEEDS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(args.output)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
