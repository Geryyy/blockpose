#!/usr/bin/env python3
"""Run the frozen real-snapshot benchmark for all Python detector modes.

The written JSON contains compact metrics plus per-target overlay poses for
the showcase notebook. It is a development diagnostic, never an acceptance
artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from blockpose.mode_benchmark import (  # noqa: E402
    DEFAULT_SNAPSHOTS,
    DEFAULT_TRANSFORM,
    run_mode_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--timeout-ms", type=float, default=500.0)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON destination; parent directory must already exist.",
    )
    args = parser.parse_args()
    report = run_mode_benchmark(
        args.snapshots_dir,
        args.config,
        min_score=args.min_score,
        timeout_ms=args.timeout_ms,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scene_discovery": {
                    key: report["modes"]["SCENE_DISCOVERY"].get(key)
                    for key in (
                        "n_snapshots_scored",
                        "precision",
                        "recall",
                        "exact_count_rate",
                        "translation_error_p95_m",
                        "rotation_error_p95_deg",
                        "timeout_count",
                        "end_to_end_runtime_p95_ms",
                    )
                },
                "refine_block": {
                    key: report["modes"]["REFINE_BLOCK"].get(key)
                    for key in (
                        "n_planned_targets",
                        "observed_status_counts_on_present_targets",
                        "translation_error_p95_m",
                        "end_to_end_runtime_p95_ms",
                    )
                },
                "refine_gripped": report["modes"]["REFINE_GRASPED"]["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
