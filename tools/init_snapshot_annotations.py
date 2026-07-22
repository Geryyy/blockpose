#!/usr/bin/env python3
"""Safely initialize provisional manifests for registration snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_batch import initialize_root, initialize_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "data" / "registration_snapshots",
        help="snapshot root, or one snapshot with --snapshot",
    )
    parser.add_argument("--snapshot", action="store_true", help="treat path as one snapshot")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--dedup-distance", type=float, default=0.40)
    args = parser.parse_args()
    if not np_is_positive(args.dedup_distance):
        parser.error("--dedup-distance must be positive and finite")
    kwargs = {
        "camera_calibration": args.camera_calibration,
        "dedup_distance_m": args.dedup_distance,
    }
    results = (
        [initialize_snapshot(args.path, **kwargs)]
        if args.snapshot
        else initialize_root(args.path, **kwargs)
    )
    for result in results:
        print(
            f"{result.snapshot}: {result.status}; registrations={result.registration_targets}; "
            f"rgb_fits={result.rgb_targets}; failures={len(result.failures)}"
        )
        for failure in result.failures:
            print(f"  - {failure.source}: {failure.reason}")
    return 1 if any(result.status == "failed_integrity" for result in results) else 0


def np_is_positive(value: float) -> bool:
    return value > 0 and value < float("inf")


if __name__ == "__main__":
    raise SystemExit(main())
