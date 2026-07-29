#!/usr/bin/env python3
"""Run the whole offline pipeline: sessions -> export -> viability probe.

One command to go from a directory of snapshots to a held-out viability number,
so newly-added data is ingested the same way every time.  Each stage is the same
module you can run by hand; this just chains them with shared paths.

    python -m experiments.rgb_candidate_prior.pipeline \\
        --snapshots-dir data/registration_snapshots --export-root out/rgb_candidate_prior
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_MODULE = "experiments.rgb_candidate_prior"


def _run(step: str, args: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, "-m", f"{_MODULE}.{step}", *args], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--gap-seconds", type=float, default=120.0)
    parser.add_argument("--session-overrides", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--skip-probe", action="store_true", help="Stop after export.")
    parser.add_argument(
        "--probe-real-only",
        action="store_true",
        help="Probe on real PCL-sibling candidates only (least gameable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_root = args.export_root
    export_root.mkdir(parents=True, exist_ok=True)
    sessions_yaml = export_root / "sessions.yaml"

    session_args = [
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--output",
        str(sessions_yaml),
        "--gap-seconds",
        str(args.gap_seconds),
    ]
    if args.session_overrides is not None:
        session_args += ["--overrides", str(args.session_overrides)]
    _run("sessions", session_args)

    export_args = [
        "--snapshots-dir",
        str(args.snapshots_dir),
        "--export-root",
        str(export_root),
        "--sessions",
        str(sessions_yaml),
    ]
    if args.calibration is not None:
        export_args += ["--calibration", str(args.calibration)]
    _run("export", export_args)

    if not args.skip_probe:
        probe_args = ["--export-root", str(export_root), "--cache", str(export_root / "emb.npz")]
        if args.probe_real_only:
            probe_args.append("--real-only")
        _run("probe", probe_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
