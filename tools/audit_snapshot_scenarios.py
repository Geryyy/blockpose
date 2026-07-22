#!/usr/bin/env python3
"""Audit snapshot scenario coverage and relations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from blockpose.snapshot_scenarios import DEFAULT_SCENARIOS, audit_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=ROOT / "data/registration_snapshots")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SCENARIOS)
    args = parser.parse_args()
    report = audit_manifest(args.snapshots_dir, args.manifest)
    print(json.dumps(report, indent=2))
    return int(bool(report["missing"] or report["extra"]))


if __name__ == "__main__":
    raise SystemExit(main())
