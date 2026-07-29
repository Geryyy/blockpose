#!/usr/bin/env python3
"""Assign snapshots to recording sessions for held-out evaluation.

Held-out splits must be by session, never by frame/crop/candidate.  Real session
metadata is preferred; until it exists this clusters snapshots by the Unix
timestamp in their directory name, treating a gap larger than ``--gap-seconds``
as a session boundary.  These are *provisional* sessions -- honest enough for a
held-out proxy, but flagged as such so nobody mistakes them for verified runs.

An optional ``--overrides`` YAML (``snapshot_name: session_id``) wins over the
automatic assignment, so once true session labels are known they are dropped in
without changing anything downstream.

    python -m experiments.rgb_candidate_prior.sessions \\
        --snapshots-dir data/registration_snapshots --output sessions.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "registration_snapshots"

# Directory names begin with a Unix-seconds timestamp, e.g. 1783427923_...seq5.
_TIMESTAMP = re.compile(r"^(\d+)_")


def _snapshot_seconds(name: str) -> int | None:
    match = _TIMESTAMP.match(name)
    return int(match.group(1)) if match else None


def assign_sessions(
    snapshot_names: list[str], *, gap_seconds: float, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Map each snapshot to a ``session_NN`` id; overrides take precedence.

    Snapshots without a parseable timestamp are grouped under ``session_unknown``
    so they never silently merge into a timestamped cluster.
    """
    overrides = overrides or {}
    timestamped = sorted(
        (seconds, name)
        for name in snapshot_names
        if (seconds := _snapshot_seconds(name)) is not None
    )
    mapping: dict[str, str] = {}
    index = 0
    previous: int | None = None
    for seconds, name in timestamped:
        if previous is None or seconds - previous > gap_seconds:
            index += 1
        mapping[name] = f"session_{index:02d}"
        previous = seconds
    for name in snapshot_names:
        if _snapshot_seconds(name) is None:
            mapping[name] = "session_unknown"
    mapping.update({name: overrides[name] for name in snapshot_names if name in overrides})
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=120.0,
        help="Time gap (s) above which a new session starts (default: 120).",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="YAML of snapshot_name -> session_id to force known groupings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.snapshots_dir.is_dir():
        print(f"snapshots dir not found: {args.snapshots_dir}", file=sys.stderr)
        return 1
    names = [p.name for p in args.snapshots_dir.iterdir() if (p / "tf.yaml").is_file()]
    overrides = yaml.safe_load(args.overrides.read_text()) if args.overrides else None
    mapping = assign_sessions(names, gap_seconds=args.gap_seconds, overrides=overrides or {})
    args.output.write_text(yaml.safe_dump(mapping, sort_keys=True))

    sessions = sorted(set(mapping.values()))
    print(f"{len(names)} snapshots -> {len(sessions)} sessions (gap {args.gap_seconds:g}s)")
    for session in sessions:
        members = sorted(name for name, sid in mapping.items() if sid == session)
        print(f"  {session}: {len(members)} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
