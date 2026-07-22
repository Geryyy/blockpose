#!/usr/bin/env python3
"""Mark reviewed snapshots complete using their full world-space cloud bounds."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_batch import discover_snapshots  # noqa: E402
from blockpose.annotation_roi import prepare_roi_completion  # noqa: E402
from blockpose.annotations import load_annotations  # noqa: E402


def _write_atomic(path: Path, document: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        # Leave a failed temporary publication for inspection; repository policy
        # intentionally forbids deleting evidence.
        raise


def complete_snapshots(root: Path, *, write: bool = False):
    """Validate every active snapshot, optionally publishing all manifests."""
    snapshots = discover_snapshots(root)
    completions = [prepare_roi_completion(snapshot) for snapshot in snapshots]
    if write:
        for completion in completions:
            _write_atomic(completion.snapshot / "annotations.yaml", completion.document)
        for completion in completions:
            load_annotations(completion.snapshot / "annotations.yaml")
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=REPO_ROOT / "data" / "registration_snapshots"
    )
    parser.add_argument(
        "--write", action="store_true", help="write only after every snapshot validates"
    )
    args = parser.parse_args()

    completions = complete_snapshots(args.root, write=args.write)
    action = "completed" if args.write else "validated (dry run)"
    print(f"{action}: {len(completions)} active snapshots")
    for completion in completions:
        print(f"  {completion.snapshot.name}: min={completion.roi_min} max={completion.roi_max}")


if __name__ == "__main__":
    main()
