"""Versioned, non-destructive exclusions for unusable or duplicate snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

EXCLUSIONS_FILENAME = "snapshot_exclusions.yaml"
EXCLUSIONS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SnapshotExclusion:
    snapshot: str
    reason: str
    retained_snapshot: str | None = None


def load_snapshot_exclusions(root: str | Path) -> dict[str, SnapshotExclusion]:
    """Load and validate exclusions; a missing file means no exclusions."""
    root = Path(root)
    path = root / EXCLUSIONS_FILENAME
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict) or data.get("schema_version") != EXCLUSIONS_SCHEMA_VERSION:
        raise ValueError(f"unsupported or malformed exclusion manifest: {path}")
    entries = data.get("excluded_snapshots", [])
    if not isinstance(entries, list):
        raise ValueError("excluded_snapshots must be a list")
    result: dict[str, SnapshotExclusion] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each snapshot exclusion must be a mapping")
        snapshot = str(entry.get("snapshot", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        retained = str(entry.get("retained_snapshot", "")).strip() or None
        if not snapshot or Path(snapshot).name != snapshot or not reason:
            raise ValueError("snapshot exclusions require a direct-child name and reason")
        if snapshot in result:
            raise ValueError(f"duplicate snapshot exclusion {snapshot!r}")
        if retained is not None and Path(retained).name != retained:
            raise ValueError("retained_snapshot must be a direct-child name")
        result[snapshot] = SnapshotExclusion(snapshot, reason, retained)
    return result
