"""Versioned, auditable query taxonomy for the frozen real snapshots.

The manifest is deliberately independent of annotations: geometry may be used
to propose these labels, but never silently changes them.  ``audit_manifest``
is the explicit guard that keeps the two reviewed datasets in lock-step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from blockpose.annotations import load_annotations
from blockpose.snapshot_exclusions import load_snapshot_exclusions

DEFAULT_SCENARIOS = Path(__file__).resolve().parents[2] / "config" / "snapshot_scenarios.yaml"
SCENARIOS = frozenset({"free", "placed_isolated", "touching", "wall", "gripped"})
QUERIES = frozenset({"detect_free", "verify_placed", "estimate_gripped"})
POSE_QUALITIES = frozenset({"precise", "approximate"})


@dataclass(frozen=True)
class SnapshotScenario:
    target_id: str
    scenario: str
    query: str
    pose_quality: str
    support: str | None
    course: str | None
    neighbors: tuple[str, ...]
    occluder: str | None
    notes: str
    classification_status: str
    classification_evidence: str


def load_snapshot_scenarios(path: str | Path = DEFAULT_SCENARIOS) -> dict[str, SnapshotScenario]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if data.get("schema_version") != 1:
        raise ValueError("snapshot scenario schema_version must be 1")
    result: dict[str, SnapshotScenario] = {}
    for target_id, raw in (data.get("targets") or {}).items():
        target_id = str(target_id)
        scenario, query = str(raw["scenario"]), str(raw["query"])
        quality = str(raw["pose_quality"])
        if scenario not in SCENARIOS or query not in QUERIES or quality not in POSE_QUALITIES:
            raise ValueError(f"invalid taxonomy for {target_id}")
        expected_query = {
            "free": "detect_free",
            "gripped": "estimate_gripped",
        }.get(scenario, "verify_placed")
        if query != expected_query:
            raise ValueError(f"query is inconsistent with scenario for {target_id}")
        status = str(raw["classification_status"])
        if status not in {"provisional", "reviewed"}:
            raise ValueError(f"invalid classification status for {target_id}")
        neighbors = tuple(str(value) for value in raw["neighbors"])
        result[target_id] = SnapshotScenario(
            target_id,
            scenario,
            query,
            quality,
            raw["support"],
            raw["course"],
            neighbors,
            raw["occluder"],
            str(raw["notes"]),
            status,
            str(raw["classification_evidence"]),
        )
    _validate_relations(result)
    return result


def _validate_relations(entries: dict[str, SnapshotScenario]) -> None:
    for key, entry in entries.items():
        snapshot, _ = key.split("/", 1)
        for neighbor in entry.neighbors:
            other_key = f"{snapshot}/{neighbor}"
            if other_key not in entries or key.split("/", 1)[1] not in entries[other_key].neighbors:
                raise ValueError(f"non-reciprocal neighbor relation {key} -> {neighbor}")
        seen, support = {key}, entry.support
        while support is not None:
            support_key = f"{snapshot}/{support}"
            if support_key not in entries:
                raise ValueError(f"unknown support {support_key}")
            if support_key in seen:
                raise ValueError(f"support cycle involving {key}")
            seen.add(support_key)
            support = entries[support_key].support


def approved_target_ids(snapshots_dir: str | Path) -> set[str]:
    root = Path(snapshots_dir)
    excluded = load_snapshot_exclusions(root)
    ids: set[str] = set()
    for path in sorted(root.glob("*/annotations.yaml")):
        if path.parent.name in excluded:
            continue
        annotations = load_annotations(path)
        ids.update(
            f"{annotations.snapshot}/{target.id}"
            for target in annotations.targets
            if target.is_evaluation_target
        )
    return ids


def audit_manifest(snapshots_dir: str | Path, path: str | Path = DEFAULT_SCENARIOS) -> dict:
    entries = load_snapshot_scenarios(path)
    approved = approved_target_ids(snapshots_dir)
    configured = set(entries)
    return {
        "approved_targets": len(approved),
        "configured_targets": len(configured),
        "missing": sorted(approved - configured),
        "extra": sorted(configured - approved),
        "approximate": sorted(
            key for key, value in entries.items() if value.pose_quality == "approximate"
        ),
        "classification_status_counts": {
            status: sum(value.classification_status == status for value in entries.values())
            for status in ("provisional", "reviewed")
        },
    }


def acceptance_scenarios(
    entries: dict[str, SnapshotScenario],
) -> dict[str, SnapshotScenario]:
    """Return only reviewed classifications eligible for stratified acceptance."""
    return {
        key: value for key, value in entries.items() if value.classification_status == "reviewed"
    }


def taxonomy_acceptance_summary(entries: dict[str, SnapshotScenario]) -> dict:
    """Describe eligibility without scoring one query using another query's output."""
    eligible = acceptance_scenarios(entries)
    queries = sorted({value.query for value in eligible.values()})
    return {
        "eligible_targets": len(eligible),
        "excluded_provisional_targets": len(entries) - len(eligible),
        "queries": queries,
        "status": (
            "available"
            if not queries or queries == ["detect_free"]
            else "unavailable_until_query_specific_evaluation"
        ),
    }
