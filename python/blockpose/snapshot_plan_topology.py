"""Leakage-safe runtime topology for snapshot VerifyPlaced diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blockpose.snapshot_plan_fixtures import DEFAULT_FIXTURES, load_snapshot_plan_fixtures
from blockpose.verify_placed import CourseConstraint, PlacedBlockPrior

DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[2] / "config/snapshot_plan_topology.yaml"


@dataclass(frozen=True)
class SnapshotPlanTopology:
    priors_by_snapshot: dict[str, tuple[PlacedBlockPrior, ...]]
    constraints_by_snapshot: dict[str, tuple[CourseConstraint, ...]]
    provenance: dict[str, object]


def load_snapshot_plan_topology(
    path: str | Path = DEFAULT_TOPOLOGY,
    *,
    fixture_path: str | Path = DEFAULT_FIXTURES,
) -> SnapshotPlanTopology:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if raw.get("schema_version") != 1:
        raise ValueError("snapshot topology schema_version must be 1")
    if float(raw.get("search_window_m", -1.0)) != 0.10:
        raise ValueError("snapshot topology search_window_m must be 0.10")
    fixtures = load_snapshot_plan_fixtures(fixture_path)
    grouped: dict[str, list[PlacedBlockPrior]] = {}
    target_to_id: dict[str, str] = {}
    for target_id, fixture in fixtures.items():
        if fixture.prior is None:
            continue
        snapshot_id, local_id = target_id.split("/", 1)
        if snapshot_id in fixture.source_snapshot_ids:
            raise ValueError(f"topology leaks evaluated snapshot for {target_id}")
        block_id = local_id
        target_to_id[target_id] = block_id
        covariance = (
            np.diag(np.asarray(fixture.covariance_diagonal, dtype=np.float64))
            if fixture.covariance_diagonal is not None
            else None
        )
        grouped.setdefault(snapshot_id, []).append(
            PlacedBlockPrior(
                block_id,
                fixture.prior,
                covariance,
                source_snapshot_ids=fixture.source_snapshot_ids,
            )
        )

    constraints: dict[str, list[CourseConstraint]] = {}
    seen_courses: set[tuple[str, str]] = set()
    for item in raw.get("courses") or []:
        snapshot_id = str(item["snapshot_id"])
        course_id = str(item["course_id"])
        key = (snapshot_id, course_id)
        if key in seen_courses:
            raise ValueError(f"duplicate snapshot course {key!r}")
        seen_courses.add(key)
        known = {prior.block_id for prior in grouped.get(snapshot_id, [])}
        block_ids = tuple(str(value) for value in item["block_ids"])
        if set(block_ids) - known:
            raise ValueError(f"unknown topology block in course {key!r}")
        offsets: dict[tuple[str, str], np.ndarray] = {}
        for offset in item.get("nominal_offsets") or []:
            pair = (str(offset["from"]), str(offset["to"]))
            if pair in offsets:
                raise ValueError(f"duplicate topology offset {pair!r}")
            offsets[pair] = np.asarray(offset["offset"], dtype=np.float64)
        constraints.setdefault(snapshot_id, []).append(
            CourseConstraint(
                course_id=course_id,
                block_ids=block_ids,
                nominal_offsets=offsets,
                shared_yaw_sigma_deg=float(item.get("shared_yaw_sigma_deg", 1.0)),
                joint_gap_m=(
                    float(item["joint_gap_m"]) if item.get("joint_gap_m") is not None else None
                ),
                translation_sigma_m=float(item.get("translation_sigma_m", 0.05)),
                plan_name=(str(item["plan_name"]) if item.get("plan_name") is not None else None),
                source_snapshot_ids=tuple(
                    str(value) for value in item.get("source_snapshot_ids", ())
                ),
                evaluated_snapshot_id=snapshot_id,
            )
        )
    return SnapshotPlanTopology(
        {key: tuple(value) for key, value in grouped.items()},
        {key: tuple(value) for key, value in constraints.items()},
        {
            "topology_file": str(Path(path)),
            "fixture_file": str(Path(fixture_path)),
            "prior_source": "cross_snapshot_loso",
            "ground_truth_loaded": False,
            "target_ids": sorted(target_to_id),
        },
    )
