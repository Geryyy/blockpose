"""Leakage-safe plan priors for snapshot evaluation.

Topology/slot association may be curated using reviewed GT, but a target's
evaluated pose is never used to build its prior.  Fixtures record every source
snapshot so this invariant remains machine-auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from blockpose.pose import BLOCK_DIMS, BlockPose, quat_xyzw_to_rotmat

DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "config/snapshot_plan_priors.yaml"
DEFAULT_PLAN_STYLE = Path(__file__).resolve().parents[2] / "assets/wall_plans.yaml"


@dataclass(frozen=True)
class SnapshotPlanFixture:
    target_id: str
    stable_slot_id: str | None
    prior: BlockPose | None
    covariance_diagonal: tuple[float, ...] | None
    search_window_m: float
    source_snapshot_ids: tuple[str, ...]
    exclusion_reason: str | None
    plan_style: str | None
    prior_source: str


def _validate_plan_style(path: str | Path) -> set[str]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    dims = tuple(float(v) for v in raw.get("defaults", {}).get("block_size", ()))
    if dims != BLOCK_DIMS:
        raise ValueError(f"plan style must use canonical dimensions {BLOCK_DIMS}, got {dims}")
    plans = set(raw.get("wall_plans") or {})
    if not plans:
        raise ValueError("plan style has no wall_plans")
    return plans


def load_snapshot_plan_fixtures(
    path: str | Path = DEFAULT_FIXTURES, *, plan_style_path: str | Path = DEFAULT_PLAN_STYLE
) -> dict[str, SnapshotPlanFixture]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if raw.get("schema_version") != 1:
        raise ValueError("snapshot plan fixture schema_version must be 1")
    result = {}
    named_plans = {
        str(item["plan_style"])
        for item in (raw.get("targets") or {}).values()
        if item.get("plan_style") is not None
    }
    if named_plans:
        available = _validate_plan_style(plan_style_path)
        unknown = named_plans - available
        if unknown:
            raise ValueError(f"unknown named plan styles: {sorted(unknown)}")
    for target_id, item in (raw.get("targets") or {}).items():
        evaluated_snapshot = str(target_id).split("/", 1)[0]
        sources = tuple(str(value) for value in item.get("source_snapshot_ids", ()))
        if evaluated_snapshot in sources:
            raise ValueError(f"leaking source snapshot for {target_id}")
        window = float(item["search_window_m"])
        if window != 0.10:
            raise ValueError(f"search window for {target_id} must be 0.10 m")
        pose = None
        covariance = None
        pose_raw = item.get("prior")
        if pose_raw is not None:
            if not sources:
                raise ValueError(f"prior for {target_id} has no source snapshots")
            if tuple(float(v) for v in pose_raw["dimensions"]) != BLOCK_DIMS:
                raise ValueError(f"non-canonical prior dimensions for {target_id}")
            pose = BlockPose(
                np.asarray(pose_raw["position"], dtype=np.float64),
                quat_xyzw_to_rotmat(np.asarray(pose_raw["orientation_xyzw"], dtype=np.float64)),
                source="snapshot_plan_loso",
                dims=BLOCK_DIMS,
            )
            covariance = tuple(float(v) for v in item["covariance_diagonal"])
            if len(covariance) != 6:
                raise ValueError(f"covariance for {target_id} must have six entries")
            if any(value <= 0.0 for value in covariance) or any(
                value > np.deg2rad(10.0) ** 2 + 1e-5 for value in covariance[3:]
            ):
                raise ValueError(f"unsafe covariance for {target_id}")
        exclusion = item.get("exclusion_reason")
        if (pose is None) == (exclusion is None):
            raise ValueError(f"{target_id} must have exactly one prior or exclusion")
        result[str(target_id)] = SnapshotPlanFixture(
            str(target_id),
            item.get("stable_slot_id"),
            pose,
            covariance,
            window,
            sources,
            exclusion,
            str(item["plan_style"]) if item.get("plan_style") is not None else None,
            str(item["prior_source"]),
        )
    return result
