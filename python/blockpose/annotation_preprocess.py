"""Thin native preprocessing adapters used by the annotation GUI."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class GroundRemovalParams:
    z_percentile: float = 5.0
    thickness: float = 0.15
    ransac_distance: float = 0.05
    normal_min_z: float = 0.7
    ransac_iterations: int = 500
    ransac_seed: int = 0
    local_cell_size: float = 0.5
    local_clearance: float = 0.05


@dataclass(frozen=True)
class Pzs100SelfFilterParams:
    """Portable PZS100 rail-box profile consumed by the native core."""

    outboard_extent_m: float = 0.30
    cross_rail_extent_m: float = 0.90
    rail_end_margin_m: float = 0.00
    # Additional exclusion volume for the jaw carrier/boom/arm structurally
    # above both jaws -- not covered by the outward-facing rail boxes above.
    # Both default to 0.0 (feature off): `filter_pzs100_gripper` forwards
    # these fields generically to the native `Pzs100SelfFilterParameters`, so
    # callers that don't set them keep the pre-existing rail-only behaviour.
    above_exclusion_height_m: float = 0.00
    above_exclusion_margin_m: float = 0.00


@dataclass(frozen=True)
class Pzs100RailConfig:
    name: str
    parent_frame: str
    rail_frame: str
    outward_axis_parent: np.ndarray


@dataclass(frozen=True)
class Pzs100SelfFilterConfig:
    enabled: bool
    parameters: Pzs100SelfFilterParams
    rails: tuple[Pzs100RailConfig, ...]


def load_pzs100_self_filter_config(path: str | Path) -> Pzs100SelfFilterConfig:
    """Read and validate the checked-in, ROS-free PZS100 filter profile."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if payload.get("schema_version") != 1 or payload.get("model") != "pzs100":
        raise ValueError("expected schema_version: 1 and model: pzs100")
    rails = []
    for item in payload.get("rails", []):
        axis = np.asarray(item["outward_axis_parent"], dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-9:
            raise ValueError(f"rail {item.get('name', '?')!r} needs a finite non-zero outward axis")
        rails.append(
            Pzs100RailConfig(
                name=str(item["name"]),
                parent_frame=str(item["parent_frame"]),
                rail_frame=str(item["rail_frame"]),
                outward_axis_parent=axis,
            )
        )
    if len(rails) != 2:
        raise ValueError("PZS100 profile must define exactly two rails")
    parameters = Pzs100SelfFilterParams(
        outboard_extent_m=float(payload["outboard_extent_m"]),
        cross_rail_extent_m=float(payload["cross_rail_extent_m"]),
        rail_end_margin_m=float(payload["rail_end_margin_m"]),
        # Defaulted rather than required: existing profiles are schema_version
        # 1 and predate this feature, and must keep loading unchanged.
        above_exclusion_height_m=float(payload.get("above_exclusion_height_m", 0.0)),
        above_exclusion_margin_m=float(payload.get("above_exclusion_margin_m", 0.0)),
    )
    if (
        not np.all(np.isfinite(tuple(vars(parameters).values())))
        or parameters.outboard_extent_m <= 0
        or parameters.cross_rail_extent_m <= 0
        or parameters.rail_end_margin_m < 0
        or parameters.above_exclusion_height_m < 0
        or parameters.above_exclusion_margin_m < 0
    ):
        raise ValueError("PZS100 extents must be finite and positive (margin non-negative)")
    return Pzs100SelfFilterConfig(bool(payload.get("enabled", True)), parameters, tuple(rails))


def _core():
    return importlib.import_module("blockpose._core")


def _points(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape((-1, 3))


def remove_ground(
    points: np.ndarray, params: GroundRemovalParams | None = None
) -> tuple[np.ndarray, dict]:
    core = _core()
    native = core.GroundParameters()
    for name, value in vars(params or GroundRemovalParams()).items():
        setattr(native, name, value)
    result = core.remove_ground(np.ascontiguousarray(points, dtype=np.float64), native)
    return _points(result["points"]), dict(
        result["ground"],
        input_count=int(result["input_count"]),
        removed_count=int(result["removed_count"]),
    )


def filter_workspace(
    points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> tuple[np.ndarray, int]:
    result = _core().filter_workspace(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(minimum, dtype=np.float64),
        np.ascontiguousarray(maximum, dtype=np.float64),
    )
    return _points(result["points"]), int(result["removed_count"])


def filter_exclusion_boxes(
    points: np.ndarray, centers: np.ndarray, rotations: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, int]:
    result = _core().filter_exclusion_boxes(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(centers, dtype=np.float64),
        np.ascontiguousarray(rotations, dtype=np.float64),
        np.ascontiguousarray(sizes, dtype=np.float64),
    )
    return _points(result["points"]), int(result["removed_count"])


def pzs100_self_filter_available() -> tuple[bool, str]:
    """Return whether the installed native extension exposes the PZS100 API."""
    try:
        core = _core()
    except ImportError as error:
        return False, f"native blockpose._core unavailable ({error})"
    required = ("Pzs100SelfFilterParameters", "filter_pzs100_gripper")
    missing = [name for name in required if not hasattr(core, name)]
    if missing:
        return False, "native blockpose._core lacks " + ", ".join(missing)
    return True, ""


def filter_pzs100_gripper(
    points: np.ndarray,
    parent_positions: np.ndarray,
    rail_positions: np.ndarray,
    parent_rotations: np.ndarray,
    outward_axes_parent: np.ndarray,
    params: Pzs100SelfFilterParams,
) -> tuple[np.ndarray, int, dict[str, np.ndarray]]:
    """Apply the native rail-box self-filter and return its diagnostic boxes."""
    available, reason = pzs100_self_filter_available()
    if not available:
        raise ImportError(reason)
    core = _core()
    native = core.Pzs100SelfFilterParameters()
    for name, value in vars(params).items():
        setattr(native, name, value)
    result = core.filter_pzs100_gripper(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(parent_positions, dtype=np.float64),
        np.ascontiguousarray(rail_positions, dtype=np.float64),
        np.ascontiguousarray(parent_rotations, dtype=np.float64),
        np.ascontiguousarray(outward_axes_parent, dtype=np.float64),
        native,
    )
    boxes = {
        name: np.asarray(result[name], dtype=np.float64)
        for name in (
            "centers",
            "rotations",
            "sizes",
            "rail_starts",
            "rail_ends",
            # Above-gripper exclusion volume: kept under separate keys (not
            # appended to centers/rotations/sizes) so the existing GUI overlay
            # keeps seeing only the two jaw boxes there and can render this
            # volume distinctly. Shape (0, 3) etc. when the feature is off.
            "above_centers",
            "above_rotations",
            "above_sizes",
        )
    }
    return _points(result["points"]), int(result["removed_count"]), boxes
