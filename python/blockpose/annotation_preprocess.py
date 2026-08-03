"""Thin native preprocessing adapters used by the annotation GUI."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np


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
