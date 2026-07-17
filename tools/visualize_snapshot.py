#!/usr/bin/env python3
"""Visualize a registration snapshot point cloud and ground-truth blocks with Open3D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from blockpose.io import load_snapshot  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
BLOCK_EXTENTS_M = np.array([0.6, 0.6, 0.9], dtype=np.float64)
POINT_CLOUD_COLOR = [0.55, 0.55, 0.55]
GROUND_TRUTH_COLOR = [0.0, 0.8, 0.0]


def _repo_relative(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _validate_snapshot(snapshot: Path) -> None:
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {snapshot}")
    if not snapshot.is_dir():
        raise NotADirectoryError(f"Snapshot path is not a directory: {snapshot}")
    for filename in ("cloud.pcd", "registrations.yaml"):
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required snapshot file: {path}")


def _make_point_cloud(pts_world: np.ndarray) -> o3d.geometry.PointCloud:
    pts_world = np.asarray(pts_world, dtype=np.float64)
    if pts_world.ndim != 2 or pts_world.shape[1] != 3:
        raise ValueError(f"Expected point cloud with shape (N, 3), got {pts_world.shape}")
    if pts_world.shape[0] == 0:
        raise ValueError("Point cloud is empty")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_world)
    point_cloud.paint_uniform_color(POINT_CLOUD_COLOR)
    return point_cloud


def _make_ground_truth_box(pose: BlockPose) -> o3d.geometry.OrientedBoundingBox:
    """Create a world-frame oriented box from a ground-truth block pose."""
    box = o3d.geometry.OrientedBoundingBox(
        center=np.asarray(pose.position, dtype=np.float64),
        R=np.asarray(pose.rotation, dtype=np.float64),
        extent=BLOCK_EXTENTS_M,
    )
    box.color = GROUND_TRUTH_COLOR
    return box


def _build_geometries(
    pts_world: np.ndarray,
    ground_truth: list[BlockPose],
) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [
        _make_point_cloud(pts_world),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    geometries.extend(_make_ground_truth_box(pose) for pose in ground_truth)
    return geometries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Snapshot directory to visualize.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to transforms YAML, resolved relative to the repository root when relative.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    config = _repo_relative(args.config).resolve()

    _validate_snapshot(snapshot)
    if not config.is_file():
        raise FileNotFoundError(f"Transform config does not exist: {config}")

    pts_world, ground_truth = load_snapshot(snapshot, config)
    geometries = _build_geometries(pts_world, ground_truth)

    print(
        f"Loaded {len(pts_world)} world-frame points and "
        f"{len(ground_truth)} ground-truth block(s)."
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"BlockPose snapshot: {snapshot.name}",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
