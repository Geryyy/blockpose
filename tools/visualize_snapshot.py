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

from blockpose.io import (  # noqa: E402
    CameraModel,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_T_world_camera,
)
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.projection import colorize_world_points  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
DEFAULT_CAMERA_CALIB = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
BLOCK_EXTENTS_M = np.array([0.6, 0.6, 0.9], dtype=np.float64)
POINT_CLOUD_COLOR = [0.55, 0.55, 0.55]
GROUND_TRUTH_COLOR = [0.0, 0.8, 0.0]
CAMERA_FRUSTUM_COLOR = [1.0, 0.5, 0.0]
CAMERA_FRUSTUM_DEPTH_M = 1.0


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


def _make_point_cloud(
    pts_world: np.ndarray, colors: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    pts_world = np.asarray(pts_world, dtype=np.float64)
    if pts_world.ndim != 2 or pts_world.shape[1] != 3:
        raise ValueError(f"Expected point cloud with shape (N, 3), got {pts_world.shape}")
    if pts_world.shape[0] == 0:
        raise ValueError("Point cloud is empty")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts_world)
    if colors is None:
        point_cloud.paint_uniform_color(POINT_CLOUD_COLOR)
    else:
        point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
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


def _make_camera_frustum(
    T_world_camera: np.ndarray,
    camera: CameraModel,
    depth: float = CAMERA_FRUSTUM_DEPTH_M,
) -> o3d.geometry.LineSet:
    """Build a world-frame camera frustum wireframe from a pose and intrinsics.

    Assumes the standard pinhole/optical convention (X right, Y down, Z
    forward out of the lens) for the frame `T_world_camera` maps into, matching
    `sensor_msgs/CameraInfo` K/D/R/P — the same convention as `config/calibration/*.yaml`.
    Uses the linear intrinsics only (ignores distortion) for the wireframe corners.
    """
    K_inv = np.linalg.inv(camera.K)
    corners_px = np.array(
        [
            [0, 0, 1],
            [camera.width, 0, 1],
            [camera.width, camera.height, 1],
            [0, camera.height, 1],
        ],
        dtype=np.float64,
    )
    corners_cam = (K_inv @ corners_px.T).T * depth
    apex_cam = np.zeros((1, 3))
    points_cam = np.vstack([apex_cam, corners_cam])

    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    points_world = (R @ points_cam.T).T + t

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.colors = o3d.utility.Vector3dVector([CAMERA_FRUSTUM_COLOR] * len(lines))
    return frustum


def _build_geometries(
    pts_world: np.ndarray,
    ground_truth: list[BlockPose],
    T_world_camera: np.ndarray,
    camera: CameraModel,
    colors: np.ndarray | None = None,
) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [
        _make_point_cloud(pts_world, colors),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3
        ).transform(T_world_camera),
        _make_camera_frustum(T_world_camera, camera),
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
    parser.add_argument(
        "--camera-calib",
        type=Path,
        default=DEFAULT_CAMERA_CALIB,
        help="Path to camera calibration YAML, resolved relative to the repository "
        "root when relative.",
    )
    parser.add_argument(
        "--color",
        choices=("rgb", "uniform"),
        default="rgb",
        help="Point cloud coloring: 'rgb' projects each point into the Blackfly "
        "image and samples its color; 'uniform' paints a flat gray.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    config = _repo_relative(args.config).resolve()
    camera_calib = _repo_relative(args.camera_calib).resolve()

    _validate_snapshot(snapshot)
    if not config.is_file():
        raise FileNotFoundError(f"Transform config does not exist: {config}")
    if not camera_calib.is_file():
        raise FileNotFoundError(f"Camera calibration does not exist: {camera_calib}")

    pts_world, ground_truth = load_snapshot(snapshot, config)
    T_world_camera = load_T_world_camera(config)
    camera = load_camera_model(camera_calib)

    colors = None
    hit_count = 0
    if args.color == "rgb":
        rgb_path = snapshot / "rgb.png"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB image for --color rgb: {rgb_path}")
        rgb = load_rgb(rgb_path)
        colors, hit = colorize_world_points(pts_world, T_world_camera, camera, rgb)
        hit_count = int(hit.sum())

    geometries = _build_geometries(
        pts_world, ground_truth, T_world_camera, camera, colors
    )

    print(
        f"Loaded {len(pts_world)} world-frame points, "
        f"{len(ground_truth)} ground-truth block(s), and the camera frustum "
        f"(camera origin in world: {T_world_camera[:3, 3]})."
    )
    if args.color == "rgb":
        pct = 100.0 * hit_count / max(len(pts_world), 1)
        print(
            f"Colored {hit_count}/{len(pts_world)} points ({pct:.1f}%) from the "
            f"Blackfly image; the rest fall outside the camera FOV (shown gray)."
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
