"""Data loading utilities for blockpose registration snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from blockpose.pose import BlockPose, quat_xyzw_to_rotmat

# Default path relative to repo root (two levels up from tools/, one up from python/)
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_TRANSFORM_CONFIG = _REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"


def load_pcd(path: str | Path) -> np.ndarray:
    """Load a PCD file and return points as (N, 3) float32 array in sensor frame."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Loaded empty point cloud from {path}")
    pts = np.asarray(pcd.points).copy().astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {pts.shape}")
    return pts


def load_T_world_seyond(config_path: str | Path | None = None) -> np.ndarray:
    """Load T_world_seyond as a (4, 4) float64 homogeneous transform matrix."""
    if config_path is None:
        config_path = _DEFAULT_TRANSFORM_CONFIG
    config_path = Path(config_path)
    with config_path.open() as f:
        data = yaml.safe_load(f)
    for entry in data["transforms"]:
        if entry["name"] == "T_world_seyond":
            T = np.array(entry["matrix"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
            return T
    raise KeyError("T_world_seyond not found in transform config")


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a (4, 4) homogeneous transform to (N, 3) points.

    Returns (N, 3) float64 in the target frame.
    Avoids allocating an (N, 4) array by applying R and t separately.
    """
    pts = np.asarray(pts, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t


def load_ground_truth(path: str | Path) -> list[BlockPose]:
    """Load ground-truth block poses from a registrations.yaml file.

    Only records where registration_ok is True are returned.
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    poses = []
    for rec in data.get("records", []):
        if not rec.get("registration_ok", False):
            continue
        pose_data = rec["pose"]
        position = np.array(pose_data["position"], dtype=np.float64)
        q = np.array(pose_data["orientation_xyzw"], dtype=np.float64)
        rotation = quat_xyzw_to_rotmat(q)
        poses.append(
            BlockPose(
                position=position,
                rotation=rotation,
                confidence=float(rec.get("confidence", 0.0)),
                source=str(rec.get("assigned_id", "")),
            )
        )
    return poses


def load_snapshot(
    snapshot_dir: str | Path,
    config_path: str | Path | None = None,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Load a registration snapshot: returns (pts_world, gt_poses).

    pts_world: (N, 3) float64 point cloud transformed to world frame.
    gt_poses: list of BlockPose objects from registrations.yaml.
    """
    snapshot_dir = Path(snapshot_dir)
    T = load_T_world_seyond(config_path)
    pts_sensor = load_pcd(snapshot_dir / "cloud.pcd")
    pts_world = apply_transform(pts_sensor, T)
    gt_poses = load_ground_truth(snapshot_dir / "registrations.yaml")
    return pts_world, gt_poses
