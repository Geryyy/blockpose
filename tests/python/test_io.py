"""Tests for blockpose.io data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from blockpose.io import (
    apply_transform,
    load_ground_truth,
    load_pcd,
    load_snapshot,
    load_T_world_seyond,
)

REPO_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"

# Use the first snapshot for deterministic tests
SNAPSHOT_1 = sorted(SNAPSHOTS_DIR.iterdir())[0]


def test_load_pcd_shape() -> None:
    pts = load_pcd(SNAPSHOT_1 / "cloud.pcd")
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert pts.dtype == np.float32
    assert pts.shape[0] > 1000


def test_load_pcd_returns_copy() -> None:
    pts = load_pcd(SNAPSHOT_1 / "cloud.pcd")
    pts_copy = pts.copy()
    pts[0, 0] = 999.0
    # Modifying the array shouldn't affect a second load (proves it's a copy)
    pts2 = load_pcd(SNAPSHOT_1 / "cloud.pcd")
    assert pts2[0, 0] != 999.0 or np.allclose(pts2[0, 0], pts_copy[0, 0])


def test_load_T_world_seyond_shape() -> None:
    T = load_T_world_seyond(TRANSFORM_CONFIG)
    assert T.shape == (4, 4)
    assert T.dtype == np.float64


def test_load_T_world_seyond_is_valid_SE3() -> None:
    T = load_T_world_seyond(TRANSFORM_CONFIG)
    # Bottom row must be [0, 0, 0, 1]
    np.testing.assert_allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-10)
    # Rotation block must be SO(3)
    R = T[:3, :3]
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_apply_transform_seyond_origin() -> None:
    """Seyond frame origin maps to the translation column of T_world_seyond."""
    T = load_T_world_seyond(TRANSFORM_CONFIG)
    origin = np.array([[0.0, 0.0, 0.0]])
    result = apply_transform(origin, T)
    expected = T[:3, 3]
    np.testing.assert_allclose(result[0], expected, atol=1e-4)


def test_apply_transform_output_dtype() -> None:
    T = load_T_world_seyond(TRANSFORM_CONFIG)
    pts = np.ones((5, 3), dtype=np.float32)
    result = apply_transform(pts, T)
    assert result.dtype == np.float64
    assert result.shape == (5, 3)


def test_load_ground_truth_count() -> None:
    # All 3 snapshots have 2-3 blocks; check first snapshot has at least 1
    gt = load_ground_truth(SNAPSHOT_1 / "registrations.yaml")
    assert len(gt) >= 1


def test_load_ground_truth_positions_finite() -> None:
    gt = load_ground_truth(SNAPSHOT_1 / "registrations.yaml")
    for pose in gt:
        assert np.all(np.isfinite(pose.position))
        assert np.all(np.isfinite(pose.rotation))


def test_load_ground_truth_rotations_valid_SO3() -> None:
    gt = load_ground_truth(SNAPSHOT_1 / "registrations.yaml")
    for pose in gt:
        R = pose.rotation
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_load_snapshot_world_frame_z_range() -> None:
    pts_world, gt = load_snapshot(SNAPSHOT_1, TRANSFORM_CONFIG)
    assert pts_world.shape[1] == 3
    assert pts_world.dtype == np.float64
    # Outdoor snapshots can include tall background objects above the work area.
    assert float(pts_world[:, 2].max()) < 25.0
    assert len(gt) >= 1
