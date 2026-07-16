"""Tests for blockpose.detection pipeline on real snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from blockpose.detection import detect_blocks
from blockpose.evaluate import evaluate
from blockpose.io import load_snapshot

REPO_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"

SNAPSHOT_DIRS = sorted(SNAPSHOTS_DIR.iterdir())


@pytest.fixture(scope="module")
def snapshot1_data():
    """Load first snapshot once for all tests in this module."""
    return load_snapshot(SNAPSHOT_DIRS[0], TRANSFORM_CONFIG)


@pytest.fixture(scope="module")
def snapshot1_detections(snapshot1_data):
    """Run detection on first snapshot once."""
    pts_world, _ = snapshot1_data
    return detect_blocks(pts_world)


def test_detect_returns_nonempty(snapshot1_detections) -> None:
    assert len(snapshot1_detections) >= 1


def test_detect_positions_reasonable(snapshot1_detections) -> None:
    """All detected block centers should be in a plausible Z range in world frame."""
    for pose in snapshot1_detections:
        z = float(pose.position[2])
        assert 0.0 <= z <= 3.0, (
            f"Detected block Z={z:.3f} out of range — likely coordinate frame error"
        )


def test_detect_rotation_is_valid_SO3(snapshot1_detections) -> None:
    for pose in snapshot1_detections:
        R = pose.rotation
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)


@pytest.mark.xfail(strict=False, reason="prototype: block count may differ from GT")
def test_detect_count_close_to_gt(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    assert abs(len(snapshot1_detections) - len(gt)) <= 1


@pytest.mark.xfail(strict=False, reason="prototype: translation threshold pre-ICP")
def test_eval_translation_error_threshold(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    result = evaluate(snapshot1_detections, gt)
    if result.matched == 0:
        pytest.skip("No matched pairs to evaluate")
    assert result.mean_translation_m < 0.25


@pytest.mark.xfail(strict=False, reason="prototype: rotation threshold pre-ICP")
def test_eval_rotation_error_threshold(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    result = evaluate(snapshot1_detections, gt)
    if result.matched == 0:
        pytest.skip("No matched pairs to evaluate")
    assert result.mean_rotation_deg < 30.0


def test_all_snapshots_run_without_error() -> None:
    """Smoke test: pipeline runs on all snapshots without raising."""
    for snap_dir in SNAPSHOT_DIRS:
        pts_world, _ = load_snapshot(snap_dir, TRANSFORM_CONFIG)
        detections = detect_blocks(pts_world)
        # Just verify types are correct
        for pose in detections:
            assert pose.position.shape == (3,)
            assert pose.rotation.shape == (3, 3)


def _make_flat_ground(n: int = 2000, z: float = 0.0, spread: float = 10.0) -> np.ndarray:
    """Dense flat ground plane at Z=z, spread over XY."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-spread, spread, size=(n, 2))
    zs = np.full((n, 1), z)
    return np.hstack([xy, zs])


def test_dimension_gating_rejects_tall_cluster() -> None:
    """Tall narrow structure (dz=8m) must be rejected by Z-extent gate."""
    rng = np.random.default_rng(1)
    # Points in a 0.5x0.5x8 m column at origin, z in [0.2, 8.2]
    n = 1000
    xyz = rng.uniform([0.0, 0.0, 0.2], [0.5, 0.5, 8.2], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )


def test_dimension_gating_rejects_wide_cluster() -> None:
    """Wide wall face (dx=5m) must be rejected by XY-extent gate."""
    rng = np.random.default_rng(2)
    # Points in a 5x0.3x0.9 m wall, z in [0.2, 1.1]
    n = 1000
    xyz = rng.uniform([0.0, 0.0, 0.2], [5.0, 0.3, 1.1], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )


def test_dimension_gating_rejects_high_altitude_cluster() -> None:
    """Block-sized box at high altitude (z_center~4.45m) must be rejected by z-center gate."""
    rng = np.random.default_rng(3)
    # Block-sized box at z in [4.0, 4.9], center = 4.45 >> 3.0 m gate
    n = 500
    xyz = rng.uniform([0.0, 0.0, 4.0], [0.6, 0.6, 4.9], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )
