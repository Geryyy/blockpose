"""Tests for blockpose.projection point projection and RGB colorization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from blockpose.io import (
    CameraModel,
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_T_world_camera,
)
from blockpose.projection import (
    colorize_world_points,
    project_points,
    world_to_camera,
)

REPO_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
CAMERA_CALIB = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
SNAPSHOT_1 = sorted(SNAPSHOTS_DIR.iterdir())[0]


def _pinhole(fx=500.0, fy=520.0, cx=320.0, cy=240.0, width=640, height=480, D=None):
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return CameraModel(K=K, width=width, height=height, D=D if D is not None else [])


def test_optical_axis_maps_to_principal_point() -> None:
    cam = _pinhole()
    pts = np.array([[0.0, 0.0, 3.0]])
    uv, in_front = project_points(pts, cam)
    np.testing.assert_allclose(uv[0], [cam.cx, cam.cy], atol=1e-9)
    assert bool(in_front[0])


def test_pinhole_projection_no_distortion() -> None:
    cam = _pinhole()
    pts = np.array([[0.5, -0.25, 2.0]])
    uv, _ = project_points(pts, cam)
    expected_u = cam.fx * (0.5 / 2.0) + cam.cx
    expected_v = cam.fy * (-0.25 / 2.0) + cam.cy
    np.testing.assert_allclose(uv[0], [expected_u, expected_v], atol=1e-9)


def test_points_behind_camera_flagged() -> None:
    cam = _pinhole()
    pts = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0], [0.0, 0.0, 0.0]])
    _, in_front = project_points(pts, cam)
    np.testing.assert_array_equal(in_front, [True, False, False])


def test_zero_distortion_matches_pinhole() -> None:
    cam_ideal = _pinhole(D=[])
    cam_zero = _pinhole(D=[0.0] * 8)
    pts = np.array([[0.6, 0.4, 2.5], [-0.3, 0.2, 1.8]])
    uv_ideal, _ = project_points(pts, cam_ideal)
    uv_zero, _ = project_points(pts, cam_zero)
    np.testing.assert_allclose(uv_ideal, uv_zero, atol=1e-12)


def test_world_to_camera_inverts_pose() -> None:
    # Camera looking down the world +Y axis, translated.
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    t = np.array([1.0, 2.0, 3.0])
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = R
    T_world_camera[:3, 3] = t
    pts_cam = np.array([[0.1, 0.2, 5.0], [-1.0, 0.0, 2.0]])
    pts_world = (R @ pts_cam.T).T + t
    recovered = world_to_camera(pts_world, T_world_camera)
    np.testing.assert_allclose(recovered, pts_cam, atol=1e-9)


def test_colorize_samples_correct_pixel() -> None:
    # Identity camera pose: world frame == camera (optical) frame.
    cam = _pinhole(fx=100.0, fy=100.0, cx=1.5, cy=1.5, width=4, height=4)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[3, 2] = [255, 0, 0]  # row v=3, col u=2

    # Point that projects to (u=2, v=3): x/z*fx+cx=2 -> x/z=0.005; y/z=0.015
    pt = np.array([[0.005 * 5.0, 0.015 * 5.0, 5.0]])
    colors, hit = colorize_world_points(pt, np.eye(4), cam, rgb)
    assert bool(hit[0])
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1e-9)


def test_colorize_out_of_frame_uses_background() -> None:
    cam = _pinhole(fx=100.0, fy=100.0, cx=1.5, cy=1.5, width=4, height=4)
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    bg = (0.1, 0.2, 0.3)
    behind = np.array([[0.0, 0.0, -1.0]])
    colors, hit = colorize_world_points(behind, np.eye(4), cam, rgb, background=bg)
    assert not bool(hit[0])
    np.testing.assert_allclose(colors[0], bg, atol=1e-12)


def test_colorize_rejects_mismatched_image_size() -> None:
    cam = _pinhole(width=4, height=4)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    try:
        colorize_world_points(np.zeros((1, 3)), np.eye(4), cam, rgb)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on image/camera size mismatch")


def test_real_snapshot_colorization_hits_most_points() -> None:
    pts_world, _ = load_snapshot(SNAPSHOT_1, TRANSFORM_CONFIG)
    T_world_camera = load_T_world_camera(TRANSFORM_CONFIG)
    cam = load_camera_model(CAMERA_CALIB)
    rgb = load_rgb(SNAPSHOT_1 / "rgb.png")
    colors, hit = colorize_world_points(pts_world, T_world_camera, cam, rgb)
    assert colors.shape == (len(pts_world), 3)
    assert 0.0 <= colors.min() and colors.max() <= 1.0
    # These snapshots are captured looking at the scene; most points fall in-frame.
    assert hit.mean() > 0.5
