"""Project LiDAR points into the camera image and colorize them from RGB.

The registration snapshots pair a Seyond LiDAR cloud with a Blackfly RGB image.
The transforms in ``config/transforms_pzs100_initial_pose_1.yaml`` and the
intrinsics/distortion in ``config/calibration/calib_blackfly_to_seyond.yaml``
let us move points between the ``seyond``, ``blackfly_rotated`` (camera), and
``world`` frames and colorize the cloud by sampling the RGB image.

Frame conventions (both from ``sensor_msgs/CameraInfo``):
    - camera / optical frame: X right, Y down, Z forward out of the lens;
    - image pixels: u right (column), v down (row).
"""

from __future__ import annotations

import numpy as np

from blockpose.io import CameraModel, apply_transform


def world_to_camera(pts_world: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Transform (N, 3) world-frame points into the camera (optical) frame."""
    return apply_transform(pts_world, np.linalg.inv(T_world_camera))


def _apply_distortion(x: np.ndarray, y: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply OpenCV plumb-bob / rational-polynomial distortion to normalized coords.

    ``D`` has length 5 ([k1,k2,p1,p2,k3]) or 8 ([k1,k2,p1,p2,k3,k4,k5,k6]);
    an empty ``D`` is treated as an ideal pinhole (no distortion).
    """
    if D.size == 0:
        return x, y
    coeffs = np.zeros(8, dtype=np.float64)
    coeffs[: D.size] = D
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def project_points(
    pts_cam: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixel coordinates.

    Args:
        pts_cam: (N, 3) float points in the camera (optical) frame.
        camera: intrinsics + distortion.

    Returns:
        (uv, in_front) where uv is (N, 2) float pixel coordinates [u, v] and
        in_front is an (N,) bool mask of points ahead of the camera (Z > 0).
        Pixel coordinates for points with Z <= 0 are meaningless; callers should
        gate on both in_front and the image bounds (see ``colorize_world_points``).
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    z = pts_cam[:, 2]
    in_front = z > 0
    z_safe = np.where(in_front, z, 1.0)
    x = pts_cam[:, 0] / z_safe
    y = pts_cam[:, 1] / z_safe

    x_d, y_d = _apply_distortion(x, y, camera.D)
    u = camera.fx * x_d + camera.cx
    v = camera.fy * y_d + camera.cy
    return np.column_stack([u, v]), in_front


def colorize_world_points(
    pts_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    rgb: np.ndarray,
    background: tuple[float, float, float] = (0.55, 0.55, 0.55),
) -> tuple[np.ndarray, np.ndarray]:
    """Colorize a world-frame cloud by projecting it into the RGB image.

    Args:
        pts_world: (N, 3) world-frame points.
        T_world_camera: (4, 4) camera -> world transform.
        camera: intrinsics + distortion + image size.
        rgb: (H, W, 3) uint8 image (H == camera.height, W == camera.width).
        background: fallback color (floats in [0, 1]) for points that do not
            project into the image (behind the camera or out of frame).

    Returns:
        (colors, hit) where colors is (N, 3) float64 in [0, 1] and hit is an
        (N,) bool mask marking points that landed inside the image.
    """
    rgb = np.asarray(rgb)
    if rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            f"RGB image shape {rgb.shape[:2]} does not match camera "
            f"({camera.height}, {camera.width})"
        )

    pts_cam = world_to_camera(pts_world, T_world_camera)
    uv, in_front = project_points(pts_cam, camera)

    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    hit = in_front & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

    colors = np.tile(np.asarray(background, dtype=np.float64), (len(pts_world), 1))
    colors[hit] = rgb[v[hit], u[hit], :].astype(np.float64) / 255.0
    return colors, hit
