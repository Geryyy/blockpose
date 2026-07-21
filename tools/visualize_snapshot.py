#!/usr/bin/env python3
"""Visualize a registration snapshot point cloud and ground-truth blocks with Open3D.

If the snapshot directory contains a `tf.yaml` (per-snapshot TF transforms
captured at recording time), the cloud- and camera-to-world transforms are
taken from it instead of the static `--config` file, so everything renders
aligned even when the crane's joints were not at the config's reference pose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

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
    load_tf_snapshot,
)
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.projection import (  # noqa: E402
    colorize_world_points,
    project_points,
    world_to_camera,
)

DEFAULT_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
DEFAULT_CAMERA_CALIB = REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"
# Fallback extents when a block's own points can't be measured (see
# _estimate_block_extents). registrations.yaml only carries position +
# orientation, not size, and this "upright" shape is wrong for the many
# blocks that rest lying on a long face (config/wall_plans.yaml default
# block_size is [0.9, 0.6, 0.6]) -- it's a last resort, not an assumption.
BLOCK_EXTENTS_M = np.array([0.6, 0.6, 0.9], dtype=np.float64)
MASK_PROXIMITY_RADIUS_M = 0.6
MIN_BLOCK_EXTENT_M = 0.3
MAX_BLOCK_EXTENT_M = 1.2
MIN_BLOCK_MASK_POINTS = 20
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


def _make_ground_truth_box(
    pose: BlockPose, extents: np.ndarray = BLOCK_EXTENTS_M
) -> o3d.geometry.OrientedBoundingBox:
    """Create a world-frame oriented box from a ground-truth block pose."""
    box = o3d.geometry.OrientedBoundingBox(
        center=np.asarray(pose.position, dtype=np.float64),
        R=np.asarray(pose.rotation, dtype=np.float64),
        extent=np.asarray(extents, dtype=np.float64),
    )
    box.color = GROUND_TRUTH_COLOR
    return box


def _load_detection_ids(registrations_path: Path) -> list[int]:
    """Detection ids for the registered ground-truth poses, in the same
    order/filter as `blockpose.io.load_ground_truth` (registration_ok only),
    so the result lines up index-for-index with its returned pose list.
    """
    with registrations_path.open() as f:
        data = yaml.safe_load(f)
    return [
        int(rec["detection_id"])
        for rec in data.get("records", [])
        if rec.get("registration_ok", False)
    ]


def _load_candidate_masks(snapshot: Path, metadata_path: Path) -> dict[int, np.ndarray]:
    """Load each candidate mask (per metadata.yaml), keyed by detection_id."""
    with metadata_path.open() as f:
        metadata = yaml.safe_load(f)
    masks: dict[int, np.ndarray] = {}
    for entry in metadata.get("candidate_masks", []):
        mask_path = snapshot / entry["file"]
        if mask_path.is_file():
            mask = np.asarray(o3d.io.read_image(str(mask_path)))
            masks[int(entry["detection_id"])] = mask > 0
    return masks


def _estimate_block_extents(
    pose: BlockPose,
    pts_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    mask: np.ndarray,
) -> np.ndarray:
    """Estimate a ground-truth block's (x, y, z) extents, in its own pose
    frame, from the LiDAR points inside its detection mask.

    A block can rest standing (0.6 x 0.6 x 0.9) or lying on a long face
    (0.9 x 0.6 x 0.6, per config/wall_plans.yaml), and registrations.yaml
    doesn't say which -- so rather than assume one, measure it from the
    points the detector actually assigned to this block. Points are further
    restricted to a small radius around the registered position, since the
    mask alone can bleed onto background/clutter (e.g. the crane, ground).
    Falls back to BLOCK_EXTENTS_M when too few points survive to measure.
    """
    pts_cam = world_to_camera(pts_world, T_world_camera)
    uv, in_front = project_points(pts_cam, camera)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    in_bounds = in_front & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

    idx = np.flatnonzero(in_bounds)
    idx = idx[mask[v[idx], u[idx]]]
    pts = pts_world[idx]

    near = np.linalg.norm(pts[:, :2] - pose.position[:2], axis=1) < MASK_PROXIMITY_RADIUS_M
    pts = pts[near]
    if len(pts) < MIN_BLOCK_MASK_POINTS:
        return BLOCK_EXTENTS_M

    local = (pose.rotation.T @ (pts - pose.position).T).T
    extents = np.percentile(local, 98, axis=0) - np.percentile(local, 2, axis=0)
    return np.clip(extents, MIN_BLOCK_EXTENT_M, MAX_BLOCK_EXTENT_M)


def _estimate_all_block_extents(
    snapshot: Path,
    ground_truth: list[BlockPose],
    pts_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
) -> list[np.ndarray]:
    """Per-block extents for every ground-truth pose, falling back to
    BLOCK_EXTENTS_M for any block whose mask/metadata isn't available.
    """
    metadata_path = snapshot / "metadata.yaml"
    registrations_path = snapshot / "registrations.yaml"
    if not metadata_path.is_file():
        return [BLOCK_EXTENTS_M] * len(ground_truth)

    detection_ids = _load_detection_ids(registrations_path)
    masks = _load_candidate_masks(snapshot, metadata_path)
    if len(detection_ids) != len(ground_truth):
        # Filtering got out of sync with load_ground_truth; don't guess.
        return [BLOCK_EXTENTS_M] * len(ground_truth)

    extents = []
    for pose, detection_id in zip(ground_truth, detection_ids, strict=True):
        mask = masks.get(detection_id)
        if mask is None:
            extents.append(BLOCK_EXTENTS_M)
        else:
            extents.append(
                _estimate_block_extents(pose, pts_world, T_world_camera, camera, mask)
            )
    return extents


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


def _origin_view_extrinsic() -> np.ndarray:
    """World -> camera extrinsic that plants the viewer at the world origin,
    looking down -x with world +z up, needing no manual rotation on open.

    Built from camera axes expressed in world coordinates (the same
    T_world_camera convention used elsewhere in this file): forward (+z) is
    world -x, down (+y) is world -z (so screen "up" is world +z), and right
    (+x) completes the right-handed frame. `draw_geometries`'s front/up/zoom
    can only aim the view, not pin an exact eye position, so this sets the
    extrinsic directly via the low-level Visualizer/ViewControl API.
    """
    forward = np.array([-1.0, 0.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, forward)
    R_world_camera = np.column_stack([right, down, forward])
    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = R_world_camera
    return np.linalg.inv(T_world_camera)


def _build_geometries(
    pts_world: np.ndarray,
    ground_truth: list[BlockPose],
    gt_extents: list[np.ndarray],
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
    geometries.extend(
        _make_ground_truth_box(pose, extents)
        for pose, extents in zip(ground_truth, gt_extents, strict=True)
    )
    return geometries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Snapshot directory to visualize.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the static transforms YAML, resolved relative to the repository "
        "root when relative. Ignored for the cloud/camera transforms when the "
        "snapshot has a tf.yaml.",
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

    tf_path = snapshot / "tf.yaml"
    if tf_path.is_file():
        tf_snapshot = load_tf_snapshot(tf_path)
        T_world_cloud = tf_snapshot.T_world_cloud
        T_world_camera = tf_snapshot.T_world_camera
        print(
            f"Using per-snapshot tf.yaml: {tf_snapshot.cloud_frame!r} and "
            f"{tf_snapshot.image_frame!r} transforms relative to "
            f"{tf_snapshot.reference_frame!r} (captured at recording time, "
            "not the static config)."
        )
    else:
        T_world_cloud = None  # load_snapshot falls back to the static config.
        T_world_camera = load_T_world_camera(config)

    pts_world, ground_truth = load_snapshot(snapshot, config, T_world_cloud=T_world_cloud)
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

    gt_extents = _estimate_all_block_extents(
        snapshot, ground_truth, pts_world, T_world_camera, camera
    )
    geometries = _build_geometries(
        pts_world, ground_truth, gt_extents, T_world_camera, camera, colors
    )

    print(
        f"Loaded {len(pts_world)} world-frame points, "
        f"{len(ground_truth)} ground-truth block(s), and the camera frustum "
        f"(camera origin in world: {T_world_camera[:3, 3]})."
    )
    for pose, extents in zip(ground_truth, gt_extents, strict=True):
        print(f"  {pose.source}: extents (x,y,z) = {np.round(extents, 3).tolist()}")
    if args.color == "rgb":
        pct = 100.0 * hit_count / max(len(pts_world), 1)
        print(
            f"Colored {hit_count}/{len(pts_world)} points ({pct:.1f}%) from the "
            f"Blackfly image; the rest fall outside the camera FOV (shown gray)."
        )
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BlockPose snapshot: {snapshot.name}")
    for geometry in geometries:
        vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = _origin_view_extrinsic()
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
