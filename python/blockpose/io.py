"""Data loading utilities for blockpose registration snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from blockpose.frames import PointFrame, PoseTrajectory
from blockpose.pose import BlockPose, quat_xyzw_to_rotmat

# Default path relative to repo root (two levels up from tools/, one up from python/)
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_TRANSFORM_CONFIG = _REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
_DEFAULT_CAMERA_CALIB = _REPO_ROOT / "config" / "calibration" / "calib_blackfly_to_seyond.yaml"


def load_pcd(path: str | Path) -> np.ndarray:
    """Load a PCD file and return points as (N, 3) float32 array in sensor frame."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Loaded empty point cloud from {path}")
    pts = np.asarray(pcd.points).copy().astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {pts.shape}")
    return pts


def load_point_frame(
    path: str | Path,
    *,
    timestamp: float = 0.0,
    frame_id: str = "sensor",
    sensor_trajectory: PoseTrajectory | None = None,
) -> PointFrame:
    """Load a legacy XYZ PCD as a rich frame with unavailable fields explicit."""
    return PointFrame(
        load_pcd(path),
        timestamp=timestamp,
        frame_id=frame_id,
        sensor_trajectory=sensor_trajectory,
    )


def load_transform(name: str, config_path: str | Path | None = None) -> np.ndarray:
    """Load a named transform as a (4, 4) float64 homogeneous transform matrix.

    `name` must match a `transforms[].name` entry in the transform config, e.g.
    "T_world_seyond" or "T_world_blackfly_rotated".
    """
    if config_path is None:
        config_path = _DEFAULT_TRANSFORM_CONFIG
    config_path = Path(config_path)
    with config_path.open() as f:
        data = yaml.safe_load(f)
    for entry in data["transforms"]:
        if entry["name"] == name:
            T = np.array(entry["matrix"], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {T.shape}")
            return T
    raise KeyError(f"{name} not found in transform config {config_path}")


def load_T_world_seyond(config_path: str | Path | None = None) -> np.ndarray:
    """Load T_world_seyond (lidar -> world) as a (4, 4) float64 transform matrix."""
    return load_transform("T_world_seyond", config_path)


def load_T_world_camera(config_path: str | Path | None = None) -> np.ndarray:
    """Load T_world_blackfly_rotated (camera -> world) as a (4, 4) float64 transform matrix."""
    return load_transform("T_world_blackfly_rotated", config_path)


@dataclass
class TFSnapshot:
    """Per-snapshot TF transforms captured at recording time (see `tf.yaml`).

    Unlike the static `transforms_*.yaml` config, these reflect the crane's
    actual joint state when the snapshot was taken, so they are the ones to
    use to get the cloud/image aligned with the recorded ground truth.
    """

    reference_frame: str
    image_frame: str
    cloud_frame: str
    transforms: dict[str, np.ndarray]  # name -> (4, 4) float64 matrix

    def lookup(self, parent: str, child: str) -> np.ndarray:
        name = f"T_{parent}_{child}"
        if name not in self.transforms:
            raise KeyError(f"{name} not found in tf snapshot (have: {sorted(self.transforms)})")
        return self.transforms[name]

    @property
    def T_world_cloud(self) -> np.ndarray:
        return self.lookup(self.reference_frame, self.cloud_frame)

    @property
    def T_world_camera(self) -> np.ndarray:
        return self.lookup(self.reference_frame, self.image_frame)


def load_tf_snapshot(path: str | Path) -> TFSnapshot:
    """Load a snapshot's `tf.yaml`: the TF transforms captured at recording time."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    transforms: dict[str, np.ndarray] = {}
    for entry in data["transforms"]:
        if not entry.get("available", True):
            continue
        T = np.array(entry["matrix"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix for {entry['name']!r}, got {T.shape}")
        transforms[entry["name"]] = T
    return TFSnapshot(
        reference_frame=data["reference_frame"],
        image_frame=data["image_frame"],
        cloud_frame=data["cloud_frame"],
        transforms=transforms,
    )


@dataclass
class CameraModel:
    """Pinhole camera model with optional radial-tangential distortion.

    Conventions match `sensor_msgs/CameraInfo` (row-major K, optical frame:
    X right, Y down, Z forward out of the lens), i.e. the layout used in
    `config/calibration/*.yaml`.
    """

    K: np.ndarray  # (3, 3) float64 intrinsic matrix
    width: int
    height: int
    # OpenCV distortion coefficients. Length 5 (plumb-bob) or 8
    # (rational_polynomial); empty for an ideal pinhole. Order:
    # [k1, k2, p1, p2, k3, (k4, k5, k6)].
    D: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(self.D, dtype=np.float64).ravel()

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


def load_camera_model(calib_path: str | Path | None = None) -> CameraModel:
    """Load a `CameraModel` (intrinsics + distortion + image size) from a calib YAML."""
    if calib_path is None:
        calib_path = _DEFAULT_CAMERA_CALIB
    calib_path = Path(calib_path)
    with calib_path.open() as f:
        data = yaml.safe_load(f)
    K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
    width = int(data["image_width"])
    height = int(data["image_height"])
    D = np.array(data.get("D", []), dtype=np.float64).ravel()
    return CameraModel(K=K, width=width, height=height, D=D)


def load_camera_intrinsics(
    calib_path: str | Path | None = None,
) -> tuple[np.ndarray, int, int]:
    """Load pinhole intrinsics from a camera calibration YAML.

    Returns (K, width, height). Prefer `load_camera_model` when distortion is
    needed (e.g. projecting points into the image).
    """
    cam = load_camera_model(calib_path)
    return cam.K, cam.width, cam.height


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image as an (H, W, 3) uint8 array."""
    img = o3d.io.read_image(str(path))
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {arr.shape}")
    return arr


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
    T_world_cloud: np.ndarray | None = None,
    *,
    require_registrations: bool = True,
) -> tuple[np.ndarray, list[BlockPose]]:
    """Load a registration snapshot: returns (pts_world, gt_poses).

    pts_world: (N, 3) float64 point cloud transformed to world frame.
    gt_poses: list of BlockPose objects from registrations.yaml. Set
    `require_registrations=False` for a replay capture that has no legacy
    registration output; it then returns an empty pose list.

    `T_world_cloud` overrides the static config's T_world_seyond, e.g. with a
    per-snapshot transform from `tf.yaml` (see `load_tf_snapshot`).
    """
    snapshot_dir = Path(snapshot_dir)
    T = T_world_cloud if T_world_cloud is not None else load_T_world_seyond(config_path)
    pts_sensor = load_pcd(snapshot_dir / "cloud.pcd")
    pts_world = apply_transform(pts_sensor, T)
    registrations = snapshot_dir / "registrations.yaml"
    if not registrations.is_file():
        if require_registrations:
            raise FileNotFoundError(f"Missing registrations file: {registrations}")
        return pts_world, []
    gt_poses = load_ground_truth(registrations)
    return pts_world, gt_poses
