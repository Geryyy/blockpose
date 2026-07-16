"""Plane-based free block detector (DetectFree prototype)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from blockpose.pose import BlockPose


@dataclass
class DetectionParams:
    voxel_size: float = 0.04
    ground_z_percentile: float = 5.0
    ground_thickness: float = 0.15
    ground_ransac_dist: float = 0.05
    ground_ransac_iters: int = 500
    ground_normal_min_z: float = 0.7
    dbscan_eps: float = 0.06
    dbscan_min_points: int = 5
    cluster_min_size: int = 30
    cluster_max_size: int = 50000
    cluster_min_extent_xy: float = 0.3
    cluster_max_extent_xy: float = 1.5
    cluster_min_extent_z: float = 0.3
    cluster_max_extent_z: float = 3.5
    cluster_max_center_z: float = 3.0
    ransac_dist_thresh: float = 0.02
    ransac_n: int = 3
    ransac_num_iterations: int = 1000
    max_planes: int = 5
    min_inliers: int = 40
    top_plane_angle_deg: float = 30.0
    side_plane_angle_deg: float = 20.0
    max_plane_center_dist: float = 0.6
    block_dims: tuple[float, float, float] = (0.6, 0.6, 0.9)


def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample (N, 3) float64 points using Open3D tensor API. Returns (M, 3) float64."""
    import open3d.core as o3c

    pcd = o3d.t.geometry.PointCloud(
        o3c.Tensor(pts.astype(np.float32), dtype=o3c.float32)
    )
    down = pcd.voxel_down_sample(voxel_size)
    return down.point.positions.numpy().astype(np.float64)


def _remove_ground(pts: np.ndarray, params: DetectionParams) -> np.ndarray:
    """Remove ground-level points using RANSAC plane fitting.

    Fits a dominant ground plane on a coarse downsampled copy, then removes
    all input points within ground_thickness above the plane. Falls back to the
    global Z-percentile method if the fitted plane is not roughly horizontal.
    """
    # Coarse downsample for fast RANSAC
    import open3d.core as o3c

    pcd_coarse = o3d.t.geometry.PointCloud(
        o3c.Tensor(pts.astype(np.float32), dtype=o3c.float32)
    )
    pcd_coarse = pcd_coarse.voxel_down_sample(0.1)
    coarse_pts = pcd_coarse.point.positions.numpy().astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coarse_pts)
    plane_model, _ = pcd.segment_plane(
        distance_threshold=params.ground_ransac_dist,
        ransac_n=3,
        num_iterations=params.ground_ransac_iters,
    )
    a, b, c, d = float(plane_model[0]), float(plane_model[1]), float(plane_model[2]), float(plane_model[3])
    normal_len = np.sqrt(a * a + b * b + c * c)

    if normal_len < 1e-8 or abs(c / normal_len) < params.ground_normal_min_z:
        # Degenerate or non-horizontal plane — fall back to percentile method
        ground_z = float(np.percentile(pts[:, 2], params.ground_z_percentile))
        return pts[pts[:, 2] >= ground_z + params.ground_thickness]

    # Ensure normal points upward (c > 0) so signed distance is positive above ground
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    # Signed distance from plane: positive = above ground
    signed_dist = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / normal_len
    return pts[signed_dist >= params.ground_thickness]


def _cluster_points(pts: np.ndarray, params: DetectionParams) -> list[np.ndarray]:
    """DBSCAN clustering. Returns list of (K_i, 3) arrays for accepted clusters."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.array(
        pcd.cluster_dbscan(
            eps=params.dbscan_eps,
            min_points=params.dbscan_min_points,
            print_progress=False,
        )
    )
    clusters = []
    for label in np.unique(labels):
        if label < 0:
            continue
        mask = labels == label
        cluster = pts[mask]
        if not (params.cluster_min_size <= len(cluster) <= params.cluster_max_size):
            continue
        # AABB dimension gating: reject clusters incompatible with block geometry
        cmin = cluster.min(axis=0)
        cmax = cluster.max(axis=0)
        dx, dy, dz = cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]
        z_center = (cmin[2] + cmax[2]) / 2.0
        if not (params.cluster_min_extent_xy <= dx <= params.cluster_max_extent_xy):
            continue
        if not (params.cluster_min_extent_xy <= dy <= params.cluster_max_extent_xy):
            continue
        if not (params.cluster_min_extent_z <= dz <= params.cluster_max_extent_z):
            continue
        if z_center > params.cluster_max_center_z:
            continue
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters


def _fit_planes(
    pts: np.ndarray, params: DetectionParams
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Iteratively fit planes using RANSAC.

    Returns list of (normal, centroid, inlier_pts) tuples.
    """
    remaining = pts.copy()
    planes = []
    for _ in range(params.max_planes):
        if len(remaining) < params.min_inliers:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining)
        plane_model, inlier_idx = pcd.segment_plane(
            distance_threshold=params.ransac_dist_thresh,
            ransac_n=params.ransac_n,
            num_iterations=params.ransac_num_iterations,
        )
        inlier_idx = list(inlier_idx)
        if len(inlier_idx) < params.min_inliers:
            break
        normal = np.array(plane_model[:3], dtype=np.float64)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            break
        normal /= norm_len
        inliers = remaining[inlier_idx]
        centroid = inliers.mean(axis=0)
        planes.append((normal, centroid, inliers))
        mask = np.ones(len(remaining), dtype=np.bool_)
        mask[inlier_idx] = False
        remaining = remaining[mask]
    return planes


def _classify_planes(
    planes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: DetectionParams,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Classify fitted planes into top and side candidates.

    Returns (top_normal, top_centroid, side_normal, side_centroid) or Nones.
    """
    world_z = np.array([0.0, 0.0, 1.0])
    cos_top = np.cos(np.radians(params.top_plane_angle_deg))
    sin_side = np.sin(np.radians(params.side_plane_angle_deg))

    top_candidate = None
    top_centroid = None
    top_dot = -1.0

    for normal, centroid, _ in planes:
        dot = float(np.dot(normal, world_z))
        absdot = abs(dot)
        if absdot > cos_top and absdot > top_dot:
            # Flip to ensure upward-pointing normal
            flipped = normal if dot > 0 else -normal
            top_candidate = flipped
            top_centroid = centroid
            top_dot = absdot

    if top_candidate is None:
        return None, None, None, None

    side_candidate = None
    side_centroid = None
    best_dist = np.inf

    for normal, centroid, _ in planes:
        dot = float(np.dot(normal, world_z))
        if abs(dot) >= sin_side:
            continue  # Too tilted — not a side plane
        dist = float(np.linalg.norm(centroid - top_centroid))
        if dist > params.max_plane_center_dist:
            continue
        if dist < best_dist:
            side_candidate = normal.copy()
            side_centroid = centroid
            best_dist = dist

    return top_candidate, top_centroid, side_candidate, side_centroid


def _synthesize_pose(
    top_normal: np.ndarray,
    top_centroid: np.ndarray,
    side_normal: np.ndarray,
    block_dims: tuple[float, float, float],
    cluster_pts: np.ndarray | None = None,
) -> BlockPose:
    """Closed-form MVP pose from fitted planes and the cluster AABB.

    Plane centroids are biased toward visible faces in a single LiDAR view. Use
    the cluster AABB center for XY and the top plane for Z.
    """
    z_world = top_normal / np.linalg.norm(top_normal)

    # Project side_normal perpendicular to z_world
    x_raw = side_normal - np.dot(side_normal, z_world) * z_world
    x_len = np.linalg.norm(x_raw)
    if x_len < 1e-8:
        x_raw = np.array([1.0, 0.0, 0.0])
        x_raw = x_raw - np.dot(x_raw, z_world) * z_world
        x_len = np.linalg.norm(x_raw)
    x_world = x_raw / x_len
    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)

    R = np.column_stack([x_world, y_world, z_world])

    half_height = block_dims[2] / 2.0
    position = top_centroid - z_world * half_height
    if cluster_pts is not None and len(cluster_pts) > 0:
        mn = cluster_pts.min(axis=0)
        mx = cluster_pts.max(axis=0)
        center = (mn + mx) / 2.0
        ext = mx - mn
        position[:2] = center[:2]
        if ext[2] < block_dims[2] * 1.25:
            position[2] = center[2]

    return BlockPose(position=position, rotation=R, confidence=1.0, source="plane_fit")


def _synthesize_pose_top_only(
    top_normal: np.ndarray,
    top_centroid: np.ndarray,
    cluster_pts: np.ndarray,
    block_dims: tuple[float, float, float],
) -> BlockPose:
    """Fallback pose from top plane only (no side plane).

    Yaw is estimated from the principal axis of the cluster projected onto the
    horizontal plane. This gives a PartiallyConstrained pose.
    """
    z_world = top_normal / np.linalg.norm(top_normal)

    # Project cluster points onto horizontal plane and find principal axis
    pts_horizontal = cluster_pts[:, :2] - cluster_pts[:, :2].mean(axis=0)
    cov = pts_horizontal.T @ pts_horizontal / max(len(pts_horizontal) - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    x_horiz = np.array([vecs[0, -1], vecs[1, -1], 0.0])
    x_horiz_len = np.linalg.norm(x_horiz)
    if x_horiz_len < 1e-8:
        x_horiz = np.array([1.0, 0.0, 0.0])
    else:
        x_horiz /= x_horiz_len

    x_world = x_horiz - np.dot(x_horiz, z_world) * z_world
    x_len = np.linalg.norm(x_world)
    if x_len < 1e-8:
        x_world = np.array([1.0, 0.0, 0.0])
    else:
        x_world /= x_len

    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)

    R = np.column_stack([x_world, y_world, z_world])
    half_height = block_dims[2] / 2.0
    position = top_centroid - z_world * half_height

    return BlockPose(
        position=position, rotation=R, confidence=0.5, source="top_plane_only"
    )


def detect_blocks(
    pts_world: np.ndarray,
    params: DetectionParams | None = None,
) -> list[BlockPose]:
    """Detect concrete blocks in a world-frame point cloud.

    Args:
        pts_world: (N, 3) float64 point cloud in world frame (Z = up).
        params: Detection parameters; uses defaults if None.

    Returns:
        List of BlockPose, one per detected block, sorted by Z descending.
        Note: stacked blocks that fall within the same DBSCAN cluster yield one
        detection (the top block); the lower block will be unmatched (FN).
    """
    if params is None:
        params = DetectionParams()

    pts = np.asarray(pts_world, dtype=np.float64)

    # Stage 1: Voxel downsample
    pts = _voxel_downsample(pts, params.voxel_size)

    # Stage 2: Ground removal
    pts = _remove_ground(pts, params)
    if len(pts) == 0:
        return []

    # Stage 3: DBSCAN clustering
    clusters = _cluster_points(pts, params)

    # Stage 4–6: Per-cluster plane fitting and pose synthesis
    results: list[BlockPose] = []
    for cluster in clusters:
        planes = _fit_planes(cluster, params)
        if not planes:
            continue
        top_n, top_c, side_n, side_c = _classify_planes(planes, params)
        if top_n is None:
            continue
        if side_n is not None:
            pose = _synthesize_pose(top_n, top_c, side_n, params.block_dims, cluster)
        else:
            pose = _synthesize_pose_top_only(top_n, top_c, cluster, params.block_dims)
        results.append(pose)

    # Filter out physically implausible poses (center below ground or above height cap)
    results = [p for p in results if 0.0 <= p.position[2] <= params.cluster_max_center_z]
    results.sort(key=lambda p: p.position[2], reverse=True)
    return results
