"""Plane-based free block detector (DetectFree prototype)."""

from __future__ import annotations

from dataclasses import dataclass

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
    region_max_extent_xy: float = 3.0
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


@dataclass(frozen=True)
class GroundPlane:
    """Normalized upward-facing plane ``normal·point + offset = 0``."""

    normal: np.ndarray
    offset: float
    source: str

    def height(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.offset


@dataclass(frozen=True)
class PlanePatch:
    normal: np.ndarray
    centroid: np.ndarray
    points: np.ndarray
    residual_mad: float


@dataclass(frozen=True)
class HypothesisEvidence:
    support_points: int
    top_height_error_m: float
    score: float


@dataclass(frozen=True)
class CuboidHypothesis:
    pose: BlockPose
    evidence: HypothesisEvidence


def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample (N, 3) float64 points using Open3D tensor API. Returns (M, 3) float64."""
    import open3d.core as o3c

    pcd = o3d.t.geometry.PointCloud(
        o3c.Tensor(pts.astype(np.float32), dtype=o3c.float32)
    )
    down = pcd.voxel_down_sample(voxel_size)
    result = down.point.positions.numpy().astype(np.float64)
    return result[np.lexsort((result[:, 2], result[:, 1], result[:, 0]))]


def _remove_ground(
    pts: np.ndarray, params: DetectionParams
) -> tuple[np.ndarray, GroundPlane]:
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

    def percentile_fallback() -> tuple[np.ndarray, GroundPlane]:
        ground_z = float(np.percentile(pts[:, 2], params.ground_z_percentile))
        ground = GroundPlane(np.array([0.0, 0.0, 1.0]), -ground_z, "percentile")
        return pts[ground.height(pts) >= params.ground_thickness], ground

    # Tensor voxelization can collapse an otherwise valid cloud below the
    # RANSAC sample size. Collinear samples also cannot define a plane.
    if len(coarse_pts) < 3 or np.linalg.matrix_rank(coarse_pts - coarse_pts.mean(axis=0)) < 2:
        return percentile_fallback()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coarse_pts)
    plane_model, _ = pcd.segment_plane(
        distance_threshold=params.ground_ransac_dist,
        ransac_n=3,
        num_iterations=params.ground_ransac_iters,
    )
    a, b, c, d = (float(value) for value in plane_model)
    normal_len = np.sqrt(a * a + b * b + c * c)

    if normal_len < 1e-8 or abs(c / normal_len) < params.ground_normal_min_z:
        # Degenerate or non-horizontal plane — fall back to percentile method
        return percentile_fallback()

    # Ensure normal points upward (c > 0) so signed distance is positive above ground
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    # Signed distance from plane: positive = above ground
    ground = GroundPlane(np.array([a, b, c]) / normal_len, d / normal_len, "ransac")
    signed_dist = ground.height(pts)
    return pts[signed_dist >= params.ground_thickness], ground


def _cluster_points(
    pts: np.ndarray, params: DetectionParams, ground: GroundPlane | None = None
) -> list[np.ndarray]:
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
        center = (cmin + cmax) / 2.0
        center_height = float(ground.height(center)) if ground is not None else float(center[2])
        if not (params.cluster_min_extent_xy <= dx <= params.region_max_extent_xy):
            continue
        if not (params.cluster_min_extent_xy <= dy <= params.region_max_extent_xy):
            continue
        if not (params.cluster_min_extent_z <= dz <= params.cluster_max_extent_z):
            continue
        if center_height > params.cluster_max_center_z:
            continue
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters


def _fit_planes(
    pts: np.ndarray, params: DetectionParams
) -> list[PlanePatch]:
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
        residuals = (inliers - centroid) @ normal
        residual_mad = float(np.median(np.abs(residuals - np.median(residuals))))
        planes.append(PlanePatch(normal, centroid, inliers, residual_mad))
        mask = np.ones(len(remaining), dtype=np.bool_)
        mask[inlier_idx] = False
        remaining = remaining[mask]
    return planes


def _classify_planes(
    planes: list[PlanePatch],
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

    for patch in planes:
        normal, centroid = patch.normal, patch.centroid
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

    for patch in planes:
        normal, centroid = patch.normal, patch.centroid
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


def _candidate_plane_sets(
    planes: list[PlanePatch], params: DetectionParams, ground: GroundPlane | None = None
) -> list[tuple[PlanePatch, PlanePatch | None]]:
    """Pair observed tops with sides sharing the same layer boundary/support."""
    up = ground.normal if ground is not None else np.array([0.0, 0.0, 1.0])
    cos_top = np.cos(np.radians(params.top_plane_angle_deg))
    sin_side = np.sin(np.radians(params.side_plane_angle_deg))
    tops = [patch for patch in planes if abs(float(patch.normal @ up)) > cos_top]
    sides = [patch for patch in planes if abs(float(patch.normal @ up)) < sin_side]
    tops.sort(key=lambda patch: (-float(patch.centroid @ up), *patch.centroid[:2]))
    candidates = []
    for top in tops:
        ranked: list[tuple[float, PlanePatch]] = []
        top_height = float(top.centroid @ up)
        for side in sides:
            side_normal = side.normal - float(side.normal @ up) * up
            norm = np.linalg.norm(side_normal)
            if norm < 1e-8:
                continue
            side_normal /= norm
            side_heights = side.points @ up
            centroid_drop = top_height - float(side.centroid @ up)
            if not 0.15 <= centroid_drop <= params.block_dims[2] - 0.1:
                continue
            # The side support must reach the top layer and extend materially
            # below it; this prevents pairing a top with another stack layer.
            if side_heights.max() < top_height - 0.15:
                continue
            if side_heights.min() > top_height - 0.3:
                continue
            plane_dist = np.abs((top.points - side.centroid) @ side_normal)
            boundary_gap = float(plane_dist.min())
            if boundary_gap > 0.1:
                continue
            tangent = np.cross(up, side_normal)
            tangent /= np.linalg.norm(tangent)
            top_range = np.array([np.min(top.points @ tangent), np.max(top.points @ tangent)])
            side_range = np.array([np.min(side.points @ tangent), np.max(side.points @ tangent)])
            overlap = min(top_range[1], side_range[1]) - max(top_range[0], side_range[0])
            if overlap < 0.15:
                continue
            height_error = abs(centroid_drop - params.block_dims[2] / 2.0)
            ranked.append((boundary_gap + height_error, side))
        side = min(ranked, key=lambda item: (item[0], *item[1].centroid))[1] if ranked else None
        candidates.append((top, side))
    return candidates


def _split_connected_region(cluster: np.ndarray, params: DetectionParams) -> list[np.ndarray]:
    """Split a long connected row into known-width spatial candidate regions.

    This deliberately handles only an observable horizontal repetition. Vertical
    stacks with buried interfaces are left connected until stronger evidence is
    available.
    """
    extents = np.ptp(cluster, axis=0)
    axis = int(np.argmax(extents[:2]))
    width = params.block_dims[axis]
    count = int(round(extents[axis] / width))
    if count < 2 or count > 8 or abs(extents[axis] - count * width) > 0.12:
        return [cluster]
    low = float(cluster[:, axis].min())
    # Extent alone is not evidence of multiple objects: a continuous elongated
    # cuboid has exactly the same outer dimensions as a seamless row. Require a
    # repeatable return-density valley at every proposed interface.
    values = cluster[:, axis]
    valley_half_width = max(params.voxel_size * 1.5, 0.04)
    shoulder_width = max(params.voxel_size * 3.0, 0.08)
    for index in range(1, count):
        boundary = low + index * width
        valley = np.count_nonzero(np.abs(values - boundary) <= valley_half_width)
        left = np.count_nonzero(
            (values >= boundary - shoulder_width - valley_half_width)
            & (values < boundary - valley_half_width)
        )
        right = np.count_nonzero(
            (values > boundary + valley_half_width)
            & (values <= boundary + shoulder_width + valley_half_width)
        )
        expected = (left + right) * valley_half_width / shoulder_width
        if expected <= 0.0 or valley > 0.45 * expected:
            return [cluster]
    regions = []
    for index in range(count):
        lower = low + index * width
        upper = low + (index + 1) * width
        mask = (cluster[:, axis] >= lower - 1e-6) & (
            cluster[:, axis] <= upper + (1e-6 if index == count - 1 else 0.0)
        )
        region = cluster[mask]
        if len(region) >= params.cluster_min_size:
            regions.append(region)
    return regions if len(regions) == count else [cluster]


def _make_hypothesis(
    pose: BlockPose,
    top_centroid: np.ndarray,
    support_points: int,
    ground: GroundPlane,
    params: DetectionParams,
) -> CuboidHypothesis:
    top_height = float(ground.height(top_centroid))
    layer = max(round(top_height / params.block_dims[2]), 1)
    height_error = abs(top_height - layer * params.block_dims[2])
    score = min(support_points / 500.0, 1.0) - 3.0 * height_error
    return CuboidHypothesis(pose, HypothesisEvidence(support_points, height_error, score))


def _select_hypotheses(hypotheses: list[CuboidHypothesis]) -> list[CuboidHypothesis]:
    """Greedy deterministic NMS: score governs selection; only duplicates collapse."""
    ordered = sorted(
        hypotheses,
        key=lambda h: (-h.evidence.score, *np.round(h.pose.position, 9).tolist()),
    )
    selected: list[CuboidHypothesis] = []
    for candidate in ordered:
        duplicate = any(
            np.linalg.norm(candidate.pose.position - accepted.pose.position) < 0.2
            for accepted in selected
        )
        if not duplicate:
            selected.append(candidate)
    return selected


def _synthesize_pose(
    top_normal: np.ndarray,
    top_centroid: np.ndarray,
    side_normal: np.ndarray,
    side_centroid: np.ndarray,
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
    if cluster_pts is not None and len(cluster_pts) > 0:
        # Plane normals from RANSAC have arbitrary sign. Orient this axis from
        # the observed side face toward the block interior.
        inward_evidence = float(np.dot(cluster_pts.mean(axis=0) - side_centroid, x_world))
        if inward_evidence < 0.0:
            x_world = -x_world
    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)

    R = np.column_stack([x_world, y_world, z_world])

    half_height = block_dims[2] / 2.0
    position = top_centroid - z_world * half_height
    if cluster_pts is not None and len(cluster_pts) > 0:
        # Constrain the center using the observed side-plane equation and the
        # known half-width. The orthogonal horizontal direction remains
        # partially constrained by the top-face centroid.
        current = float(np.dot(position - side_centroid, x_world))
        position += x_world * (block_dims[0] / 2.0 - current)

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
        Multiple poses may be returned from one connected region only when
        distinct top patches have compatible same-layer side/boundary support.
        Fully buried stack interfaces remain unobservable and are not inferred.
    """
    if params is None:
        params = DetectionParams()

    pts = np.asarray(pts_world)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts_world must have shape (N, 3), got {pts.shape}")
    if not np.issubdtype(pts.dtype, np.number) or np.iscomplexobj(pts):
        raise ValueError("pts_world must contain real numeric coordinates")
    pts = pts.astype(np.float64, copy=False)
    if not np.all(np.isfinite(pts)):
        raise ValueError("pts_world must contain only finite coordinates")
    # Open3D's voxel and RANSAC routines report opaque errors for these valid
    # but uninformative clouds.  Treat them as containing no detectable block.
    if len(pts) < 3:
        return []
    pts = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]

    # Stage 1: Voxel downsample
    pts = _voxel_downsample(pts, params.voxel_size)
    if len(pts) < 3:
        return []

    # Stage 2: Ground removal
    pts, ground = _remove_ground(pts, params)
    if len(pts) == 0:
        return []

    # Stage 3: DBSCAN clustering
    clusters = _cluster_points(pts, params, ground)

    # Stage 4–6: Per-cluster plane fitting and pose synthesis
    hypotheses: list[CuboidHypothesis] = []
    for cluster in clusters:
        regions = _split_connected_region(cluster, params)
        if len(regions) == 1 and np.any(
            np.ptp(cluster, axis=0)[:2] > params.cluster_max_extent_xy
        ):
            # A large connected footprint without internal boundary evidence is
            # an elongated object, not a license to tile it with known boxes.
            continue
        for region in regions:
            planes = _fit_planes(region, params)
            if not planes:
                continue
            for top, side in _candidate_plane_sets(planes, params, ground):
                top_n = top.normal if top.normal[2] > 0 else -top.normal
                if side is not None:
                    local_support = np.vstack([top.points, side.points])
                    pose = _synthesize_pose(
                        top_n,
                        top.centroid,
                        side.normal,
                        side.centroid,
                        params.block_dims,
                        local_support,
                    )
                    support = len(top.points) + len(side.points)
                else:
                    pose = _synthesize_pose_top_only(
                        top_n, top.centroid, region, params.block_dims
                    )
                    support = len(top.points)
                hypothesis = _make_hypothesis(
                    pose, top.centroid, support, ground, params
                )
                # The observed top must reach one known block height. Higher
                # top patches are retained as independently observed layers.
                if float(ground.height(top.centroid)) >= params.block_dims[2] - 0.08:
                    hypotheses.append(hypothesis)

    results = [hypothesis.pose for hypothesis in _select_hypotheses(hypotheses)]

    # Filter out physically implausible poses (center below ground or above height cap)
    results = [
        p for p in results if 0.0 <= float(ground.height(p.position)) <= params.cluster_max_center_z
    ]
    results.sort(key=lambda p: p.position[2], reverse=True)
    return results
