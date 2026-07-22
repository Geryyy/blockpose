"""Plane-based free block detector (DetectFree prototype)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from blockpose.pose import BLOCK_DIMS, BlockPose
from blockpose.ransac import segment_plane
from blockpose.refine import refine_poses


@dataclass
class DetectionParams:
    voxel_size: float = 0.04
    ground_z_percentile: float = 5.0
    ground_thickness: float = 0.15
    ground_ransac_dist: float = 0.05
    ground_ransac_iters: int = 500
    ground_normal_min_z: float = 0.7
    dbscan_eps: float = 0.07
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
    # Bound only candidate-plane search for oversized proposal regions. The
    # winner is still scored and least-squares-refitted against the full region.
    ransac_search_max_points: int | None = 2048
    # Seed for the in-repo deterministic RANSAC (blockpose.ransac).
    ransac_seed: int = 0
    max_planes: int = 5
    min_inliers: int = 40
    top_plane_angle_deg: float = 30.0
    side_plane_angle_deg: float = 20.0
    max_plane_center_dist: float = 0.6
    block_dims: tuple[float, float, float] = BLOCK_DIMS
    # Orientations to hypothesise when the block's resting pose is unknown. The
    # same 0.9 x 0.6 x 0.6 block lies on a long face (the default, matching
    # every wall course in config/wall_plans.yaml) or stands upright as
    # (0.6, 0.6, 0.9) -- a detector configured for one sees NONE of the other.
    #
    # Empty means "use block_dims only". Multi-orientation is NOT yet the default
    # because the evidence model cannot reliably rank the two: `height_consistency`
    # is blind wherever a top height is a common multiple of 0.9 and 0.6 (1.8 m,
    # i.e. any two-course upright stack), and with that tie-breaker gone the
    # remaining features let a laid hypothesis win on upright geometry -- measured
    # as a 0.20 m centre error plus false positives on clutter. Ranking
    # orientations needs a feature that penalises a box protruding into space the
    # returns show is empty; see docs/detector-plan.md.
    #
    # Set explicitly to enable, e.g. ((0.6, 0.6, 0.9),) when a scene says the
    # blocks stand upright, or both entries to accept the current precision cost.
    candidate_dims: tuple[tuple[float, float, float], ...] = ()
    # Model-based refinement (blockpose.refine). The closed-form pose uses a top
    # plane plus at most one side plane; refinement fits the exact cuboid to
    # every assigned return, which is what removes the single-plane yaw fragility.
    refine_enabled: bool = True
    refine_band: float = 0.10  # assignment distance to a hypothesis surface (m)
    refine_iterations: int = 2
    refine_min_points: int = 20
    refine_max_translation: float = 0.15  # reject corrections larger than this
    refine_max_rotation_deg: float = 20.0
    # Minimum weighted evidence score (see _SCORE_WEIGHTS) to keep a hypothesis
    # at all, applied before NMS selection. 0.0 (accept everything that
    # survives clustering/plane-fitting) matches all prior behaviour and every
    # synthetic scenario this is gated against. Real snapshots have real
    # clutter -- e.g. the crane/gripper hardware above a block -- that can
    # still produce a shaped-enough cluster to reach pose synthesis; measured
    # on real data, spurious hypotheses like that score under ~0.2 while every
    # true block scores over ~0.4, a threshold around there is a first,
    # deliberately conservative cut, not a tuned classifier.
    min_score: float = 0.0
    local_ground_cell_size: float = 0.5
    local_ground_clearance: float = 0.05
    proposal_scales: tuple[float, ...] = (0.06, 0.09, 0.13)
    proposal_range_gain: float = 0.08
    proposal_max_components: int = 8
    proposal_max_points: int = 30000
    multiscale_proposals: bool = False


@dataclass(frozen=True)
class GroundPlane:
    """Normalized upward-facing plane ``normal·point + offset = 0``."""

    normal: np.ndarray
    offset: float
    source: str

    def height(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.offset


@dataclass(frozen=True)
class LocalGroundModel:
    """Deterministic 0.5 m low-surface grid with planar fallback."""

    normal: np.ndarray
    offset: float
    source: str
    cell_size: float
    cell_xy: np.ndarray
    cell_z: np.ndarray

    def support_z(self, points: np.ndarray) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if not len(self.cell_xy):
            return -(query[:, :2] @ self.normal[:2] + self.offset) / self.normal[2]
        _, indices = cKDTree(self.cell_xy).query(query[:, :2], k=1)
        fallback = -(query[:, :2] @ self.normal[:2] + self.offset) / self.normal[2]
        cell_fallback = -(
            self.cell_xy[np.asarray(indices)] @ self.normal[:2] + self.offset
        ) / self.normal[2]
        local = fallback + self.cell_z[np.asarray(indices)] - cell_fallback
        # A remote cell is not local evidence; use the stable planar fallback.
        distance = np.linalg.norm(query[:, :2] - self.cell_xy[np.asarray(indices)], axis=1)
        return np.where(distance <= self.cell_size * 1.75, local, fallback)

    def height(self, points: np.ndarray) -> np.ndarray:
        query = np.atleast_2d(np.asarray(points, dtype=np.float64))
        result = query[:, 2] - self.support_z(query)
        return result if np.asarray(points).ndim == 2 else result[0]


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
    expected_visible_faces: int = 0
    covered_visible_faces: int = 0
    free_space_violations: int = 0
    supported_rays: int = 0
    observed_geometry_faces: int = 0
    # Rays geometrically aimed at the visible faces, whatever they actually hit.
    # Denominator for `surface_explained`; keeps that feature scale-free.
    incident_rays: int = 0


@dataclass(frozen=True)
class CuboidHypothesis:
    pose: BlockPose
    evidence: HypothesisEvidence
    proposal_scale_m: float | None = None
    support_height_m: float | None = None


class Observability(str, Enum):
    LIMITED = "limited"
    PARTIAL = "partial"
    FULL = "full"


@dataclass(frozen=True)
class DetectionDiagnostics:
    evidence: HypothesisEvidence
    observability: Observability
    features: dict[str, float]
    calibrated_confidence: float | None = None
    proposal_scale_m: float | None = None
    support_height_m: float | None = None


@dataclass(frozen=True)
class DetectionResult:
    poses: list[BlockPose]
    diagnostics: list[DetectionDiagnostics]
    stage_counts: dict[str, int] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorContext:
    """Optional ray evidence in the same world frame as the input cloud.

    ``miss_directions`` holds beams that returned nothing. They are optional
    because a raw point cloud cannot recover them, but when the source can
    supply them they are the strongest available free-space evidence.
    """

    origin: np.ndarray
    ray_directions: np.ndarray
    ranges: np.ndarray
    miss_directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @classmethod
    def from_points(cls, origin: np.ndarray, points: np.ndarray) -> SensorContext:
        origin = np.asarray(origin, dtype=np.float64)
        vectors = np.asarray(points, dtype=np.float64) - origin
        ranges = np.linalg.norm(vectors, axis=1)
        valid = ranges > 1e-9
        return cls(origin, vectors[valid] / ranges[valid, None], ranges[valid])


def _validate_sensor_context(context: SensorContext) -> SensorContext:
    origin = np.asarray(context.origin, dtype=np.float64)
    directions = np.asarray(context.ray_directions, dtype=np.float64)
    ranges = np.asarray(context.ranges, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("sensor_context.origin must have shape (3,) with finite coordinates")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("sensor_context.ray_directions must have shape (N, 3)")
    if ranges.ndim != 1:
        raise ValueError("sensor_context.ranges must have shape (N,)")
    if len(directions) != len(ranges):
        raise ValueError("sensor_context directions and ranges must have the same length")
    if not np.all(np.isfinite(directions)):
        raise ValueError("sensor_context.ray_directions must be finite")
    if not np.all(np.isfinite(ranges)) or np.any(ranges < 0.0):
        raise ValueError("sensor_context.ranges must be finite and non-negative")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("sensor_context.ray_directions must have non-zero finite norms")
    misses = np.asarray(context.miss_directions, dtype=np.float64)
    if misses.size and (misses.ndim != 2 or misses.shape[1] != 3):
        raise ValueError("sensor_context.miss_directions must have shape (M, 3)")
    if misses.size and not np.all(np.isfinite(misses)):
        raise ValueError("sensor_context.miss_directions must be finite")
    return SensorContext(origin, directions / norms[:, None], ranges, misses)


def _ray_box_interval(
    origin_local: np.ndarray, dirs_local: np.ndarray, half: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slab test of rays against an origin-centred axis-aligned box.

    Returns ``(t_enter, t_exit, intersects)``. Rays parallel to a slab are
    handled by IEEE infinities; the NaN from ``0 * inf`` is scrubbed.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / dirs_local
        t_low = (-half - origin_local) * inverse
        t_high = (half - origin_local) * inverse
    t_low = np.nan_to_num(t_low, nan=-np.inf)
    t_high = np.nan_to_num(t_high, nan=np.inf)
    t_enter = np.minimum(t_low, t_high).max(axis=1)
    t_exit = np.maximum(t_low, t_high).min(axis=1)
    return t_enter, t_exit, t_exit > np.maximum(t_enter, 0.0)


def visibility_evidence(
    pose: BlockPose,
    context: SensorContext,
    dims: tuple[float, float, float],
    tolerance: float = 0.06,
) -> tuple[int, int, int, int]:
    """Return ``(expected, covered, violations, supported, incident)`` ray evidence.

    Face coverage is measured per visible face with a *generous* rectangle
    (+tolerance), because a return near a face edge still supports that face.

    Free-space violations are measured against the block *volume* shrunk by
    tolerance, not against extended face planes. The plane formulation counted a
    ray as violating whenever it crossed an enlarged face rectangle, which
    included rays skimming past the true silhouette and terminating on the
    ground metres behind -- so violations fired in the hundreds on perfectly
    correct poses. Requiring the ray to traverse the whole shrunk volume, and to
    measure a range beyond the exit point, makes a violation mean what its name
    says: the beam passed through where the block is claimed to be.

    Rays ending before the block are occlusion and stay neutral. Faces oriented
    away from the sensor are not expected and also stay neutral.
    """
    context = _validate_sensor_context(context)
    origin_local = pose.rotation.T @ (context.origin - pose.position)
    half = np.asarray(dims, dtype=np.float64) / 2.0

    # A ray intersecting the tolerance-expanded box must intersect its bounding
    # sphere. Use that exact necessary condition to avoid applying six slab/face
    # tests to the entire scan for every raw hypothesis. This changes no evidence
    # semantics: every potentially incident/supporting/violating ray is retained.
    center_vector = pose.position - context.origin
    projections = context.ray_directions @ center_vector
    perpendicular_sq = np.maximum(
        float(center_vector @ center_vector) - projections * projections, 0.0
    )
    radius_sq = float(np.sum((half + tolerance) ** 2))
    if float(center_vector @ center_vector) <= radius_sq:
        # The sensor lies inside the conservative sphere; no angular rejection
        # is safe (including rays initially directed away from its centre).
        relevant = np.ones(len(context.ray_directions), dtype=bool)
    else:
        relevant = (projections > 0.0) & (perpendicular_sq <= radius_sq + 1e-12)
    directions = context.ray_directions[relevant]
    ranges = context.ranges[relevant]
    dirs_local = directions @ pose.rotation

    expected_faces = covered_faces = supported_rays = incident_rays = 0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            face_coord = sign * half[axis]
            if sign * (origin_local[axis] - face_coord) <= 0.0:
                continue
            expected_faces += 1
            denom = dirs_local[:, axis]
            valid = np.abs(denom) > 1e-9
            distance = np.zeros(len(denom), dtype=np.float64)
            distance[valid] = (face_coord - origin_local[axis]) / denom[valid]
            valid &= distance > 0.0
            intersections = np.zeros_like(dirs_local)
            intersections[valid] = origin_local + distance[valid, None] * dirs_local[valid]
            for other in range(3):
                if other != axis:
                    valid &= np.abs(intersections[:, other]) <= half[other] + tolerance
            if not np.any(valid):
                continue
            incident_rays += int(np.count_nonzero(valid))
            supported = int(
                np.count_nonzero(np.abs(ranges[valid] - distance[valid]) <= tolerance)
            )
            supported_rays += supported
            if supported:
                covered_faces += 1

    shrunk = np.maximum(half - tolerance, 1e-3)
    _, t_exit, intersects = _ray_box_interval(origin_local, dirs_local, shrunk)
    violations = int(np.count_nonzero(intersects & (ranges > t_exit + tolerance)))
    return expected_faces, covered_faces, violations, supported_rays, incident_rays


def free_space_violations_from_misses(
    pose: BlockPose,
    origin: np.ndarray,
    miss_directions: np.ndarray,
    dims: tuple[float, float, float],
    tolerance: float = 0.06,
) -> int:
    """Count no-return rays that pass through the hypothesised block volume.

    A ray that reported no return travelled to (at least) the sensor's maximum
    range through empty space, so any such ray crossing the block volume is
    conclusive evidence against the hypothesis -- stronger than a terminating
    ray, which could always be explained by a surface behind the block.
    """
    directions = np.asarray(miss_directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or len(directions) == 0:
        return 0
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-12] / norms[norms > 1e-12, None]
    origin_local = pose.rotation.T @ (np.asarray(origin, dtype=np.float64) - pose.position)
    dirs_local = directions @ pose.rotation
    half = np.maximum(np.asarray(dims, dtype=np.float64) / 2.0 - tolerance, 1e-3)
    _, _, intersects = _ray_box_interval(origin_local, dirs_local, half)
    return int(np.count_nonzero(intersects))


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
) -> tuple[np.ndarray, GroundPlane | LocalGroundModel]:
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
    # Open3D's voxel downsample does not guarantee output ordering, and seeded
    # RANSAC samples by index, so an unsorted cloud reintroduces exactly the
    # nondeterminism the seeded fitter removes. `_voxel_downsample` already
    # sorts for this reason; the coarse pass must too.
    coarse_pts = coarse_pts[np.lexsort((coarse_pts[:, 2], coarse_pts[:, 1], coarse_pts[:, 0]))]

    def percentile_fallback() -> tuple[np.ndarray, GroundPlane]:
        ground_z = float(np.percentile(pts[:, 2], params.ground_z_percentile))
        ground = GroundPlane(np.array([0.0, 0.0, 1.0]), -ground_z, "percentile")
        return pts[ground.height(pts) >= params.ground_thickness], ground

    # Tensor voxelization can collapse an otherwise valid cloud below the
    # RANSAC sample size. Collinear samples also cannot define a plane.
    if len(coarse_pts) < 3 or np.linalg.matrix_rank(coarse_pts - coarse_pts.mean(axis=0)) < 2:
        return percentile_fallback()

    plane_model, _ = segment_plane(
        coarse_pts,
        distance_threshold=params.ground_ransac_dist,
        ransac_n=3,
        num_iterations=params.ground_ransac_iters,
        seed=params.ransac_seed,
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
    # Use a low local surface so slopes and modest terrain changes do not erase
    # object returns. Cells with too few samples are deliberately absent and
    # fall back to the dominant plane.
    cell_index = np.floor(pts[:, :2] / params.local_ground_cell_size).astype(np.int64)
    cell_xy, cell_z = [], []
    order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
    ordered_cells, ordered_pts = cell_index[order], pts[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(np.any(np.diff(ordered_cells, axis=0), axis=1)) + 1,
        len(pts),
    ]
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        key = ordered_cells[begin]
        samples = ordered_pts[begin:end]
        if len(samples) >= 3:
            center = (np.asarray(key, dtype=np.float64) + 0.5) * params.local_ground_cell_size
            planar_z = -(samples[:, :2] @ ground.normal[:2] + ground.offset) / ground.normal[2]
            center_planar_z = -(center @ ground.normal[:2] + ground.offset) / ground.normal[2]
            residual = samples[:, 2] - planar_z
            low = float(np.percentile(residual, 10.0))
            # A cell seeing only an object top is not terrain evidence. Local
            # terrain may deviate, but must include a low surface near the
            # globally fitted support plane.
            if low <= params.ground_thickness:
                cell_xy.append(center)
                cell_z.append(center_planar_z + low)
    local = LocalGroundModel(
        ground.normal,
        ground.offset,
        ground.source,
        params.local_ground_cell_size,
        np.asarray(cell_xy, dtype=np.float64).reshape(-1, 2),
        np.asarray(cell_z, dtype=np.float64),
    )
    signed_dist = local.height(pts)
    return pts[signed_dist >= params.local_ground_clearance], local


def _cluster_points_multiscale(
    pts: np.ndarray,
    params: DetectionParams,
    ground: GroundPlane | LocalGroundModel | None = None,
    sensor_origin: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Range-aware multi-scale deterministic KD-tree components.

    Components from all scales are unioned, then near-identical point sets are
    removed. The point/component caps bound pathological outdoor scenes.
    """
    # Retain the 5 cm-above-support cloud for downstream face fitting, but do
    # not let residual ground texture connect every object proposal together.
    all_pts = pts
    if ground is not None:
        pts = pts[np.asarray(ground.height(pts)) >= params.ground_thickness]
    if len(pts) == 0:
        return []
    if len(pts) > params.proposal_max_points:
        # Canonical evenly spaced sampling avoids input-order dependence.
        indices = np.linspace(0, len(pts) - 1, params.proposal_max_points, dtype=int)
        pts = pts[indices]
    tree = cKDTree(pts)
    ranges = (
        np.linalg.norm(pts - sensor_origin, axis=1)
        if sensor_origin is not None
        else np.full(len(pts), 5.0)
    )
    candidates: list[tuple[np.ndarray, float, frozenset[int]]] = []
    for scale in params.proposal_scales:
        max_radius = scale * (1.0 + params.proposal_range_gain * float(ranges.max()))
        # A capped neighbour graph avoids the quadratic pair materialisation of
        # query_pairs on dense surfaces while remaining sufficient to connect a
        # sampled 2-D face lattice.
        distances, neighbours = tree.query(
            pts, k=min(16, len(pts)), distance_upper_bound=max_radius
        )
        parent = np.arange(len(pts))

        def find(value: int, links: np.ndarray = parent) -> int:
            while links[value] != value:
                links[value] = links[links[value]]
                value = int(links[value])
            return value

        for first in range(len(pts)):
            for distance, second in zip(distances[first, 1:], neighbours[first, 1:], strict=True):
                if second >= len(pts) or not np.isfinite(distance):
                    continue
                radius = scale * (
                    1.0 + params.proposal_range_gain * min(ranges[first], ranges[second])
                )
                if distance > radius:
                    continue
                left, right = find(first), find(int(second))
                if left != right:
                    parent[max(left, right)] = min(left, right)
        groups: dict[int, list[int]] = {}
        for index in range(len(pts)):
            groups.setdefault(find(index), []).append(index)
        for indices in groups.values():
            if len(indices) < min(params.cluster_min_size, 12):
                continue
            index_set = frozenset(indices)
            if any(
                len(index_set & prior[2]) / min(len(index_set), len(prior[2])) >= 0.85
                for prior in candidates
            ):
                continue
            candidates.append((pts[np.asarray(indices)], scale, index_set))

    clusters: list[tuple[np.ndarray, float]] = []
    for cluster, scale, index_set in candidates:
        median_range = float(np.median(ranges[np.asarray(tuple(index_set))]))
        density_scale = float(np.clip((5.0 / max(median_range, 1.0)) ** 2, 0.25, 1.0))
        local_min_size = max(12, int(round(params.cluster_min_size * density_scale)))
        if not (local_min_size <= len(cluster) <= params.cluster_max_size):
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
        # Reattach nearby low side/boundary returns after connectivity is
        # decided. They improve fitting without acting as graph bridges.
        if len(all_pts) != len(pts):
            distance, _ = cKDTree(cluster).query(all_pts, k=1)
            expanded = all_pts[distance <= max(scale * 1.5, 0.09)]
            if len(expanded) <= params.cluster_max_size:
                cluster = expanded
        clusters.append((cluster, scale))
    clusters.sort(key=lambda item: (-len(item[0]), item[1], *item[0].mean(axis=0)))
    return clusters[: params.proposal_max_components]


def _cluster_points(
    pts: np.ndarray,
    params: DetectionParams,
    ground: GroundPlane | LocalGroundModel | None = None,
    sensor_origin: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Build bounded proposals, using the measured-safe fixed scale by default."""
    if params.multiscale_proposals:
        return _cluster_points_multiscale(pts, params, ground, sensor_origin)
    seed = pts
    if ground is not None:
        seed = pts[np.asarray(ground.height(pts)) >= params.ground_thickness]
    if len(seed) == 0:
        return []
    clusters: list[tuple[np.ndarray, float]] = []
    passes = [(seed, params.dbscan_eps, params.dbscan_min_points, params.cluster_min_size)]
    for pass_points, scale, min_points, min_size in passes:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pass_points)
        labels = np.asarray(
            cloud.cluster_dbscan(eps=scale, min_points=min_points, print_progress=False)
        )
        for label in np.unique(labels):
            if label < 0:
                continue
            cluster = pass_points[labels == label]
            if not (min_size <= len(cluster) <= params.cluster_max_size):
                continue
            extent = np.ptp(cluster, axis=0)
            center = (cluster.min(axis=0) + cluster.max(axis=0)) / 2.0
            if not (
                params.cluster_min_extent_xy <= extent[0] <= params.region_max_extent_xy
                and params.cluster_min_extent_xy <= extent[1] <= params.region_max_extent_xy
                and params.cluster_min_extent_z <= extent[2] <= params.cluster_max_extent_z
            ):
                continue
            if ground is not None and float(ground.height(center)) > params.cluster_max_center_z:
                continue
            # Fixed/far passes can rediscover the same object. Bounding-box
            # overlap is sufficient here because these are proposal regions.
            if any(
                np.linalg.norm(cluster.mean(axis=0) - prior.mean(axis=0)) < 0.15
                for prior, _ in clusters
            ):
                continue
            clusters.append((cluster, scale))
    clusters.sort(key=lambda item: (-len(item[0]), *item[0].mean(axis=0)))
    return clusters[: params.proposal_max_components]


def _fit_planes(
    pts: np.ndarray, params: DetectionParams
) -> tuple[list[PlanePatch], dict[str, int], dict[str, float]]:
    """Iteratively fit planes using RANSAC.

    Returns list of (normal, centroid, inlier_pts) tuples.
    """
    remaining = pts.copy()
    planes = []
    counts = {
        "plane_fit_calls": 0,
        "plane_search_points": 0,
        "plane_full_points_scored": 0,
        "plane_trials_evaluated": 0,
        "plane_valid_trials": 0,
    }
    timings = {"plane_search": 0.0, "plane_full_score_refit": 0.0}
    for index in range(params.max_planes):
        if len(remaining) < params.min_inliers:
            break
        # Vary the seed per extraction so successive planes are not drawn from
        # the same sample sequence, while staying reproducible overall.
        plane_model, inlier_idx, fit_diag = segment_plane(
            remaining,
            distance_threshold=params.ransac_dist_thresh,
            ransac_n=params.ransac_n,
            num_iterations=params.ransac_num_iterations,
            seed=params.ransac_seed + index,
            max_search_points=params.ransac_search_max_points,
            return_diagnostics=True,
        )
        counts["plane_fit_calls"] += 1
        counts["plane_search_points"] += fit_diag.search_points
        counts["plane_full_points_scored"] += fit_diag.input_points
        counts["plane_trials_evaluated"] += fit_diag.trials_evaluated
        counts["plane_valid_trials"] += fit_diag.valid_trials
        timings["plane_search"] += fit_diag.search_ms
        timings["plane_full_score_refit"] += fit_diag.full_score_refit_ms
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
    return planes, counts, timings


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


def _candidate_dims(params: DetectionParams) -> tuple[tuple[float, float, float], ...]:
    """Orientations to hypothesise, de-duplicated, defaulting to block_dims.

    Each candidate with a non-square horizontal cross-section is expanded with
    its yaw-swapped twin. `_synthesize_pose` constrains the centre using
    ``dims[0] / 2`` along the observed side normal, which silently assumes the
    footprint is square: for a laid 0.9 x 0.6 block the visible side may be the
    long face or the short one, and guessing wrong puts the centre 0.15 m out
    with a 90 degree yaw error. Both readings describe the same physical block,
    so both are proposed and the evidence score picks.
    """
    candidates = tuple(params.candidate_dims or ()) or (tuple(params.block_dims),)
    seen: list[tuple[float, float, float]] = []
    for dims in candidates:
        entry = tuple(float(value) for value in dims)
        swapped = (entry[1], entry[0], entry[2])
        for variant in (entry, swapped):
            if variant not in seen:
                seen.append(variant)
    return tuple(seen)


def _candidate_plane_sets(
    planes: list[PlanePatch],
    params: DetectionParams,
    ground: GroundPlane | None = None,
    dims: tuple[float, float, float] | None = None,
) -> list[tuple[PlanePatch, PlanePatch | None]]:
    """Pair observed tops with sides sharing the same layer boundary/support."""
    dims = tuple(dims) if dims is not None else tuple(params.block_dims)
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
            if not 0.15 <= centroid_drop <= dims[2] - 0.1:
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
            height_error = abs(centroid_drop - dims[2] / 2.0)
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


def _evidence_features(evidence: HypothesisEvidence) -> dict[str, float]:
    """Normalized [0, 1] quality features. Shared by scoring and diagnostics."""
    expected = evidence.expected_visible_faces
    total_rays = evidence.supported_rays + evidence.free_space_violations
    return {
        "support": min(evidence.support_points / 400.0, 1.0),
        "height_consistency": max(0.0, 1.0 - evidence.top_height_error_m / 0.2),
        "visible_face_coverage": evidence.covered_visible_faces / expected if expected else 0.0,
        "free_space_consistency": (
            evidence.supported_rays / total_rays if total_rays else 0.0
        ),
        # What fraction of the beams aimed at this cuboid actually terminate on
        # it. Distinguishes a well-placed hypothesis from a shifted one even when
        # both have full face coverage and no free-space violations.
        "surface_explained": (
            evidence.supported_rays / evidence.incident_rays if evidence.incident_rays else 0.0
        ),
    }


# Weights sum to 1.0, so score is bounded in [0, 1] and comparable across scenes.
_SCORE_WEIGHTS = {
    "support": 0.25,
    "height_consistency": 0.15,
    "visible_face_coverage": 0.20,
    "free_space_consistency": 0.20,
    "surface_explained": 0.20,
}


def _make_hypothesis(
    pose: BlockPose,
    top_centroid: np.ndarray,
    support_points: int,
    ground: GroundPlane | LocalGroundModel,
    params: DetectionParams,
    sensor_context: SensorContext | None = None,
    observed_geometry_faces: int = 1,
    dims: tuple[float, float, float] | None = None,
    proposal_scale_m: float | None = None,
) -> CuboidHypothesis:
    dims = tuple(dims) if dims is not None else tuple(params.block_dims)
    # Support-relative consistency: the candidate bottom should meet its local
    # support. Unlike layer quantisation, this does not penalise uneven terrain
    # or elevated blocks supported by observed geometry.
    bottom = pose.position - pose.rotation[:, 2] * (dims[2] / 2.0)
    support_height = float(ground.height(bottom))
    height_error = abs(support_height)
    expected = covered = violations = supported_rays = incident_rays = 0
    if sensor_context is not None:
        expected, covered, violations, supported_rays, incident_rays = visibility_evidence(
            pose, sensor_context, dims
        )
        # No-return beams crossing the volume are conclusive counter-evidence.
        violations += free_space_violations_from_misses(
            pose, sensor_context.origin, sensor_context.miss_directions, dims
        )
    evidence = HypothesisEvidence(
        support_points,
        height_error,
        0.0,
        expected,
        covered,
        violations,
        supported_rays,
        observed_geometry_faces,
        incident_rays,
    )
    # Every term is normalized and weighted. The previous formulation summed raw
    # ray counts (0.01 * supported_rays - 0.02 * violations), which dominated the
    # semantic terms by an order of magnitude and made the score a proxy for
    # proximity to the sensor: an identical block scored 8.4 near and 1.9 far.
    features = _evidence_features(evidence)
    score = sum(weight * features[name] for name, weight in _SCORE_WEIGHTS.items())
    return CuboidHypothesis(
        pose,
        replace(evidence, score=score),
        proposal_scale_m,
        support_height,
    )


def _canonicalize_pose(pose: BlockPose) -> BlockPose:
    """Express yaw-swapped laid boxes in canonical 0.9 x 0.6 x 0.6 form."""
    if np.allclose(pose.dims, (BLOCK_DIMS[1], BLOCK_DIMS[0], BLOCK_DIMS[2])):
        rotation = pose.rotation @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return BlockPose(pose.position, rotation, pose.confidence, pose.source, BLOCK_DIMS)
    return pose


def _boxes_overlap(first: BlockPose, second: BlockPose, shrink: float) -> bool:
    """Separating-axis test between two oriented boxes, each shrunk by `shrink`.

    Exact rather than a centre-distance heuristic, because competing hypotheses
    are not near-coincident: an upright and a laid interpretation of the same
    block sit ~0.21 m apart, just outside a 0.2 m radius, yet describe the same
    physical object and cannot both be true.

    Shrinking matters for the wall case: real courses sit 2 cm apart and touching
    blocks merely abut, so neither must suppress its neighbour. Only genuine
    interpenetration counts as a conflict.
    """
    half_a = np.maximum(np.asarray(first.dims, dtype=np.float64) / 2.0 - shrink, 1e-3)
    half_b = np.maximum(np.asarray(second.dims, dtype=np.float64) / 2.0 - shrink, 1e-3)
    rotation_a, rotation_b = first.rotation, second.rotation
    relative = rotation_a.T @ (second.position - first.position)
    coupling = rotation_a.T @ rotation_b
    absolute = np.abs(coupling) + 1e-9

    for axis in range(3):  # faces of the first box
        if abs(relative[axis]) > half_a[axis] + float(absolute[axis] @ half_b):
            return False
    for axis in range(3):  # faces of the second box
        reach = float(absolute[:, axis] @ half_a) + half_b[axis]
        if abs(float(relative @ coupling[:, axis])) > reach:
            return False
    for i in range(3):  # edge-edge cross products
        for j in range(3):
            reach_a = sum(
                half_a[k] * absolute[(i + offset) % 3, j]
                for k, offset in (((i + 2) % 3, 1), ((i + 1) % 3, 2))
            )
            reach_b = sum(
                half_b[k] * absolute[i, (j + offset) % 3]
                for k, offset in (((j + 2) % 3, 1), ((j + 1) % 3, 2))
            )
            distance = abs(
                relative[(i + 2) % 3] * coupling[(i + 1) % 3, j]
                - relative[(i + 1) % 3] * coupling[(i + 2) % 3, j]
            )
            if distance > reach_a + reach_b:
                return False
    return True


def _select_hypotheses(
    hypotheses: list[CuboidHypothesis], overlap_shrink: float = 0.08
) -> list[CuboidHypothesis]:
    """Greedy deterministic NMS: highest score wins; interpenetrating rivals lose."""
    ordered = sorted(
        hypotheses,
        key=lambda h: (-h.evidence.score, *np.round(h.pose.position, 9).tolist()),
    )
    selected: list[CuboidHypothesis] = []
    for candidate in ordered:
        conflict = any(
            _boxes_overlap(candidate.pose, accepted.pose, overlap_shrink)
            for accepted in selected
        )
        if not conflict:
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


def _detect_blocks_impl(
    pts_world: np.ndarray,
    params: DetectionParams | None = None,
    *,
    sensor_context: SensorContext | None = None,
) -> tuple[list[BlockPose], list[CuboidHypothesis], dict[str, int], dict[str, float]]:
    """Detect concrete blocks in a world-frame point cloud.

    Args:
        pts_world: (N, 3) float64 point cloud in world frame (Z = up).
        params: Detection parameters; uses defaults if None.
        sensor_context: Optional sensor origin and hit rays for visibility,
            occlusion, free-space evidence, and range-aware sampling thresholds.

    Returns:
        List of BlockPose, one per detected block, sorted by Z descending.
        Multiple poses may be returned from one connected region only when
        distinct top patches have compatible same-layer side/boundary support.
        Fully buried stack interfaces remain unobservable and are not inferred.
    """
    if params is None:
        params = DetectionParams()
    started = perf_counter()
    timings: dict[str, float] = {
        "voxel_downsample": 0.0,
        "ground_model": 0.0,
        "proposals": 0.0,
        "region_split": 0.0,
        "plane_search": 0.0,
        "plane_full_score_refit": 0.0,
        "hypothesis_generation_evidence": 0.0,
        "plane_hypotheses": 0.0,
        "selection": 0.0,
        "refinement": 0.0,
        "total": 0.0,
    }
    counts: dict[str, int] = {
        "input_points": len(pts_world),
        "downsampled_points": 0,
        "above_support_points": 0,
        "proposal_components": 0,
        "plane_regions": 0,
        "plane_fit_calls": 0,
        "plane_search_points": 0,
        "plane_full_points_scored": 0,
        "plane_trials_evaluated": 0,
        "plane_valid_trials": 0,
        "raw_hypotheses": 0,
        "selected_hypotheses": 0,
    }

    if sensor_context is not None:
        sensor_context = _validate_sensor_context(sensor_context)

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
        timings["total"] = (perf_counter() - started) * 1000.0
        return [], [], counts, timings
    pts = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]

    # Stage 1: Voxel downsample
    pts = _voxel_downsample(pts, params.voxel_size)
    timings["voxel_downsample"] = (perf_counter() - started) * 1000.0
    counts["downsampled_points"] = len(pts)
    if len(pts) < 3:
        timings["total"] = (perf_counter() - started) * 1000.0
        return [], [], counts, timings

    # Stage 2: Ground removal
    stage_started = perf_counter()
    pts, ground = _remove_ground(pts, params)
    timings["ground_model"] = (perf_counter() - stage_started) * 1000.0
    counts["above_support_points"] = len(pts)
    if len(pts) == 0:
        timings["total"] = (perf_counter() - started) * 1000.0
        return [], [], counts, timings

    # Stage 3: DBSCAN clustering
    stage_started = perf_counter()
    clusters = _cluster_points(
        pts,
        params,
        ground,
        sensor_context.origin if sensor_context is not None else None,
    )
    timings["proposals"] = (perf_counter() - stage_started) * 1000.0
    counts["proposal_components"] = len(clusters)

    # Stage 4–6: Per-cluster plane fitting and pose synthesis
    hypotheses: list[CuboidHypothesis] = []
    plane_subtimings = {"plane_search": 0.0, "plane_full_score_refit": 0.0}
    stage_started = perf_counter()
    for cluster, proposal_scale in clusters:
        if sensor_context is not None:
            median_range = float(np.median(np.linalg.norm(cluster - sensor_context.origin, axis=1)))
            density_scale = float(np.clip((5.0 / max(median_range, 1.0)) ** 2, 0.25, 1.0))
            cluster_params = replace(
                params,
                min_inliers=max(12, int(round(params.min_inliers * density_scale))),
                cluster_min_size=max(12, int(round(params.cluster_min_size * density_scale))),
            )
        else:
            cluster_params = params
        split_started = perf_counter()
        regions = _split_connected_region(cluster, cluster_params)
        timings["region_split"] += (perf_counter() - split_started) * 1000.0
        if len(regions) == 1 and np.any(
            np.ptp(cluster, axis=0)[:2] > params.cluster_max_extent_xy
        ):
            # A large connected footprint without internal boundary evidence is
            # an elongated object, not a license to tile it with known boxes.
            continue
        for region in regions:
            counts["plane_regions"] += 1
            planes, fit_counts, fit_timings = _fit_planes(region, cluster_params)
            for name, value in fit_counts.items():
                counts[name] += value
            for name, value in fit_timings.items():
                plane_subtimings[name] += value
            if not planes:
                continue
            hypothesis_started = perf_counter()
            # One hypothesis set per candidate orientation. The wrong orientation
            # puts faces where no returns are and leaves observed surface
            # unexplained, so `surface_explained` and `free_space_consistency`
            # separate them; selection then keeps the better-scoring one.
            for dims in _candidate_dims(params):
                for top, side in _candidate_plane_sets(planes, cluster_params, ground, dims):
                    top_n = top.normal if top.normal[2] > 0 else -top.normal
                    if side is not None:
                        local_support = np.vstack([top.points, side.points])
                        pose = _synthesize_pose(
                            top_n, top.centroid, side.normal, side.centroid, dims, local_support
                        )
                        support = len(top.points) + len(side.points)
                    else:
                        pose = _synthesize_pose_top_only(top_n, top.centroid, region, dims)
                        support = len(top.points)
                    pose.dims = dims
                    pose = _canonicalize_pose(pose)
                    hypothesis = _make_hypothesis(
                        pose,
                        top.centroid,
                        support,
                        ground,
                        params,
                        sensor_context,
                        observed_geometry_faces=2 if side is not None else 1,
                        dims=pose.dims,
                        proposal_scale_m=proposal_scale,
                    )
                    # Support-relative gating allows for terrain/support model
                    # uncertainty while rejecting planes too low to be a block
                    # top. Exact layer multiples are intentionally not assumed.
                    top_support_height = float(ground.height(top.centroid))
                    first_course_supported = not (
                        top_support_height <= dims[2] * 1.1
                        and abs(float(hypothesis.support_height_m or 0.0)) > 0.135
                    )
                    if top_support_height >= dims[2] * 0.70 and first_course_supported:
                        hypotheses.append(hypothesis)
            timings["hypothesis_generation_evidence"] += (
                perf_counter() - hypothesis_started
            ) * 1000.0

    timings["plane_hypotheses"] = (perf_counter() - stage_started) * 1000.0
    timings.update(plane_subtimings)
    counts["raw_hypotheses"] = len(hypotheses)

    selection_started = perf_counter()
    hypotheses = [h for h in hypotheses if h.evidence.score >= params.min_score]
    selected = _select_hypotheses(hypotheses)
    timings["selection"] = (perf_counter() - selection_started) * 1000.0

    if params.refine_enabled and selected:
        refinement_started = perf_counter()
        # Refine against the above-ground cloud, then rebuild evidence from the
        # refined pose so diagnostics describe the pose actually returned. The
        # top-face centre is recomputed from the pose rather than reusing the
        # original plane centroid, which no longer corresponds to it.
        refined = refine_poses(
            [hypothesis.pose for hypothesis in selected],
            pts,
            None,  # per-pose extents; a mixed-orientation scene has no single dims
            band=params.refine_band,
            iterations=params.refine_iterations,
            min_points=params.refine_min_points,
            max_translation=params.refine_max_translation,
            max_rotation_deg=params.refine_max_rotation_deg,
        )
        selected = [
            _make_hypothesis(
                pose,
                pose.position + pose.rotation[:, 2] * (pose.dims[2] / 2.0),
                hypothesis.evidence.support_points,
                ground,
                params,
                sensor_context,
                hypothesis.evidence.observed_geometry_faces,
                dims=pose.dims,
                proposal_scale_m=hypothesis.proposal_scale_m,
            )
            for hypothesis, pose in zip(selected, refined, strict=True)
        ]
        timings["refinement"] = (perf_counter() - refinement_started) * 1000.0

    selected = [
        hypothesis
        for hypothesis in selected
        if 0.0
        <= float(ground.height(hypothesis.pose.position))
        <= params.cluster_max_center_z
    ]
    selected.sort(key=lambda hypothesis: hypothesis.pose.position[2], reverse=True)
    counts["selected_hypotheses"] = len(selected)
    timings["total"] = (perf_counter() - started) * 1000.0
    return [hypothesis.pose for hypothesis in selected], selected, counts, timings


def _diagnostics(hypothesis: CuboidHypothesis) -> DetectionDiagnostics:
    evidence = hypothesis.evidence
    features = _evidence_features(evidence)
    observed_faces = max(evidence.covered_visible_faces, evidence.observed_geometry_faces)
    # A ratio, not an exact-zero test. Requiring free_space_violations == 0 made
    # FULL unreachable while the violation count was buggy, and would stay
    # brittle now that mixed pixels can put a stray point inside the volume.
    clean = features["free_space_consistency"]
    if observed_faces >= 3 and clean >= 0.98:
        observability = Observability.FULL
    elif observed_faces >= 2 and clean >= 0.90:
        observability = Observability.PARTIAL
    else:
        observability = Observability.LIMITED
    return DetectionDiagnostics(
        evidence,
        observability,
        features,
        None,
        hypothesis.proposal_scale_m,
        hypothesis.support_height_m,
    )


def detect_blocks_detailed(
    pts_world: np.ndarray,
    params: DetectionParams | None = None,
    *,
    sensor_context: SensorContext | None = None,
) -> DetectionResult:
    """Return poses plus normalized evidence and geometry observability."""
    poses, hypotheses, counts, timings = _detect_blocks_impl(
        pts_world, params, sensor_context=sensor_context
    )
    diagnostics = [_diagnostics(hypothesis) for hypothesis in hypotheses]
    return DetectionResult(poses, diagnostics, counts, timings)


def detect_blocks(
    pts_world: np.ndarray,
    params: DetectionParams | None = None,
    *,
    sensor_context: SensorContext | None = None,
) -> list[BlockPose]:
    """Compatibility API returning only poses, sorted by descending world Z."""
    return detect_blocks_detailed(
        pts_world, params, sensor_context=sensor_context
    ).poses


def detect_scene_sample(sample, params: DetectionParams | None = None) -> list[BlockPose]:
    """Detect a rich sample carrying points, sensor origin, rays, and ranges.

    The argument is intentionally duck-typed to avoid coupling the detector to
    the synthetic generator's ``SceneSample`` class.
    """
    context = SensorContext(
        np.asarray(sample.sensor_origin, dtype=np.float64),
        np.asarray(sample.ray_directions, dtype=np.float64),
        np.asarray(sample.ranges, dtype=np.float64),
        np.asarray(getattr(sample, "miss_directions", np.zeros((0, 3))), dtype=np.float64),
    )
    return detect_blocks(sample.points, params, sensor_context=context)
