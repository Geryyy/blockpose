"""Typed Python data containers for the native C++ detector API.

These classes are transport values only; all discovery, scoring, and
refinement are implemented by :mod:`blockpose._core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from blockpose.pose import BLOCK_DIMS, BlockPose


@dataclass
class DetectionParameters:
    """Compatibility input for the explicitly supported native parameters."""

    voxel_size: float = 0.04
    ground_thickness: float = 0.15
    ground_ransac_dist: float = 0.05
    ground_ransac_iters: int = 500
    ground_normal_min_z: float = 0.7
    dbscan_eps: float = 0.07
    dbscan_min_points: int = 5
    cluster_min_size: int = 30
    cluster_max_size: int = 50000
    cluster_min_extent_xy: float = 0.3
    # 2.25, not 1.5: pipeline.hpp drops any unsplit component wider than this
    # BEFORE fitting a plane to it, and blocks touching in a wall merge into
    # exactly such a component -- measured over 62 annotated blocks, one in an
    # over-gate component was missed 91% of the time against 25% otherwise.
    # Raising it to 2.25 gives recall 62.9% -> 74.2% AND precision 75.0% ->
    # 78.0%, losing no previously-detected block. The result is flat from 2.25
    # to 3.5, so this is the knee: the gate was rejecting valid clusters rather
    # than guarding against merged ones. See experiments/detector_cluster_failures.md.
    cluster_max_extent_xy: float = 2.25
    region_max_extent_xy: float = 3.0
    cluster_min_extent_z: float = 0.3
    cluster_max_extent_z: float = 3.5
    cluster_max_center_z: float = 3.0
    ransac_dist_thresh: float = 0.02
    ransac_num_iterations: int = 1000
    ransac_search_max_points: int = 2048
    ransac_seed: int = 0
    max_planes: int = 5
    min_inliers: int = 40
    top_plane_angle_deg: float = 30.0
    side_plane_angle_deg: float = 20.0
    block_dims: tuple[float, float, float] = BLOCK_DIMS
    candidate_dims: tuple[tuple[float, float, float], ...] = ()
    refine_band: float = 0.10
    refine_iterations: int = 2
    refine_min_points: int = 20
    refine_max_translation: float = 0.15
    refine_max_rotation_deg: float = 20.0
    min_score: float = 0.0
    local_ground_cell_size: float = 0.5
    local_ground_clearance: float = 0.05
    proposal_max_components: int = 8
    conflict_alternatives: int = 1


class Observability(str, Enum):
    LIMITED = "limited"
    PARTIAL = "partial"
    FULL = "full"


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
    incident_rays: int = 0


@dataclass(frozen=True)
class DetectionDiagnostics:
    evidence: HypothesisEvidence
    observability: Observability
    features: dict[str, float]
    calibrated_confidence: float | None = None
    proposal_scale_m: float | None = None
    support_height_m: float | None = None


@dataclass(frozen=True)
class TraceHypothesis:
    id: str
    pose: BlockPose
    score: float
    stage: str
    fate: str


@dataclass(frozen=True)
class RawHypothesisLineage:
    id: str
    proposal_component: int
    region: int
    top_plane: int
    side_plane: int | None
    top_normal: np.ndarray
    top_centroid: np.ndarray
    side_normal: np.ndarray | None
    side_centroid: np.ndarray | None
    candidate_dims: tuple[float, float, float]
    top_only: bool
    synthesized_pose: BlockPose
    evidence: HypothesisEvidence
    top_support_height_m: float
    accepted_to_raw: bool
    passed_score_threshold: bool
    fate: str


@dataclass(frozen=True)
class InferenceTrace:
    downsampled_points: np.ndarray
    above_support_points: np.ndarray
    proposal_points: tuple[np.ndarray, ...]
    raw_hypotheses: tuple[TraceHypothesis, ...]
    threshold_hypotheses: tuple[TraceHypothesis, ...]
    final_hypotheses: tuple[TraceHypothesis, ...]
    pre_refinement_hypotheses: tuple[TraceHypothesis, ...] = ()
    raw_lineage: tuple[RawHypothesisLineage, ...] = ()


@dataclass(frozen=True)
class DetectionResult:
    poses: list[BlockPose]
    diagnostics: list[DetectionDiagnostics]
    stage_counts: dict[str, int] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    trace: InferenceTrace | None = None


@dataclass(frozen=True)
class SensorContext:
    origin: np.ndarray
    ray_directions: np.ndarray
    ranges: np.ndarray
    miss_directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    ray_origins: np.ndarray | None = None
    miss_origins: np.ndarray | None = None
    # Volumes known to block the sensor's view -- in practice the robot's own
    # gripper, whose pose comes from FK. Rays these stop carry no information
    # about anything behind them, so the native evidence code drops them rather
    # than counting them against a hypothesis standing in the shadow. Pass the
    # centre/rotation/size triple `filter_pzs100_gripper` already returns.
    occluders: dict[str, np.ndarray] | None = None

    @classmethod
    def from_points(
        cls,
        origin: np.ndarray,
        points: np.ndarray,
        *,
        ray_origins: np.ndarray | None = None,
        occluders: dict[str, np.ndarray] | None = None,
    ) -> "SensorContext":
        origin, points = np.asarray(origin, dtype=np.float64), np.asarray(points, dtype=np.float64)
        origins = (
            np.broadcast_to(origin, points.shape)
            if ray_origins is None
            else np.asarray(ray_origins, dtype=np.float64)
        )
        if (
            origin.shape != (3,)
            or origins.shape != points.shape
            or not np.all(np.isfinite(origins))
        ):
            raise ValueError("ray_origins must have one finite world origin per point")
        vectors = points - origins
        ranges = np.linalg.norm(vectors, axis=1)
        valid = ranges > 1e-9
        return cls(
            origin,
            vectors[valid] / ranges[valid, None],
            ranges[valid],
            ray_origins=origins[valid],
            occluders=occluders,
        )
