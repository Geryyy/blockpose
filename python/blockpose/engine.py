"""Hybrid detector facade for free, placed, and gripped block queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from blockpose.core_bridge import detect_blocks_detailed_cpp
from blockpose.core_types import DetectionParameters, SensorContext
from blockpose.frames import FrameBurst, PointFrame
from blockpose.pose import BlockPose


class QueryStatus(str, Enum):
    DETECTED = "Detected"
    CONFIRMED = "Confirmed"
    DEVIATED = "Deviated"
    MISSING = "Missing"
    NOT_OBSERVABLE = "NotObservable"


class ResultObservability(str, Enum):
    WELL_CONSTRAINED = "WellConstrained"
    PARTIALLY_CONSTRAINED = "PartiallyConstrained"
    SYMMETRY_AMBIGUOUS = "SymmetryAmbiguous"
    LAYOUT_INFERRED = "LayoutInferred"
    UNOBSERVABLE = "Unobservable"


class QueryErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE_TRANSFORM = "unavailable_transform"
    EMPTY_ROI = "empty_roi"
    TIMEOUT = "timeout"
    INSUFFICIENT_OBSERVABILITY = "insufficient_observability"
    UNSUPPORTED_MODE = "unsupported_mode"


@dataclass(frozen=True)
class QueryFailure:
    code: QueryErrorCode
    message: str


@dataclass(frozen=True)
class BlockResult:
    pose: BlockPose
    dimensions: tuple[float, float, float]
    covariance: np.ndarray | None
    confidence: float | None
    observability: ResultObservability
    unconstrained_directions: tuple[str, ...]
    source: str
    score_features: dict[str, float]
    status: QueryStatus
    prior_deviation_m: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    blocks: tuple[BlockResult, ...]
    failures: tuple[QueryFailure, ...]
    runtime_ms: float
    unavailable_evidence: tuple[str, ...]


FrameInput = PointFrame | FrameBurst | np.ndarray


class HybridBlockDetector:
    """Stable facade over C++ free discovery; placed modes await C++ ports."""

    def __init__(
        self,
        params: DetectionParameters | None = None,
        *,
        timeout_ms: float = 500.0,
        verify_placed_v1: bool = False,
    ):
        self.params = params or DetectionParameters(min_score=0.4)
        self.timeout_ms = float(timeout_ms)
        if not np.isfinite(self.timeout_ms) or self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive and finite")
        self._world_state: tuple[BlockPose, ...] = ()
        self.verify_placed_v1 = bool(verify_placed_v1)

    def set_world_state(self, blocks: list[BlockPose]) -> None:
        self._validate_poses(blocks)
        self._world_state = tuple(blocks)

    @staticmethod
    def _validate_poses(poses: list[BlockPose]) -> None:
        for pose in poses:
            rotation = np.asarray(pose.rotation, dtype=np.float64)
            dims = np.asarray(pose.dims, dtype=np.float64)
            if (
                pose.position.shape != (3,)
                or not np.all(np.isfinite(pose.position))
                or rotation.shape != (3, 3)
                or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
                or dims.shape != (3,)
                or not np.all(np.isfinite(dims))
                or np.any(dims <= 0)
            ):
                message = "block poses require finite centers, positive dimensions, and SO(3)"
                raise ValueError(message)

    def _points(
        self, source: FrameInput, *, deadline_s: float | None = None
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if isinstance(source, FrameBurst):
            fused = source.fuse(deadline_s=deadline_s)
            origin = (
                fused.point_origins_world[-1]
                if fused.point_origins_world is not None and len(fused.point_origins_world)
                else fused.no_return_origins_world[-1]
                if fused.no_return_origins_world is not None
                and len(fused.no_return_origins_world)
                else None
            )
            return (
                fused.points_world,
                fused.unavailable_evidence,
                origin,
                fused.no_return_directions_world,
                fused.point_origins_world,
                fused.no_return_origins_world,
            )
        if isinstance(source, PointFrame):
            points = source.points_in_world()
            origins = source.point_origins_in_world()
            misses = source.no_return_rays_in_world()
            if origins is None:
                return points, source.unavailable_evidence, None, None, None, None
            origin = (
                origins[-1]
                if len(origins)
                else misses[0][-1]
                if misses is not None and len(misses[0])
                else None
            )
            return (
                points,
                source.unavailable_evidence,
                origin,
                None if misses is None else misses[1],
                origins,
                None if misses is None else misses[0],
            )
        points = np.asarray(source, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("point input must have shape (N, 3) and be finite")
        unavailable = (
            "intensity",
            "relative_time",
            "beam_id",
            "no_return_rays",
            "per_point_sensor_origins",
            "no_return_ray_origins",
            "gripper_trajectory",
        )
        return points, unavailable, None, None, None, None

    @staticmethod
    def _sensor_context(
        points: np.ndarray,
        origin: np.ndarray | None,
        misses: np.ndarray | None,
        ray_origins: np.ndarray | None = None,
        miss_origins: np.ndarray | None = None,
    ) -> SensorContext | None:
        if origin is None:
            return None
        context = SensorContext.from_points(origin, points, ray_origins=ray_origins)
        if misses is not None:
            context = SensorContext(
                context.origin,
                context.ray_directions,
                context.ranges,
                misses,
                context.ray_origins,
                miss_origins,
            )
        return context

    def detect_free(
        self, source: FrameInput, workspace_roi: tuple[np.ndarray, np.ndarray] | None = None
    ) -> QueryResult:
        started = time.perf_counter()
        try:
            points, unavailable, origin, misses, ray_origins, miss_origins = self._points(
                source, deadline_s=started + self.timeout_ms / 1000.0
            )
        except TimeoutError as exc:
            return self._failure(started, QueryErrorCode.TIMEOUT, str(exc))
        except ValueError as exc:
            code = (
                QueryErrorCode.UNAVAILABLE_TRANSFORM
                if isinstance(source, (PointFrame, FrameBurst))
                else QueryErrorCode.INVALID_INPUT
            )
            return self._failure(started, code, str(exc))
        if workspace_roi is not None:
            try:
                low, high = (np.asarray(v, dtype=np.float64) for v in workspace_roi)
            except (TypeError, ValueError):
                return self._failure(started, QueryErrorCode.INVALID_INPUT, "invalid workspace ROI")
            if (
                low.shape != (3,)
                or high.shape != (3,)
                or not np.all(np.isfinite(low))
                or not np.all(np.isfinite(high))
                or np.any(low >= high)
            ):
                return self._failure(started, QueryErrorCode.INVALID_INPUT, "invalid workspace ROI")
            mask = np.all((points >= low) & (points <= high), axis=1)
            points = points[mask]
            if ray_origins is not None:
                ray_origins = ray_origins[mask]
        if not len(points):
            return self._failure(started, QueryErrorCode.EMPTY_ROI, "no points in workspace ROI")
        context = self._sensor_context(points, origin, misses, ray_origins, miss_origins)
        result = detect_blocks_detailed_cpp(points, params=self.params, sensor_context=context)
        blocks = []
        for pose, diag in zip(result.poses, result.diagnostics, strict=True):
            obs = {
                "full": ResultObservability.WELL_CONSTRAINED,
                "partial": ResultObservability.PARTIALLY_CONSTRAINED,
                "limited": ResultObservability.SYMMETRY_AMBIGUOUS,
            }[diag.observability.value]
            blocks.append(
                BlockResult(
                    pose,
                    pose.dims,
                    None,
                    diag.calibrated_confidence,
                    obs,
                    ("yaw",) if obs == ResultObservability.SYMMETRY_AMBIGUOUS else (),
                    pose.source or "detect_free",
                    diag.features,
                    QueryStatus.DETECTED,
                    diagnostics={"raw_score": diag.evidence.score},
                )
            )
        return self._finish(started, blocks, unavailable)

    def verify_placed(self, source: FrameInput, expected_blocks: list[BlockPose]) -> QueryResult:
        started = time.perf_counter()
        # Placed verification and wall-plan priors were Python prototypes. They
        # are intentionally unavailable until their native C++ equivalents land.
        return self._failure(
            started,
            QueryErrorCode.UNSUPPORTED_MODE,
            "verify_placed is unavailable: its Python implementation was removed; "
            "use detect_free until the C++ placed-verification API is available",
        )
    def estimate_gripped(self, source: FrameInput, grasp_prior: BlockPose) -> QueryResult:
        return self._failure(
            time.perf_counter(),
            QueryErrorCode.UNSUPPORTED_MODE,
            "estimate_gripped is unavailable until the C++ placed/gripper API is available",
        )

    def _finish(
        self, started: float, blocks: list[BlockResult], unavailable: tuple[str, ...]
    ) -> QueryResult:
        runtime = (time.perf_counter() - started) * 1000.0
        if runtime > self.timeout_ms:
            message = f"query exceeded {self.timeout_ms:g} ms"
            # This is a post-hoc soft deadline: Python work cannot be interrupted,
            # so discard completed blocks rather than presenting stale results.
            return QueryResult(
                (), (QueryFailure(QueryErrorCode.TIMEOUT, message),), runtime, unavailable
            )
        return QueryResult(tuple(blocks), (), runtime, unavailable)

    def _failure(self, started: float, code: QueryErrorCode, message: str) -> QueryResult:
        runtime = (time.perf_counter() - started) * 1000
        return QueryResult((), (QueryFailure(code, message),), runtime, ())
