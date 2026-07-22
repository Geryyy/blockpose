"""Hybrid detector facade for free, placed, and gripped block queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from blockpose.detection import DetectionParams, SensorContext, detect_blocks_detailed
from blockpose.frames import FrameBurst, PointFrame
from blockpose.plan_prior import detect_with_plan
from blockpose.pose import BlockPose
from blockpose.verify_placed import (
    PlacedObservability,
    PlacedStatus,
    VerifyPlacedParams,
    verify_placed_blocks,
)


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
    """Stable facade over the current blind and plan-prior Python prototypes."""

    def __init__(
        self,
        params: DetectionParams | None = None,
        *,
        timeout_ms: float = 500.0,
        verify_placed_v1: bool = False,
    ):
        self.params = params or DetectionParams(min_score=0.4)
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
        self, source: FrameInput
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray | None, np.ndarray | None]:
        if isinstance(source, FrameBurst):
            fused = source.fuse()
            return fused.points_world, fused.unavailable_evidence, None, None
        if isinstance(source, PointFrame):
            points = source.points_in_world()
            if source.sensor_trajectory is None:
                return points, source.unavailable_evidence, None, None
            transform = source.sensor_trajectory.sample(np.array([source.timestamp]))[0]
            origin = transform[:3, 3]
            misses = None
            if source.no_return_rays is not None:
                misses = source.no_return_rays @ transform[:3, :3].T
            return points, source.unavailable_evidence, origin, misses
        points = np.asarray(source, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("point input must have shape (N, 3) and be finite")
        unavailable = (
            "intensity",
            "relative_time",
            "beam_id",
            "no_return_rays",
            "gripper_trajectory",
        )
        return points, unavailable, None, None

    @staticmethod
    def _sensor_context(
        points: np.ndarray, origin: np.ndarray | None, misses: np.ndarray | None
    ) -> SensorContext | None:
        if origin is None:
            return None
        context = SensorContext.from_points(origin, points)
        if misses is not None:
            context = SensorContext(context.origin, context.ray_directions, context.ranges, misses)
        return context

    def detect_free(
        self, source: FrameInput, workspace_roi: tuple[np.ndarray, np.ndarray] | None = None
    ) -> QueryResult:
        started = time.perf_counter()
        try:
            points, unavailable, origin, misses = self._points(source)
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
            points = points[np.all((points >= low) & (points <= high), axis=1)]
        if not len(points):
            return self._failure(started, QueryErrorCode.EMPTY_ROI, "no points in workspace ROI")
        context = self._sensor_context(points, origin, misses)
        result = detect_blocks_detailed(points, self.params, sensor_context=context)
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
        try:
            points, unavailable, origin, misses = self._points(source)
        except ValueError as exc:
            code = (
                QueryErrorCode.UNAVAILABLE_TRANSFORM
                if isinstance(source, (PointFrame, FrameBurst))
                else QueryErrorCode.INVALID_INPUT
            )
            return self._failure(started, code, str(exc))
        if not expected_blocks:
            return self._finish(started, [], unavailable)
        try:
            self._validate_poses(expected_blocks)
        except ValueError as exc:
            return self._failure(started, QueryErrorCode.INVALID_INPUT, str(exc))
        context = self._sensor_context(points, origin, misses)
        if self.verify_placed_v1:
            verified = verify_placed_blocks(
                points,
                expected_blocks,
                sensor_context=context,
                params=VerifyPlacedParams(
                    max_translation_m=0.10,
                    max_rotation_deg=10.0,
                    min_support_points=self.params.refine_min_points,
                ),
            )
            status_map = {
                PlacedStatus.CONFIRMED: QueryStatus.CONFIRMED,
                PlacedStatus.DEVIATED: QueryStatus.DEVIATED,
                PlacedStatus.MISSING: QueryStatus.MISSING,
                PlacedStatus.NOT_OBSERVABLE: QueryStatus.NOT_OBSERVABLE,
            }
            observability_map = {
                PlacedObservability.WELL_CONSTRAINED: ResultObservability.WELL_CONSTRAINED,
                PlacedObservability.PARTIALLY_CONSTRAINED: (
                    ResultObservability.PARTIALLY_CONSTRAINED
                ),
                PlacedObservability.SYMMETRY_AMBIGUOUS: (ResultObservability.SYMMETRY_AMBIGUOUS),
                PlacedObservability.UNOBSERVABLE: ResultObservability.UNOBSERVABLE,
            }
            blocks = [
                BlockResult(
                    result.pose,
                    result.pose.dims,
                    result.covariance,
                    None,
                    observability_map[result.observability],
                    result.unconstrained_directions,
                    "verify_placed_v1",
                    {
                        "support_points": float(result.support_points),
                        "free_space_rays": float(result.free_space_rays),
                        **(
                            {"residual_rms_m": float(result.residual_rms_m)}
                            if result.residual_rms_m is not None
                            else {}
                        ),
                    },
                    status_map[result.status],
                    result.translation_from_plan_m,
                    {
                        **result.diagnostics,
                        "rotation_from_plan_deg": result.rotation_from_plan_deg,
                        "visible_faces": result.visible_faces,
                        "occluded_faces": result.occluded_faces,
                    },
                )
                for result in verified
            ]
            return self._finish(started, blocks, unavailable)
        results = detect_with_plan(points, expected_blocks, self.params, sensor_context=context)
        blocks = []
        for result in results:
            observable = (
                result.refined
                and result.support_points >= self.params.refine_min_points
                and result.score >= self.params.min_score
            )
            status = QueryStatus.NOT_OBSERVABLE
            observability = ResultObservability.UNOBSERVABLE
            if observable:
                status = (
                    QueryStatus.CONFIRMED
                    if result.translation_from_plan <= 0.05
                    else QueryStatus.DEVIATED
                )
                observability = ResultObservability.PARTIALLY_CONSTRAINED
            elif (
                context is not None
                and result.support_points == 0
                and result.incident_rays > 0
                and result.free_space_violations > 0
            ):
                status = QueryStatus.MISSING
            blocks.append(
                BlockResult(
                    result.pose,
                    result.pose.dims,
                    None,
                    None,
                    observability,
                    () if observable else ("x", "y", "z", "roll", "pitch", "yaw"),
                    "plan_prior",
                    result.features,
                    status,
                    result.translation_from_plan,
                    {
                        "support_points": result.support_points,
                        "refined": result.refined,
                        "raw_score": result.score,
                        "incident_rays": result.incident_rays,
                    },
                )
            )
        return self._finish(started, blocks, unavailable)

    def estimate_gripped(self, source: FrameInput, grasp_prior: BlockPose) -> QueryResult:
        result = self.verify_placed(source, [grasp_prior])
        if not result.blocks:
            return result
        block = result.blocks[0]
        # Gripper geometry is not yet modelled. Be explicit: an unsupported prior
        # is not a measurement, and a refined pose remains only partially constrained.
        block = replace(
            block,
            source="grasp_prior",
            status=(
                QueryStatus.DEVIATED
                if block.status == QueryStatus.DEVIATED
                else QueryStatus.CONFIRMED
                if block.status == QueryStatus.CONFIRMED
                else QueryStatus.NOT_OBSERVABLE
            ),
        )
        return replace(result, blocks=(block,))

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
