"""Plan-prior detection: verify and re-estimate blocks at expected poses.

Blind detection discovers blocks from geometry alone. That is the right default,
but it cannot separate blocks laid in contact or with a 2 cm joint: at the
sensor's resolution the abutting faces are coplanar and continuous, and the
returns that mark the gap are stripped by ground removal and clustering before
any splitter sees them (measured; see docs/detector-plan.md). A wall being built
is exactly the case where the plan IS known -- the robot placed the blocks -- so
the expected poses are available and should be used.

This module takes expected poses (from `wall_plans`, or the world model of
already-placed blocks) and, for each, fits the exact cuboid to nearby returns
with the same SDF refinement the blind detector uses. It then scores the fit
with the same evidence model, so a block that was moved, is missing, or was never
placed scores low rather than being reported blindly at its expected pose.

Two uses, both named by the user:
  * verify/re-estimate a wall course the blind detector cannot separate;
  * update the pose of an already-placed block before stacking the next on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blockpose.detection import (
    DetectionParams,
    GroundPlane,
    SensorContext,
    _evidence_features,
    _make_hypothesis,
    _remove_ground,
    _validate_sensor_context,
    _voxel_downsample,
)
from blockpose.pose import BlockPose
from blockpose.refine import refine_pose


@dataclass(frozen=True)
class PlanPriorResult:
    """One expected block, its refined pose, and how well the returns support it."""

    expected: BlockPose
    pose: BlockPose
    score: float
    features: dict[str, float]
    support_points: int
    refined: bool

    @property
    def translation_from_plan(self) -> float:
        return float(np.linalg.norm(self.pose.position - self.expected.position))


def detect_with_plan(
    pts_world: np.ndarray,
    expected: list[BlockPose],
    params: DetectionParams | None = None,
    *,
    sensor_context: SensorContext | None = None,
    assign_band: float = 0.12,
) -> list[PlanPriorResult]:
    """Fit each expected block to nearby returns and score the fit.

    Points are assigned to the nearest expected surface within `assign_band`, so
    neighbouring wall blocks 2 cm apart take their own returns even though blind
    clustering cannot separate them. Refinement then snaps each pose to its
    points, and the shared evidence model scores it. Ordering follows `expected`;
    the caller thresholds on `score` to decide what to trust.
    """
    if params is None:
        params = DetectionParams()
    points = np.asarray(pts_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"pts_world must have shape (N, 3), got {points.shape}")
    if not expected:
        return []
    if sensor_context is not None:
        sensor_context = _validate_sensor_context(sensor_context)

    # Ground plane from the same routine the blind detector uses, so height
    # scoring is measured against the same reference.
    if len(points) >= 3:
        downsampled = _voxel_downsample(points, params.voxel_size)
        object_points, ground = _remove_ground(downsampled, params)
    else:
        object_points = points
        ground = GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0, "empty")

    # Assign object points to the nearest expected block surface, exclusively, so
    # one block's face cannot also feed its neighbour's fit.
    from blockpose.refine import surface_distance

    if len(object_points):
        distances = np.stack(
            [np.abs(surface_distance(pose, object_points, pose.dims)) for pose in expected],
            axis=1,
        )
        owner = np.argmin(distances, axis=1)
        within = distances[np.arange(len(object_points)), owner] <= assign_band
    else:
        owner = np.zeros(0, dtype=int)
        within = np.zeros(0, dtype=bool)

    results: list[PlanPriorResult] = []
    for index, prior in enumerate(expected):
        owned = object_points[within & (owner == index)] if len(object_points) else object_points
        pose, refined = refine_pose(
            prior,
            owned,
            prior.dims,
            max_translation=params.refine_max_translation,
            max_rotation_deg=params.refine_max_rotation_deg,
            min_points=params.refine_min_points,
        )
        pose = BlockPose(pose.position, pose.rotation, source="plan_prior", dims=prior.dims)
        top_centroid = pose.position + pose.rotation[:, 2] * (pose.dims[2] / 2.0)
        hypothesis = _make_hypothesis(
            pose,
            top_centroid,
            len(owned),
            ground,
            params,
            sensor_context,
            observed_geometry_faces=2 if refined else 1,
            dims=pose.dims,
        )
        results.append(
            PlanPriorResult(
                expected=prior,
                pose=pose,
                score=hypothesis.evidence.score,
                features=_evidence_features(hypothesis.evidence),
                support_points=len(owned),
                refined=refined,
            )
        )
    return results
