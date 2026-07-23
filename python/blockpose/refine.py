"""Model-based pose refinement: fit the exact block cuboid to observed returns.

The plane-fit detector produces a closed-form pose from a top plane plus at most
one side plane. That uses roughly a third of the visible surface and makes yaw
depend on a single RANSAC normal, which is the pipeline's most fragile estimate.

This module refines that guess by minimizing the analytic signed distance from
every assigned return to the cuboid surface, over the full 6-DOF pose. It is a
direct model fit rather than an ICP correspondence loop: a box has a closed-form
SDF, so no nearest-point search is needed and each iteration is a dense
least-squares solve.

Measured on isolated blocks (12 seeds per noise level), refining the detector's
own output:

    range noise   rotation p95        translation p95
    0.005 m       0.553 -> 0.161 deg  0.019 -> 0.002 m
    0.010 m       0.478 -> 0.341 deg  0.014 -> 0.002 m
    0.020 m       6.512 -> 0.761 deg  0.028 -> 0.005 m
    0.030 m       1.827 -> 1.202 deg  0.038 -> 0.006 m
    0.050 m       3.428 -> 2.296 deg  0.059 -> 0.019 m

The 0.02 m row is the important one: it removes the noise cliff where yaw from a
single plane normal degrades sharply.

Multi-block scenes need point assignment first. A stack's cluster contains both
blocks, and fitting one cuboid to all of it drags the pose toward the midpoint,
so `assign_points` splits returns between competing hypotheses before fitting.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from blockpose.pose import BlockPose


def box_sdf(points_local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance from origin-centred axis-aligned box. Negative inside."""
    delta = np.abs(points_local) - half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(delta.max(axis=1), 0.0)
    return outside + inside


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )


def _exp_so3(vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map; identity for a vanishing rotation vector."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = _skew(vector / angle)
    return np.eye(3) + np.sin(angle) * axis + (1.0 - np.cos(angle)) * axis @ axis


def surface_distance(pose: BlockPose, points: np.ndarray, dims) -> np.ndarray:
    """Distance from each world point to the posed cuboid's surface."""
    half = np.asarray(dims, dtype=np.float64) / 2.0
    return box_sdf((np.asarray(points, dtype=np.float64) - pose.position) @ pose.rotation, half)


def _dims_for(pose: BlockPose, dims):
    """Explicit `dims` wins; otherwise use the pose's own extents.

    A scene can mix orientations -- upright staging blocks beside laid wall
    courses -- so a single dims argument cannot describe every hypothesis.
    """
    return tuple(dims) if dims is not None else tuple(pose.dims)


def assign_points(
    poses: list[BlockPose],
    points: np.ndarray,
    dims,
    band: float,
) -> list[np.ndarray]:
    """Split returns between competing hypotheses by nearest surface.

    A point is assigned to the hypothesis whose surface it lies closest to, and
    only if it lies within `band` of that surface. Points near no hypothesis
    (ground remnants, clutter, mixed pixels) are dropped rather than forced onto
    the nearest block, so they cannot drag a fit.
    """
    points = np.asarray(points, dtype=np.float64)
    if not poses or len(points) == 0:
        return [np.zeros((0, 3)) for _ in poses]
    distances = np.stack(
        [np.abs(surface_distance(pose, points, _dims_for(pose, dims))) for pose in poses], axis=1
    )
    owner = np.argmin(distances, axis=1)
    within = distances[np.arange(len(points)), owner] <= band
    return [points[within & (owner == index)] for index in range(len(poses))]


def refine_pose(
    pose: BlockPose,
    points: np.ndarray,
    dims,
    *,
    huber_scale: float = 0.05,
    max_translation: float = 0.15,
    max_rotation_deg: float = 20.0,
    min_points: int = 20,
) -> tuple[BlockPose, bool]:
    """Fit the cuboid to `points`. Returns ``(pose, refined)``.

    The initial pose is kept whenever refinement is not trustworthy: too few
    points, a failed solve, or a correction larger than the guards allow. A
    large correction means the assignment was wrong (usually a hypothesis
    straddling two blocks), and in that case the closed-form guess is safer than
    a confidently wrong fit.
    """
    refined, success, _ = refine_pose_with_diagnostics(
        pose,
        points,
        dims,
        huber_scale=huber_scale,
        max_translation=max_translation,
        max_rotation_deg=max_rotation_deg,
        min_points=min_points,
    )
    return refined, success


def refine_pose_with_diagnostics(
    pose: BlockPose,
    points: np.ndarray,
    dims,
    *,
    huber_scale: float = 0.05,
    max_translation: float = 0.15,
    max_rotation_deg: float = 20.0,
    min_points: int = 20,
) -> tuple[BlockPose, bool, dict[str, object]]:
    """Run :func:`refine_pose` and expose solver facts for parity fixtures.

    This deliberately shares the production solver path.  The returned facts
    describe the unconstrained SciPy result *before* the safety guard decides
    whether it is safe to publish, so an optimizer mismatch cannot be confused
    with a guard rejection.
    """
    points = np.asarray(points, dtype=np.float64)
    diagnostic: dict[str, object] = {
        "attempted": False,
        "point_count": int(len(points)),
        "success": False,
        "status": None,
        "nfev": 0,
        "njev": 0,
        "cost": None,
        "optimality": None,
        "solution_x": None,
        "translation_norm": None,
        "rotation_deg": None,
        "guard_accepted": False,
    }
    if len(points) < min_points:
        return pose, False, diagnostic
    diagnostic["attempted"] = True
    half = np.asarray(dims, dtype=np.float64) / 2.0
    rotation0, position0 = pose.rotation, pose.position

    def residuals(x: np.ndarray) -> np.ndarray:
        rotation = rotation0 @ _exp_so3(x[:3])
        return box_sdf((points - (position0 + x[3:])) @ rotation, half)

    try:
        solution = least_squares(
            residuals,
            np.zeros(6),
            method="trf",
            loss="huber",
            f_scale=huber_scale,
            max_nfev=200,
        )
    except (ValueError, np.linalg.LinAlgError):
        return pose, False, diagnostic
    diagnostic.update(
        {
            "success": bool(solution.success),
            "status": int(solution.status),
            "nfev": int(solution.nfev),
            "njev": int(solution.njev) if solution.njev is not None else None,
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
            "solution_x": solution.x.tolist(),
        }
    )
    if not solution.success:
        return pose, False, diagnostic

    delta_rotation = _exp_so3(solution.x[:3])
    angle = np.degrees(
        np.arccos(np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    translation_norm = float(np.linalg.norm(solution.x[3:]))
    diagnostic["translation_norm"] = translation_norm
    diagnostic["rotation_deg"] = float(angle)
    if translation_norm > max_translation or angle > max_rotation_deg:
        return pose, False, diagnostic

    diagnostic["guard_accepted"] = True
    return (
        BlockPose(
            position=position0 + solution.x[3:],
            rotation=rotation0 @ delta_rotation,
            confidence=pose.confidence,
            source=f"{pose.source}+sdf",
            dims=pose.dims,
        ),
        True,
        diagnostic,
    )


def refine_poses(
    poses: list[BlockPose],
    points: np.ndarray,
    dims,
    *,
    band: float = 0.10,
    iterations: int = 2,
    **refine_kwargs,
) -> list[BlockPose]:
    """Alternate assignment and fitting so neighbouring blocks stop contaminating.

    Two passes is enough in practice: the first cleans up assignment using the
    closed-form poses, the second fits against the corrected split.
    """
    if not poses:
        return []
    current = list(poses)
    for _ in range(max(iterations, 1)):
        assigned = assign_points(current, points, dims, band)
        current = [
            refine_pose(pose, owned, _dims_for(pose, dims), **refine_kwargs)[0]
            for pose, owned in zip(current, assigned, strict=True)
        ]
    return current
