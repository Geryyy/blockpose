from __future__ import annotations

import numpy as np
from blockpose.annotation_snap import SnapParams, snap_block_to_cloud
from blockpose.annotation_tool import (
    AnnotationController,
    AnnotationDocument,
    EditableTarget,
)


def _cuboid_surface(
    center: np.ndarray,
    dims: tuple[float, float, float],
    yaw_deg: float,
    *,
    seed: int = 4,
    faces: tuple[tuple[int, int, int], ...] = ((2, 1, 650), (0, -1, 300), (1, -1, 300)),
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rotation = np.array(
        [
            [np.cos(np.deg2rad(yaw_deg)), -np.sin(np.deg2rad(yaw_deg)), 0],
            [np.sin(np.deg2rad(yaw_deg)), np.cos(np.deg2rad(yaw_deg)), 0],
            [0, 0, 1],
        ]
    )
    half = np.asarray(dims) / 2
    local = []
    # Dense top plus both visible vertical faces resembles the annotation data.
    for axis, sign, count in faces:
        samples = rng.uniform(-half, half, size=(count, 3))
        samples[:, axis] = sign * half[axis]
        local.append(samples)
    points = np.vstack(local) @ rotation.T + center
    points += rng.normal(0.0, 0.006, points.shape)
    outliers = center + rng.uniform(-1.0, 1.0, size=(120, 3))
    return np.vstack([points, outliers]), rotation


def _angle_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    value = np.clip((np.trace(first.T @ second) - 1) / 2, -1.0, 1.0)
    angle = np.rad2deg(np.arccos(value))
    return min(float(angle), abs(180.0 - float(angle)))


def test_snap_laid_block_with_noise_and_outliers() -> None:
    center = np.array([2.1, -0.7, 0.8])
    points, rotation = _cuboid_surface(center, (0.9, 0.6, 0.6), 27.0)
    result = snap_block_to_cloud(points, points[20])
    assert result.success, result.reason
    assert result.position is not None and np.linalg.norm(result.position - center) < 0.045
    assert result.rotation is not None and _angle_error_degrees(result.rotation, rotation) < 5.0
    assert result.dimensions == (0.9, 0.6, 0.6)


def test_snap_upright_block_and_is_permutation_deterministic() -> None:
    center = np.array([1.7, 0.5, 1.05])
    points, _ = _cuboid_surface(center, (0.6, 0.6, 0.9), 12.0)
    first = snap_block_to_cloud(points, points[10])
    second = snap_block_to_cloud(points[::-1], points[10])
    assert first.success and second.success
    assert first.dimensions == second.dimensions == (0.9, 0.6, 0.6)
    assert abs(first.rotation[2, 0]) > 0.99
    assert np.array_equal(first.position, second.position)
    assert np.array_equal(first.rotation, second.rotation)


def test_snap_failure_is_structured() -> None:
    result = snap_block_to_cloud(np.zeros((5, 3)), np.zeros(3))
    assert not result.success
    assert result.position is None and result.rotation is None
    assert "need" in result.reason


def test_controller_snap_is_one_undo_and_keeps_pending() -> None:
    original = EditableTarget("draft", np.zeros(3), np.eye(3), review_status="approved")
    controller = AnnotationController(AnnotationDocument("scene", [original]))
    position = np.array([1.0, 2.0, 3.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    controller.apply_snapped_pose(position, rotation, (0.6, 0.6, 0.9))
    assert controller.target is not None
    assert np.array_equal(controller.target.position, position)
    assert controller.target.dimensions == (0.9, 0.6, 0.6)
    assert abs(controller.target.rotation[2, 0]) > 0.99
    assert controller.target.review_status == "pending"
    assert controller.undo()
    assert controller.target is not None
    assert np.array_equal(controller.target.position, np.zeros(3))
    assert controller.target.dimensions == (0.9, 0.6, 0.6)
    assert controller.target.review_status == "approved"


def test_snap_rejects_sparse_plane_even_with_relaxed_score() -> None:
    points = np.column_stack([np.linspace(-0.1, 0.1, 20), np.zeros(20), np.ones(20)])
    result = snap_block_to_cloud(points, points[0], SnapParams(min_score=0.0))
    assert not result.success


def test_partial_top_and_front_fit_from_top_or_side_click() -> None:
    center = np.array([1.2, -0.3, 0.7])
    points, _ = _cuboid_surface(
        center,
        (0.9, 0.6, 0.6),
        33.0,
        faces=((2, 1, 420), (0, -1, 240)),
    )
    for clicked in (points[10], points[450]):
        result = snap_block_to_cloud(points, clicked)
        assert result.success, result.reason
        assert result.position is not None
        assert np.linalg.norm(result.position - center) < 0.04


def test_top_without_vertical_face_fails_safely() -> None:
    points, _ = _cuboid_surface(
        np.array([1.0, 0.0, 0.6]),
        (0.9, 0.6, 0.6),
        15.0,
        faces=((2, 1, 500),),
    )
    result = snap_block_to_cloud(points, points[0])
    assert not result.success
    assert "vertical" in result.reason or "top/side" in result.reason


def test_nearby_blocks_never_snap_to_displaced_compromise() -> None:
    first_center = np.array([0.0, 0.0, 0.6])
    for gap in (0.0, 0.15, 0.30):
        first, _ = _cuboid_surface(first_center, (0.9, 0.6, 0.6), 0.0, seed=11)
        second, _ = _cuboid_surface(
            np.array([0.9 + gap, 0.0, 0.6]),
            (0.9, 0.6, 0.6),
            0.0,
            seed=12,
        )
        result = snap_block_to_cloud(np.vstack([first, second]), first[30])
        if result.success:
            assert result.position is not None
            assert np.linalg.norm(result.position - first_center) <= 0.04
        else:
            assert "ambiguous" in result.reason or "supports" in result.reason


def test_ground_and_clutter_do_not_substitute_for_vertical_block_face() -> None:
    rng = np.random.default_rng(18)
    top, _ = _cuboid_surface(
        np.array([0.0, 0.0, 0.6]),
        (0.9, 0.6, 0.6),
        0.0,
        faces=((2, 1, 350),),
    )
    ground = np.column_stack(
        [rng.uniform(-1, 1, (500, 2)), rng.normal(0.0, 0.005, 500)]
    )
    clutter = rng.uniform([-0.8, -0.8, 0.05], [0.8, 0.8, 1.1], (80, 3))
    result = snap_block_to_cloud(np.vstack([top, ground, clutter]), top[5])
    assert not result.success


def test_broad_unrelated_wall_cannot_support_canonical_top() -> None:
    rng = np.random.default_rng(22)
    top, _ = _cuboid_surface(
        np.array([0.0, 0.0, 0.6]),
        (0.9, 0.6, 0.6),
        0.0,
        faces=((2, 1, 420),),
    )
    wall = np.column_stack(
        [
            rng.normal(-0.45, 0.005, 700),
            rng.uniform(-1.0, 1.0, 700),
            rng.uniform(0.05, 1.0, 700),
        ]
    )
    result = snap_block_to_cloud(np.vstack([top, wall]), top[12])
    assert not result.success


def test_invalid_snap_parameters_fail_without_computation() -> None:
    points = np.ones((100, 3))
    invalid = (
        SnapParams(crop_radius_m=0.0),
        SnapParams(voxel_m=np.nan),
        SnapParams(plane_tolerance_m=-0.1),
        SnapParams(min_score=1.1),
        SnapParams(ambiguity_score_ratio=0.0),
    )
    for params in invalid:
        result = snap_block_to_cloud(points, np.ones(3), params)
        assert not result.success
        assert "must" in result.reason
