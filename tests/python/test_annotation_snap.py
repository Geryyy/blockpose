from __future__ import annotations

import numpy as np
from blockpose.annotation_snap import (
    FACE_CLASSES,
    FACE_NORMALS,
    SnapParams,
    snap_block_face_to_plane,
    snap_block_to_cloud,
)
from blockpose.annotation_tool import (
    AnnotationController,
    AnnotationDocument,
    EditableTarget,
)
from blockpose.pose import BLOCK_DIMS
from scipy.spatial.transform import Rotation


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
    ground = np.column_stack([rng.uniform(-1, 1, (500, 2)), rng.normal(0.0, 0.005, 500)])
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


def test_snap_discards_nonfinite_depth_outliers() -> None:
    center = np.array([2.1, -0.7, 0.8])
    points, _ = _cuboid_surface(center, (0.9, 0.6, 0.6), 27.0)
    contaminated = np.vstack([points, np.array([[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]])])
    result = snap_block_to_cloud(contaminated, points[20])
    assert result.success, result.reason
    assert result.position is not None
    assert np.linalg.norm(result.position - center) < 0.045


def test_selected_face_snaps_to_local_support_plane() -> None:
    rng = np.random.default_rng(3)
    plane = np.column_stack(
        [rng.uniform(-0.4, 0.4, 180), rng.uniform(-0.3, 0.3, 180), np.full(180, 0.0)]
    )
    result = snap_block_face_to_plane(
        plane,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.8]),
        np.eye(3),
        (0.9, 0.6, 0.6),
        "Long face",
    )
    assert result.success, result.reason
    assert result.position is not None
    # The block starts above the plane, so the resolved long face is -Z and
    # the block lands ON the plane rather than under it.
    np.testing.assert_allclose(result.position, [0.0, 0.0, 0.3], atol=1e-6)
    np.testing.assert_array_equal(result.rotation, np.eye(3))


def test_selected_face_snap_rejects_sparse_support() -> None:
    result = snap_block_face_to_plane(
        np.zeros((4, 3)), np.zeros(3), np.zeros(3), np.eye(3), (0.9, 0.6, 0.6), "Long face"
    )
    assert not result.success
    assert "insufficient" in result.reason


def test_world_z_yaw_preserves_roll_and_pitch_axes() -> None:
    base = Rotation.from_euler("xyz", [17.0, -23.0, 11.0], degrees=True).as_matrix()
    expected = Rotation.from_euler("z", 90.0, degrees=True).as_matrix() @ base
    # This is the transform used by the GUI's Rotate 90° yaw action.
    actual = Rotation.from_euler(
        "xyz", Rotation.from_matrix(expected).as_euler("xyz", degrees=True), degrees=True
    ).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-10)


# --- Face classes: "Auto"/"End"/"Long face" resolve to a concrete signed axis ---


def test_face_classes_partition_the_six_signed_axes() -> None:
    assert set(FACE_CLASSES["End"]) == {"+X", "-X"}
    assert set(FACE_CLASSES["Long face"]) == {"+Y", "-Y", "+Z", "-Z"}
    assert set(FACE_CLASSES["Auto"]) == set(FACE_NORMALS)


def test_unknown_face_string_is_rejected_without_computation() -> None:
    result = snap_block_face_to_plane(
        np.zeros((30, 3)), np.zeros(3), np.zeros(3), np.eye(3), BLOCK_DIMS, "Middle"
    )
    assert not result.success
    assert "unknown face class" in result.reason


def _ground_plane(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-0.4, 0.4, 220), rng.uniform(-0.4, 0.4, 220), rng.normal(0.0, 0.002, 220)]
    )


def test_auto_resolves_to_the_face_actually_touching_the_ground() -> None:
    """A block resting above a ground plane should snap by its bottom (-Z) face.

    Both +Z and -Z are parallel to a horizontal plane, so this is exactly the
    ambiguity `snap_block_face_to_plane` must resolve using the block's
    current pose rather than an arbitrary PCA sign.
    """
    ground = _ground_plane()
    position = np.array([0.0, 0.0, 0.8])
    click = np.array([0.0, 0.0, 0.0])
    rotation = np.eye(3)
    auto = snap_block_face_to_plane(ground, click, position, rotation, BLOCK_DIMS, "Auto")
    long_face = snap_block_face_to_plane(ground, click, position, rotation, BLOCK_DIMS, "Long face")
    assert auto.success and long_face.success
    np.testing.assert_allclose(auto.position, long_face.position, atol=1e-9)
    # Resolution must land the block ON the plane (centre half a height above
    # it), i.e. pick the near/-Z face, not translate it down onto the far one.
    # The block travels along the FITTED normal, which is only near-vertical
    # for a randomly sampled plane, so allow it to drift slightly off-axis.
    np.testing.assert_allclose(auto.position, [0.0, 0.0, 0.3], atol=2e-3)


def test_auto_and_all_four_equivalent_long_face_axes_agree() -> None:
    """The block's square cross-section gives 4-fold symmetry about local +X.

    Four different rotation matrices can describe the exact same physical
    placement, each labeling a different local axis as "the face touching
    the ground" (see `symmetry_equivalents` in pose.py). Auto must resolve
    each representation to whichever axis is actually correct for it, and
    every representation must agree on the resulting world-space position.
    """
    ground = _ground_plane(seed=6)
    position = np.array([0.0, 0.0, 0.8])
    click = np.array([0.0, 0.0, 0.0])
    reference = snap_block_face_to_plane(ground, click, position, np.eye(3), BLOCK_DIMS, "Auto")
    assert reference.success
    for k in range(4):
        rotation = Rotation.from_euler("x", 90.0 * k, degrees=True).as_matrix()
        # Whichever long-face axis this representative rotation currently
        # assigns to world "down" is the one an explicit signed-axis request
        # should independently discover.
        auto = snap_block_face_to_plane(ground, click, position, rotation, BLOCK_DIMS, "Auto")
        long_face = snap_block_face_to_plane(
            ground, click, position, rotation, BLOCK_DIMS, "Long face"
        )
        assert auto.success, auto.reason
        assert long_face.success, long_face.reason
        np.testing.assert_allclose(auto.position, reference.position, atol=1e-9)
        np.testing.assert_allclose(long_face.position, reference.position, atol=1e-9)


def test_end_and_long_face_classes_resolve_to_different_faces() -> None:
    """A vertical wall is parallel to the End pair but not the Long-face quad."""
    rng = np.random.default_rng(9)
    wall = np.column_stack(
        [rng.normal(0.0, 0.002, 220), rng.uniform(-0.3, 0.3, 220), rng.uniform(-0.3, 0.3, 220)]
    )
    position = np.array([0.6, 0.0, 0.0])
    click = np.array([0.0, 0.0, 0.0])
    rotation = np.eye(3)
    end = snap_block_face_to_plane(wall, click, position, rotation, BLOCK_DIMS, "End")
    long_face = snap_block_face_to_plane(wall, click, position, rotation, BLOCK_DIMS, "Long face")
    auto = snap_block_face_to_plane(wall, click, position, rotation, BLOCK_DIMS, "Auto")
    assert end.success, end.reason
    assert not long_face.success
    assert auto.success
    np.testing.assert_allclose(auto.position, end.position, atol=1e-9)
    # The resolved face is the NEAR one: the block ends up on the wall's near
    # side (x = +0.45), not pulled through it to the far side.
    assert end.position[0] > 0.0
