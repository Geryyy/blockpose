from __future__ import annotations

import time

import numpy as np
from blockpose.detection import SensorContext
from blockpose.engine import HybridBlockDetector
from blockpose.evaluate import rotation_error_deg
from blockpose.pose import BlockPose
from blockpose.synth import BlockSpec, SynthParams, generate_scene
from blockpose.verify_placed import (
    PlacedStatus,
    VerifyPlacedParams,
    _rotation_deviation_deg,
    expected_visibility,
    verify_placed_blocks,
)


def _context(sample) -> SensorContext:
    return SensorContext(
        sample.sensor_origin, sample.ray_directions, sample.ranges, sample.miss_directions
    )


def _face_cloud(shift_x: float = 0.0, *, single_face: bool = False) -> np.ndarray:
    across = np.linspace(-0.27, 0.27, 15)
    high = np.linspace(0.03, 0.57, 15)
    first = np.array([[shift_x - 0.45, y, z] for y in across for z in high], dtype=np.float64)
    if single_face:
        return first
    top = np.array(
        [[shift_x + x, y, 0.6] for x in np.linspace(-0.42, 0.42, 15) for y in across],
        dtype=np.float64,
    )
    return np.vstack((first, top))


def _three_face_cloud(dx: float, dy: float, yaw_deg: float = 0.0) -> np.ndarray:
    grid = np.linspace(-0.27, 0.27, 15)
    long_grid = np.linspace(-0.42, 0.42, 15)
    local = np.vstack(
        (
            [[-0.45, y, z] for y in grid for z in np.linspace(-0.27, 0.27, 15)],
            [[x, -0.3, z] for x in long_grid for z in np.linspace(-0.27, 0.27, 15)],
            [[x, y, 0.3] for x in long_grid for y in grid],
        )
    ).astype(np.float64)
    angle = np.deg2rad(yaw_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    return local @ rotation.T + np.array([dx, dy, 0.3])


def test_touching_blocks_keep_identity_and_refine_inside_bounds() -> None:
    sample = generate_scene(
        [BlockSpec(0.04, -0.008), BlockSpec(0.04, 0.628)],
        SynthParams(ang_res_deg=0.4, range_noise_std=0.005),
        np.random.default_rng(4),
    )
    priors = [
        BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3)),
        BlockPose(np.array([0.0, 0.62, 0.3]), np.eye(3)),
    ]
    results = verify_placed_blocks(sample.points, priors, sensor_context=_context(sample))
    assert [result.status for result in results] == [PlacedStatus.DEVIATED] * 2
    assert all(result.pose.position[1] < 0.31 for result in results[:1])
    assert all(result.pose.position[1] > 0.31 for result in results[1:])
    assert (
        max(
            np.linalg.norm(result.pose.position - target.position)
            for result, target in zip(results, sample.ground_truth, strict=True)
        )
        <= 0.02
    )
    assert (
        max(
            rotation_error_deg(result.pose, target)
            for result, target in zip(results, sample.ground_truth, strict=True)
        )
        <= 1.0
    )


def test_raw_xyz_silence_is_not_missing_or_a_ground_measurement() -> None:
    sample = generate_scene(
        [],
        SynthParams(look_at=(0.0, 0.0, 0.3), ang_res_deg=0.4),
        np.random.default_rng(1),
    )
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    result = verify_placed_blocks(sample.points, [prior])[0]
    assert result.status == PlacedStatus.NOT_OBSERVABLE
    assert result.pose is prior


def test_actual_through_volume_rays_can_call_missing() -> None:
    sample = generate_scene(
        [],
        SynthParams(look_at=(0.0, 0.0, 0.3), ang_res_deg=0.4),
        np.random.default_rng(2),
    )
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    result = verify_placed_blocks(sample.points, [prior], sensor_context=_context(sample))[0]
    assert result.status == PlacedStatus.MISSING
    assert result.free_space_rays >= 3


def test_hidden_plan_neighbor_is_not_observable() -> None:
    front = BlockSpec(-0.9, 0.0)
    hidden = BlockSpec(0.0, 0.0)
    sample = generate_scene(
        [front, hidden],
        SynthParams(
            sensor_pos=(-5.0, 0.0, 0.3),
            look_at=(-0.4, 0.0, 0.3),
            ang_res_deg=0.3,
            ground=False,
        ),
        np.random.default_rng(3),
    )
    results = verify_placed_blocks(
        sample.points, sample.ground_truth, sensor_context=_context(sample)
    )
    assert results[1].status == PlacedStatus.NOT_OBSERVABLE
    assert results[1].occluded_faces


def test_point_permutation_is_stable_and_five_blocks_are_fast() -> None:
    specs = [BlockSpec(0.02, (index - 2) * 0.62) for index in range(5)]
    sample = generate_scene(
        specs,
        SynthParams(ang_res_deg=0.4, range_noise_std=0.005),
        np.random.default_rng(9),
    )
    priors = [BlockPose(np.array([0.0, (index - 2) * 0.62, 0.3]), np.eye(3)) for index in range(5)]
    started = time.perf_counter()
    first = verify_placed_blocks(sample.points, priors, sensor_context=_context(sample))
    runtime_ms = (time.perf_counter() - started) * 1000
    permutation = np.random.default_rng(10).permutation(len(sample.points))
    second = verify_placed_blocks(
        sample.points[permutation], priors, sensor_context=_context(sample)
    )
    assert runtime_ms <= 250.0
    assert [result.status for result in first] == [result.status for result in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.pose.position, right.pose.position, atol=1e-9)
        np.testing.assert_allclose(left.pose.rotation, right.pose.rotation, atol=1e-9)


def test_fixed_seed_translation_status_regression() -> None:
    correct = 0
    total = 0
    translation_errors = []
    rotation_errors = []
    runtimes = []
    for seed in range(12):
        count = 1 + seed % 5
        specs = []
        priors = []
        expected_statuses = []
        for index in range(count):
            planned_y = (index - (count - 1) / 2.0) * 0.62
            x_deviation = 0.045 if seed % 3 == 0 else 0.0
            y_deviation = 0.008 * (index % 3 - 1)
            yaw_deviation = 0.0
            specs.append(BlockSpec(x_deviation, planned_y + y_deviation, yaw_deg=yaw_deviation))
            priors.append(BlockPose(np.array([0.0, planned_y, 0.3]), np.eye(3)))
            expected_statuses.append(
                PlacedStatus.DEVIATED if x_deviation > 0.03 else PlacedStatus.CONFIRMED
            )
        sample = generate_scene(
            specs,
            SynthParams(ang_res_deg=0.4, range_noise_std=0.005),
            np.random.default_rng(seed),
        )
        started = time.perf_counter()
        results = verify_placed_blocks(sample.points, priors, sensor_context=_context(sample))
        runtimes.append((time.perf_counter() - started) * 1000.0)
        for result, target, expected_status in zip(
            results, sample.ground_truth, expected_statuses, strict=True
        ):
            total += 1
            correct += result.status == expected_status
            translation_errors.append(np.linalg.norm(result.pose.position - target.position))
            rotation_errors.append(rotation_error_deg(result.pose, target))
    assert correct / total >= 0.95
    # Disabled v1 prototype regression ceiling; the product gate remains 20 mm.
    assert np.percentile(translation_errors, 95) <= 0.03
    assert np.percentile(rotation_errors, 95) <= 1.0
    assert np.percentile(runtimes, 95) <= 250.0


def test_single_face_cannot_be_reported_as_measurement() -> None:
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    result = verify_placed_blocks(_face_cloud(single_face=True), [prior])[0]
    assert result.status == PlacedStatus.NOT_OBSERVABLE
    assert result.diagnostics["supported_face_count"] == 1
    assert result.covariance is None


def test_beyond_window_geometry_is_not_clamped_to_deviated() -> None:
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    result = verify_placed_blocks(_face_cloud(0.15), [prior])[0]
    assert result.status == PlacedStatus.NOT_OBSERVABLE
    assert result.pose is prior


def test_euclidean_translation_window_rejects_diagonal_corrections() -> None:
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    for shift in (0.08, 0.09):
        result = verify_placed_blocks(
            _three_face_cloud(shift, shift),
            [prior],
            params=VerifyPlacedParams(assign_band_m=0.15),
        )[0]
        assert result.status == PlacedStatus.NOT_OBSERVABLE
        assert result.pose is prior
        assert result.diagnostics["translation_correction_norm_m"] > 0.10


def test_outside_rotation_search_never_falls_through_to_missing() -> None:
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    origin = np.array([-5.0, 0.0, 4.0])
    aims = _three_face_cloud(0.0, 0.0)[::20]
    vectors = aims - origin
    ranges = np.linalg.norm(vectors, axis=1) + 2.0
    context = SensorContext(origin, vectors / np.linalg.norm(vectors, axis=1)[:, None], ranges)
    for yaw in (12.0, 15.0):
        result = verify_placed_blocks(
            _three_face_cloud(0.0, 0.0, yaw),
            [prior],
            sensor_context=context,
            params=VerifyPlacedParams(assign_band_m=0.15),
        )[0]
        assert result.status == PlacedStatus.NOT_OBSERVABLE
        assert result.status != PlacedStatus.MISSING
        assert result.diagnostics["fit_reason"] in {
            "search_bound",
            "rotation_search_exceeded",
        }


def test_displaced_realistic_block_with_support_is_not_missing() -> None:
    sample = generate_scene(
        [BlockSpec(0.09, 0.09)],
        SynthParams(
            sensor_pos=(-5.0, -5.0, 4.0),
            look_at=(0.0, 0.0, 0.3),
            ang_res_deg=0.3,
            range_noise_std=0.005,
        ),
        np.random.default_rng(17),
    )
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    result = verify_placed_blocks(sample.points, [prior], sensor_context=_context(sample))[0]
    assert result.support_points >= VerifyPlacedParams().min_support_points
    assert result.free_space_rays > 0
    assert result.status == PlacedStatus.NOT_OBSERVABLE
    assert result.status != PlacedStatus.MISSING


def test_rotation_deviation_respects_cuboid_symmetry() -> None:
    prior = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    equivalent = BlockPose(prior.position.copy(), np.diag([-1.0, -1.0, 1.0]), dims=prior.dims)
    assert _rotation_deviation_deg(prior, equivalent) == 0.0


def test_face_grid_keeps_partially_exposed_face_visible() -> None:
    target = BlockPose(np.array([0.0, 0.0, 0.3]), np.eye(3))
    center_blocker = BlockPose(np.array([-1.0, 0.25, 0.3]), np.eye(3))
    visible, occluded = expected_visibility([target, center_blocker], np.array([-5.0, 0.0, 0.3]))[0]
    assert "x-" in visible
    assert "x-" not in occluded


def test_direct_api_rejects_noncanonical_dimensions() -> None:
    noncanonical = BlockPose(np.array([0.0, 0.0, 0.45]), np.eye(3), dims=(0.6, 0.6, 0.9))
    with np.testing.assert_raises_regex(ValueError, "canonical dimensions"):
        verify_placed_blocks(np.empty((0, 3)), [noncanonical])


def test_known_touching_row_yaw_limitation_keeps_v1_disabled() -> None:
    specs = [
        BlockSpec(0.0, (index - 2) * 0.62 + 0.008 * (index % 3 - 1), yaw_deg=3.0)
        for index in range(5)
    ]
    sample = generate_scene(
        specs,
        SynthParams(ang_res_deg=0.4, range_noise_std=0.005),
        np.random.default_rng(4),
    )
    priors = [BlockPose(np.array([0.0, (index - 2) * 0.62, 0.3]), np.eye(3)) for index in range(5)]
    results = verify_placed_blocks(sample.points, priors, sensor_context=_context(sample))
    # The explicit-face v1 fit currently under-corrects this 3-degree yaw and
    # therefore cannot satisfy the 1-degree acceptance gate.
    assert (
        max(
            rotation_error_deg(result.pose, target)
            for result, target in zip(results, sample.ground_truth, strict=True)
        )
        > 1.0
    )
    assert HybridBlockDetector().verify_placed_v1 is False
