"""Tests for blockpose.detection pipeline on real snapshots."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from blockpose.detection import (
    CuboidHypothesis,
    DetectionDiagnostics,
    DetectionParams,
    DetectionResult,
    GroundPlane,
    HypothesisEvidence,
    Observability,
    PlanePatch,
    SensorContext,
    _make_hypothesis,
    _proposal_size_first_key,
    _remove_ground,
    _select_conflict_alternatives,
    _select_hypotheses,
    _split_connected_region,
    _split_plane_patch_in_plane,
    _synthesize_pose,
    _validate_sensor_context,
    detect_blocks,
    detect_blocks_detailed,
    detect_scene_sample,
    free_space_violations_from_misses,
    visibility_evidence,
)
from blockpose.evaluate import evaluate, rotation_error_deg
from blockpose.io import load_snapshot
from blockpose.pose import BlockPose
from blockpose.snapshot_exclusions import load_snapshot_exclusions
from blockpose.synth import BlockSpec, ClutterParams, SynthParams, generate_scene
from conftest import snapshot_with_ground_truth

REPO_ROOT = Path(__file__).parent.parent.parent
SNAPSHOTS_DIR = REPO_ROOT / "data" / "registration_snapshots"
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"


def test_in_plane_patch_split_is_deterministic_and_retains_unsplit_alternative() -> None:
    grid = np.array(
        [[x, y, 0.6] for x in np.linspace(0.0, 0.4, 8) for y in np.linspace(0.0, 0.4, 8)]
    )
    points = np.vstack((grid, grid + np.array([0.8, 0.0, 0.0])))
    patch = PlanePatch(np.array([0.0, 0.0, 1.0]), points.mean(axis=0), points, 0.0)
    params = DetectionParams(min_inliers=20)
    first = _split_plane_patch_in_plane(patch, params)
    second = _split_plane_patch_in_plane(patch, params)
    assert [len(item.points) for item in first] == [128, 64, 64]
    assert [item.centroid.tolist() for item in first] == [item.centroid.tolist() for item in second]


def test_conflict_alternatives_are_bounded() -> None:
    evidence = HypothesisEvidence(100, 0.0, 0.8)
    hypotheses = [
        CuboidHypothesis(BlockPose(np.array([offset, 0.0, 0.3]), np.eye(3)), evidence)
        for offset in (0.0, 0.05, 0.10)
    ]
    selected = _select_conflict_alternatives(hypotheses, 2)
    assert len(selected) == 2
    with pytest.raises(ValueError, match="at least one"):
        _select_conflict_alternatives(hypotheses, 0)


def test_proposal_order_is_deterministic_size_first_then_compatibility() -> None:
    params = DetectionParams()
    compatible = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.6, 0.0], [0.9, 0.6, 0.6]])
    incompatible = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.2]])
    larger_but_incompatible = np.vstack((incompatible, [[0.1, 0.1, 0.1]]))
    proposals = [compatible, larger_but_incompatible, incompatible]
    first = sorted(proposals, key=lambda points: _proposal_size_first_key(points, params))
    second = sorted(
        reversed(proposals), key=lambda points: _proposal_size_first_key(points, params)
    )
    assert first[0] is larger_but_incompatible
    assert first[1] is compatible
    assert first[2] is incompatible
    assert second[0] is larger_but_incompatible
    assert second[1] is compatible
    assert second[2] is incompatible


_EXCLUDED_SNAPSHOTS = load_snapshot_exclusions(SNAPSHOTS_DIR)
SNAPSHOT_DIRS = sorted(
    path
    for path in SNAPSHOTS_DIR.iterdir()
    if path.is_dir() and path.name not in _EXCLUDED_SNAPSHOTS
)


@pytest.fixture(scope="module")
def snapshot1_data():
    """Load a snapshot with confirmed ground truth once for all tests in this
    module (see conftest.snapshot_with_ground_truth: some real snapshots have
    zero registered blocks, so "the first one" isn't reliably meaningful)."""
    return load_snapshot(snapshot_with_ground_truth(SNAPSHOTS_DIR), TRANSFORM_CONFIG)


@pytest.fixture(scope="module")
def snapshot1_detections(snapshot1_data):
    """Run detection on first snapshot once."""
    pts_world, _ = snapshot1_data
    return detect_blocks(pts_world)


def test_detect_returns_nonempty(snapshot1_detections) -> None:
    assert len(snapshot1_detections) >= 1


def test_detect_positions_reasonable(snapshot1_detections) -> None:
    """All detected block centers should be in a plausible Z range in world frame."""
    for pose in snapshot1_detections:
        z = float(pose.position[2])
        assert 0.0 <= z <= 3.0, (
            f"Detected block Z={z:.3f} out of range — likely coordinate frame error"
        )


def test_detect_rotation_is_valid_SO3(snapshot1_detections) -> None:
    for pose in snapshot1_detections:
        R = pose.rotation
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)


@pytest.mark.xfail(strict=False, reason="prototype: block count may differ from GT")
def test_detect_count_close_to_gt(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    assert abs(len(snapshot1_detections) - len(gt)) <= 1


@pytest.mark.xfail(strict=False, reason="prototype: translation threshold pre-ICP")
def test_eval_translation_error_threshold(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    result = evaluate(snapshot1_detections, gt)
    if result.matched == 0:
        pytest.skip("No matched pairs to evaluate")
    assert result.mean_translation_m < 0.25


@pytest.mark.xfail(strict=False, reason="prototype: rotation threshold pre-ICP")
def test_eval_rotation_error_threshold(snapshot1_data, snapshot1_detections) -> None:
    _, gt = snapshot1_data
    result = evaluate(snapshot1_detections, gt)
    if result.matched == 0:
        pytest.skip("No matched pairs to evaluate")
    assert result.mean_rotation_deg < 30.0


def test_all_snapshots_run_without_error() -> None:
    """Smoke test: pipeline runs on all snapshots without raising."""
    for snap_dir in SNAPSHOT_DIRS:
        pts_world, _ = load_snapshot(snap_dir, TRANSFORM_CONFIG)
        detections = detect_blocks(pts_world)
        # Just verify types are correct
        for pose in detections:
            assert pose.position.shape == (3,)
            assert pose.rotation.shape == (3, 3)


def _make_flat_ground(n: int = 2000, z: float = 0.0, spread: float = 10.0) -> np.ndarray:
    """Dense flat ground plane at Z=z, spread over XY."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(-spread, spread, size=(n, 2))
    zs = np.full((n, 1), z)
    return np.hstack([xy, zs])


@pytest.mark.parametrize("pts", [np.empty((0, 3)), np.zeros((1, 3)), np.zeros((2, 3))])
def test_empty_and_sparse_clouds_return_no_detections(pts: np.ndarray) -> None:
    assert detect_blocks(pts) == []


@pytest.mark.parametrize("pts", [np.zeros((3,)), np.zeros((2, 2)), np.zeros((2, 3, 1))])
def test_malformed_cloud_has_clear_error(pts: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        detect_blocks(pts)


def test_nonfinite_cloud_has_clear_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        detect_blocks(np.array([[0.0, np.nan, 0.0]]))


def test_complex_cloud_is_rejected() -> None:
    with pytest.raises(ValueError, match="real numeric"):
        detect_blocks(np.ones((3, 3), dtype=np.complex128))


def test_ground_removal_returns_local_sloped_ground_frame() -> None:
    grid = np.linspace(-3.0, 3.0, 45)
    xx, yy = np.meshgrid(grid, grid)
    zz = 1.2 + 0.08 * xx - 0.04 * yy
    ground_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    object_points = ground_points[::8].copy()
    object_points[:, 2] += 0.6
    above, ground = _remove_ground(
        np.vstack([ground_points, object_points]), DetectionParams(ground_ransac_iters=200)
    )
    assert ground.source == "ransac"
    assert ground.normal[2] > 0.99
    assert np.median(ground.height(ground_points)) == pytest.approx(0.0, abs=0.01)
    assert len(above) >= len(object_points) * 0.9


def test_local_ground_preserves_points_five_cm_above_support() -> None:
    grid = np.linspace(-1.0, 1.0, 25)
    xx, yy = np.meshgrid(grid, grid)
    ground = np.column_stack([xx.ravel(), yy.ravel(), 0.04 * xx.ravel()])
    raised = ground[::20].copy()
    raised[:, 2] += 0.06
    above, model = _remove_ground(
        np.vstack([ground, raised]),
        DetectionParams(local_ground_clearance=0.05, ground_ransac_iters=200),
    )
    assert model.source == "ransac"
    assert len(above) >= len(raised)


def test_ground_removal_exact_three_point_cloud_is_safe() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0]])
    above, ground = _remove_ground(points, DetectionParams())
    assert above.shape == (0, 3)
    assert ground.normal.shape == (3,)
    assert detect_blocks(points) == []


def test_ground_removal_falls_back_when_coarse_voxels_collapse() -> None:
    # Distinct 4 cm detector voxels, but all points occupy fewer than three
    # voxels at the 10 cm ground-RANSAC resolution.
    points = np.array([[0.001, 0.001, 0.0], [0.045, 0.001, 0.0], [0.085, 0.001, 0.2]])
    above, ground = _remove_ground(points, DetectionParams())
    assert ground.source == "percentile"
    assert above.shape == (1, 3)
    assert detect_blocks(points) == []


def test_side_plane_equation_recovers_known_dimension_center() -> None:
    cluster = np.array(
        [
            [-0.3, -0.2, 0.3],
            [-0.3, 0.2, 0.7],
            [0.1, -0.2, 0.9],
            [0.2, 0.2, 0.9],
        ]
    )
    pose = _synthesize_pose(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.08, 0.1, 0.9]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.3, 0.0, 0.5]),
        (0.6, 0.6, 0.9),
        cluster,
    )
    assert pose.position[0] == pytest.approx(0.0, abs=1e-9)
    assert pose.position[2] == pytest.approx(0.45, abs=1e-9)


def test_continuous_elongated_cuboid_is_not_split_by_extent() -> None:
    rng = np.random.default_rng(91)
    # Dense continuous top and two sides of a 0.6 x 1.8 x 0.9 cuboid.
    top = np.column_stack(
        [rng.uniform(-0.3, 0.3, 1000), rng.uniform(-0.9, 0.9, 1000), np.full(1000, 0.9)]
    )
    side = np.column_stack(
        [np.full(800, -0.3), rng.uniform(-0.9, 0.9, 800), rng.uniform(0.15, 0.9, 800)]
    )
    cluster = np.vstack([top, side])
    regions = _split_connected_region(cluster, DetectionParams())
    assert len(regions) == 1


def test_evidence_selection_collapses_duplicates_but_keeps_neighbors() -> None:
    def hypothesis(position, score):
        pose = _make_pose_for_selection(position)
        return CuboidHypothesis(pose, HypothesisEvidence(100, 0.0, score))

    candidates = [
        hypothesis((0.0, 0.0, 0.45), 0.7),
        hypothesis((0.03, 0.0, 0.45), 0.9),
        hypothesis((0.6, 0.0, 0.45), 0.6),
        hypothesis((0.0, 0.0, 1.35), 0.5),
    ]
    selected = _select_hypotheses(candidates)
    assert [item.evidence.score for item in selected] == [0.9, 0.6, 0.5]


def _make_pose_for_selection(position):
    from blockpose.pose import BlockPose

    # Pinned upright: _boxes_overlap uses pose.dims, and the 0.6 m spacing
    # cases in this file assume the 0.6x0.6 square footprint, not the
    # ambient default.
    return BlockPose(np.asarray(position, dtype=np.float64), np.eye(3), dims=(0.6, 0.6, 0.9))


def test_detection_is_stable_under_point_permutation() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=31.0)],
        SynthParams(range_noise_std=0.0),
        np.random.default_rng(17),
    )
    baseline = detect_blocks(sample.points)
    shuffled = detect_blocks(np.random.default_rng(99).permutation(sample.points))
    assert len(baseline) == len(shuffled) == 1
    # Canonical ordering removes order-dependent voxel/DBSCAN behavior. Open3D
    # RANSAC has no per-call RNG API, so retain a tight geometric tolerance for
    # its remaining sampling variation rather than mutating its global RNG.
    np.testing.assert_allclose(baseline[0].position, shuffled[0].position, atol=0.02)
    assert rotation_error_deg(baseline[0], shuffled[0]) <= 2.0


def test_adversarial_offset_stack_has_no_cross_layer_pairing_bias() -> None:
    upright = (0.6, 0.6, 0.9)
    sample = generate_scene(
        [
            BlockSpec(0.0, 0.0, z=0.45, yaw_deg=30.0, dims=upright),
            BlockSpec(0.34641016, 0.2, z=1.35, yaw_deg=30.0, dims=upright),
        ],
        SynthParams(look_at=(0.15, 0.1, 0.9), dims=upright),
        np.random.default_rng(23),
    )
    detections = detect_blocks(sample.points, DetectionParams(block_dims=upright))
    result = evaluate(detections, sample.ground_truth)
    assert result.unmatched_det == 0
    if result.matched:
        assert max(result.translation_errors_m) <= 0.04
    permuted = detect_blocks(
        np.random.default_rng(101).permutation(sample.points), DetectionParams(block_dims=upright)
    )
    stability = evaluate(permuted, detections, max_match_dist=0.1)
    assert stability.matched == len(detections)
    assert max(stability.translation_errors_m, default=0.0) <= 0.03


def _cuboid_ray_context():
    """Ray context for a single block, raycast so the background is consistent.

    Built from the generator rather than hand-placed returns: a hand-built
    context containing *only* rays that terminate on the true block has no
    background, so a shifted hypothesis has nothing to contradict it and scores
    identically to the true one. Ground returns beside and in front of the block
    -- correctly shadowed where the block occludes them -- are what make a
    displaced cuboid measurably worse.
    """
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, dims=(0.6, 0.6, 0.9))],
        SynthParams(sensor_pos=(-5.0, 0.0, 3.0), range_noise_std=0.0, ang_res_deg=0.5),
        np.random.default_rng(7),
    )
    return SensorContext(
        sample.sensor_origin, sample.ray_directions, sample.ranges, sample.miss_directions
    )


def test_visibility_score_prefers_true_cuboid_and_wrong_height_loses() -> None:
    context = _cuboid_ray_context()
    ground = GroundPlane(np.array([0.0, 0.0, 1.0]), 0.0, "test")
    true_pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    shifted_pose = _make_pose_for_selection((0.0, 0.25, 0.45))
    wrong_height = _make_pose_for_selection((0.0, 0.0, 0.7))
    params = DetectionParams(block_dims=(0.6, 0.6, 0.9))
    true = _make_hypothesis(true_pose, np.array([0.0, 0.0, 0.9]), 100, ground, params, context)
    shifted = _make_hypothesis(
        shifted_pose, np.array([0.0, 0.25, 0.9]), 100, ground, params, context
    )
    wrong = _make_hypothesis(wrong_height, np.array([0.0, 0.0, 1.15]), 100, ground, params, context)
    assert true.evidence.score > shifted.evidence.score
    assert true.evidence.score > wrong.evidence.score


def test_hidden_faces_are_neutral_and_free_space_is_penalized() -> None:
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    context = _cuboid_ray_context()
    expected, covered, violations, supported, incident = visibility_evidence(
        pose, context, (0.6, 0.6, 0.9)
    )
    assert expected == 2  # near side and top; back/bottom faces are hidden
    assert covered == 2
    assert violations == 0
    assert supported > 0

    through = SensorContext(context.origin, context.ray_directions, context.ranges + 1.0)
    _, _, through_violations, _, _ = visibility_evidence(pose, through, (0.6, 0.6, 0.9))
    assert through_violations > 0
    occluded = SensorContext(context.origin, context.ray_directions, context.ranges - 0.5)
    _, _, occluded_violations, _, _ = visibility_evidence(pose, occluded, (0.6, 0.6, 0.9))
    assert occluded_violations == 0


def test_rich_sample_entry_point_recovers_far_case() -> None:
    from blockpose.scenario_matrix import generate_scenario_matrix

    _, sample = next(item for item in generate_scenario_matrix(0) if item[0].name == "isolated_far")
    detections = detect_scene_sample(sample, DetectionParams(multiscale_proposals=True))
    result = evaluate(detections, sample.ground_truth)
    assert result.recall == 1.0
    assert result.precision == 1.0


@pytest.mark.parametrize(
    "context",
    [
        SensorContext(np.array([0.0, 0.0]), np.ones((1, 3)), np.ones(1)),
        SensorContext(np.zeros(3), np.ones((1, 2)), np.ones(1)),
        SensorContext(np.zeros(3), np.ones((1, 3)), np.ones((1, 1))),
        SensorContext(np.zeros(3), np.ones((1, 3)), np.array([-1.0])),
        SensorContext(np.zeros(3), np.zeros((1, 3)), np.ones(1)),
    ],
)
def test_invalid_sensor_context_is_rejected_even_for_sparse_cloud(context) -> None:
    with pytest.raises(ValueError, match="sensor_context"):
        detect_blocks(np.empty((0, 3)), sensor_context=context)


def test_scaled_ray_directions_are_normalized_invariantly() -> None:
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    context = _cuboid_ray_context()
    scaled = SensorContext(context.origin, context.ray_directions * 7.0, context.ranges)
    points = context.origin + context.ranges[:, None] * context.ray_directions
    a = detect_blocks(points, sensor_context=context)
    b = detect_blocks(points, sensor_context=scaled)
    assert len(a) == len(b)
    assert visibility_evidence(pose, context, (0.6, 0.6, 0.9)) == visibility_evidence(
        pose, _validate_sensor_context(scaled), (0.6, 0.6, 0.9)
    )


def test_parallel_rays_do_not_emit_runtime_warning() -> None:
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    context = SensorContext(
        np.array([-5.0, 0.0, 3.0]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([2.0, 5.0]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        visibility_evidence(pose, context, (0.6, 0.6, 0.9))


def test_visibility_keeps_intersecting_rays_when_sensor_is_inside_bounding_sphere() -> None:
    """The angular prefilter must become a no-op when the sensor is inside it."""
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    # Put the sensor at the cuboid centre. Both rays initially have zero
    # projection toward the centre, but traverse the cuboid and terminate well
    # beyond it. An unconditional projection filter would incorrectly drop both.
    context = SensorContext(
        pose.position.copy(),
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
    )
    expected, covered, violations, supported, incident = visibility_evidence(
        pose, context, (0.6, 0.6, 0.9)
    )
    # This is the exact unfiltered slab-test result: a sensor inside the box
    # sees no outward-facing surface, while both rays cross its shrunk volume.
    assert (expected, covered, violations, supported, incident) == (0, 0, 2, 0, 0)


def test_per_ray_miss_origins_prevent_false_free_space_evidence() -> None:
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    directions = np.array([[1.0, 0.0, 0.0]])
    # The legacy scalar origin intersects the planned volume, while the actual
    # moving-sensor ray at y=2 m does not. Using the scalar would falsely turn
    # absent returns into counter-evidence.
    assert free_space_violations_from_misses(
        pose, np.array([-5.0, 0.0, 0.45]), directions, (0.6, 0.6, 0.9)
    ) == 1
    assert free_space_violations_from_misses(
        pose, np.array([[-5.0, 2.0, 0.45]]), directions, (0.6, 0.6, 0.9)
    ) == 0


def test_per_ray_origin_evidence_is_permutation_stable() -> None:
    pose = _make_pose_for_selection((0.0, 0.0, 0.45))
    origins = np.array([[-5.0, 0.0, 0.45], [-5.0, 2.0, 0.45], [-5.0, 0.0, 0.45]])
    directions = np.tile(np.array([[1.0, 0.0, 0.0]]), (3, 1))
    first = free_space_violations_from_misses(pose, origins, directions, (0.6, 0.6, 0.9))
    order = np.array([2, 0, 1])
    second = free_space_violations_from_misses(
        pose, origins[order], directions[order], (0.6, 0.6, 0.9)
    )
    assert (first, second) == (2, 2)


def test_far_block_survives_nearby_nonoccluding_clutter() -> None:
    upright = (0.6, 0.6, 0.9)
    sensor = (-5.0, 0.0, 4.0)
    far = generate_scene(
        [BlockSpec(8.0, 0.0, yaw_deg=61.0, dims=upright)],
        SynthParams(sensor_pos=sensor, look_at=(8.0, 0.0, 0.45), dims=upright),
        np.random.default_rng(31),
    )
    near = generate_scene(
        [],
        SynthParams(sensor_pos=sensor, look_at=(0.0, -2.0, 0.45), az_fov_deg=40.0, dims=upright),
        np.random.default_rng(32),
        ClutterParams(
            trees=False,
            container=False,
            logs=False,
            near_field_boxes=1,
            near_field_offset=2.0,
        ),
    )
    points = np.vstack([far.points, near.points])
    context = SensorContext(
        np.asarray(sensor),
        np.vstack([far.ray_directions, near.ray_directions]),
        np.concatenate([far.ranges, near.ranges]),
    )
    result = evaluate(
        detect_blocks(
            points,
            DetectionParams(block_dims=upright, multiscale_proposals=True),
            sensor_context=context,
        ),
        far.ground_truth,
    )
    assert result.recall == 1.0
    assert result.unmatched_det == 0


def test_detailed_api_preserves_pose_api_and_has_finite_normalized_features() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=19.0)],
        SynthParams(range_noise_std=0.0),
        np.random.default_rng(71),
    )
    context = SensorContext(sample.sensor_origin, sample.ray_directions, sample.ranges)
    detailed = detect_blocks_detailed(sample.points, sensor_context=context)
    plain = detect_blocks(sample.points, sensor_context=context)
    assert isinstance(detailed, DetectionResult)
    assert len(detailed.poses) == len(detailed.diagnostics) == len(plain) == 1
    np.testing.assert_allclose(detailed.poses[0].position, plain[0].position, atol=0.03)
    for value in detailed.diagnostics[0].features.values():
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert detailed.diagnostics[0].calibrated_confidence is None
    assert detailed.stage_counts["proposal_components"] >= 1
    assert detailed.stage_counts["selected_hypotheses"] == 1
    assert detailed.stage_timings_ms["total"] > 0.0
    assert detailed.diagnostics[0].proposal_scale_m > 0.0
    assert np.isfinite(detailed.diagnostics[0].support_height_m)


def test_detailed_api_reports_stable_plane_work_diagnostics_for_empty_input() -> None:
    result = detect_blocks_detailed(np.empty((0, 3), dtype=np.float64))
    assert result.poses == []
    assert result.stage_counts == {
        "input_points": 0,
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
        "refinement_candidates": 0,
        "selected_hypotheses": 0,
    }
    expected_timings = {
        "voxel_downsample",
        "ground_model",
        "proposals",
        "region_split",
        "plane_search",
        "plane_full_score_refit",
        "hypothesis_generation_evidence",
        "plane_hypotheses",
        "selection",
        "refinement",
        "total",
    }
    assert set(result.stage_timings_ms) == expected_timings
    assert all(value >= 0.0 for value in result.stage_timings_ms.values())


@pytest.mark.parametrize(
    ("covered", "violations", "expected"),
    [(1, 0, Observability.LIMITED), (2, 0, Observability.PARTIAL), (3, 0, Observability.FULL)],
)
def test_observability_is_geometry_derived(covered, violations, expected) -> None:
    from blockpose.detection import _diagnostics

    evidence = HypothesisEvidence(100, 0.0, 1.0, 3, covered, violations, 20)
    hypothesis = CuboidHypothesis(_make_pose_for_selection((0.0, 0.0, 0.45)), evidence)
    diagnostics = _diagnostics(hypothesis)
    assert isinstance(diagnostics, DetectionDiagnostics)
    assert diagnostics.observability is expected


def test_detailed_observability_stable_under_permutation() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=31.0)], SynthParams(), np.random.default_rng(81)
    )
    context = SensorContext(sample.sensor_origin, sample.ray_directions, sample.ranges)
    first = detect_blocks_detailed(sample.points, sensor_context=context)
    second = detect_blocks_detailed(
        np.random.default_rng(82).permutation(sample.points), sensor_context=context
    )
    assert [item.observability for item in first.diagnostics] == [
        item.observability for item in second.diagnostics
    ]
    np.testing.assert_allclose(first.poses[0].position, second.poses[0].position, atol=0.03)


def test_dimension_gating_rejects_tall_cluster() -> None:
    """Tall narrow structure (dz=8m) must be rejected by Z-extent gate."""
    rng = np.random.default_rng(1)
    # Points in a 0.5x0.5x8 m column at origin, z in [0.2, 8.2]
    n = 1000
    xyz = rng.uniform([0.0, 0.0, 0.2], [0.5, 0.5, 8.2], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )


def test_dimension_gating_rejects_wide_cluster() -> None:
    """Wide wall face (dx=5m) must be rejected by XY-extent gate."""
    rng = np.random.default_rng(2)
    # Points in a 5x0.3x0.9 m wall, z in [0.2, 1.1]
    n = 1000
    xyz = rng.uniform([0.0, 0.0, 0.2], [5.0, 0.3, 1.1], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )


def test_dimension_gating_rejects_high_altitude_cluster() -> None:
    """Block-sized box at high altitude (z_center~4.45m) must be rejected by z-center gate."""
    rng = np.random.default_rng(3)
    # Block-sized box at z in [4.0, 4.9], center = 4.45 >> 3.0 m gate
    n = 500
    xyz = rng.uniform([0.0, 0.0, 4.0], [0.6, 0.6, 4.9], size=(n, 3))
    ground = _make_flat_ground(n=3000)
    pts = np.vstack([ground, xyz]).astype(np.float64)
    result = detect_blocks(pts)
    assert len(result) == 0, (
        f"Expected 0 detections, got {len(result)}: {[p.position for p in result]}"
    )
