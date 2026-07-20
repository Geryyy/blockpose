"""Tests for the synthetic scene generator (blockpose.synth)."""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.synth import (
    BlockSpec,
    ClutterParams,
    SynthParams,
    generate_scene,
    random_scene,
)

# Wide FOV so the side clutter is in view; no noise for exact comparisons.
_WIDE = SynthParams(az_fov_deg=130.0, el_fov_deg=80.0, range_noise_std=0.0)


def test_determinism_same_seed() -> None:
    spec = [BlockSpec(x=0.0, y=0.0, yaw_deg=20.0)]
    a, _ = generate_scene(spec, SynthParams(range_noise_std=0.01), np.random.default_rng(42))
    b, _ = generate_scene(spec, SynthParams(range_noise_std=0.01), np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_gt_matches_specs_and_is_valid_so3() -> None:
    specs = [BlockSpec(0.0, 0.0, yaw_deg=30.0), BlockSpec(1.5, 0.0, yaw_deg=-10.0, pitch_deg=2.0)]
    _, gt = generate_scene(specs, SynthParams(), np.random.default_rng(0))
    assert len(gt) == len(specs)
    for pose in gt:
        assert np.all(np.isfinite(pose.position))
        R = pose.rotation
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_cloud_nonempty_and_on_block_surface() -> None:
    # Single axis-aligned block, no noise: top face is at z = height, fully visible.
    dims = (0.6, 0.6, 0.9)
    pts, gt = generate_scene(
        [BlockSpec(0.0, 0.0, yaw_deg=0.0, dims=dims)],
        SynthParams(range_noise_std=0.0),
        np.random.default_rng(0),
    )
    assert len(pts) > 1000
    # Top face fully in view -> max Z hits the block top (0.9) exactly (no noise).
    assert abs(float(pts[:, 2].max()) - dims[2]) < 0.02
    # The near side face (-x toward the sensor at x=-5) should be sampled.
    assert float(pts[:, 0].min()) < -dims[0] / 2 + 0.05


def test_empty_scene_has_ground_only() -> None:
    pts, gt = generate_scene([], SynthParams(), np.random.default_rng(0))
    assert len(gt) == 0
    assert len(pts) > 100  # ground is still raycast
    # No block -> nothing rises above the ground plane.
    assert float(pts[:, 2].max()) < 0.15


def test_random_scene_block_count() -> None:
    pts, gt = random_scene(3, SynthParams(range_noise_std=0.01), np.random.default_rng(1))
    assert len(gt) == 3
    assert len(pts) > 1000
    for pose in gt:
        assert pose.source == "synthetic"


def test_clutter_adds_side_objects_but_no_ground_truth() -> None:
    specs = [BlockSpec(0.0, 0.0, yaw_deg=15.0)]
    _, gt0 = generate_scene(specs, _WIDE, np.random.default_rng(1))
    pts, gt = generate_scene(specs, _WIDE, np.random.default_rng(1), clutter=ClutterParams())
    # Clutter never adds ground truth.
    assert len(gt) == len(gt0) == 1
    # Trees (canopies, z>1), logs (-Y, raised), and container (+Y, tall) all appear.
    assert (pts[(np.abs(pts[:, 1]) > 2.5) & (pts[:, 2] > 1.0)]).shape[0] > 500  # trees
    assert (pts[(pts[:, 1] < -3.0) & (pts[:, 2] > 0.1) & (pts[:, 2] < 0.45)]).shape[0] > 100  # logs
    assert (pts[(pts[:, 1] > 3.0) & (pts[:, 2] > 0.5)]).shape[0] > 500  # container


def test_clutter_keeps_working_area_free() -> None:
    specs = [BlockSpec(0.0, 0.0, yaw_deg=15.0)]
    pts0, _ = generate_scene(specs, _WIDE, np.random.default_rng(1))
    pts1, _ = generate_scene(specs, _WIDE, np.random.default_rng(1), clutter=ClutterParams())
    # With no range noise, clutter (all at |y| >= ~2.5) must not touch the central
    # working band, so the block + ground points there are byte-identical.
    wa0 = int((np.abs(pts0[:, 1]) < 1.5).sum())
    wa1 = int((np.abs(pts1[:, 1]) < 1.5).sum())
    assert wa0 == wa1


def test_clutter_deterministic() -> None:
    specs = [BlockSpec(0.0, 0.0)]
    a, _ = generate_scene(specs, _WIDE, np.random.default_rng(7), clutter=ClutterParams())
    b, _ = generate_scene(specs, _WIDE, np.random.default_rng(7), clutter=ClutterParams())
    assert np.array_equal(a, b)


def test_tilted_default_block_is_physically_settled() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0, pitch_deg=12.0, roll_deg=-7.0)],
        SynthParams(range_noise_std=0.0),
        np.random.default_rng(4),
    )
    pose = sample.ground_truth[0]
    half = np.asarray((0.6, 0.6, 0.9)) / 2
    corners = np.array(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    world = (pose.rotation @ corners.T).T + pose.position
    assert world[:, 2].min() == pytest.approx(0.0, abs=1e-9)


def test_rich_sample_labels_and_legacy_unpacking() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0)], SynthParams(range_noise_std=0.0), np.random.default_rng(2)
    )
    pts, gt = sample
    assert pts is sample.points and gt is sample.ground_truth
    assert sample.instance_ids.shape == (len(pts),)
    assert sample.face_ids.shape == (len(pts),)
    assert sample.ray_directions.shape == pts.shape
    assert np.any(sample.instance_ids == 0)
    assert sample.manifest["generator"] == "blockpose.synth.v2"


def test_degradation_profile_is_deterministic_and_labeled() -> None:
    params = SynthParams(
        range_noise_std=0.0,
        range_noise_slope=0.002,
        dropout=0.1,
        stripe_dropout=0.05,
        outlier_fraction=0.01,
    )
    a = generate_scene([BlockSpec(0.0, 0.0)], params, np.random.default_rng(8))
    b = generate_scene([BlockSpec(0.0, 0.0)], params, np.random.default_rng(8))
    assert np.array_equal(a.points, b.points)
    assert len(a.points) == len(a.instance_ids) == len(a.ranges)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("az_fov_deg", 0.0),
        ("el_fov_deg", np.inf),
        ("ang_res_deg", -0.1),
        ("ground_size", 0.0),
        ("range_noise_std", -0.1),
        ("range_noise_slope", np.nan),
        ("dropout", 1.1),
        ("stripe_dropout", -0.1),
        ("outlier_fraction", np.inf),
        ("outlier_radius", 0.049),
    ],
)
def test_invalid_synth_parameters_raise_clear_error(field: str, value: float) -> None:
    params = SynthParams(**{field: value})
    with pytest.raises(ValueError, match=field):
        generate_scene([], params, np.random.default_rng(1))


@pytest.mark.parametrize(
    "params",
    [
        SynthParams(sensor_pos=(0.0, np.nan, 1.0)),
        SynthParams(look_at=(np.inf, 0.0, 0.0)),
        SynthParams(sensor_pos=(1.0, 2.0, 3.0), look_at=(1.0, 2.0, 3.0)),
    ],
)
def test_invalid_sensor_geometry_raises_clear_error(params: SynthParams) -> None:
    with pytest.raises(ValueError, match="sensor_pos|look_at"):
        generate_scene([], params, np.random.default_rng(1))


def test_full_stripe_dropout_removes_every_azimuth_column() -> None:
    sample = generate_scene(
        [BlockSpec(0.0, 0.0)],
        SynthParams(range_noise_std=0.0, stripe_dropout=1.0),
        np.random.default_rng(3),
    )
    assert sample.points.shape == (0, 3)
    assert sample.ray_directions.shape == (0, 3)


@pytest.mark.parametrize(
    "spec",
    [
        BlockSpec(np.nan, 0.0),
        BlockSpec(0.0, 0.0, z=np.inf),
        BlockSpec(0.0, 0.0, yaw_deg=np.nan),
        BlockSpec(0.0, 0.0, pitch_deg=np.inf),
        BlockSpec(0.0, 0.0, dims=(0.6, 0.6, 0.0)),
        BlockSpec(0.0, 0.0, dims=(0.6, np.nan, 0.9)),
        BlockSpec(0.0, 0.0, dims=(0.6, 0.9)),
    ],
)
def test_invalid_block_specs_raise_before_mesh_construction(spec: BlockSpec) -> None:
    with pytest.raises(ValueError, match=r"blocks\[0\]"):
        generate_scene([spec], SynthParams(), np.random.default_rng(1))
