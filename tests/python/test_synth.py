"""Tests for the synthetic scene generator (blockpose.synth)."""

from __future__ import annotations

import numpy as np
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
