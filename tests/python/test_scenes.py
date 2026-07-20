"""Tests for YAML scene definitions (blockpose.scenes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from blockpose.scenes import (
    _expand_pattern,
    generate_from_config,
    load_scene_config,
)

REPO_ROOT = Path(__file__).parent.parent.parent
SCENES_DIR = REPO_ROOT / "config" / "scenes"
DIMS = (0.6, 0.6, 0.9)

EXPECTED_BLOCKS = {
    "storage_yard": 4 + 4 + 1,
    "stacked_blocks": 3 + 2 + 2,
    "wall": 3 * 5 + 1,
}


@pytest.mark.parametrize("name,count", EXPECTED_BLOCKS.items())
def test_example_scene_loads_with_expected_block_count(name, count) -> None:
    cfg = load_scene_config(SCENES_DIR / f"{name}.yaml")
    assert len(cfg.specs) == count
    assert cfg.clutter is not None  # all example scenes enable clutter


def test_pattern_row_positions() -> None:
    specs = _expand_pattern(
        {"type": "row", "count": 3, "start": [0.0, 0.0], "step": [0.0, 1.0], "yaw_deg": 5}, DIMS
    )
    ys = [s.y for s in specs]
    assert ys == [0.0, 1.0, 2.0]
    assert all(s.yaw_deg == 5 for s in specs)


def test_pattern_stack_z_centers() -> None:
    specs = _expand_pattern({"type": "stack", "count": 3, "base": [1.0, 2.0]}, DIMS)
    zs = [round(s.z, 3) for s in specs]
    assert zs == [0.45, 1.35, 2.25]  # block height 0.9 -> centers stacked
    assert all(s.x == 1.0 and s.y == 2.0 for s in specs)


def test_pattern_wall_running_bond() -> None:
    specs = _expand_pattern(
        {"type": "wall", "origin": [0.0, 0.0], "courses": 2, "length": 2,
         "axis": "y", "running_bond": True}, DIMS
    )
    assert len(specs) == 4
    course0 = sorted(s.y for s in specs if abs(s.z - 0.45) < 1e-6)
    course1 = sorted(s.y for s in specs if abs(s.z - 1.35) < 1e-6)
    np.testing.assert_allclose(course0, [0.0, 0.6], atol=1e-9)
    np.testing.assert_allclose(course1, [0.3, 0.9], atol=1e-9)  # offset by half a block


def test_unknown_pattern_raises() -> None:
    with pytest.raises(ValueError):
        _expand_pattern({"type": "spiral", "count": 3}, DIMS)


def test_generate_from_config_produces_cloud_and_gt() -> None:
    cfg = load_scene_config(SCENES_DIR / "stacked_blocks.yaml")
    pts, gt = generate_from_config(cfg, np.random.default_rng(0))
    assert pts.shape[1] == 3 and len(pts) > 1000
    assert len(gt) == EXPECTED_BLOCKS["stacked_blocks"]
    for pose in gt:
        assert pose.source == "synthetic"
        np.testing.assert_allclose(np.linalg.det(pose.rotation), 1.0, atol=1e-6)


def test_scene_yaml_forwards_all_degradation_parameters(tmp_path: Path) -> None:
    scene_path = tmp_path / "degraded.yaml"
    scene_path.write_text(
        """
blocks: [{x: 0.0, y: 0.0}]
sensor:
  range_noise_slope: 0.003
  stripe_dropout: 0.2
  outlier_fraction: 0.04
  outlier_radius: 0.7
"""
    )
    params = load_scene_config(scene_path).params
    assert params.range_noise_slope == pytest.approx(0.003)
    assert params.stripe_dropout == pytest.approx(0.2)
    assert params.outlier_fraction == pytest.approx(0.04)
    assert params.outlier_radius == pytest.approx(0.7)
