"""Tests for the multi-source candidate-block computation used by the annotator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from blockpose.annotation_candidates import (
    CandidateSourcesResult,
    SourceCandidate,
    compute_candidate_sources,
)
from blockpose.io import CameraModel
from blockpose.synth import BlockSpec, SynthParams, generate_scene


def _scene():
    """A single detectable block, no clutter, deterministic RNG (matches test_detection.py)."""
    params = SynthParams(range_noise_std=0.0)
    sample = generate_scene([BlockSpec(0.0, 0.0, yaw_deg=31.0)], params, np.random.default_rng(17))
    return sample, params


def _look_at_transform(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera (X right, Y down, Z forward) pose looking from `camera_pos` at `target`."""
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, forward])
    transform[:3, 3] = camera_pos
    return transform


def _camera_and_rgb(
    sample, params, rng: np.random.Generator
) -> tuple[CameraModel, np.ndarray, np.ndarray]:
    """A wide-FOV camera roughly co-located with the synthetic LiDAR sensor."""
    camera = CameraModel(
        K=np.array([[400.0, 0.0, 640.0], [0.0, 400.0, 480.0], [0.0, 0.0, 1.0]]),
        width=1280,
        height=960,
    )
    target = sample.ground_truth[0].position if sample.ground_truth else np.zeros(3)
    T_world_camera = _look_at_transform(np.asarray(params.sensor_pos), target)
    rgb = rng.integers(0, 256, size=(camera.height, camera.width, 3), dtype=np.uint8)
    return camera, T_world_camera, rgb


def test_pcl_only_populates_when_no_rgb_or_checkpoints_given() -> None:
    sample, _ = _scene()
    result = compute_candidate_sources(sample.points, None, None, None)

    assert isinstance(result, CandidateSourcesResult)
    assert len(result.pcl) > 0
    assert all(isinstance(item, SourceCandidate) for item in result.pcl)
    assert any(item.accepted for item in result.pcl), "at least the final detection should survive"

    assert result.rgb_edge_prior == ()
    assert result.dino_proposer == ()
    assert result.dino_verifier_reranked == ()
    # The geometry-only verifier view needs no RGB/checkpoint at all.
    assert len(result.dino_verifier_geometry_only) == len(result.pcl)

    assert any("rgb_edge_prior unavailable" in w for w in result.warnings)
    assert any(
        w == "dino_proposer unavailable: no checkpoint given" for w in result.warnings
    )
    assert any(
        w == "dino_verifier unavailable: no checkpoint given" for w in result.warnings
    )


def test_missing_sensor_origin_warns_and_changes_pcl_scores() -> None:
    """`T_world_cloud` must reach `detect_blocks_detailed` as a real sensor context.

    Regression test for a bug where `compute_candidate_sources` had no way to
    receive the sensor origin, so it silently ran with `sensor_context=None`,
    zeroing ~0.60 of the score weight (free-space/visibility evidence) and
    changing which hypotheses survive -- confirmed on real data where max PCL
    score dropped from 0.818 (real sensor context) to 0.292 (without).
    """
    sample, params = _scene()

    without_origin = compute_candidate_sources(sample.points, None, None, None)
    assert any("sensor_context unavailable" in w for w in without_origin.warnings)

    T_world_cloud = np.eye(4)
    T_world_cloud[:3, 3] = params.sensor_pos
    with_origin = compute_candidate_sources(
        sample.points, None, None, None, T_world_cloud=T_world_cloud
    )
    assert not any("sensor_context unavailable" in w for w in with_origin.warnings)

    max_score_without = max(item.score for item in without_origin.pcl)
    max_score_with = max(item.score for item in with_origin.pcl)
    assert max_score_with > max_score_without


def test_rgb_edge_prior_populates_when_rgb_camera_and_tf_are_given() -> None:
    from blockpose import _core

    if not getattr(_core, "vision_enabled", False):
        pytest.skip("blockpose._core was built without optional RGB vision support")

    sample, params = _scene()
    camera, T_world_camera, rgb = _camera_and_rgb(sample, params, np.random.default_rng(3))

    result = compute_candidate_sources(sample.points, rgb, T_world_camera, camera)

    assert len(result.rgb_edge_prior) == len(result.pcl)
    assert {item.id for item in result.rgb_edge_prior} == {item.id for item in result.pcl}
    assert not any("rgb_edge_prior unavailable" in w for w in result.warnings)
    # Still unavailable: no DINO checkpoints given.
    assert result.dino_proposer == ()
    assert result.dino_verifier_reranked == ()


def test_unavailable_rgb_edge_prior_does_not_discard_pcl_candidates(monkeypatch) -> None:
    """A geometry-only native build must still support PCL annotation."""
    from blockpose import _core

    sample, params = _scene()
    camera, T_world_camera, rgb = _camera_and_rgb(sample, params, np.random.default_rng(31))
    monkeypatch.setattr(_core, "vision_enabled", False)

    result = compute_candidate_sources(sample.points, rgb, T_world_camera, camera)

    assert len(result.pcl) > 0
    assert result.rgb_edge_prior == ()
    assert any(
        "rgb_edge_prior unavailable" in warning
        and "not enabled" in warning
        for warning in result.warnings
    )


def test_dino_sources_report_a_clear_warning_without_crashing_when_no_checkpoint_given() -> None:
    from blockpose import _core

    sample, params = _scene()
    camera, T_world_camera, rgb = _camera_and_rgb(sample, params, np.random.default_rng(4))

    result = compute_candidate_sources(
        sample.points,
        rgb,
        T_world_camera,
        camera,
        dino_proposer_checkpoint=None,
        dino_verifier_checkpoint=None,
    )

    assert result.dino_proposer == ()
    assert result.dino_verifier_reranked == ()
    assert "dino_proposer unavailable: no checkpoint given" in result.warnings
    assert "dino_verifier unavailable: no checkpoint given" in result.warnings
    # Every other source still computed -- one missing optional source never
    # blanks out the rest.
    assert len(result.pcl) > 0
    if getattr(_core, "vision_enabled", False):
        assert len(result.rgb_edge_prior) == len(result.pcl)
    else:
        assert result.rgb_edge_prior == ()
        assert any("rgb_edge_prior unavailable" in warning for warning in result.warnings)
    assert len(result.dino_verifier_geometry_only) == len(result.pcl)


class _FakeBackboneConfig:
    hidden_size = 8
    patch_size = 14


class _FakeBackbone:
    config = _FakeBackboneConfig()

    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


def _patch_frozen_backbone(monkeypatch, torch, *, patch_size: int):
    """Stub the shared frozen-backbone loader: no network/GPU required.

    Mirrors the `_patch_dino_stack`/`_FakeBackbone` pattern in
    `tests/python/test_rgb_verifier.py` and `test_rgb_region_proposer.py`.
    """

    class _FakeProcessor:
        def __call__(self, images, return_tensors="pt", do_resize=True, do_center_crop=True):
            width, height = images.size
            return {"pixel_values": torch.zeros(1, 3, height, width)}

    def fake_forward(self, **inputs):
        _, _, height, width = inputs["pixel_values"].shape
        grid_h, grid_w = height // patch_size, width // patch_size
        tokens = torch.zeros(1, grid_h * grid_w + 1, _FakeBackboneConfig.hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)

    monkeypatch.setattr(_FakeBackbone, "__call__", fake_forward, raising=False)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda model_id: _FakeBackbone())
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", lambda model_id: _FakeProcessor()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_stubbed_dino_proposer_populates_dino_proposer_field(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _patch_frozen_backbone(monkeypatch, torch, patch_size=14)
    from blockpose.rgb_region_proposer import (
        DinoRegionProposerConfig,
        DinoRegionProposerHead,
        save_checkpoint,
    )

    config = DinoRegionProposerConfig(hidden_dim=4)
    head = DinoRegionProposerHead(config, device="cpu")
    head._load()
    # Force the dense head to predict foreground everywhere, independent of
    # the (all-zero, stubbed) backbone features, so proposal extraction is
    # deterministic instead of depending on random initialization.
    last_conv = list(head.dense_head.children())[-1]
    with torch.no_grad():
        last_conv.weight.zero_()
        last_conv.bias.fill_(10.0)
    checkpoint_dir = tmp_path / "dino_proposer_ckpt"
    save_checkpoint(checkpoint_dir, head, config)

    sample, params = _scene()
    camera, T_world_camera, rgb = _camera_and_rgb(sample, params, np.random.default_rng(5))

    result = compute_candidate_sources(
        sample.points, rgb, T_world_camera, camera, dino_proposer_checkpoint=checkpoint_dir
    )

    assert not any("dino_proposer unavailable" in w for w in result.warnings)
    assert len(result.dino_proposer) > 0
    assert all(item.id.startswith("dino_proposer:") for item in result.dino_proposer)


def test_stubbed_dino_verifier_populates_reranked_field(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    _patch_frozen_backbone(monkeypatch, torch, patch_size=32)
    from blockpose.rgb_verifier import DinoVerifierConfig, DinoVerifierHead, save_checkpoint

    config = DinoVerifierConfig(hidden_dim=4, max_score_contribution=0.2)
    head = DinoVerifierHead(config, device="cpu")
    head._load()
    # Force a strong, confident, non-abstained positive score regardless of
    # input features: bias = [score_logit, log_uncertainty].
    mlp_last = list(head.mlp.children())[-1]
    with torch.no_grad():
        mlp_last.weight.zero_()
        mlp_last.bias[0] = 10.0
        mlp_last.bias[1] = -10.0
    checkpoint_dir = tmp_path / "dino_verifier_ckpt"
    save_checkpoint(checkpoint_dir, head, config, calibration_temperature=1.0)

    sample, params = _scene()
    camera, T_world_camera, rgb = _camera_and_rgb(sample, params, np.random.default_rng(6))

    result = compute_candidate_sources(
        sample.points, rgb, T_world_camera, camera, dino_verifier_checkpoint=checkpoint_dir
    )

    assert not any("dino_verifier unavailable" in w for w in result.warnings)
    assert len(result.dino_verifier_reranked) == len(result.pcl)
    # Every candidate got real (non-abstained) DINO evidence -- the rigged
    # head always predicts a strong positive, confident score.
    assert all(item.reason == "ok" for item in result.dino_verifier_reranked)
    assert all(item.score == pytest.approx(0.2, abs=1e-3) for item in result.dino_verifier_reranked)
