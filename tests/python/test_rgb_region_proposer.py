"""Tests for the full-image DINOv2 `dino_proposer` region head."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from blockpose.rgb_proposals import ImageProposal
from blockpose.rgb_region_proposer import (
    DinoRegionProposerConfig,
    DinoRegionProposerHead,
    _patch_aligned_size,
    extract_region_proposals,
    load_dino_region_proposer,
    save_checkpoint,
)


def test_config_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="model_id"):
        DinoRegionProposerConfig(model_id="")
    with pytest.raises(ValueError, match="hidden_dim"):
        DinoRegionProposerConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="max_image_side"):
        DinoRegionProposerConfig(max_image_side=0)
    with pytest.raises(ValueError, match="foreground_threshold"):
        DinoRegionProposerConfig(foreground_threshold=1.5)
    with pytest.raises(ValueError, match="foreground_threshold"):
        DinoRegionProposerConfig(foreground_threshold=-0.1)
    with pytest.raises(ValueError, match="min_component_area_px"):
        DinoRegionProposerConfig(min_component_area_px=-1)


def test_patch_aligned_size_preserves_aspect_and_aligns_to_patch_grid() -> None:
    height, width = _patch_aligned_size(1536, 2048, max_image_side=896, patch_size=14)
    assert height % 14 == 0 and width % 14 == 0
    assert max(height, width) == pytest.approx(896, abs=14)
    # Aspect ratio is approximately preserved.
    assert height / width == pytest.approx(1536 / 2048, abs=0.02)

    # A max_image_side far below patch_size must never round a dimension to zero.
    height, width = _patch_aligned_size(1000, 1000, max_image_side=1, patch_size=14)
    assert height == 14 and width == 14


def test_extract_region_proposals_splits_multiple_blobs_and_drops_small_ones() -> None:
    probability_map = np.zeros((40, 50), dtype=np.float64)
    probability_map[2:10, 2:10] = 0.9  # 8x8 = 64 px, kept
    probability_map[20:28, 20:30] = 0.8  # 8x10 = 80 px, kept
    probability_map[35:37, 35:37] = 0.95  # 2x2 = 4 px, dropped

    config = DinoRegionProposerConfig(foreground_threshold=0.5, min_component_area_px=10)
    proposals = extract_region_proposals(probability_map, config, model_id="test-model")

    assert len(proposals) == 2
    areas = sorted(int(p.mask.sum()) for p in proposals)
    assert areas == [64, 80]
    for proposal in proposals:
        assert isinstance(proposal, ImageProposal)
        assert proposal.provider == "dino_region_proposer"
        assert proposal.model_id == "test-model"
        assert proposal.mask.shape == probability_map.shape
        assert 0.0 <= proposal.score <= 1.0
    scores = sorted(round(p.score, 3) for p in proposals)
    assert scores == [0.8, 0.9]


def test_extract_region_proposals_score_is_mean_probability_within_component() -> None:
    probability_map = np.zeros((20, 20), dtype=np.float64)
    probability_map[5:9, 5:9] = 0.6  # 16 px at 0.6
    probability_map[6, 6] = 1.0  # bump one pixel inside the same component to 1.0

    config = DinoRegionProposerConfig(foreground_threshold=0.5, min_component_area_px=1)
    proposals = extract_region_proposals(probability_map, config)

    assert len(proposals) == 1
    expected_mean = (15 * 0.6 + 1.0) / 16
    assert proposals[0].score == pytest.approx(expected_mean)


def test_extract_region_proposals_returns_nothing_for_an_all_background_map() -> None:
    probability_map = np.full((20, 20), 0.1, dtype=np.float64)
    config = DinoRegionProposerConfig(foreground_threshold=0.5)
    assert extract_region_proposals(probability_map, config) == ()


def test_extract_region_proposals_rejects_malformed_probability_map() -> None:
    config = DinoRegionProposerConfig()
    with pytest.raises(ValueError, match="non-empty"):
        extract_region_proposals(np.zeros((0, 0)), config)
    with pytest.raises(ValueError, match="2-D"):
        extract_region_proposals(np.zeros((4, 4, 1)), config)
    with pytest.raises(ValueError, match="finite"):
        extract_region_proposals(np.full((4, 4), np.nan), config)


def test_dino_region_proposer_head_is_lazy() -> None:
    head = DinoRegionProposerHead(DinoRegionProposerConfig())
    assert not head.is_loaded


@pytest.mark.parametrize("missing_module", ["torch", "transformers"])
def test_dino_region_proposer_head_reports_missing_optional_dependency(monkeypatch, missing_module):
    monkeypatch.setitem(sys.modules, missing_module, None)
    head = DinoRegionProposerHead(DinoRegionProposerConfig())
    with pytest.raises(RuntimeError, match="uv sync --extra sam") as excinfo:
        head._load()
    assert missing_module in str(excinfo.value)


class _FakeBackboneConfig:
    hidden_size = 6
    patch_size = 14


class _FakeBackbone:
    config = _FakeBackboneConfig()

    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors, do_resize, do_center_crop):
        import torch

        assert do_resize is False and do_center_crop is False
        width, height = images.size
        return {"pixel_values": torch.zeros(1, 3, height, width)}


def _patch_dino_stack(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda model_id: _FakeBackbone()
    )
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", lambda model_id: _FakeProcessor()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_checkpoint_roundtrip_preserves_config_and_head_weights(tmp_path, monkeypatch) -> None:
    _patch_dino_stack(monkeypatch)
    import torch

    config = DinoRegionProposerConfig(hidden_dim=8, max_image_side=112)
    head = DinoRegionProposerHead(config, device="cpu")
    head._load()
    checkpoint_dir = tmp_path / "dino_region_proposer_ckpt"
    save_checkpoint(checkpoint_dir, head, config)
    assert (checkpoint_dir / "config.json").is_file()
    assert (checkpoint_dir / "dense_head.pt").is_file()

    loaded_head, loaded_config = load_dino_region_proposer(checkpoint_dir)
    assert loaded_config == config
    original_state = head.dense_head.state_dict()
    loaded_state = loaded_head.dense_head.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])


def test_load_dino_region_proposer_rejects_missing_or_unversioned_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dino_region_proposer(tmp_path / "does-not-exist")

    bad = tmp_path / "bad_checkpoint"
    bad.mkdir()
    (bad / "dense_head.pt").write_bytes(b"not a real state dict")
    (bad / "config.json").write_text('{"schema_version": 999}')
    with pytest.raises(ValueError, match="schema"):
        load_dino_region_proposer(bad)


def test_predict_proposal_map_upsamples_frozen_features_to_input_resolution(monkeypatch) -> None:
    _patch_dino_stack(monkeypatch)
    import torch

    patch_size, grid_h, grid_w, hidden_size = 14, 4, 3, _FakeBackboneConfig.hidden_size
    n_patches = grid_h * grid_w

    def fake_forward(self, **inputs):
        return SimpleNamespace(last_hidden_state=torch.zeros(1, n_patches + 1, hidden_size))

    monkeypatch.setattr(_FakeBackbone, "__call__", fake_forward, raising=False)

    height, width = grid_h * patch_size, grid_w * patch_size  # exact match, no rounding surprise
    config = DinoRegionProposerConfig(max_image_side=height, hidden_dim=4)
    head = DinoRegionProposerHead(config, device="cpu")

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    probability_map = head.predict_proposal_map(rgb)

    assert probability_map.shape == (height, width)
    assert np.all(np.isfinite(probability_map))
    assert np.all((probability_map >= 0.0) & (probability_map <= 1.0))


def test_patch_features_exposes_frozen_grid_for_external_training_loops(monkeypatch) -> None:
    _patch_dino_stack(monkeypatch)
    import torch

    patch_size, grid_h, grid_w, hidden_size = 14, 2, 2, _FakeBackboneConfig.hidden_size
    n_patches = grid_h * grid_w

    def fake_forward(self, **inputs):
        tokens = torch.arange(n_patches + 1, dtype=torch.float32)
        tokens = tokens.reshape(1, n_patches + 1, 1).expand(1, n_patches + 1, hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)

    monkeypatch.setattr(_FakeBackbone, "__call__", fake_forward, raising=False)

    height = width = grid_h * patch_size
    config = DinoRegionProposerConfig(max_image_side=height)
    head = DinoRegionProposerHead(config, device="cpu")
    features = head.patch_features(np.zeros((height, width, 3), dtype=np.uint8))

    assert features.shape == (grid_h, grid_w, hidden_size)
    # CLS token (value 0) is dropped; patch tokens are 1..n_patches in row-major order.
    expected = np.arange(1, n_patches + 1).reshape(grid_h, grid_w)
    np.testing.assert_allclose(features[..., 0], expected)
