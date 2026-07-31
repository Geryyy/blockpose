"""Tests for the candidate-conditioned DINOv2 `dino_verifier` evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from blockpose.rgb_verifier import (
    DinoVerifierConfig,
    DinoVerifierHead,
    DinoVerifierScore,
    load_dino_verifier,
    pool_patch_tokens,
    region_masks,
    save_checkpoint,
    score_candidate_with_dino,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_ablate_module():
    """Import `tools/ablate_rgb_verifier.py` without requiring torch at import time."""
    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "ablate_rgb_verifier", REPO_ROOT / "tools" / "ablate_rgb_verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _conditioning(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[20:40, 20:40, 1] = 255  # G: filled silhouette
    image[20, 20:40, 0] = 255  # R: one visible edge on the silhouette's top row
    return image


def test_region_masks_derive_interior_edge_exterior_from_r_and_g_only() -> None:
    masks = region_masks(_conditioning(), edge_band_px=2)
    assert set(masks) == {"interior", "edge_band", "exterior", "gripper_occlusion"}
    # interior is exactly the filled G silhouette.
    assert masks["interior"].sum() == 20 * 20
    assert masks["interior"][30, 30] and not masks["interior"][0, 0]
    # edge_band is a dilation of the R row, strictly bigger than the raw edge.
    raw_edge_pixels = 20
    assert raw_edge_pixels < masks["edge_band"].sum() < masks["interior"].sum()
    # exterior excludes both interior and the edge band.
    assert not np.any(masks["exterior"] & masks["interior"])
    assert not np.any(masks["exterior"] & masks["edge_band"])
    # B is never populated by render_conditioning today; the occlusion region
    # is a deliberate always-empty gap, not a heuristic guess.
    assert not np.any(masks["gripper_occlusion"])


def test_region_masks_rejects_malformed_conditioning() -> None:
    with pytest.raises(ValueError, match="conditioning"):
        region_masks(np.zeros((8, 8), dtype=np.uint8), edge_band_px=1)
    with pytest.raises(ValueError, match="edge_band_px"):
        region_masks(_conditioning(), edge_band_px=-1)


def test_pool_patch_tokens_averages_under_mask_and_zeros_empty_region() -> None:
    grid = 8
    dim = 4
    tokens = np.zeros((grid, grid, dim), dtype=np.float64)
    tokens[:4, :, :] = 1.0  # top half of the patch grid
    tokens[4:, :, :] = 5.0  # bottom half

    masks = {
        "interior": np.zeros((grid, grid), dtype=bool),
        "edge_band": np.zeros((grid, grid), dtype=bool),
        "exterior": np.zeros((grid, grid), dtype=bool),
        "gripper_occlusion": np.zeros((grid, grid), dtype=bool),
    }
    masks["interior"][:4, :] = True  # exactly the top half -> mean 1.0
    masks["exterior"][4:, :] = True  # exactly the bottom half -> mean 5.0
    # edge_band and gripper_occlusion stay empty -> zero vectors.

    pooled = pool_patch_tokens(tokens, masks)
    assert pooled.shape == (4 * dim,)
    interior_vec, edge_vec, exterior_vec, occlusion_vec = np.split(pooled, 4)
    np.testing.assert_allclose(interior_vec, 1.0)
    np.testing.assert_allclose(edge_vec, 0.0)
    np.testing.assert_allclose(exterior_vec, 5.0)
    np.testing.assert_allclose(occlusion_vec, 0.0)


def test_pool_patch_tokens_resamples_a_finer_pixel_mask_to_the_patch_grid() -> None:
    grid = 4
    tokens = np.arange(grid * grid, dtype=np.float64).reshape(grid, grid, 1)
    # A pixel-resolution mask much finer than the patch grid, covering exactly
    # the first patch cell (top-left corner) at 16x oversampling.
    fine = np.zeros((64, 64), dtype=bool)
    fine[:16, :16] = True
    masks = {
        "interior": fine,
        "edge_band": np.zeros((64, 64), dtype=bool),
        "exterior": np.zeros((64, 64), dtype=bool),
        "gripper_occlusion": np.zeros((64, 64), dtype=bool),
    }
    pooled = pool_patch_tokens(tokens, masks)
    interior_vec = pooled[:1]
    np.testing.assert_allclose(interior_vec, tokens[0, 0], atol=1e-6)


def test_pool_patch_tokens_rejects_non_square_grid() -> None:
    with pytest.raises(ValueError, match="square"):
        pool_patch_tokens(np.zeros((4, 5, 3)), {name: np.zeros((4, 5), bool) for name in ()})


class _FakeHead:
    """Stands in for `DinoVerifierHead` in tests that must not require torch."""

    def __init__(self, raw_score: float, log_uncertainty: float = 0.0):
        self.raw_score = raw_score
        self.log_uncertainty = log_uncertainty
        self.calls: list[tuple] = []

    def forward(self, rgb_crop: np.ndarray, conditioning: np.ndarray) -> tuple[float, float]:
        self.calls.append((rgb_crop, conditioning))
        return self.raw_score, self.log_uncertainty


def _crop_and_conditioning(size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((size, size, 3), dtype=np.uint8), _conditioning(size)


def test_score_candidate_with_dino_caps_contribution_at_max_score_contribution() -> None:
    config = DinoVerifierConfig(max_score_contribution=0.2)
    rgb, cond = _crop_and_conditioning()
    confident_positive = score_candidate_with_dino(_FakeHead(1.0), config, rgb, cond)
    confident_negative = score_candidate_with_dino(_FakeHead(0.0), config, rgb, cond)
    neutral = score_candidate_with_dino(_FakeHead(0.5), config, rgb, cond)
    assert not confident_positive.abstained
    assert confident_positive.contribution == pytest.approx(0.2, abs=1e-5)
    assert confident_negative.contribution == pytest.approx(-0.2, abs=1e-5)
    assert neutral.contribution == pytest.approx(0.0, abs=1e-9)


def test_score_candidate_with_dino_is_safely_neutral_on_missing_or_malformed_inputs() -> None:
    config = DinoVerifierConfig()
    rgb, cond = _crop_and_conditioning()
    head = _FakeHead(0.9)

    missing_head = score_candidate_with_dino(None, config, rgb, cond)
    missing_crop = score_candidate_with_dino(head, config, None, cond)
    mismatched_shapes = score_candidate_with_dino(head, config, rgb, cond[:-1])
    non_finite = score_candidate_with_dino(head, config, rgb.astype(np.float64) * np.nan, cond)

    for result in (missing_head, missing_crop, mismatched_shapes, non_finite):
        assert isinstance(result, DinoVerifierScore)
        assert result.abstained
        assert result.contribution == 0.0
    assert missing_crop.reason == "missing_dino_or_inputs"
    assert mismatched_shapes.reason == "incompatible_crop_or_conditioning"
    assert non_finite.reason == "non_finite_input"


def test_score_candidate_with_dino_abstains_on_high_calibrated_uncertainty() -> None:
    config = DinoVerifierConfig(uncertainty_abstain_threshold=1.0)
    rgb, cond = _crop_and_conditioning()
    # log_uncertainty=1.0 -> uncertainty=e^1 ~ 2.72, well above the threshold.
    result = score_candidate_with_dino(_FakeHead(0.95, log_uncertainty=1.0), config, rgb, cond)
    assert result.abstained
    assert result.reason == "uncertain"
    assert result.contribution == 0.0


def test_score_candidate_with_dino_propagates_backend_unavailable_as_abstain() -> None:
    class _BrokenHead:
        def forward(self, rgb_crop: np.ndarray, conditioning: np.ndarray) -> tuple[float, float]:
            raise RuntimeError("dino_verifier is missing 'torch'")

    rgb, cond = _crop_and_conditioning()
    result = score_candidate_with_dino(_BrokenHead(), DinoVerifierConfig(), rgb, cond)
    assert result.abstained and result.reason == "dino_unavailable"


def test_bounded_reranking_never_lets_dino_cross_a_geometry_band() -> None:
    """Rigged case: DINO strongly prefers a losing candidate in a worse band.

    Mirrors `ablate_rgb_edge_prior.py`'s `_rank` enforcement, reused here from
    `tools/ablate_rgb_verifier.py` -- RGB evidence may only reorder candidates
    *within* one geometry score band, never override geometry across bands.
    """
    rank = _load_ablate_module()._rank
    # "winner" has clearly better geometry (band 0); "loser" is a whole band
    # behind but DINO is maximally confident it is correct, "winner" maximally
    # not. Without the band restriction DINO would flip the top-1 pick.
    candidates = [
        {
            "id": "winner",
            "geometry_score": 1.0,
            "dino_contribution": -0.2,
            "dino_abstained": False,
        },
        {
            "id": "loser",
            "geometry_score": 0.80,  # one full band (0.05) behind -> band 1+
            "dino_contribution": 0.2,
            "dino_abstained": False,
        },
    ]
    ranked = rank(
        [dict(c) for c in candidates], score_band=0.05, dino_bands={0, 1, 2, 3, 4, 5, 6, 7, 8}
    )
    assert ranked[0]["id"] == "winner"

    # Within the same band, DINO is free to break the tie.
    same_band = [
        {"id": "a", "geometry_score": 1.00, "dino_contribution": -0.2, "dino_abstained": False},
        {"id": "b", "geometry_score": 0.99, "dino_contribution": 0.2, "dino_abstained": False},
    ]
    reranked = rank([dict(c) for c in same_band], score_band=0.05, dino_bands={0})
    assert reranked[0]["id"] == "b"


def test_ablate_tool_requires_a_checkpoint_and_does_not_silently_no_op(
    tmp_path, monkeypatch
) -> None:
    module = _load_ablate_module()
    missing = tmp_path / "nonexistent_checkpoint"
    output = tmp_path / "o.json"
    argv = ["ablate_rgb_verifier.py", "--checkpoint", str(missing), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        module.main()


@pytest.mark.parametrize("missing_module", ["torch", "transformers"])
def test_dino_verifier_head_reports_missing_optional_dependency(monkeypatch, missing_module):
    # Only fake the actually-missing module: a `SimpleNamespace()` stub for the
    # *other* one has no `nn`/attrs of its own, so `from torch import nn` would
    # fail on that alone regardless of which module this case means to test.
    # Leaving the other module untouched (real or already-imported) means the
    # only forced failure is the one this parametrization claims to exercise.
    monkeypatch.setitem(sys.modules, missing_module, None)
    head = DinoVerifierHead(DinoVerifierConfig())
    with pytest.raises(RuntimeError, match="uv sync --extra sam") as excinfo:
        head._load()
    assert missing_module in str(excinfo.value)


def test_dino_verifier_head_is_lazy() -> None:
    head = DinoVerifierHead(DinoVerifierConfig())
    assert not head.is_loaded


def test_checkpoint_roundtrip_preserves_config_and_calibration(tmp_path, monkeypatch) -> None:
    """Exercise `save_checkpoint`/`load_dino_verifier` with a stubbed torch/transformers.

    This covers the on-disk schema/JSON round-trip and MLP state-dict
    save/load plumbing without downloading real DINOv2 weights.
    """
    torch = pytest.importorskip("torch")

    class _FakeBackboneConfig:
        hidden_size = 6

    class _FakeBackbone:
        config = _FakeBackboneConfig()

        def to(self, device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, flag):
            return self

    class _FakeProcessor:
        pass

    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained", lambda model_id: _FakeBackbone()
    )
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained", lambda model_id: _FakeProcessor()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = DinoVerifierConfig(hidden_dim=8, max_score_contribution=0.15)
    head = DinoVerifierHead(config, device="cpu")
    head._load()
    checkpoint_dir = tmp_path / "dino_verifier_ckpt"
    save_checkpoint(checkpoint_dir, head, config, calibration_temperature=1.25)
    assert (checkpoint_dir / "config.json").is_file()
    assert (checkpoint_dir / "mlp_head.pt").is_file()

    loaded_head, loaded_config, loaded_temperature = load_dino_verifier(checkpoint_dir)
    assert loaded_config == config
    assert loaded_temperature == pytest.approx(1.25)
    original_state = head.mlp.state_dict()
    loaded_state = loaded_head.mlp.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], loaded_state[key])


def test_load_dino_verifier_rejects_missing_or_unversioned_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dino_verifier(tmp_path / "does-not-exist")

    bad = tmp_path / "bad_checkpoint"
    bad.mkdir()
    (bad / "mlp_head.pt").write_bytes(b"not a real state dict")
    (bad / "config.json").write_text('{"schema_version": 999}')
    with pytest.raises(ValueError, match="schema"):
        load_dino_verifier(bad)
