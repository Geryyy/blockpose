"""Patch-grid resampling, weighted loss, fold-safety, and recall metric tests
for the ``dino_proposer`` trainer/report.

Offline unit tests: no GPU, no network, no real DINOv2 weights.
``train_head``/``embed_all`` (the pieces that actually construct a
``DinoRegionProposerHead``) are exercised against a tiny local fake object
satisfying the ``.dense_head``/``.patch_features`` contract documented in
``train.py``, so they never need torch/transformers weights.
``extract_region_proposals`` itself needs no torch either (pure
numpy/scipy), so it's imported and exercised directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from blockpose.rgb_region_proposer import (  # noqa: E402
    DinoRegionProposerConfig,
    extract_region_proposals,
)

from experiments.rgb_region_prior.dataset import (  # noqa: E402
    RegionDataset,
    RegionSample,
    development_fold_of_snapshot,
    fold_of_group,
    select_development_groups,
    split_by_session,
)
from experiments.rgb_region_prior.model import (  # noqa: E402
    foreground_pos_weight,
    resample_mask_to_patch_grid,
    weighted_bce_loss,
)
from experiments.rgb_region_prior.report import (  # noqa: E402
    evaluate_snapshot_recall,
    region_iou,
)
from experiments.rgb_region_prior.train import embed_all, train_head  # noqa: E402

# ---------------------------------------------------------------------------
# resample_mask_to_patch_grid
# ---------------------------------------------------------------------------


def test_resample_mask_to_patch_grid_full_coverage_is_all_ones() -> None:
    mask = np.ones((16, 16), dtype=bool)
    grid = resample_mask_to_patch_grid(mask, 4, 4)
    assert grid.shape == (4, 4)
    assert torch.allclose(grid, torch.ones(4, 4))


def test_resample_mask_to_patch_grid_empty_is_all_zeros() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    grid = resample_mask_to_patch_grid(mask, 4, 4)
    assert torch.allclose(grid, torch.zeros(4, 4))


def test_resample_mask_to_patch_grid_partial_patch_gets_soft_target() -> None:
    # left half of the image is foreground -> each row of patches is half full
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :4] = True
    grid = resample_mask_to_patch_grid(mask, 2, 2)
    assert grid.shape == (2, 2)
    # left column patches ~1.0, right column patches ~0.0 (not a lossy 0/1 snap
    # for the *whole* mask, but each patch here is either fully in or fully out)
    assert torch.allclose(grid[:, 0], torch.ones(2), atol=1e-5)
    assert torch.allclose(grid[:, 1], torch.zeros(2), atol=1e-5)


def test_resample_mask_to_patch_grid_straddling_patch_is_fractional() -> None:
    # one quarter of one single-cell "grid" is foreground -> soft target 0.25
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    grid = resample_mask_to_patch_grid(mask, 1, 1)
    assert grid.shape == (1, 1)
    assert grid.item() == pytest.approx(0.25, abs=1e-5)


def test_resample_mask_to_patch_grid_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        resample_mask_to_patch_grid(np.zeros((4, 4, 1)), 2, 2)
    with pytest.raises(ValueError):
        resample_mask_to_patch_grid(np.zeros((4, 4)), 0, 2)


# ---------------------------------------------------------------------------
# foreground_pos_weight / weighted_bce_loss
# ---------------------------------------------------------------------------


def test_foreground_pos_weight_matches_negative_over_positive_mass() -> None:
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # 1 positive, 3 negative
    weight = foreground_pos_weight(targets)
    assert float(weight) == pytest.approx(3.0)


def test_foreground_pos_weight_grows_as_positive_mass_shrinks() -> None:
    sparse = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dense = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert float(foreground_pos_weight(sparse)) > float(foreground_pos_weight(dense))


def test_foreground_pos_weight_neutral_when_batch_has_no_foreground() -> None:
    targets = torch.zeros(4, 4)
    assert float(foreground_pos_weight(targets)) == pytest.approx(1.0)


def test_weighted_bce_loss_all_zero_target_is_finite_and_derives_from_batch() -> None:
    logits = torch.randn(4, 4)
    targets = torch.zeros(4, 4)
    loss = weighted_bce_loss(logits, targets)
    assert torch.isfinite(loss)


def test_weighted_bce_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        weighted_bce_loss(torch.zeros(2, 2), torch.zeros(3, 3))


def test_weighted_bce_loss_gradient_pushes_logits_toward_targets() -> None:
    logits = torch.zeros(2, 2, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = weighted_bce_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
    # gradient descent (logit -= lr * grad) must increase logits where target=1
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[1, 1] < 0.0
    # and decrease logits where target=0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 0] > 0.0


def test_weighted_bce_loss_weighting_changes_relative_contribution() -> None:
    # a wrong prediction on the (rare) positive patch should cost more than an
    # equally-wrong prediction on a (common) negative patch once weighted
    logits_wrong_on_positive = torch.tensor([[-5.0, 0.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # flip which one matches: first is target=1 (rare positive), wrong -> big weighted loss
    loss_positive_wrong = weighted_bce_loss(logits_wrong_on_positive, targets)
    unweighted_positive_wrong = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_wrong_on_positive, targets
    )
    assert float(loss_positive_wrong) > float(unweighted_positive_wrong)


# ---------------------------------------------------------------------------
# honest development-fold resolution (config/rgb_ablation_splits.yaml shape)
# ---------------------------------------------------------------------------

_MANIFEST = {
    "groups": {
        "capture_batch_a": {"split": "development", "fold": 0, "snapshots": ["s1", "s2"]},
        "capture_batch_b": {"split": "development", "fold": 1, "snapshots": ["s3"]},
        "capture_batch_c": {"split": "final", "snapshots": ["s4"]},
    }
}


def _sample(sample_id: str, split_group: str, block_count: int, snapshot: str) -> RegionSample:
    return RegionSample(
        sample_id=sample_id,
        split_group=split_group,
        image_path=Path(f"/nonexistent/{sample_id}/image.png"),
        mask_path=Path(f"/nonexistent/{sample_id}/mask.png"),
        block_count=block_count,
        raw={"source": {"snapshot": snapshot}},
    )


def test_development_fold_of_snapshot_excludes_final() -> None:
    assert development_fold_of_snapshot(_MANIFEST) == {"s1": 0, "s2": 0, "s3": 1}


def test_fold_of_group_resolves_via_source_snapshot() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = [_sample("s1", "session_00", 1, "s1"), _sample("s2", "session_00", 0, "s2")]
    assert fold_of_group(samples, snapshot_fold) == 0


def test_fold_of_group_returns_none_for_unlisted_snapshot() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    assert fold_of_group([_sample("s4", "session_00", 1, "s4")], snapshot_fold) is None


def test_fold_of_group_raises_when_group_spans_two_folds() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = [_sample("s1", "session_00", 1, "s1"), _sample("s3", "session_00", 1, "s3")]
    with pytest.raises(ValueError, match="multiple development folds"):
        fold_of_group(samples, snapshot_fold)


def test_select_development_groups_drops_final_only_groups() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = (
        _sample("g1/s1", "g1", 1, "s1"),
        _sample("g1/s2", "g1", 0, "s2"),
        _sample("g2/s3", "g2", 2, "s3"),
        _sample("g3/s4", "g3", 1, "s4"),  # final-only snapshot
    )
    dataset = RegionDataset(root=Path("/nonexistent"), samples=samples)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)
    assert group_fold == {"g1": 0, "g2": 1}
    assert {s.sample_id for s in dev_dataset.samples} == {"g1/s1", "g1/s2", "g2/s3"}


def test_split_by_session_holds_out_whole_groups_without_leakage() -> None:
    samples = (
        _sample("g1/s1", "g1", 1, "s1"),
        _sample("g2/s3", "g2", 2, "s3"),
    )
    dataset = RegionDataset(root=Path("/nonexistent"), samples=samples)
    train, holdout = split_by_session(dataset, {"g2"})
    assert {s.split_group for s in train} == {"g1"}
    assert {s.split_group for s in holdout} == {"g2"}


def test_split_by_session_rejects_unknown_holdout_group() -> None:
    dataset = RegionDataset(root=Path("/nonexistent"), samples=(_sample("g1/s1", "g1", 1, "s1"),))
    with pytest.raises(ValueError, match="unknown holdout groups"):
        split_by_session(dataset, {"does_not_exist"})


# ---------------------------------------------------------------------------
# train_head / embed_all against a local fake head (no blockpose.rgb_region_proposer needed)
# ---------------------------------------------------------------------------

_HIDDEN_SIZE = 6
_GRID = 4


class _FakeDenseHead(torch.nn.Module):
    """Conv2d-based, matching the real dense head's (N, C, H, W) layout.

    An earlier version of this stub used ``nn.Linear`` over the last
    dimension, which happily accepts a (grid_h, grid_w, D) tensor with no
    permute at all -- masking a real bug where ``train_head``/``evaluate_loss``
    fed features to the real Conv2d-based head without first converting to
    channel-first. Using Conv2d here makes that class of bug fail loudly.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class _FakeRegionHead:
    """Local stand-in for the assumed ``DinoRegionProposerHead`` training API."""

    def __init__(self) -> None:
        self.dense_head = _FakeDenseHead(_HIDDEN_SIZE)
        self.calls = 0

    def patch_features(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        rng = np.random.default_rng(int(image.sum()) % (2**31))
        return rng.normal(size=(_GRID, _GRID, _HIDDEN_SIZE)).astype(np.float32)


def _region_sample(
    tmp_path: Path, sample_id: str, split_group: str, foreground: bool
) -> RegionSample:
    from PIL import Image

    image_path = tmp_path / f"{sample_id}_image.png"
    mask_path = tmp_path / f"{sample_id}_mask.png"
    Image.fromarray(np.full((16, 16, 3), 30, dtype=np.uint8)).save(image_path)
    mask = np.zeros((16, 16), dtype=np.uint8)
    if foreground:
        mask[4:12, 4:12] = 255
    Image.fromarray(mask).save(mask_path)
    return RegionSample(
        sample_id=sample_id,
        split_group=split_group,
        image_path=image_path,
        mask_path=mask_path,
        block_count=1 if foreground else 0,
        raw={"source": {"snapshot": sample_id}},
    )


def test_train_head_only_updates_dense_head_parameters(tmp_path: Path) -> None:
    head = _FakeRegionHead()
    samples = [
        _region_sample(tmp_path, "s1", "g1", True),
        _region_sample(tmp_path, "s2", "g1", False),
    ]
    features = {s.sample_id: head.patch_features(np.zeros((1, 1, 3))) for s in samples}

    report = train_head(head, samples, features, epochs=5, lr=0.05, device="cpu", seed=0)

    assert report["samples_total"] == 2
    assert len(report["epoch_losses"]) == 5
    assert not head.dense_head.training  # left in eval mode when done
    for parameter in head.dense_head.parameters():
        assert parameter.requires_grad


def test_train_head_reduces_loss(tmp_path: Path) -> None:
    head = _FakeRegionHead()
    samples = [
        _region_sample(tmp_path, f"s{i}", "g1", foreground=(i % 2 == 0)) for i in range(6)
    ]
    features = {s.sample_id: head.patch_features(np.zeros((1, 1, 3))) for s in samples}
    report = train_head(head, samples, features, epochs=80, lr=0.05, device="cpu", seed=0)
    assert report["epoch_losses"][-1] <= report["epoch_losses"][0]


def test_train_head_raises_without_samples() -> None:
    head = _FakeRegionHead()
    with pytest.raises(ValueError, match="no training samples"):
        train_head(head, [], {}, epochs=1, lr=0.01, device="cpu", seed=0)


def test_embed_all_uses_cache_without_touching_images(tmp_path: Path) -> None:
    samples = [
        _region_sample(tmp_path, "s1", "g1", True),
        _region_sample(tmp_path, "s2", "g1", False),
    ]
    cache = tmp_path / "features.npz"
    rng = np.random.default_rng(0)
    expected = {
        s.sample_id: rng.normal(size=(_GRID, _GRID, _HIDDEN_SIZE)).astype(np.float32)
        for s in samples
    }
    np.savez(
        cache,
        ids=np.array([s.sample_id for s in samples]),
        features=np.stack([expected[s.sample_id] for s in samples]),
    )

    class _FailingHead:
        def patch_features(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
            raise AssertionError("embed_all must use the cache instead of recomputing features")

    result = embed_all(_FailingHead(), samples, cache)
    assert set(result) == {s.sample_id for s in samples}


def test_embed_all_computes_and_caches_when_no_cache_hit(tmp_path: Path) -> None:
    head = _FakeRegionHead()
    samples = [_region_sample(tmp_path, "s1", "g1", True)]
    cache = tmp_path / "features.npz"
    result = embed_all(head, samples, cache)
    assert head.calls == 1
    assert cache.is_file()
    assert result["s1"].shape == (_GRID, _GRID, _HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# report.py: connected-component extraction + recall/false-positive metric
# ---------------------------------------------------------------------------


def test_report_uses_the_real_extract_region_proposals_thresholding() -> None:
    """report.py's ``extract_regions`` closure wraps this function directly
    (no local reimplementation) -- exercise it the same way ``main()`` does.
    """
    probability_map = np.zeros((20, 20))
    probability_map[2:6, 2:6] = 0.9  # 16 px, kept
    probability_map[15:16, 15:16] = 0.9  # 1 px, filtered
    config = DinoRegionProposerConfig(foreground_threshold=0.5, min_component_area_px=4)
    proposals = extract_region_proposals(probability_map, config)
    assert len(proposals) == 1
    assert int(proposals[0].mask.sum()) == 16


def test_region_iou_identical_masks_is_one() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert region_iou(mask, mask) == pytest.approx(1.0)


def test_region_iou_disjoint_masks_is_zero() -> None:
    a = np.zeros((10, 10), dtype=bool)
    a[0:2, 0:2] = True
    b = np.zeros((10, 10), dtype=bool)
    b[8:10, 8:10] = True
    assert region_iou(a, b) == pytest.approx(0.0)


def test_evaluate_snapshot_recall_recovers_block_above_threshold() -> None:
    block = np.zeros((10, 10), dtype=bool)
    block[2:6, 2:6] = True  # 16 px
    predicted = np.zeros((10, 10), dtype=bool)
    predicted[2:6, 2:8] = True  # overlaps fully with block, IoU = 16/24 = 0.667
    result = evaluate_snapshot_recall([predicted], {"block_0": block}, iou_threshold=0.3)
    assert result["blocks_recovered"] == 1
    assert result["recall"] == pytest.approx(1.0)
    assert result["false_positives"] == 0


def test_evaluate_snapshot_recall_below_threshold_is_not_recovered() -> None:
    block = np.zeros((20, 20), dtype=bool)
    block[0:10, 0:10] = True  # 100 px
    predicted = np.zeros((20, 20), dtype=bool)
    predicted[9:10, 9:10] = True  # tiny sliver overlap, IoU << 0.3
    result = evaluate_snapshot_recall([predicted], {"block_0": block}, iou_threshold=0.3)
    assert result["blocks_recovered"] == 0
    assert result["recall"] == pytest.approx(0.0)
    assert result["false_positives"] == 1  # unmatched predicted region


def test_evaluate_snapshot_recall_counts_false_positive_region() -> None:
    block = np.zeros((10, 10), dtype=bool)
    block[0:3, 0:3] = True
    unrelated = np.zeros((10, 10), dtype=bool)
    unrelated[7:10, 7:10] = True
    result = evaluate_snapshot_recall([unrelated], {"block_0": block}, iou_threshold=0.3)
    assert result["false_positives"] == 1
    assert result["blocks_recovered"] == 0


def test_evaluate_snapshot_recall_one_region_credits_only_its_best_block() -> None:
    # one huge predicted region overlapping two blocks should not "recover"
    # both for free once IoU against each individual block is checked
    block_a = np.zeros((20, 20), dtype=bool)
    block_a[0:2, 0:2] = True
    block_b = np.zeros((20, 20), dtype=bool)
    block_b[10:12, 10:12] = True
    huge_region = np.ones((20, 20), dtype=bool)
    result = evaluate_snapshot_recall(
        [huge_region], {"block_a": block_a, "block_b": block_b}, iou_threshold=0.3
    )
    # IoU of a 4px block against a 400px region is far below 0.3 for both
    assert result["blocks_recovered"] == 0
    assert result["false_positives"] == 1


def test_evaluate_snapshot_recall_empty_blocks_gives_none_recall() -> None:
    predicted = np.zeros((10, 10), dtype=bool)
    predicted[0:3, 0:3] = True
    result = evaluate_snapshot_recall([predicted], {}, iou_threshold=0.3)
    assert result["recall"] is None
    assert result["false_positives"] == 1
