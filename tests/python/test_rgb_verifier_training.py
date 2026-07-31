"""Ranking loss, honest fold splitting, and calibration for the dino_verifier trainer.

These are offline unit tests: no GPU, no network, no real DINOv2 weights. Where
``experiments.rgb_candidate_prior.train`` needs a ``DinoVerifierHead`` instance
(``train_head``/``evaluate_blocks``), ``DinoVerifierHead._load`` is monkeypatched
to install a tiny real ``nn.Sequential`` MLP instead of loading the frozen
backbone, so the actual training/optimizer code path still runs for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from blockpose.rgb_verifier import DinoVerifierConfig, DinoVerifierHead

from experiments.rgb_candidate_prior.dataset import CandidateDataset, Sample, split_by_session
from experiments.rgb_candidate_prior.model import (
    apply_calibration,
    development_fold_of_snapshot,
    first_correct_rank,
    fit_temperature,
    fold_of_group,
    has_valid_pair,
    ranking_margin_loss,
    select_development_groups,
)
from experiments.rgb_candidate_prior.train import embed_all, evaluate_blocks, train_head

# ---------------------------------------------------------------------------
# ranking_margin_loss
# ---------------------------------------------------------------------------


def test_ranking_margin_loss_zero_when_margin_already_satisfied() -> None:
    scores = torch.tensor([0.9, 0.1])
    loss = ranking_margin_loss(scores, ["positive", "negative"], margin=0.1)
    assert float(loss) == pytest.approx(0.0)


def test_ranking_margin_loss_positive_when_margin_violated() -> None:
    scores = torch.tensor([0.5, 0.5])
    loss = ranking_margin_loss(scores, ["positive", "negative"], margin=0.2)
    assert float(loss) == pytest.approx(0.2)


def test_ranking_margin_loss_averages_over_all_pairs() -> None:
    # positives [0.9, 0.5] vs negatives [0.6]; margin 0.1:
    # pair (0.9, 0.6): satisfied -> 0; pair (0.5, 0.6): violated by 0.2 -> 0.2
    scores = torch.tensor([0.9, 0.5, 0.6])
    loss = ranking_margin_loss(scores, ["positive", "positive", "negative"], margin=0.1)
    assert float(loss) == pytest.approx(0.1)


def test_ranking_margin_loss_ignores_ignore_labelled_samples() -> None:
    # the "ignore" sample has the worst score but must not affect the loss
    with_ignore = torch.tensor([0.9, 0.1, 0.0])
    without_ignore = torch.tensor([0.9, 0.1])
    loss_with = ranking_margin_loss(with_ignore, ["positive", "negative", "ignore"], margin=0.1)
    loss_without = ranking_margin_loss(without_ignore, ["positive", "negative"], margin=0.1)
    assert float(loss_with) == pytest.approx(float(loss_without))


def test_ranking_margin_loss_gradient_pushes_positive_up_and_negative_down() -> None:
    scores = torch.tensor([0.5, 0.5], requires_grad=True)
    loss = ranking_margin_loss(scores, ["positive", "negative"], margin=0.2)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0] < 0.0  # gradient descent will increase the positive score
    assert scores.grad[1] > 0.0  # gradient descent will decrease the negative score


def test_ranking_margin_loss_no_valid_pair_returns_non_differentiable_zero() -> None:
    scores = torch.tensor([0.9, 0.8], requires_grad=True)
    loss = ranking_margin_loss(scores, ["positive", "positive"], margin=0.1)
    assert float(loss) == 0.0
    assert not loss.requires_grad
    with pytest.raises(RuntimeError):
        loss.backward()


def test_ranking_margin_loss_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ranking_margin_loss(torch.tensor([0.1, 0.2]), ["positive"])


def test_ranking_margin_loss_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        ranking_margin_loss(torch.tensor([0.1]), ["maybe"])


def test_has_valid_pair() -> None:
    assert has_valid_pair(["positive", "negative", "ignore"])
    assert not has_valid_pair(["positive", "positive"])
    assert not has_valid_pair(["negative", "ignore"])
    assert not has_valid_pair([])


# ---------------------------------------------------------------------------
# first_correct_rank
# ---------------------------------------------------------------------------


def test_first_correct_rank_basic() -> None:
    scores = [0.9, 0.5, 0.1]
    labels = ["negative", "positive", "negative"]
    assert first_correct_rank(scores, labels) == 1


def test_first_correct_rank_top1() -> None:
    assert first_correct_rank([0.9, 0.1], ["positive", "negative"]) == 0


def test_first_correct_rank_none_without_positive() -> None:
    assert first_correct_rank([0.9, 0.1], ["negative", "negative"]) is None


def test_first_correct_rank_drops_ignore_candidates() -> None:
    # the "ignore" sample outranks everything but must not count as the answer
    scores = [0.99, 0.5]
    labels = ["ignore", "positive"]
    assert first_correct_rank(scores, labels) == 0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_apply_calibration_identity_at_temperature_one() -> None:
    assert apply_calibration(0.5, 1.0) == pytest.approx(0.5, abs=1e-6)
    assert apply_calibration(0.8, 1.0) == pytest.approx(0.8, abs=1e-6)


def test_apply_calibration_softens_toward_half_as_temperature_grows() -> None:
    baseline = apply_calibration(0.9, 1.0)
    softened = apply_calibration(0.9, 3.0)
    assert 0.5 < softened < baseline


def test_apply_calibration_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError):
        apply_calibration(0.5, 0.0)


def test_fit_temperature_softens_overconfident_predictions() -> None:
    # extreme confidence but only 6/8 correct -> a well-fit temperature must exceed 1
    raw_scores = torch.tensor([0.97] * 6 + [0.97, 0.03])
    labels = torch.tensor([1.0] * 6 + [0.0, 1.0])
    temperature = fit_temperature(raw_scores, labels)
    assert temperature > 1.0


def test_fit_temperature_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_temperature(torch.tensor([]), torch.tensor([]))


def test_fit_temperature_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_temperature(torch.tensor([0.1, 0.2]), torch.tensor([1.0]))


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


def _sample(
    sample_id: str,
    split_group: str,
    label: str,
    snapshot: str,
    provenance: str = "raw_hypothesis",
) -> Sample:
    return Sample(
        sample_id=sample_id,
        split_group=split_group,
        label=label,
        provenance=provenance,
        rgb_path=Path(f"/nonexistent/{sample_id}/rgb.png"),
        conditioning_path=Path(f"/nonexistent/{sample_id}/cond.png"),
        raw={"source": {"snapshot": snapshot}},
    )


def test_development_fold_of_snapshot_excludes_final() -> None:
    mapping = development_fold_of_snapshot(_MANIFEST)
    assert mapping == {"s1": 0, "s2": 0, "s3": 1}


def test_fold_of_group_resolves_via_source_snapshot() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = [
        _sample("g/1", "session_00", "positive", "s1"),
        _sample("g/2", "session_00", "negative", "s2"),
    ]
    assert fold_of_group(samples, snapshot_fold) == 0


def test_fold_of_group_returns_none_for_unlisted_snapshot() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = [_sample("g/1", "session_00", "positive", "s4")]
    assert fold_of_group(samples, snapshot_fold) is None


def test_fold_of_group_raises_when_group_spans_two_folds() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = [
        _sample("g/1", "session_00", "positive", "s1"),
        _sample("g/2", "session_00", "negative", "s3"),
    ]
    with pytest.raises(ValueError, match="multiple development folds"):
        fold_of_group(samples, snapshot_fold)


def test_select_development_groups_drops_final_only_groups() -> None:
    snapshot_fold = development_fold_of_snapshot(_MANIFEST)
    samples = (
        _sample("g1/blockA/gt", "g1", "positive", "s1"),
        _sample("g1/blockA/raw_0", "g1", "negative", "s2"),
        _sample("g2/blockB/gt", "g2", "positive", "s3"),
        _sample("g3/blockC/gt", "g3", "positive", "s4"),  # final-only snapshot
    )
    dataset = CandidateDataset(root=Path("/nonexistent"), samples=samples)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)

    assert group_fold == {"g1": 0, "g2": 1}
    kept_ids = {sample.sample_id for sample in dev_dataset.samples}
    assert kept_ids == {"g1/blockA/gt", "g1/blockA/raw_0", "g2/blockB/gt"}


# ---------------------------------------------------------------------------
# split_by_session: no candidate-set/session leakage
# ---------------------------------------------------------------------------


def test_split_by_session_holds_out_whole_groups_without_leakage() -> None:
    samples = (
        _sample("g1/blockA/gt", "g1", "positive", "s1"),
        _sample("g1/blockA/raw_0", "g1", "negative", "s1"),
        _sample("g2/blockB/gt", "g2", "positive", "s3"),
        _sample("g2/blockB/raw_0", "g2", "negative", "s3"),
    )
    dataset = CandidateDataset(root=Path("/nonexistent"), samples=samples)
    train, holdout = split_by_session(dataset, {"g2"})

    train_groups = {sample.split_group for sample in train}
    holdout_groups = {sample.split_group for sample in holdout}
    assert train_groups == {"g1"}
    assert holdout_groups == {"g2"}
    assert train_groups.isdisjoint(holdout_groups)

    train_ids = {sample.sample_id for sample in train}
    holdout_ids = {sample.sample_id for sample in holdout}
    assert train_ids.isdisjoint(holdout_ids)

    # every candidate set (block) stays entirely on one side of the split
    train_blocks = CandidateDataset(root=dataset.root, samples=tuple(train)).candidate_sets()
    holdout_blocks = CandidateDataset(root=dataset.root, samples=tuple(holdout)).candidate_sets()
    assert set(train_blocks) == {"g1/blockA"}
    assert set(holdout_blocks) == {"g2/blockB"}


def test_split_by_session_rejects_unknown_holdout_group() -> None:
    samples = (_sample("g1/blockA/gt", "g1", "positive", "s1"),)
    dataset = CandidateDataset(root=Path("/nonexistent"), samples=samples)
    with pytest.raises(ValueError, match="unknown holdout groups"):
        split_by_session(dataset, {"does_not_exist"})


# ---------------------------------------------------------------------------
# train_head / evaluate_blocks / embed_all, with a stubbed (no-network) head
# ---------------------------------------------------------------------------

_HIDDEN_SIZE = 8


def _stub_backbone_load(monkeypatch: pytest.MonkeyPatch, config: DinoVerifierConfig) -> None:
    """Replace `DinoVerifierHead._load` with a real tiny MLP, no torch backbone/network."""

    def fake_load(self: DinoVerifierHead) -> None:
        if self._backbone is not None:
            return
        self.device = "cpu"
        self._backbone = object()  # sentinel: never invoked by train_head/evaluate_blocks
        self._processor = object()
        head = torch.nn.Sequential(
            torch.nn.Linear(_HIDDEN_SIZE * 4, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_dim, 2),
        )
        head.eval()
        self._head = head
        self._torch = torch

    monkeypatch.setattr(DinoVerifierHead, "_load", fake_load)


def _random_features(sample_ids: list[str], seed: int = 0) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        sample_id: rng.normal(size=_HIDDEN_SIZE * 4).astype(np.float32) for sample_id in sample_ids
    }


def test_train_head_only_updates_mlp_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DinoVerifierConfig(hidden_dim=16)
    _stub_backbone_load(monkeypatch, config)

    samples = [
        _sample("block/gt", "g1", "positive", "s1"),
        _sample("block/raw_0", "g1", "negative", "s1"),
        _sample("block/raw_1", "g1", "negative", "s1"),
    ]
    features = _random_features([s.sample_id for s in samples])
    blocks = {"block": samples}

    head, report = train_head(
        config, blocks, features, epochs=5, lr=0.05, margin=0.1, device="cpu", seed=0
    )

    assert report["candidate_sets_usable"] == 1
    assert report["skipped_no_valid_pair"] == 0
    assert len(report["epoch_losses"]) == 5
    assert not head.mlp.training  # train_head leaves the head in eval mode when done
    for parameter in head.mlp.parameters():
        assert parameter.requires_grad


def test_train_head_reduces_ranking_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DinoVerifierConfig(hidden_dim=16)
    _stub_backbone_load(monkeypatch, config)

    samples = [
        _sample(f"block/{i}", "g1", "positive" if i % 2 == 0 else "negative", "s1")
        for i in range(8)
    ]
    features = _random_features([s.sample_id for s in samples])
    blocks = {"block": samples}

    _, report = train_head(
        config, blocks, features, epochs=100, lr=0.05, margin=0.1, device="cpu", seed=0
    )
    assert report["epoch_losses"][-1] <= report["epoch_losses"][0]


def test_train_head_raises_without_any_usable_candidate_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DinoVerifierConfig(hidden_dim=16)
    _stub_backbone_load(monkeypatch, config)
    samples = [_sample("block/gt", "g1", "positive", "s1")]
    features = _random_features([s.sample_id for s in samples])
    with pytest.raises(ValueError, match="no candidate set"):
        train_head(
            config,
            {"block": samples},
            features,
            epochs=1,
            lr=0.01,
            margin=0.1,
            device="cpu",
            seed=0,
        )


def test_evaluate_blocks_reports_top1_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DinoVerifierConfig(hidden_dim=16)
    _stub_backbone_load(monkeypatch, config)
    samples = [
        _sample("block/gt", "g1", "positive", "s1"),
        _sample("block/raw_0", "g1", "negative", "s1"),
    ]
    features = _random_features([s.sample_id for s in samples])
    head, _ = train_head(
        config, {"block": samples}, features, epochs=50, lr=0.05, margin=0.1, device="cpu", seed=0
    )
    report = evaluate_blocks(head.mlp, features, {"block": samples}, "cpu")
    assert report["candidate_sets_evaluated"] == 1
    assert report["top1_accuracy"] in (0.0, 1.0)


def test_embed_all_uses_cache_without_touching_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    config = DinoVerifierConfig(hidden_dim=16)
    samples = [
        _sample("block/gt", "g1", "positive", "s1"),
        _sample("block/raw_0", "g1", "negative", "s1"),
    ]
    cache = tmp_path / "features.npz"
    rng = np.random.default_rng(0)
    expected = {s.sample_id: rng.normal(size=4).astype(np.float32) for s in samples}
    np.savez(
        cache,
        ids=np.array([s.sample_id for s in samples]),
        features=np.stack([expected[s.sample_id] for s in samples]),
    )

    def fail_embed_regions(self, rgb, conditioning):  # pragma: no cover - must not be called
        raise AssertionError("embed_all must use the cache instead of recomputing features")

    monkeypatch.setattr(DinoVerifierHead, "embed_regions", fail_embed_regions)
    head = DinoVerifierHead(config)
    result = embed_all(head, samples, cache)
    assert set(result) == {s.sample_id for s in samples}
