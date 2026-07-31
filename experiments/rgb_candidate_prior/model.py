"""Training-only pieces for the ``dino_verifier`` head: loss, splits, calibration.

This module does not define the model architecture -- that is owned by
``blockpose.rgb_verifier.DinoVerifierHead`` (frozen DINOv2 backbone + trainable
MLP). It adds what ``train.py``/``report.py`` need on top:

- a margin/ranking loss over one candidate set at a time (design doc:
  "positive should outrank same-target negatives");
- honest development-fold resolution against
  ``config/rgb_ablation_splits.yaml``, since a sample's ``split_group`` may be
  a session id rather than a raw snapshot name;
- a first-correct-rank helper mirroring
  ``tools/ablate_rgb_edge_prior.py``'s ``_first_correct_rank`` convention;
- a post-hoc temperature-scaling calibration fit so a raw ``[0, 1]`` head
  score becomes the design doc's bounded, calibrated ``rgb_probability``.

Torch is imported at module scope: unlike ``blockpose.rgb_verifier`` (which
must stay importable without torch for the runtime/annotation paths), this
module is only ever used by the GPU-only, ``sam``-extra training scripts.
"""

from __future__ import annotations

import math

import torch

from experiments.rgb_candidate_prior.dataset import CandidateDataset, Sample

_VALID_LABELS = frozenset({"positive", "negative", "ignore"})


def has_valid_pair(labels: list[str]) -> bool:
    """Whether a candidate set has at least one positive and one negative label."""
    labels_set = set(labels)
    return "positive" in labels_set and "negative" in labels_set


def ranking_margin_loss(
    scores: torch.Tensor,
    labels: list[str],
    *,
    margin: float = 0.1,
    log_uncertainties: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise hinge loss over one block's candidate set.

    Every positive score must exceed every negative score by at least
    ``margin``; ``ignore``-labelled samples take no part in the loss. Returns
    a zero tensor with no gradient when the set has no positive/negative pair
    to compare -- callers must check ``has_valid_pair`` before calling
    ``.backward()``, since a detached zero cannot be backpropagated.

    ``log_uncertainties`` (one per candidate, same order as ``scores``) turns
    this into a heteroscedastic loss: each pair's hinge term is scaled by
    ``exp(-mean_log_uncertainty)`` plus that mean added back as a
    regularizer, so a head can lower its loss on a genuinely ambiguous pair by
    reporting higher uncertainty, but pays a linear cost for doing so. Without
    this, ``log_uncertainty`` never receives a gradient and stays at random
    init -- pass it whenever the head's second output column is meant to be
    used for abstention (``score_candidate_with_dino``'s
    ``uncertainty_abstain_threshold``).
    """
    if scores.ndim != 1 or scores.shape[0] != len(labels):
        raise ValueError("scores and labels must be the same length, one per candidate")
    unknown = set(labels) - _VALID_LABELS
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    positive_idx = [i for i, label in enumerate(labels) if label == "positive"]
    negative_idx = [i for i, label in enumerate(labels) if label == "negative"]
    if not positive_idx or not negative_idx:
        return scores.new_zeros(())
    positives = scores[positive_idx]
    negatives = scores[negative_idx]
    diffs = margin - (positives.unsqueeze(1) - negatives.unsqueeze(0))
    hinge = torch.clamp(diffs, min=0.0)
    if log_uncertainties is None:
        return hinge.mean()
    if log_uncertainties.shape != scores.shape:
        raise ValueError("log_uncertainties must have the same shape as scores")
    pos_logvar = log_uncertainties[positive_idx]
    neg_logvar = log_uncertainties[negative_idx]
    pair_logvar = 0.5 * (pos_logvar.unsqueeze(1) + neg_logvar.unsqueeze(0))
    return (hinge * torch.exp(-pair_logvar) + pair_logvar).mean()


def first_correct_rank(scores: list[float], labels: list[str]) -> int | None:
    """Index of the first positive label when ranked by descending score.

    Mirrors ``tools/ablate_rgb_edge_prior.py``'s ``_first_correct_rank``
    convention. ``ignore``-labelled candidates are dropped before ranking.
    Returns ``None`` if no positive label remains to rank against.
    """
    ranked = sorted(
        (
            (score, label)
            for score, label in zip(scores, labels, strict=True)
            if label in ("positive", "negative")
        ),
        key=lambda item: -item[0],
    )
    for index, (_, label) in enumerate(ranked):
        if label == "positive":
            return index
    return None


def development_fold_of_snapshot(manifest: dict) -> dict[str, int]:
    """Map snapshot name -> development fold id from a loaded split manifest.

    Only ``development``-split groups are included; ``final`` snapshots
    (``selection_eligible: false``) are never returned, so callers cannot
    accidentally train or select on them.
    """
    mapping: dict[str, int] = {}
    for group in manifest["groups"].values():
        if group["split"] != "development":
            continue
        for snapshot in group["snapshots"]:
            mapping[snapshot] = group["fold"]
    return mapping


def fold_of_group(samples: list[Sample], snapshot_fold: dict[str, int]) -> int | None:
    """Development fold id for one ``split_group``, resolved via true snapshot names.

    ``split_group`` may be a session id assigned by ``sessions.py``, not a raw
    snapshot name, so this always resolves through each sample's recorded
    ``source.snapshot`` rather than string-matching ``split_group`` against
    the manifest. Returns ``None`` if no member snapshot is in a development
    fold (i.e. the group is entirely ``final``/unlisted data).
    """
    folds = {
        snapshot_fold[snapshot]
        for sample in samples
        if (snapshot := sample.raw.get("source", {}).get("snapshot")) in snapshot_fold
    }
    if not folds:
        return None
    if len(folds) > 1:
        raise ValueError(
            f"split_group spans multiple development folds ({sorted(folds)}); "
            "held-out splitting requires exactly one fold per split_group"
        )
    return next(iter(folds))


def select_development_groups(
    dataset: CandidateDataset, snapshot_fold: dict[str, int]
) -> tuple[CandidateDataset, dict[str, int]]:
    """Restrict a dataset to samples whose ``split_group`` resolves to a development fold.

    Returns the restricted dataset plus the resolved ``{split_group: fold}``
    map, so callers can build per-fold held-out splits with
    ``dataset.split_by_session``.

    A ``split_group`` (session) can mix development and ``final`` snapshots
    if they happen to land in the same session bucket; keeping every sample
    of a group once *any* member resolves to a fold would leak ``final``
    (``selection_eligible: false``) candidates into training. Each sample is
    therefore also checked individually against its own recorded
    ``source.snapshot``, not just its group's resolved fold.
    """
    group_fold: dict[str, int] = {}
    for group, samples in dataset.groups().items():
        fold = fold_of_group(samples, snapshot_fold)
        if fold is not None:
            group_fold[group] = fold
    kept = tuple(
        sample
        for sample in dataset.samples
        if sample.split_group in group_fold
        and sample.raw.get("source", {}).get("snapshot") in snapshot_fold
    )
    return CandidateDataset(root=dataset.root, samples=kept), group_fold


def _logit(probability: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    clamped = probability.clamp(eps, 1.0 - eps)
    return torch.log(clamped / (1.0 - clamped))


def fit_temperature(
    raw_scores: torch.Tensor, binary_labels: torch.Tensor, *, max_iter: int = 200
) -> float:
    """Fit a scalar temperature calibrating ``raw_scores`` against binary labels.

    Post-hoc temperature scaling (Guo et al. 2017): a single scalar is fit on
    a held-out slice that the MLP was not trained on, so it can turn the
    head's raw ``[0, 1]`` score into a calibrated probability without
    touching the frozen backbone or the already-trained MLP weights.
    ``binary_labels`` must be ``1.0`` for a positive candidate and ``0.0`` for
    a negative one; the caller must filter out ``ignore`` samples first.
    """
    if raw_scores.shape != binary_labels.shape:
        raise ValueError("raw_scores and binary_labels must have matching shape")
    if raw_scores.numel() == 0:
        raise ValueError("cannot fit a calibration temperature on an empty held-out slice")
    logits = _logit(raw_scores.detach())
    labels = binary_labels.detach().float()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = loss_fn(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def apply_calibration(raw_score: float, temperature: float) -> float:
    """Calibrated ``rgb_probability`` in ``[0, 1]`` for one raw head score.

    Pure ``math`` (no tensor construction) since this runs per-sample in
    ``report.py`` over potentially many candidates.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    probability = min(max(raw_score, 1e-6), 1.0 - 1e-6)
    logit = math.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + math.exp(-logit / temperature))
