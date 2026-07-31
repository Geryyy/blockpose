#!/usr/bin/env python3
"""Train the ``dino_verifier`` MLP head on exported candidate sets.

Only the trainable MLP on top of the frozen DINOv2 backbone is fit here; the
architecture itself lives in ``blockpose.rgb_verifier.DinoVerifierHead``. This
script never calls ``.backward()`` through the backbone: region features are
extracted once per sample with ``DinoVerifierHead.embed_regions`` (which runs
the backbone under ``torch.inference_mode()`` and returns a detached numpy
array), cached, and only the small MLP is then run in training mode on those
fixed feature vectors.

This is a **first viability pass**. ``config/rgb_ablation_splits.yaml`` has
only two ``development`` folds (``capture_batch_a`` = fold 0,
``capture_batch_b`` = fold 1); ``capture_batch_c``/``capture_batch_d`` are
``final`` and explicitly flagged ``selection_eligible: false``. This script
only ever trains/evaluates on the two development folds, cross-validated both
ways (train on one, hold out the other; then the reverse). The checkpoint
actually written to disk is trained on the union of both development folds --
there is no additional held-out slice left to reserve once both are used for
the final artifact, so the calibration temperature is instead fit on the
pooled cross-validated held-out predictions (each sample scored only by a
head that did not train on its fold).

    python -m experiments.rgb_candidate_prior.train \\
        --export-root out/rgb_candidate_prior \\
        --checkpoint-out out/dino_verifier_checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402
from blockpose.rgb_verifier import (  # noqa: E402
    DinoVerifierConfig,
    DinoVerifierHead,
    save_checkpoint,
)

from experiments.embedding_cache import cached_embed_all  # noqa: E402
from experiments.rgb_candidate_prior.dataset import (  # noqa: E402
    CandidateDataset,
    Sample,
    split_by_session,
)
from experiments.rgb_candidate_prior.model import (  # noqa: E402
    development_fold_of_snapshot,
    first_correct_rank,
    fit_temperature,
    has_valid_pair,
    ranking_margin_loss,
    select_development_groups,
)

DEFAULT_EXPORT_ROOT = REPO_ROOT / "out" / "rgb_candidate_prior"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits.yaml"


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def embed_all(
    head: DinoVerifierHead, samples: list[Sample], cache: Path | None
) -> dict[str, np.ndarray]:
    """Pooled region features per ``sample_id``, cached since the backbone is frozen."""

    def _embed_one(sample: Sample) -> np.ndarray:
        rgb, cond = _load_image(sample.rgb_path), _load_image(sample.conditioning_path)
        return head.embed_regions(rgb, cond)

    return cached_embed_all(samples, cache, embed_one=_embed_one, label="dino_verifier")


def _candidate_sets(root: Path, samples: list[Sample]) -> dict[str, list[Sample]]:
    return CandidateDataset(root=root, samples=tuple(samples)).candidate_sets()


def _scores(
    mlp: torch.nn.Module, features: dict[str, np.ndarray], samples: list[Sample], device: str
) -> torch.Tensor:
    return _scores_and_log_uncertainty(mlp, features, samples, device)[0]


def _scores_and_log_uncertainty(
    mlp: torch.nn.Module, features: dict[str, np.ndarray], samples: list[Sample], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = np.stack([features[sample.sample_id] for sample in samples])
    tensor = torch.from_numpy(stacked).float().to(device)
    raw = mlp(tensor)
    return raw[:, 0], raw[:, 1]


def train_head(
    config: DinoVerifierConfig,
    blocks: dict[str, list[Sample]],
    features: dict[str, np.ndarray],
    *,
    epochs: int,
    lr: float,
    margin: float,
    device: str,
    seed: int,
) -> tuple[DinoVerifierHead, dict]:
    """Train only the MLP head via the ranking loss; the backbone never sees ``.backward()``."""
    torch.manual_seed(seed)
    head = DinoVerifierHead(config, device=device)
    mlp = head.mlp  # public property; triggers backbone+head load
    mlp.train()
    trainable = [p for p in mlp.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("dino_verifier head exposes no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    usable_blocks = {
        block_id: samples
        for block_id, samples in blocks.items()
        if has_valid_pair([sample.label for sample in samples])
    }
    if not usable_blocks:
        raise ValueError(
            "no candidate set in the training split has both a positive and a negative sample"
        )

    epoch_losses = []
    for _ in range(epochs):
        losses = []
        for samples in usable_blocks.values():
            usable = [sample for sample in samples if sample.label in ("positive", "negative")]
            logits, log_uncertainty = _scores_and_log_uncertainty(mlp, features, usable, device)
            scores = torch.sigmoid(logits)
            losses.append(
                ranking_margin_loss(
                    scores,
                    [s.label for s in usable],
                    margin=margin,
                    log_uncertainties=log_uncertainty,
                )
            )
        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_losses.append(batch_loss.item())

    mlp.eval()
    return head, {
        "epochs": epochs,
        "candidate_sets_total": len(blocks),
        "candidate_sets_usable": len(usable_blocks),
        "skipped_no_valid_pair": len(blocks) - len(usable_blocks),
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
    }


def evaluate_blocks(
    mlp: torch.nn.Module,
    features: dict[str, np.ndarray],
    blocks: dict[str, list[Sample]],
    device: str,
) -> dict:
    """First-correct-rank / top-1 accuracy over one held-out set of candidate sets."""
    ranks: dict[str, int | None] = {}
    with torch.no_grad():
        for block_id, samples in blocks.items():
            usable = [sample for sample in samples if sample.label in ("positive", "negative")]
            if not has_valid_pair([s.label for s in usable]):
                continue
            scores = torch.sigmoid(_scores(mlp, features, usable, device)).cpu().numpy().tolist()
            ranks[block_id] = first_correct_rank(scores, [s.label for s in usable])
    evaluated = [rank for rank in ranks.values() if rank is not None]
    top1 = sum(rank == 0 for rank in evaluated)
    return {
        "candidate_sets_evaluated": len(ranks),
        "top1_correct": top1,
        "top1_accuracy": top1 / len(ranks) if ranks else None,
        "ranks_by_block": ranks,
    }


def _pooled_calibration_data(
    mlp: torch.nn.Module,
    features: dict[str, np.ndarray],
    blocks: dict[str, list[Sample]],
    device: str,
) -> tuple[list[float], list[float]]:
    scores, labels = [], []
    with torch.no_grad():
        for samples in blocks.values():
            usable = [sample for sample in samples if sample.label in ("positive", "negative")]
            if not usable:
                continue
            batch_scores = torch.sigmoid(_scores(mlp, features, usable, device)).cpu().numpy()
            scores.extend(batch_scores.tolist())
            labels.extend(1.0 if sample.label == "positive" else 0.0 for sample in usable)
    return scores, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--edge-band-px", type=int, default=6)
    parser.add_argument("--max-score-contribution", type=float, default=0.20)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument(
        "--held-out-fold",
        type=int,
        default=None,
        help="If given, train ONE checkpoint on every development fold EXCEPT this one, "
        "and evaluate/calibrate only on this fold -- a genuinely held-out artifact "
        "(pair with `report.py --held-out-fold`). Without it, the default checkpoint is "
        "trained on the union of both development folds, and the cross-validated numbers "
        "in the summary are the only held-out evidence available for it.",
    )
    return parser.parse_args()


def _train_one_pass(
    config: DinoVerifierConfig,
    features: dict[str, np.ndarray],
    train_blocks: dict[str, list[Sample]],
    eval_blocks: dict[str, list[Sample]] | None,
    *,
    epochs: int,
    lr: float,
    margin: float,
    device: str,
    seed: int,
) -> tuple[DinoVerifierHead, dict, dict | None, list[float], list[float]]:
    """Train one fresh head on ``train_blocks``; evaluate + pool calibration data if given.

    Returns ``(head, train_report, eval_report_or_None, scores, labels)`` --
    ``scores``/``labels`` are ``eval_blocks``' raw predictions for calibration
    fitting (empty lists when ``eval_blocks`` is ``None``, the final pass's
    case, since no held-out slice remains once every fold trains on all data).
    Shared by the held-out-fold path, each cross-validation fold, and the
    final pass -- they differ only in which blocks go where and what happens
    to the returned scores/labels.
    """
    head, train_report = train_head(
        config, train_blocks, features,
        epochs=epochs, lr=lr, margin=margin, device=device, seed=seed,
    )
    if eval_blocks is None:
        return head, train_report, None, [], []
    eval_report = evaluate_blocks(head.mlp, features, eval_blocks, device)
    scores, labels = _pooled_calibration_data(head.mlp, features, eval_blocks, device)
    return head, train_report, eval_report, scores, labels


def main() -> int:
    args = parse_args()
    manifest = load_split_manifest(args.splits)
    snapshot_fold = development_fold_of_snapshot(manifest)

    dataset = CandidateDataset.load(args.export_root)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)
    folds = sorted(set(group_fold.values()))
    if len(folds) < 2:
        print(
            f"WARNING: only {len(folds)} development fold(s) resolved from the exported data; "
            "cross-validated held-out reporting needs at least 2.",
            file=sys.stderr,
        )
    if not dev_dataset.samples:
        raise ValueError(
            "no exported sample resolves to a development fold in "
            f"{args.splits} -- nothing to train or evaluate on"
        )

    config = DinoVerifierConfig(
        model_id=args.model_id,
        edge_band_px=args.edge_band_px,
        hidden_dim=args.hidden_dim,
        max_score_contribution=args.max_score_contribution,
    )
    # Resolve "auto" up front: `embed_all` may hit its feature cache and skip
    # `DinoVerifierHead._load()` entirely, which is the only thing that would
    # otherwise turn "auto" into a concrete "cpu"/"cuda" on `embedder.device`.
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = DinoVerifierHead(config, device=device)
    cache = args.feature_cache or (args.export_root / "dino_verifier_features.npz")
    features = embed_all(embedder, list(dev_dataset.samples), cache)

    if args.held_out_fold is not None:
        if args.held_out_fold not in folds:
            raise ValueError(
                f"--held-out-fold {args.held_out_fold} not among resolved folds {folds}"
            )
        holdout_groups = {group for group, fold in group_fold.items() if fold == args.held_out_fold}
        train_samples, eval_samples = split_by_session(dev_dataset, holdout_groups)
        train_blocks = _candidate_sets(dev_dataset.root, train_samples)
        eval_blocks = _candidate_sets(dev_dataset.root, eval_samples)

        held_out_head, train_report, eval_report, scores, labels = _train_one_pass(
            config, features, train_blocks, eval_blocks,
            epochs=args.epochs, lr=args.lr, margin=args.margin, device=device, seed=args.seed,
        )
        if scores:
            temperature = fit_temperature(torch.tensor(scores), torch.tensor(labels))
        else:
            temperature = 1.0
            print(
                "WARNING: held-out fold has no scorable candidate set; "
                "calibration_temperature defaults to 1.0 (uncalibrated)",
                file=sys.stderr,
            )
        args.checkpoint_out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(args.checkpoint_out, held_out_head, config, temperature)

        summary = {
            "development_only_pass": True,
            "mode": "single_held_out_fold",
            "held_out_fold": args.held_out_fold,
            "note": (
                "This checkpoint was trained excluding development fold "
                f"{args.held_out_fold}; `eval` below is a genuinely held-out estimate for "
                "exactly this checkpoint. `final` groups (capture_batch_c/d) remain untouched."
            ),
            "export_root": str(args.export_root),
            "checkpoint_out": str(args.checkpoint_out),
            "development_folds": folds,
            "train_snapshot_groups": sorted(set(s.split_group for s in train_samples)),
            "eval_snapshot_groups": sorted(set(s.split_group for s in eval_samples)),
            "calibration_temperature": temperature,
            "calibration_fit_on": "this_folds_true_held_out_predictions",
            "train": train_report,
            "eval": eval_report,
        }
        print(json.dumps(summary, indent=2))
        if args.summary_out is not None:
            args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        return 0

    cross_validation = {}
    pooled_scores: list[float] = []
    pooled_labels: list[float] = []
    for held_out in folds:
        holdout_groups = {group for group, fold in group_fold.items() if fold == held_out}
        train_samples, eval_samples = split_by_session(dev_dataset, holdout_groups)
        train_blocks = _candidate_sets(dev_dataset.root, train_samples)
        eval_blocks = _candidate_sets(dev_dataset.root, eval_samples)

        head, train_report, eval_report, scores, labels = _train_one_pass(
            config, features, train_blocks, eval_blocks,
            epochs=args.epochs, lr=args.lr, margin=args.margin, device=device, seed=args.seed,
        )
        cross_validation[f"fold_{held_out}_held_out"] = {
            "train_snapshot_groups": sorted(set(s.split_group for s in train_samples)),
            "eval_snapshot_groups": sorted(set(s.split_group for s in eval_samples)),
            "train": train_report,
            "eval": eval_report,
        }
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)

    # The deployed checkpoint is trained on every development sample: once
    # both folds are used there, no further held-out slice remains, so
    # calibration is fit on the pooled cross-validated held-out predictions
    # above instead (each prediction came from a head that never saw that
    # sample's fold during training).
    final_blocks = _candidate_sets(dev_dataset.root, list(dev_dataset.samples))
    final_head, final_train_report, _, _, _ = _train_one_pass(
        config, features, final_blocks, None,
        epochs=args.epochs, lr=args.lr, margin=args.margin, device=device, seed=args.seed,
    )

    if pooled_scores:
        temperature = fit_temperature(torch.tensor(pooled_scores), torch.tensor(pooled_labels))
    else:
        temperature = 1.0
        print(
            "WARNING: no pooled held-out predictions available; "
            "calibration_temperature defaults to 1.0 (uncalibrated)",
            file=sys.stderr,
        )

    args.checkpoint_out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint_out, final_head, config, temperature)

    summary = {
        "development_only_pass": True,
        "note": (
            "First viability pass: only the two `development` folds in "
            f"{args.splits} (capture_batch_a=fold 0, capture_batch_b=fold 1) were used. "
            "`final` groups (capture_batch_c/d) are selection_eligible: false and were "
            "never trained or evaluated on."
        ),
        "export_root": str(args.export_root),
        "checkpoint_out": str(args.checkpoint_out),
        "development_folds": folds,
        "candidate_sets_total": len(final_blocks),
        "calibration_temperature": temperature,
        "calibration_fit_on": "pooled_cross_validated_held_out_predictions",
        "final_model_train_report": final_train_report,
        "cross_validation": cross_validation,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out is not None:
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
