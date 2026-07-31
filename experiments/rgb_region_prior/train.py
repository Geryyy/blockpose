#!/usr/bin/env python3
"""Train the ``dino_proposer`` dense head on exported per-snapshot union masks.

Only the trainable dense head on top of the frozen DINOv2 backbone is fit
here; the architecture itself lives in
``blockpose.rgb_region_proposer.DinoRegionProposerHead``. This script never
calls ``.backward()`` through the backbone: patch-grid features are extracted
once per snapshot via ``head.patch_features(image)``, cached, and only the
small dense head (``head.dense_head``) is then run in training mode on those
fixed feature grids.

The ``blockpose.rgb_region_proposer`` import itself is deferred into
``main()`` (not module scope) so this script -- and its tests -- stay
importable without torch/transformers installed; construction is otherwise
mirrored from ``experiments.rgb_candidate_prior.train``'s
``development_fold_of_snapshot``/fold-safe cross-validation structure.

    python -m experiments.rgb_region_prior.train \\
        --export-root out/rgb_region_prior \\
        --checkpoint-out out/dino_proposer_checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402

from experiments.embedding_cache import cached_embed_all  # noqa: E402
from experiments.rgb_region_prior.dataset import (  # noqa: E402
    RegionDataset,
    RegionSample,
    development_fold_of_snapshot,
    select_development_groups,
    split_by_session,
)
from experiments.rgb_region_prior.model import (  # noqa: E402
    resample_mask_to_patch_grid,
    weighted_bce_loss,
)

if TYPE_CHECKING:
    from blockpose.rgb_region_proposer import DinoRegionProposerConfig, DinoRegionProposerHead

DEFAULT_EXPORT_ROOT = REPO_ROOT / "out" / "rgb_region_prior"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits.yaml"


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 0


def embed_all(
    head: "DinoRegionProposerHead", samples: list[RegionSample], cache: Path | None
) -> dict[str, np.ndarray]:
    """Frozen patch-grid features per ``sample_id``, cached since the backbone is frozen."""

    def _embed_one(sample: RegionSample) -> np.ndarray:
        return np.asarray(head.patch_features(_load_image(sample.image_path)))

    return cached_embed_all(samples, cache, embed_one=_embed_one, label="dino_proposer")


def _targets_for(
    samples: list[RegionSample], features: dict[str, np.ndarray]
) -> dict[str, torch.Tensor]:
    targets = {}
    for sample in samples:
        grid_h, grid_w = features[sample.sample_id].shape[:2]
        targets[sample.sample_id] = resample_mask_to_patch_grid(
            _load_mask(sample.mask_path), grid_h, grid_w
        )
    return targets


def mass_normalized_weak_weights(
    real_samples: list[RegionSample],
    weak_samples: list[RegionSample],
    mass_fraction: float,
) -> dict[str, float]:
    """Per-weak-sample weight so total weak mass is exactly ``mass_fraction`` of real mass.

    Real samples always have an implicit weight of 1.0, so their total mass is
    ``len(real_samples)``. Weak mass is fixed at ``mass_fraction`` of that,
    split evenly across however many weak samples are given -- unlike a flat
    per-sample weight, the weak contribution can never dominate just because
    there happen to be many more weak samples than real ones. Empty inputs
    (no weak samples, or no real samples to normalize against) return ``{}``,
    which ``train_head`` treats as "no weak samples in this call".
    """
    if not weak_samples or not real_samples:
        return {}
    per_sample_weight = (mass_fraction * len(real_samples)) / len(weak_samples)
    return {sample.sample_id: per_sample_weight for sample in weak_samples}


def train_head(
    head: "DinoRegionProposerHead",
    samples: list[RegionSample],
    features: dict[str, np.ndarray],
    *,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
    sample_weights: dict[str, float] | None = None,
) -> dict:
    """Train ``head``'s dense head in place via weighted BCE; the backbone never sees a gradient.

    Mirrors ``experiments.rgb_candidate_prior.train``'s ``train_head`` shape
    (there: a ranking loss over a ``DinoVerifierHead``'s MLP; here: a dense
    per-patch loss over a ``DinoRegionProposerHead``'s dense head), including
    leaving the head in ``eval()`` mode when done.

    ``sample_weights`` (``sample_id -> loss weight``) is only used to blend in
    weak-anchor samples at a subordinate weight (see ``mass_normalized_weak_weights``
    and ``--weak-mass-fraction`` in ``main()``). Left ``None`` (the default, and
    the only path exercised when ``--weak-export-root`` is not given), the batch loss is exactly
    ``torch.stack(losses).mean()`` as before -- byte-for-byte identical to the
    pre-weak-supervision behavior. A sample missing from ``sample_weights``
    defaults to weight ``1.0`` (a real sample).
    """
    if not samples:
        raise ValueError("no training samples given")
    torch.manual_seed(seed)
    dense_head = head.dense_head
    dense_head.train()
    trainable = [p for p in dense_head.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("dino_proposer head exposes no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    targets = _targets_for(samples, features)

    epoch_losses = []
    for _ in range(epochs):
        losses = []
        for sample in samples:
            feature_tensor = torch.from_numpy(features[sample.sample_id]).float().to(device)
            # dense_head is Conv2d-based: (grid_h, grid_w, D) -> (1, D, grid_h, grid_w),
            # matching blockpose.rgb_region_proposer.DinoRegionProposerHead.predict_proposal_map.
            feature_tensor = feature_tensor.permute(2, 0, 1).unsqueeze(0)
            target = targets[sample.sample_id].to(device)
            logits = dense_head(feature_tensor).reshape(target.shape)
            losses.append(weighted_bce_loss(logits, target))
        if sample_weights is None:
            batch_loss = torch.stack(losses).mean()
        else:
            weight_tensor = torch.tensor(
                [sample_weights.get(sample.sample_id, 1.0) for sample in samples],
                dtype=torch.float32,
                device=device,
            )
            batch_loss = (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_losses.append(float(batch_loss.item()))

    dense_head.eval()
    return {
        "epochs": epochs,
        "samples_total": len(samples),
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epoch_losses": epoch_losses,
    }


def evaluate_loss(
    head: "DinoRegionProposerHead",
    samples: list[RegionSample],
    features: dict[str, np.ndarray],
    device: str,
) -> dict:
    """Mean held-out weighted-BCE loss; a lightweight development signal only.

    The design doc's actual promotion metric (per-block recall / false
    positives after thresholding + connected-component extraction) lives in
    ``report.py``, which needs a saved checkpoint and the runtime inference
    path -- this is just enough signal to sanity-check a fold during training.
    """
    if not samples:
        return {"samples_evaluated": 0, "mean_loss": None}
    dense_head = head.dense_head
    dense_head.eval()
    targets = _targets_for(samples, features)
    losses = []
    with torch.no_grad():
        for sample in samples:
            feature_tensor = torch.from_numpy(features[sample.sample_id]).float().to(device)
            # dense_head is Conv2d-based: (grid_h, grid_w, D) -> (1, D, grid_h, grid_w),
            # matching blockpose.rgb_region_proposer.DinoRegionProposerHead.predict_proposal_map.
            feature_tensor = feature_tensor.permute(2, 0, 1).unsqueeze(0)
            target = targets[sample.sample_id].to(device)
            logits = dense_head(feature_tensor).reshape(target.shape)
            losses.append(float(weighted_bce_loss(logits, target).item()))
    return {"samples_evaluated": len(samples), "mean_loss": float(np.mean(losses))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=896)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--min-component-area-px", type=int, default=64)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument(
        "--held-out-fold",
        type=int,
        default=None,
        help="If given, train ONE checkpoint on every development fold EXCEPT this one, "
        "and evaluate only on this fold -- a genuinely held-out artifact (pair with "
        "`report.py --held-out-fold`). Without it, the default checkpoint is trained on "
        "the union of both development folds.",
    )
    parser.add_argument(
        "--weak-export-root",
        type=Path,
        default=None,
        help="Optional export root from `export_weak.py` (weak anchor union masks over "
        "data/seibersdorf_dataset). When given, weak samples are blended into every fold's "
        "TRAINING pool only, mass-normalized per --weak-mass-fraction, and never into "
        "held-out evaluation. Without it (the default), training is byte-for-byte identical "
        "to before this flag existed.",
    )
    parser.add_argument(
        "--weak-mass-fraction",
        type=float,
        default=0.3,
        help="Total weak-sample training mass, as a fraction of the real samples' mass "
        "(mass = sample count x weight), in EVERY individual train_head() call -- not a flat "
        "per-sample weight. A flat per-sample weight can still let weak data dominate purely "
        "because there are more of it: 210 weak samples at a flat 0.3 weight actually "
        "outweighed 19 real samples 3.3x in total mass on the first attempt at this, which "
        "made the trained head worse, not better. This fixes the weak contribution as a "
        "fraction of whatever real-sample count is actually in THIS call (which varies "
        "between the held-out-fold path, each cross-validation fold, and the final union "
        "pass), so it can never accidentally dominate regardless of how many weak samples "
        "exist. Ignored unless --weak-export-root is given.",
    )
    parser.add_argument(
        "--weak-feature-cache",
        type=Path,
        default=None,
        help="Feature cache for weak samples (default: <weak-export-root>/"
        "dino_proposer_features.npz). Kept separate from --feature-cache so weak and real "
        "caches never collide.",
    )
    return parser.parse_args()


def _train_one_pass(
    config: "DinoRegionProposerConfig",
    device: str,
    train_samples: list[RegionSample],
    weak_samples: list[RegionSample],
    features: dict[str, np.ndarray],
    *,
    epochs: int,
    lr: float,
    seed: int,
    weak_mass_fraction: float,
    eval_samples: list[RegionSample] | None,
) -> tuple["DinoRegionProposerHead", dict, dict | None, dict[str, float]]:
    """Train one fresh head on ``train_samples`` (+ mass-normalized weak samples).

    Evaluates on ``eval_samples`` if given (``None`` for the final union pass,
    which has no held-out slice left). Shared by the held-out-fold path, each
    cross-validation fold, and the final pass -- they differ only in which
    samples go where.
    """
    from blockpose.rgb_region_proposer import DinoRegionProposerHead

    sample_weights = mass_normalized_weak_weights(train_samples, weak_samples, weak_mass_fraction)
    head = DinoRegionProposerHead(config, device=device)
    train_report = train_head(
        head,
        train_samples + weak_samples,
        features,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        sample_weights=sample_weights or None,
    )
    eval_report = (
        evaluate_loss(head, eval_samples, features, device) if eval_samples is not None else None
    )
    return head, train_report, eval_report, sample_weights


def main() -> int:
    # Deferred (not module scope) so this script -- and its train_head/embed_all
    # unit tests -- stay importable without torch/transformers installed.
    from blockpose.rgb_region_proposer import (
        DinoRegionProposerConfig,
        DinoRegionProposerHead,
        save_checkpoint,
    )

    args = parse_args()
    manifest = load_split_manifest(args.splits)
    snapshot_fold = development_fold_of_snapshot(manifest)

    dataset = RegionDataset.load(args.export_root)
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
            f"no exported sample resolves to a development fold in {args.splits} -- nothing "
            "to train or evaluate on"
        )

    config = DinoRegionProposerConfig(
        model_id=args.model_id,
        hidden_dim=args.hidden_dim,
        max_image_side=args.max_image_side,
        foreground_threshold=args.foreground_threshold,
        min_component_area_px=args.min_component_area_px,
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = DinoRegionProposerHead(config, device=device)
    cache = args.feature_cache or (args.export_root / "dino_proposer_features.npz")
    features = embed_all(embedder, list(dev_dataset.samples), cache)

    # Weak-anchor blending: strictly additive to the TRAINING side only. When
    # --weak-export-root is not given, weak_samples stays empty, every
    # mass_normalized_weak_weights() call below returns {} (== None-equivalent
    # for train_head), and every train_head() call takes the exact same code
    # path as before this flag existed.
    weak_samples: list[RegionSample] = []
    if args.weak_export_root is not None:
        weak_dataset = RegionDataset.load(args.weak_export_root)
        weak_samples = list(weak_dataset.samples)
        real_ids = {s.sample_id for s in dev_dataset.samples}
        collisions = {s.sample_id for s in weak_samples} & real_ids
        if collisions:
            raise ValueError(
                f"weak sample id(s) collide with real development sample id(s): "
                f"{sorted(collisions)} -- refusing to silently merge"
            )
        weak_cache = args.weak_feature_cache or (
            args.weak_export_root / "dino_proposer_features.npz"
        )
        weak_features = embed_all(embedder, weak_samples, weak_cache)
        features.update(weak_features)
        print(
            f"blending {len(weak_samples)} weak samples from {args.weak_export_root} into "
            f"training only, mass-normalized to {args.weak_mass_fraction:.2f}x each call's real "
            "sample mass (never used for held-out eval)",
            file=sys.stderr,
        )

    if args.held_out_fold is not None:
        if args.held_out_fold not in folds:
            raise ValueError(
                f"--held-out-fold {args.held_out_fold} not among resolved folds {folds}"
            )
        holdout_groups = {g for g, f in group_fold.items() if f == args.held_out_fold}
        train_samples, eval_samples = split_by_session(dev_dataset, holdout_groups)
        held_out_head, train_report, eval_report, sample_weights = _train_one_pass(
            config, device, train_samples, weak_samples, features,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            weak_mass_fraction=args.weak_mass_fraction, eval_samples=eval_samples,
        )

        args.checkpoint_out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(args.checkpoint_out, held_out_head, config)

        summary = {
            "development_only_pass": True,
            "mode": "single_held_out_fold",
            "held_out_fold": args.held_out_fold,
            "note": (
                "This checkpoint was trained excluding development fold "
                f"{args.held_out_fold}; `eval` below is a genuinely held-out loss for exactly "
                "this checkpoint. `final` groups remain untouched. Region recall/false-positive "
                "numbers come from report.py, not this script."
            ),
            "export_root": str(args.export_root),
            "checkpoint_out": str(args.checkpoint_out),
            "development_folds": folds,
            "train_snapshot_groups": sorted({s.split_group for s in train_samples}),
            "eval_snapshot_groups": sorted({s.split_group for s in eval_samples}),
            "weak_supervision": {
                "weak_export_root": str(args.weak_export_root) if args.weak_export_root else None,
                "weak_samples_blended": len(weak_samples),
                "weak_mass_fraction": args.weak_mass_fraction if weak_samples else None,
                "effective_per_sample_weight": (
                    next(iter(sample_weights.values())) if sample_weights else None
                ),
            },
            "train": train_report,
            "eval": eval_report,
        }
        print(json.dumps(summary, indent=2))
        if args.summary_out is not None:
            args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        return 0

    cross_validation = {}
    for held_out in folds:
        holdout_groups = {g for g, f in group_fold.items() if f == held_out}
        train_samples, eval_samples = split_by_session(dev_dataset, holdout_groups)
        _, train_report, eval_report, fold_sample_weights = _train_one_pass(
            config, device, train_samples, weak_samples, features,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            weak_mass_fraction=args.weak_mass_fraction, eval_samples=eval_samples,
        )
        cross_validation[f"fold_{held_out}_held_out"] = {
            "train_snapshot_groups": sorted({s.split_group for s in train_samples}),
            "eval_snapshot_groups": sorted({s.split_group for s in eval_samples}),
            "weak_effective_per_sample_weight": (
                next(iter(fold_sample_weights.values())) if fold_sample_weights else None
            ),
            "train": train_report,
            "eval": eval_report,
        }

    final_head, final_train_report, _, final_sample_weights = _train_one_pass(
        config, device, list(dev_dataset.samples), weak_samples, features,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
        weak_mass_fraction=args.weak_mass_fraction, eval_samples=None,
    )
    args.checkpoint_out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.checkpoint_out, final_head, config)

    summary = {
        "development_only_pass": True,
        "note": (
            f"First viability pass: only the `development` folds in {args.splits} were used. "
            "`final` groups are selection_eligible: false and were never trained/evaluated on. "
            "Region recall/false-positive numbers come from report.py, not this script."
        ),
        "weak_supervision": {
            "weak_export_root": str(args.weak_export_root) if args.weak_export_root else None,
            "weak_samples_blended": len(weak_samples),
            "weak_mass_fraction": args.weak_mass_fraction if weak_samples else None,
            "final_effective_per_sample_weight": (
                next(iter(final_sample_weights.values())) if final_sample_weights else None
            ),
        },
        "export_root": str(args.export_root),
        "checkpoint_out": str(args.checkpoint_out),
        "development_folds": folds,
        "samples_total": len(dev_dataset.samples),
        "final_model_train_report": final_train_report,
        "cross_validation": cross_validation,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out is not None:
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
