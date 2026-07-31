#!/usr/bin/env python3
"""Report ``dino_verifier`` ranking accuracy vs. today's geometry-only baseline.

This is a **first viability pass**, mirroring the same development-only scope
as ``train.py``: only ``config/rgb_ablation_splits.yaml``'s two ``development``
folds (``capture_batch_a`` = fold 0, ``capture_batch_b`` = fold 1) are ever
read; ``capture_batch_c``/``capture_batch_d`` are ``final`` and explicitly
flagged ``selection_eligible: false``, so this script never touches them.

**Held-out honesty.** ``save_checkpoint``/``load_dino_verifier`` (owned by
``blockpose.rgb_verifier``) do not record which snapshots a checkpoint was
trained on, so this script cannot verify a checkpoint's own training history.
Pass ``--held-out-fold N`` to assert "this checkpoint was trained excluding
fold N" (produced by ``train.py --held-out-fold N``); this script then reports
a genuinely held-out number for exactly that fold. Without it, this script
evaluates every development sample and labels the result **in-sample**, since
the default ``train.py`` checkpoint is trained on the union of both folds.

**Geometry baseline caveat.** ``samples.jsonl`` carries no geometry score
field today: ``candidates.py``'s raw hypotheses are exported as ``(id, pose)``
only, and ``export.py``'s ``label_details`` carries only translation/rotation
error to ground truth, not a detector score (see ``export.py:_raw_candidates``
and ``candidates.py:build_candidate_set``). So the baseline here is the
loosest available proxy: "is the first ``provenance == raw_hypothesis``
candidate in exporter order the correct one." That order is the detector
trace's hypothesis order, not a verified score-sorted NMS ranking, so it is a
weaker baseline than production geometry top-1 -- this script reports that
caveat alongside every number rather than implying otherwise.

    python -m experiments.rgb_candidate_prior.train --held-out-fold 1 \\
        --checkpoint-out out/dino_verifier_fold1_heldout
    python -m experiments.rgb_candidate_prior.report --held-out-fold 1 \\
        --checkpoint out/dino_verifier_fold1_heldout --export-root out/rgb_candidate_prior
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402
from blockpose.rgb_verifier import load_dino_verifier  # noqa: E402

from experiments.rgb_candidate_prior.dataset import CandidateDataset, Sample  # noqa: E402
from experiments.rgb_candidate_prior.model import (  # noqa: E402
    apply_calibration,
    development_fold_of_snapshot,
    first_correct_rank,
    has_valid_pair,
    select_development_groups,
)

DEFAULT_EXPORT_ROOT = REPO_ROOT / "out" / "rgb_candidate_prior"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits.yaml"


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def score_block(head, calibration_temperature: float, samples: list[Sample]):
    """Raw + calibrated scores for one candidate set's positive/negative samples.

    Uses the public ``head.forward`` (inference-only, matching the runtime
    contract exactly) rather than a cached-feature fast path, since this
    script runs occasionally and correctness-by-construction matters more
    than throughput here.
    """
    raw_scores, calibrated, labels = [], [], []
    for sample in samples:
        if sample.label not in ("positive", "negative"):
            continue
        rgb = _load_image(sample.rgb_path)
        conditioning = _load_image(sample.conditioning_path)
        raw_score, _log_uncertainty = head.forward(rgb, conditioning)
        raw_scores.append(raw_score)
        calibrated.append(apply_calibration(raw_score, calibration_temperature))
        labels.append(sample.label)
    return raw_scores, calibrated, labels


def geometry_baseline_result(samples: list[Sample]) -> str:
    """"Is the first raw-geometry candidate in exporter order the correct one?"

    See the module docstring's "Geometry baseline caveat": no geometry score
    is exported today, so "first in exporter order" stands in for "first by
    geometry score." Returns ``"no_raw_hypothesis_candidate"`` when the set
    has no ``provenance == raw_hypothesis`` member to evaluate.
    """
    raw_candidates = [
        sample
        for sample in samples
        if sample.provenance == "raw_hypothesis" and sample.label in ("positive", "negative")
    ]
    if not raw_candidates:
        return "no_raw_hypothesis_candidate"
    return "correct" if raw_candidates[0].label == "positive" else "incorrect"


def calibration_summary(calibrated: list[float], labels: list[str], *, n_bins: int = 10) -> dict:
    """Reliability-diagram bins plus expected calibration error / Brier score."""
    if not calibrated:
        return {"bins": [], "expected_calibration_error": None, "brier_score": None}
    scores = np.asarray(calibrated, dtype=np.float64)
    targets = np.asarray([1.0 if label == "positive" else 0.0 for label in labels])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lo) & (scores <= hi if hi >= 1.0 else scores < hi)
        if not np.any(mask):
            continue
        mean_predicted = float(scores[mask].mean())
        empirical_rate = float(targets[mask].mean())
        count = int(mask.sum())
        bins.append(
            {
                "range": [float(lo), float(hi)],
                "count": count,
                "mean_predicted": mean_predicted,
                "empirical_positive_rate": empirical_rate,
            }
        )
        ece += (count / len(scores)) * abs(mean_predicted - empirical_rate)
    brier_score = float(np.mean((scores - targets) ** 2))
    return {"bins": bins, "expected_calibration_error": ece, "brier_score": brier_score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--held-out-fold",
        type=int,
        default=None,
        help="Assert the checkpoint was trained excluding this development fold (produced "
        "by `train.py --held-out-fold`); restricts evaluation to that fold only, a "
        "genuinely held-out number. Without it, all development data is evaluated and "
        "labelled in-sample.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    head, config, calibration_temperature = load_dino_verifier(args.checkpoint)

    manifest = load_split_manifest(args.splits)
    snapshot_fold = development_fold_of_snapshot(manifest)
    dataset = CandidateDataset.load(args.export_root)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)
    if not dev_dataset.samples:
        raise ValueError(
            f"no exported sample resolves to a development fold in {args.splits}"
        )

    if args.held_out_fold is not None:
        holdout_groups = {group for group, fold in group_fold.items() if fold == args.held_out_fold}
        if not holdout_groups:
            raise ValueError(f"fold {args.held_out_fold} has no development groups")
        eval_samples = [s for s in dev_dataset.samples if s.split_group in holdout_groups]
        held_out = True
        provenance_note = (
            f"Evaluated only on development fold {args.held_out_fold}'s snapshots. This is a "
            "genuinely held-out number ONLY IF the checkpoint was trained excluding this fold "
            f"(e.g. via `train.py --held-out-fold {args.held_out_fold}`) -- this script cannot "
            "verify that from the checkpoint file itself and trusts the caller's assertion."
        )
    else:
        eval_samples = list(dev_dataset.samples)
        held_out = False
        provenance_note = (
            "No --held-out-fold given: evaluated on ALL development-fold data. These are "
            "IN-SAMPLE numbers if the checkpoint was trained on any of it (true for the "
            "default `train.py` checkpoint, which trains on the union of both development "
            "folds) -- not a held-out estimate."
        )

    blocks = CandidateDataset(root=dev_dataset.root, samples=tuple(eval_samples)).candidate_sets()

    dino_ranks: dict[str, int | None] = {}
    geometry_results: dict[str, str] = {}
    calibrated_all: list[float] = []
    labels_all: list[str] = []
    for block_id, samples in blocks.items():
        _raw_scores, calibrated, labels = score_block(head, calibration_temperature, samples)
        if has_valid_pair(labels):
            # Ranking under temperature scaling is a monotone transform of the raw
            # score, so ranking by `calibrated` and by `_raw_scores` agree exactly.
            # A set with no negative (or no positive) has nothing to rank against;
            # `first_correct_rank` would otherwise report a vacuous top-1 "correct"
            # for an all-positive set, inflating accuracy.
            dino_ranks[block_id] = first_correct_rank(calibrated, labels)
            calibrated_all.extend(calibrated)
            labels_all.extend(labels)
        geometry_results[block_id] = geometry_baseline_result(samples)

    evaluated_dino = [rank for rank in dino_ranks.values() if rank is not None]
    dino_top1 = sum(rank == 0 for rank in evaluated_dino)

    comparable_blocks = [
        block_id
        for block_id, result in geometry_results.items()
        if result != "no_raw_hypothesis_candidate" and dino_ranks.get(block_id) is not None
    ]
    geometry_top1_comparable = sum(geometry_results[b] == "correct" for b in comparable_blocks)
    dino_top1_comparable = sum(dino_ranks[b] == 0 for b in comparable_blocks)

    groups_evaluated = sorted({sample.split_group for sample in eval_samples})
    warnings = []
    if len(groups_evaluated) < 3:
        warnings.append(
            f"only {len(groups_evaluated)} snapshot groups evaluated: held-out accuracy will "
            "be high-variance"
        )

    report = {
        "development_only_pass": True,
        "held_out": held_out,
        "provenance_note": provenance_note,
        "export_root": str(args.export_root),
        "checkpoint": str(args.checkpoint),
        "config": asdict(config),
        "calibration_temperature": calibration_temperature,
        "snapshot_groups_evaluated": groups_evaluated,
        "warnings": warnings,
        "dino_verifier": {
            "candidate_sets_with_pos_neg_pair": len(evaluated_dino),
            "top1_correct": dino_top1,
            "top1_accuracy": dino_top1 / len(evaluated_dino) if evaluated_dino else None,
        },
        "geometry_baseline": {
            "description": (
                "First provenance=='raw_hypothesis' candidate in exporter order, NOT a "
                "geometry-score-sorted ranking -- no geometry score is exported today "
                "(see candidates.py/export.py). A looser baseline than production NMS top-1."
            ),
            "candidate_sets_with_raw_hypothesis": sum(
                1 for result in geometry_results.values() if result != "no_raw_hypothesis_candidate"
            ),
            "top1_correct": sum(result == "correct" for result in geometry_results.values()),
        },
        "head_to_head_on_comparable_sets": {
            "note": "Only sets where both dino_verifier and the geometry baseline are defined.",
            "candidate_sets": len(comparable_blocks),
            "geometry_top1_correct": geometry_top1_comparable,
            "dino_top1_correct": dino_top1_comparable,
        },
        "calibration": calibration_summary(calibrated_all, labels_all),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
