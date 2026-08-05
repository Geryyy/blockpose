#!/usr/bin/env python3
"""Refit the shipped ``dino_verifier`` calibration temperature and compare it.

Phase 2a of the production-readiness audit (see
``docs/production-readiness-plan.md`` and
``experiments/dino_proposer_pooled_ablation.md``): the shipped checkpoint at
``models/production/dino_verifier`` carries ``calibration_temperature =
12.497270584106445``. At that temperature ``sigmoid(logit(r)/T)`` maps the
*entire* raw range ``r in [0.01, 0.99]`` to calibrated scores in roughly
``[0.409, 0.591]`` -- a span of ~0.073 against ``max_score_contribution``'s
+/-0.20 cap, while contested-geometry margins measured in the pooled
ablation were 0.005-0.09. So the verifier frequently cannot move a contested
decision even when it is confident in principle.

**What this script does NOT claim.** ``sigmoid(logit(r)/T)`` is strictly
monotonic in ``r`` for any ``T > 0``, so a large ``T`` never changes the
*ranking* of candidates by calibrated score -- it only compresses the score
into a narrow band around 0.5, which matters exactly because this contribution
is used as an *additive scalar* (``score_candidate_with_dino``'s
``contribution``), not as an argmax. This script is about that additive use,
not about whether the verifier can tell candidates apart.

**What this script does.**
  1. Loads the already-exported Seibersdorf candidate set from
     ``out/rgb_candidate_prior_seibersdorf`` (produced by
     ``python -m experiments.rgb_candidate_prior.export ...`` -- see
     ``docs/production-readiness-plan.md``). It refuses to regenerate that
     export if missing; a 25-snapshot export silently rebuilt here would not
     match whatever the rest of the audit is citing numbers from.
  2. Loads the shipped ``models/production/dino_verifier`` checkpoint (frozen
     MLP head; the DINOv2 backbone itself is never re-run here -- see
     "cached features" below) and scores every labelled (positive/negative)
     sample's raw ``[0, 1]`` head output.
  3. Refits the calibration temperature with proper fold discipline: for
     each leave-one-scenario-out development fold in
     ``config/rgb_ablation_splits_seibersdorf.yaml`` (resolved via
     ``experiments.fold_safety``, the same helpers ``train.py``/``report.py``
     use -- never by string-matching ``split_group``), a temperature is fit
     on every *other* fold's raw scores and applied only to the held-out
     fold, and the held-out predictions are pooled across all folds. This is
     a genuinely held-out estimate of the *calibration step* -- see the
     "in-sample raw scores" caveat below for the one thing it is not. Read
     ``temperature_objective_scan``'s output before reading any *magnitude*
     into a fitted temperature: when the objective has no finite minimum,
     what ``fit_temperature`` returns is where LBFGS stopped, not an
     estimate of anything, and its spread across folds means nothing.
  4. Also fits a single "global" temperature on the full pooled dev set --
     one scalar being what ``config.json`` holds -- and reports its
     in-sample metrics. This script does *not* recommend shipping it; see
     ``experiments/verifier_calibration_refit.md``'s recommendation.
  5. Reports NLL / expected calibration error / Brier score for the shipped
     T=12.497 and for both refit numbers, plus the resulting contribution
     span across raw 0.01..0.99.
  6. Reports where the head's raw scores actually live (min/max, per-label
     mean/std, rank AUC) and the contribution span over that *observed*
     range rather than the illustrative 0.01..0.99 grid, scans whether the
     BCE objective ``fit_temperature`` minimizes even has a finite minimum
     for this score distribution, and fits a 2-parameter Platt reference
     (diagnostic only, never shipped). Every number quoted in
     ``experiments/verifier_calibration_refit.md`` comes out of this one
     command, so the report can be re-derived rather than trusted.

**In-sample raw scores, by design.** This script does not retrain the MLP
head per fold the way ``train.py``'s cross-validation loop does -- it scores
every sample with the one frozen, already-shipped head. That head was
trained on the union of all development data, so its *raw* scores are
in-sample for every fold. Only the *temperature* is fit/evaluated with
leave-one-scenario-out discipline. This is weaker than ``train.py``'s
original per-fold retraining (see the module docstring's fold-safety note
and ``experiments/verifier_calibration_refit.md`` for the comparison against
how the shipped 12.497 was actually produced), but it is significantly
cheaper -- no backbone or MLP retraining, cached region features are reused
as-is -- and it directly answers the question this refit is for: is a
single well-fit temperature going to look this large regardless of how
carefully it's fit. If the answer is "yes", the degeneracy is in the raw
score distribution, not in a fold-safety bug in how T was fit.

    python -m tools.refit_verifier_calibration
    python -m tools.refit_verifier_calibration --output out/verifier_calibration_refit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402
from blockpose.rgb_verifier import load_dino_verifier  # noqa: E402

from experiments.embedding_cache import cached_embed_all  # noqa: E402
from experiments.fold_safety import (  # noqa: E402
    development_fold_of_snapshot,
    select_development_groups,
)
from experiments.rgb_candidate_prior.dataset import CandidateDataset, Sample  # noqa: E402
from experiments.rgb_candidate_prior.model import apply_calibration, fit_temperature  # noqa: E402

DEFAULT_EXPORT_ROOT = REPO_ROOT / "out" / "rgb_candidate_prior_seibersdorf"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "production" / "dino_verifier"
DEFAULT_SPLITS = REPO_ROOT / "config" / "rgb_ablation_splits_seibersdorf.yaml"
SHIPPED_TEMPERATURE = 12.497270584106445  # models/production/dino_verifier/config.json today


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _sample_fold(sample: Sample, snapshot_fold: dict[str, int]) -> int:
    snapshot = sample.raw.get("source", {}).get("snapshot")
    if snapshot not in snapshot_fold:
        raise KeyError(f"{sample.sample_id}: source.snapshot {snapshot!r} not in a dev fold")
    return snapshot_fold[snapshot]


def _raw_scores(
    mlp, features: dict[str, np.ndarray], samples: list[Sample], device: str
) -> list[float]:
    """Raw ``[0, 1]`` head scores for ``samples``, from cached region features.

    Mirrors ``DinoVerifierHead.forward``'s score computation
    (``sigmoid(mlp(features)[0])``) but skips re-running the frozen DINOv2
    backbone by reusing the export's cached pooled features -- the backbone
    is deterministic and frozen, so this is bit-identical to calling
    ``head.forward`` on the original crop/conditioning pair.
    """
    import torch

    stacked = np.stack([features[sample.sample_id] for sample in samples])
    tensor = torch.from_numpy(stacked).float().to(device)
    with torch.inference_mode():
        raw = mlp(tensor)
    return torch.sigmoid(raw[:, 0]).detach().cpu().tolist()


def nll(scores: list[float], labels: list[float], *, eps: float = 1e-6) -> float:
    """Mean binary negative log-likelihood of calibrated ``scores`` against ``labels``."""
    if not scores:
        return float("nan")
    total = 0.0
    for score, label in zip(scores, labels, strict=True):
        probability = min(max(score, eps), 1.0 - eps)
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    return total / len(scores)


def expected_calibration_error(
    scores: list[float], labels: list[float], *, n_bins: int = 10
) -> float:
    """Standard binned ECE (mirrors ``report.py``'s ``calibration_summary``, kept independent
    here since that function also returns reliability-diagram bins this script doesn't need)."""
    if not scores:
        return float("nan")
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores_arr >= lo) & (scores_arr <= hi if hi >= 1.0 else scores_arr < hi)
        if not np.any(mask):
            continue
        mean_predicted = float(scores_arr[mask].mean())
        empirical_rate = float(labels_arr[mask].mean())
        ece += (float(mask.sum()) / len(scores_arr)) * abs(mean_predicted - empirical_rate)
    return ece


def brier_score(scores: list[float], labels: list[float]) -> float:
    if not scores:
        return float("nan")
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    return float(np.mean((scores_arr - labels_arr) ** 2))


def contribution_span(
    temperature: float,
    max_score_contribution: float,
    *,
    low_raw: float = 0.01,
    high_raw: float = 0.99,
) -> dict:
    """Calibrated-score and additive-contribution range across a raw-score interval.

    Matches ``score_candidate_with_dino``'s ``signed = clip((calibrated - 0.5) * 2, -1, 1)``
    then ``contribution = signed * max_score_contribution``. ``low_raw``/``high_raw``
    default to the audit's illustrative ``[0.01, 0.99]`` grid; pass the head's
    *observed* raw range instead to get the span the verifier can actually
    realize (they differ by a lot here -- see ``raw_score_diagnostics``).
    """
    low_cal = apply_calibration(low_raw, temperature)
    high_cal = apply_calibration(high_raw, temperature)
    low_contrib = np.clip((low_cal - 0.5) * 2.0, -1.0, 1.0) * max_score_contribution
    high_contrib = np.clip((high_cal - 0.5) * 2.0, -1.0, 1.0) * max_score_contribution
    return {
        "raw_low": low_raw,
        "raw_high": high_raw,
        "calibrated_at_raw_low": low_cal,
        "calibrated_at_raw_high": high_cal,
        "calibrated_span": high_cal - low_cal,
        "contribution_at_raw_low": float(low_contrib),
        "contribution_at_raw_high": float(high_contrib),
        "contribution_span": float(high_contrib - low_contrib),
    }


def metrics_for_temperature(
    raw_scores: list[float], labels: list[float], temperature: float, max_score_contribution: float
) -> dict:
    calibrated = [apply_calibration(score, temperature) for score in raw_scores]
    return {
        "temperature": temperature,
        "n": len(raw_scores),
        "nll": nll(calibrated, labels),
        "expected_calibration_error": expected_calibration_error(calibrated, labels),
        "brier_score": brier_score(calibrated, labels),
        "grid_span": contribution_span(temperature, max_score_contribution),
    }


def rank_auc(raw_scores: list[float], labels: list[float]) -> float:
    """Mann-Whitney rank AUC of ``raw_scores`` against binary ``labels``, ties averaged.

    Reported because it is the one thing a large temperature provably cannot
    damage: ``apply_calibration`` is strictly monotonic in the raw score, so
    AUC is identical before and after calibration at any ``T > 0``. It is the
    evidence for "the head still discriminates" in
    ``experiments/verifier_calibration_refit.md``, separating that claim from
    the claim about additive-contribution magnitude.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    n_positive = float(targets.sum())
    n_negative = float(len(targets) - n_positive)
    if n_positive == 0.0 or n_negative == 0.0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average the ranks inside each tie group so ties count as half a win.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique) != len(scores):
        sums = np.zeros(len(unique), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float(
        (ranks[targets == 1.0].sum() - n_positive * (n_positive + 1.0) / 2.0)
        / (n_positive * n_negative)
    )


def raw_score_diagnostics(
    raw_scores: list[float], labels: list[float], temperature: float, max_score_contribution: float
) -> dict:
    """Where the head's raw scores actually live, and the span they actually realize.

    The audit's framing evaluated the calibration on an illustrative raw grid
    of ``0.01..0.99``. The head never produces scores anywhere near that wide,
    so this reports the observed distribution and re-evaluates the
    contribution span over the *observed* range -- the number that describes
    what the deployed verifier can actually contribute.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    positive = scores[targets == 1.0]
    negative = scores[targets == 0.0]
    return {
        "raw_min": float(scores.min()),
        "raw_max": float(scores.max()),
        "raw_positive_mean": float(positive.mean()) if positive.size else float("nan"),
        "raw_positive_std": float(positive.std()) if positive.size else float("nan"),
        "raw_negative_mean": float(negative.mean()) if negative.size else float("nan"),
        "raw_negative_std": float(negative.std()) if negative.size else float("nan"),
        "logit_min": float(np.log(scores.min() / (1.0 - scores.min()))),
        "logit_max": float(np.log(scores.max() / (1.0 - scores.max()))),
        "rank_auc": rank_auc(raw_scores, labels),
        "realized_span_at_shipped_temperature": contribution_span(
            temperature,
            max_score_contribution,
            low_raw=float(scores.min()),
            high_raw=float(scores.max()),
        ),
    }


def _bce_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean BCE of raw ``logits`` against ``labels``, computed the stable way.

    Used instead of ``nll(apply_calibration(...))`` for the objective scan
    below: ``apply_calibration`` raises ``OverflowError`` for temperatures
    small enough to push ``logit/T`` past ``math.exp``'s range, and the scan
    deliberately visits such temperatures.
    """
    return float(
        np.mean(np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits))))
    )


def temperature_objective_scan(raw_scores: list[float], labels: list[float]) -> dict:
    """Is the BCE objective ``fit_temperature`` minimizes actually minimized at a finite ``T``?

    This exists to stop a reader over-reading the fitted temperatures. If the
    head's logits all sit on one side of 0 and the majority label is on the
    other side, ``BCE(logits / T)`` is *strictly decreasing* in ``T`` with its
    infimum ``ln(2)`` only at ``T -> infinity`` -- there is no argmin. Under
    those conditions whatever ``fit_temperature`` returns is the point LBFGS
    stopped at, not an estimate of anything, and the spread of those numbers
    across folds carries no information about the data. This scan reports the
    monotonicity directly so the report can say which regime we are in
    instead of quoting a fitted magnitude as if it were meaningful.
    """
    probability = np.clip(np.asarray(raw_scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability))
    targets = np.asarray(labels, dtype=np.float64)
    grid = np.exp(np.linspace(math.log(1e-3), math.log(1e9), 400))
    values = np.array([_bce_from_logits(logits / t, targets) for t in grid])
    base_rate = float(targets.mean())
    best_constant = -(
        base_rate * math.log(base_rate) + (1.0 - base_rate) * math.log(1.0 - base_rate)
    )
    return {
        "grid_min_temperature": float(grid[0]),
        "grid_max_temperature": float(grid[-1]),
        "strictly_decreasing_in_temperature": bool(np.all(np.diff(values) < 0.0)),
        "argmin_on_grid": float(grid[int(np.argmin(values))]),
        "nll_at_grid_argmin": float(values.min()),
        "nll_limit_as_temperature_to_infinity": math.log(2.0),
        "base_rate": base_rate,
        "nll_of_best_constant_predictor": best_constant,
    }


def platt_reference_fit(raw_scores: list[float], labels: list[float]) -> dict:
    """Two-parameter ``sigmoid(a * logit(raw) + b)`` fit, in-sample, as a *diagnostic only*.

    Not a proposal. ``blockpose.rgb_verifier._calibrate_score`` is a
    single-scalar-temperature contract by design (it must stay torch-free and
    importable without the training stack), so an intercept is not something
    this refit can ship. It is fitted here purely to answer "is the missing
    degree of freedom the scale or the offset?" -- if a fit with an intercept
    beats every pure-temperature fit, then no value of ``T`` was ever going to
    be the fix, which is the whole conclusion of this report.

    Unlike ``fit_temperature``, this uses ``line_search_fn="strong_wolfe"``.
    LBFGS's default fixed ``lr`` step diverges on this 2-D problem -- the
    logits sit at -10..-4, so the first step along the intercept overshoots
    by ~1e10 and the fit lands at NLL 1.7e10 instead of 0.609. That is an
    optimizer artifact, not a property of the data, and a diagnostic whose
    whole job is to be compared against the temperature fits must not carry
    one.
    """
    import torch

    probability = np.clip(np.asarray(raw_scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = torch.tensor(np.log(probability / (1.0 - probability)), dtype=torch.float64)
    targets = torch.tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float64)
    parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        parameters[0] = 1.0  # scale a; intercept b starts at 0 (= the identity calibration)
    optimizer = torch.optim.LBFGS([parameters], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> "torch.Tensor":
        optimizer.zero_grad()
        loss = loss_fn(parameters[0] * logits + parameters[1], targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = float(loss_fn(parameters[0] * logits + parameters[1], targets))
    return {
        "scale_a": float(parameters[0].item()),
        "intercept_b": float(parameters[1].item()),
        "nll": final,
        "note": "diagnostic only; _calibrate_score has no intercept term",
    }


def loso_refit(raw_scores: list[float], labels: list[float], folds: list[int]) -> dict:
    """Leave-one-scenario-out nested temperature refit and its pooled held-out metrics.

    For each fold, a temperature is fit on every *other* fold's (raw score,
    label) pairs and applied only to this fold -- so every reported
    prediction is calibrated by a temperature that never saw that fold's
    labels, exactly the fold discipline ``config/rgb_ablation_splits_seibersdorf.yaml``
    is built for. The per-fold raw scores themselves are in-sample for the
    (frozen, un-retrained) head -- see the module docstring.
    """
    raw_arr = np.asarray(raw_scores)
    label_arr = np.asarray(labels)
    fold_arr = np.asarray(folds)
    present = sorted(set(folds))
    per_fold_temperature: dict[int, float] = {}
    pooled_calibrated: list[float] = []
    pooled_labels: list[float] = []
    for held_out in present:
        train_mask = fold_arr != held_out
        test_mask = fold_arr == held_out
        train_scores = raw_arr[train_mask]
        train_labels = label_arr[train_mask]
        if len(set(train_labels.tolist())) < 2:
            # A fold whose complement has only one class can't fit a
            # meaningful BCE temperature; skip it from the nested estimate
            # rather than fabricate one (mirrors train.py's "no pooled
            # held-out predictions" guard).
            continue
        import torch

        temperature = fit_temperature(torch.tensor(train_scores), torch.tensor(train_labels))
        per_fold_temperature[held_out] = temperature
        test_raw = raw_arr[test_mask].tolist()
        test_labels = label_arr[test_mask].tolist()
        for raw, label in zip(test_raw, test_labels, strict=True):
            pooled_calibrated.append(apply_calibration(raw, temperature))
            pooled_labels.append(label)
    temperatures = list(per_fold_temperature.values())
    skipped = sorted(set(present) - set(per_fold_temperature))
    return {
        "folds_used": sorted(per_fold_temperature),
        "folds_skipped_single_class_complement": skipped,
        "per_fold_temperature": per_fold_temperature,
        "temperature_mean": float(np.mean(temperatures)) if temperatures else float("nan"),
        "temperature_std": float(np.std(temperatures)) if temperatures else float("nan"),
        "temperature_min": float(np.min(temperatures)) if temperatures else float("nan"),
        "temperature_max": float(np.max(temperatures)) if temperatures else float("nan"),
        "pooled_held_out_n": len(pooled_calibrated),
        "pooled_held_out_nll": nll(pooled_calibrated, pooled_labels),
        "pooled_held_out_ece": expected_calibration_error(pooled_calibrated, pooled_labels),
        "pooled_held_out_brier": brier_score(pooled_calibrated, pooled_labels),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument(
        "--device", default=None, help="Defaults to the checkpoint head's own device."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON dump of the full report."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.export_root.is_dir() or not (args.export_root / "samples.jsonl").is_file():
        print(
            f"ERROR: {args.export_root} is missing or has no samples.jsonl -- refusing to "
            "regenerate it silently. Produce it first with:\n"
            "  python -m experiments.rgb_candidate_prior.export "
            "--snapshots-dir data/seibersdorf_dataset "
            f"--export-root {args.export_root} --sessions config/seibersdorf_sessions.yaml",
            file=sys.stderr,
        )
        return 1

    manifest = load_split_manifest(args.splits)
    snapshot_fold = development_fold_of_snapshot(manifest)

    dataset = CandidateDataset.load(args.export_root)
    dev_dataset, _group_fold = select_development_groups(dataset, snapshot_fold)
    labelled = [s for s in dev_dataset.samples if s.label in ("positive", "negative")]
    if not labelled:
        print(
            "ERROR: no positive/negative-labelled sample resolves to a development fold",
            file=sys.stderr,
        )
        return 1

    head, config, shipped_temperature = load_dino_verifier(args.checkpoint)
    if abs(shipped_temperature - SHIPPED_TEMPERATURE) > 1e-6:
        print(
            f"NOTE: checkpoint calibration_temperature={shipped_temperature!r} differs from the "
            f"{SHIPPED_TEMPERATURE!r} this script's docstring was written against -- using the "
            "checkpoint's actual value.",
            file=sys.stderr,
        )

    def _embed_one(sample: Sample) -> np.ndarray:
        rgb = _load_image(sample.rgb_path)
        conditioning = _load_image(sample.conditioning_path)
        return head.embed_regions(rgb, conditioning)

    cache = args.feature_cache or (args.export_root / "dino_verifier_features.npz")
    features = cached_embed_all(
        list(dev_dataset.samples), cache, embed_one=_embed_one, label="dino_verifier"
    )

    device = args.device or head.device
    raw_scores = _raw_scores(head.mlp, features, labelled, device)
    labels = [1.0 if s.label == "positive" else 0.0 for s in labelled]
    folds = [_sample_fold(s, snapshot_fold) for s in labelled]

    import torch

    global_refit_temperature = fit_temperature(torch.tensor(raw_scores), torch.tensor(labels))

    report = {
        "export_root": str(args.export_root),
        "checkpoint": str(args.checkpoint),
        "splits": str(args.splits),
        "n_labelled_samples": len(labelled),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
        "n_development_folds_with_data": len(set(folds)),
        "max_score_contribution": config.max_score_contribution,
        "raw_score_diagnostics": raw_score_diagnostics(
            raw_scores, labels, shipped_temperature, config.max_score_contribution
        ),
        "temperature_objective_scan": temperature_objective_scan(raw_scores, labels),
        "shipped": metrics_for_temperature(
            raw_scores, labels, shipped_temperature, config.max_score_contribution
        ),
        "global_refit_in_sample": metrics_for_temperature(
            raw_scores, labels, global_refit_temperature, config.max_score_contribution
        ),
        "loso_nested_refit": loso_refit(raw_scores, labels, folds),
        "platt_reference_fit": platt_reference_fit(raw_scores, labels),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
