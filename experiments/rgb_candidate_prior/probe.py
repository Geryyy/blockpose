#!/usr/bin/env python3
"""Minimal viability probe: do frozen DINOv2 RGB features give the PCL detector a prior?

Leave-one-session-out linear probe on frozen crop embeddings.  If a logistic
regression on DINOv2 features cannot separate correct-vs-incorrect candidates
above chance on *held-out sessions*, RGB is not a viable prior for this data.

This is a diagnostic, not the deployment model: it deliberately uses only the
RGB crop (no conditioning image, no fine-tuning) so the number reflects raw
feature separability, the cheapest possible positive signal.

    python -m experiments.rgb_candidate_prior.probe --export-root <root>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.rgb_candidate_prior.dataset import CandidateDataset, Sample  # noqa: E402

_MODEL_ID = "facebook/dinov2-small"


def _embed_crops(samples: list[Sample], cache: Path | None) -> np.ndarray:
    """Return (N, D) frozen DINOv2 CLS embeddings, cached by sample_id order."""
    if cache is not None and cache.is_file():
        blob = np.load(cache, allow_pickle=True)
        if list(blob["ids"]) == [s.sample_id for s in samples]:
            print(f"loaded cached embeddings {blob['features'].shape}")
            return blob["features"]

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    model = AutoModel.from_pretrained(_MODEL_ID).eval()
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 16):
            batch = samples[start : start + 16]
            images = [Image.open(s.rgb_path).convert("RGB") for s in batch]
            inputs = processor(images=images, return_tensors="pt")
            out = model(**inputs)
            features.append(out.last_hidden_state[:, 0].cpu().numpy())
            print(f"  embedded {min(start + 16, len(samples))}/{len(samples)}", end="\r")
    print()
    stacked = np.concatenate(features, axis=0)
    if cache is not None:
        np.savez(cache, ids=np.array([s.sample_id for s in samples]), features=stacked)
    return stacked


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    # Mann-Whitney U / rank statistic = AUROC, ties handled by average rank.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(positives), len(negatives)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run(export_root: Path, *, exclude_rgb_gt: bool, real_only: bool, cache: Path | None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    dataset = CandidateDataset.load(export_root)
    samples = [s for s in dataset.samples if s.label in ("positive", "negative")]
    if exclude_rgb_gt:
        samples = [s for s in samples if s.raw.get("gt_provenance") != "rgb_candidate_fit"]
    if real_only:
        samples = [s for s in samples if s.provenance == "raw_hypothesis"]

    features = _embed_crops(samples, cache)
    labels = np.array([1 if s.label == "positive" else 0 for s in samples])
    groups = np.array([s.split_group for s in samples])
    print(
        f"probe set: {len(samples)} samples, {labels.sum()} pos / {(labels == 0).sum()} neg, "
        f"{len(set(groups))} sessions"
    )

    pooled_labels, pooled_scores, per_session = [], [], {}
    for held in sorted(set(groups)):
        train = groups != held
        test = groups == held
        if labels[train].min() == labels[train].max() or labels[test].sum() == 0:
            per_session[held] = {"auroc": None, "reason": "degenerate_split", "n": int(test.sum())}
            continue
        scaler = StandardScaler().fit(features[train])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.transform(features[train]), labels[train])
        scores = clf.predict_proba(scaler.transform(features[test]))[:, 1]
        auroc = _auroc(labels[test], scores)
        per_session[held] = {
            "auroc": auroc,
            "n": int(test.sum()),
            "pos": int(labels[test].sum()),
        }
        pooled_labels.append(labels[test])
        pooled_scores.append(scores)

    pooled = _auroc(np.concatenate(pooled_labels), np.concatenate(pooled_scores))
    session_aurocs = [v["auroc"] for v in per_session.values() if v["auroc"] is not None]
    return {
        "model": _MODEL_ID,
        "exclude_rgb_gt": exclude_rgb_gt,
        "real_candidates_only": real_only,
        "n_samples": len(samples),
        "pooled_holdout_auroc": pooled,
        "mean_session_auroc": float(np.mean(session_aurocs)) if session_aurocs else None,
        "per_session": per_session,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--keep-rgb-gt",
        action="store_true",
        help="Keep blocks whose GT came from rgb_candidate_fit (circular).",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Use only real PCL-sibling candidates, not generated perturbations.",
    )
    parser.add_argument("--cache", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    import json

    args = parse_args()
    report = run(
        args.export_root,
        exclude_rgb_gt=not args.keep_rgb_gt,
        real_only=args.real_only,
        cache=args.cache,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
