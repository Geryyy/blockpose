"""Cached per-sample frozen-backbone embeddings, shared by every train.py.

Both RGB-prior training scripts embed once per sample through a frozen
backbone, cache the result keyed by sample id order, and print a `\\r`
progress line while doing so -- only what one sample turns into a feature
array differs (a crop pair pooled to one vector vs. a single image's patch
grid), so that's the only thing callers supply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def cached_embed_all(
    samples: list[Any],
    cache: Path | None,
    *,
    embed_one: Callable[[Any], np.ndarray],
    label: str,
) -> dict[str, np.ndarray]:
    """Return ``{sample.sample_id: feature_array}``, loading/saving ``cache`` if given."""
    ids = [sample.sample_id for sample in samples]
    if cache is not None and cache.is_file():
        blob = np.load(cache, allow_pickle=True)
        if list(blob["ids"]) == ids:
            print(f"loaded cached {label} features {blob['features'].shape}")
            return dict(zip(ids, blob["features"], strict=True))

    features = []
    for index, sample in enumerate(samples):
        features.append(np.asarray(embed_one(sample)))
        print(f"  embedded {index + 1}/{len(samples)}", end="\r")
    print()
    stacked = np.stack(features)
    if cache is not None:
        np.savez(cache, ids=np.array(ids), features=stacked)
    return dict(zip(ids, stacked, strict=True))
