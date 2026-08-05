"""Fast, GPU-free tests for the dino_proposer pooled-ablation tool's pure logic.

Covers centre-distance pooling/de-duplication, including the verifier-adjusted
score override that `tools/eval_sam_region_proposals.py`'s dedup doesn't have
-- no model download, no GPU, no dataset. `tools/` has no `__init__.py`; it is
importable as an implicit namespace package once the repo root is on
`sys.path` (mirrors `tests/python/test_sam_region_proposals.py`).
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_dino_proposer_pooled import PooledCandidate, _rate, dedupe_candidates


def _candidate(id_: str, position, score: float, source: str = "pcl") -> PooledCandidate:
    return PooledCandidate(
        id_, np.asarray(position, dtype=np.float64), np.eye(3), (0.9, 0.6, 0.6), score, source
    )


def test_dedupe_keeps_higher_geometry_score_within_radius() -> None:
    pcl = _candidate("pcl:0", [0.0, 0.0, 0.0], score=0.4, source="pcl")
    dino_close_better = _candidate("dino:0", [0.05, 0.0, 0.0], score=0.7, source="dino_proposer")
    dino_far = _candidate("dino:1", [5.0, 0.0, 0.0], score=0.1, source="dino_proposer")

    kept = dedupe_candidates([pcl, dino_close_better, dino_far], radius_m=0.25)

    kept_ids = {c.id for c in kept}
    assert kept_ids == {"dino:0", "dino:1"}


def test_dedupe_is_order_independent() -> None:
    a = _candidate("a", [0.0, 0.0, 0.0], score=0.9)
    b = _candidate("b", [0.1, 0.0, 0.0], score=0.2)

    forward = dedupe_candidates([a, b], radius_m=0.25)
    backward = dedupe_candidates([b, a], radius_m=0.25)

    assert {c.id for c in forward} == {c.id for c in backward} == {"a"}


def test_dedupe_no_duplicates_returns_everything() -> None:
    candidates = [_candidate(f"c{i}", [float(i) * 10.0, 0.0, 0.0], score=0.5) for i in range(5)]
    kept = dedupe_candidates(candidates, radius_m=0.25)
    assert len(kept) == 5


def test_dedupe_scores_override_flips_the_winner() -> None:
    """The verifier-rerank pass: a lower-geometry-score candidate can win
    if its id has a higher entry in the `scores` override -- this is the
    mechanism the reranked row depends on to differ from the pooled row."""
    lower_geometry = _candidate("pcl:0", [0.0, 0.0, 0.0], score=0.9, source="pcl")
    higher_adjusted = _candidate("dino:0", [0.05, 0.0, 0.0], score=0.3, source="dino_proposer")

    by_geometry = dedupe_candidates([lower_geometry, higher_adjusted], radius_m=0.25)
    assert {c.id for c in by_geometry} == {"pcl:0"}

    adjusted_scores = {"pcl:0": 0.1, "dino:0": 0.8}
    by_adjusted = dedupe_candidates(
        [lower_geometry, higher_adjusted], radius_m=0.25, scores=adjusted_scores
    )
    assert {c.id for c in by_adjusted} == {"dino:0"}


def test_dedupe_scores_override_falls_back_to_geometry_for_missing_ids() -> None:
    a = _candidate("a", [0.0, 0.0, 0.0], score=0.9)
    b = _candidate("b", [0.1, 0.0, 0.0], score=0.2)

    kept = dedupe_candidates([a, b], radius_m=0.25, scores={})
    assert {c.id for c in kept} == {"a"}


def test_rate_formats_percentage_and_handles_zero_denominator() -> None:
    assert _rate(6, 9) == "6/9 (66.7%)"
    assert _rate(0, 0) == "n/a"
