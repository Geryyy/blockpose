"""Fast, GPU-free tests for the SAM2 region-proposal eval tool's pure logic.

Covers mask-area filtering and centre-distance pooling/de-duplication with
synthetic inputs -- no model download, no GPU, no dataset. `tools/` has no
`__init__.py`; it is importable as an implicit namespace package once the
repo root is on `sys.path` (mirrors `tests/python/test_refinement_benchmark.py`).
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_sam_region_proposals import (
    PooledCandidate,
    dedupe_candidates,
    grid_points,
    mask_passes_area_filter,
    provenance_bucket,
)


def _candidate(id_: str, position, score: float, source: str = "pcl") -> PooledCandidate:
    return PooledCandidate(
        id_, np.asarray(position, dtype=np.float64), np.eye(3), (0.9, 0.6, 0.6), score, source
    )


def test_mask_area_filter_drops_slivers_and_near_whole_image() -> None:
    image_pixels = 2048 * 1536
    assert not mask_passes_area_filter(50, image_pixels, min_area_px=400, max_area_frac=0.35)
    assert not mask_passes_area_filter(
        int(0.9 * image_pixels), image_pixels, min_area_px=400, max_area_frac=0.35
    )
    assert mask_passes_area_filter(5000, image_pixels, min_area_px=400, max_area_frac=0.35)
    # Boundary is inclusive on both ends.
    assert mask_passes_area_filter(400, image_pixels, min_area_px=400, max_area_frac=0.35)


def test_mask_area_filter_rejects_empty_image() -> None:
    assert not mask_passes_area_filter(100, 0, min_area_px=400, max_area_frac=0.35)


def test_grid_points_count_and_inset_from_border() -> None:
    points = grid_points(1536, 2048, grid=8, margin_frac=0.02)
    assert len(points) == 64
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert min(xs) > 0.0 and max(xs) < 1536.0
    assert min(ys) > 0.0 and max(ys) < 2048.0


def test_dedupe_keeps_higher_score_within_radius() -> None:
    pcl = _candidate("pcl:0", [0.0, 0.0, 0.0], score=0.4, source="pcl")
    sam_close_better = _candidate("sam:0", [0.05, 0.0, 0.0], score=0.7, source="sam")
    sam_far = _candidate("sam:1", [5.0, 0.0, 0.0], score=0.1, source="sam")

    kept = dedupe_candidates([pcl, sam_close_better, sam_far], radius_m=0.25)

    kept_ids = {c.id for c in kept}
    assert kept_ids == {"sam:0", "sam:1"}  # higher-scoring duplicate wins, distant one survives


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


def test_provenance_bucket_collapses_unique_auto_fit_lineage_strings() -> None:
    # Every approved auto_fit block has a distinct lineage suffix; the report
    # must bucket them together, not treat each as its own category.
    assert (
        provenance_bucket("auto_fit:component-5/region-0/dims-0/top-0/side-3/refined") == "auto_fit"
    )
    assert (
        provenance_bucket("auto_fit:component-0/region-0/dims-0/top-1/side-none/refined")
        == "auto_fit"
    )
    assert provenance_bucket("rgb_mask") == "rgb_mask"
    assert provenance_bucket("rgb_mask_pca_fallback") == "rgb_mask_pca_fallback"
