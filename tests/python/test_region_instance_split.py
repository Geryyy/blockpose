"""Fast, GPU-free tests for the region instance-split probe's pure logic.

Covers coverage-fraction computation, block-to-region assignment, and the
alone/shared instance-purity aggregation with synthetic masks -- no model
download, no GPU, no dataset. `tools/` has no `__init__.py`; it is
importable as an implicit namespace package once the repo root is on
`sys.path` (mirrors `tests/python/test_sam_region_proposals.py`).

`watershed_regions` is also exercised here since it only needs `cv2` +
`scipy` -- no GPU, no model -- so a synthetic dumbbell-shaped foreground is a
fair, cheap check that it can split two touching blobs.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.rgb_region_proposer import DinoRegionProposerConfig

from tools.probe_region_instance_split import (
    _HAVE_CV2,
    BlockAssignment,
    SnapshotBlocks,
    assign_blocks_to_regions,
    coverage_fraction,
    evaluate_strategies,
    merge_purity_counts,
    purity_counts,
    threshold_regions,
    watershed_regions,
)


def _square_mask(shape, y0, x0, size) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


# --------------------------------------------------------------------------
# coverage_fraction
# --------------------------------------------------------------------------


def test_coverage_fraction_full_overlap_is_one() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 5, 5, 4)
    region = _square_mask(shape, 0, 0, 20)  # covers the whole image, including the block
    assert coverage_fraction(block, region) == pytest.approx(1.0)


def test_coverage_fraction_no_overlap_is_zero() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 0, 0, 4)
    region = _square_mask(shape, 15, 15, 4)
    assert coverage_fraction(block, region) == 0.0


def test_coverage_fraction_partial_overlap() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 0, 0, 4)  # 16 px
    region = _square_mask(shape, 2, 2, 4)  # overlaps a 2x2 = 4 px corner of the block
    assert coverage_fraction(block, region) == pytest.approx(4 / 16)


def test_coverage_fraction_empty_block_is_zero_not_nan() -> None:
    shape = (20, 20)
    block = np.zeros(shape, dtype=bool)
    region = _square_mask(shape, 0, 0, 20)
    assert coverage_fraction(block, region) == 0.0


# --------------------------------------------------------------------------
# assign_blocks_to_regions
# --------------------------------------------------------------------------


def test_assign_picks_best_covering_region_above_threshold() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 0, 0, 10)  # 100 px
    weak_region = _square_mask(shape, 0, 0, 5)  # covers 25/100 = 25%, below 30%
    strong_region = _square_mask(shape, 0, 0, 8)  # covers 64/100 = 64%
    masks = {"block_a": block}
    visibility = {"block_a": "full"}

    assignments = assign_blocks_to_regions(
        masks, visibility, [weak_region, strong_region], coverage_threshold=0.30
    )

    assert len(assignments) == 1
    a = assignments[0]
    assert a.covered is True
    assert a.region_index == 1
    assert a.coverage == pytest.approx(0.64)


def test_assign_reports_uncovered_below_threshold() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 0, 0, 10)
    weak_region = _square_mask(shape, 0, 0, 5)  # 25%, below the 30% threshold
    masks = {"block_a": block}
    visibility = {"block_a": "partial"}

    assignments = assign_blocks_to_regions(
        masks, visibility, [weak_region], coverage_threshold=0.30
    )

    assert assignments[0].covered is False
    assert assignments[0].region_index is None


def test_assign_with_no_regions_is_uncovered() -> None:
    shape = (20, 20)
    block = _square_mask(shape, 0, 0, 10)
    assignments = assign_blocks_to_regions(
        {"block_a": block}, {"block_a": "full"}, [], coverage_threshold=0.30
    )
    assert assignments[0].covered is False
    assert assignments[0].region_index is None
    assert assignments[0].coverage == 0.0


# --------------------------------------------------------------------------
# purity_counts / merge_purity_counts
# --------------------------------------------------------------------------


def _assignment(block_id, visibility, covered, region_index, coverage=1.0) -> BlockAssignment:
    return BlockAssignment(
        block_id=block_id,
        visibility=visibility,
        covered=covered,
        region_index=region_index,
        coverage=coverage,
    )


def test_purity_counts_block_alone_in_its_region() -> None:
    assignments = [_assignment("a", "full", True, region_index=0)]
    result = purity_counts(assignments)
    assert result["overall"] == {
        "total": 1,
        "covered": 1,
        "uncovered": 0,
        "alone": 1,
        "shared": 0,
        "coverage_sum": 1.0,
    }


def test_purity_counts_two_blocks_sharing_one_region() -> None:
    assignments = [
        _assignment("a", "full", True, region_index=0),
        _assignment("b", "heavy_occlusion", True, region_index=0),
    ]
    result = purity_counts(assignments)
    assert result["overall"]["alone"] == 0
    assert result["overall"]["shared"] == 2
    assert result["per_visibility"]["full"]["shared"] == 1
    assert result["per_visibility"]["heavy_occlusion"]["shared"] == 1


def test_purity_counts_mixed_alone_shared_and_uncovered() -> None:
    assignments = [
        _assignment("a", "full", True, region_index=0),  # alone
        _assignment("b", "partial", True, region_index=1),  # shares region 1 with c
        _assignment("c", "heavy_occlusion", True, region_index=1),  # shares region 1 with b
        _assignment("d", "unknown", False, region_index=None),  # uncovered
    ]
    result = purity_counts(assignments)
    overall = result["overall"]
    assert overall == {
        "total": 4,
        "covered": 3,
        "uncovered": 1,
        "alone": 1,
        "shared": 2,
        "coverage_sum": 3.0,  # only the 3 COVERED blocks contribute; the uncovered one does not
    }
    assert result["per_visibility"]["full"]["alone"] == 1
    assert result["per_visibility"]["partial"]["shared"] == 1
    assert result["per_visibility"]["heavy_occlusion"]["shared"] == 1
    assert result["per_visibility"]["unknown"]["uncovered"] == 1


def test_purity_counts_reproduces_established_ratio_on_synthetic_data() -> None:
    """Aggregation arithmetic at the real run's scale: 17 alone / 45 shared / 62 covered.

    This checks only that the counter adds up on synthetic assignments shaped
    like the pooled-ablation report's table. It is NOT a check that the real
    measurement lands on 17/45 -- nothing here touches the model, the dataset,
    or the probability map, and this test would pass unchanged if the real
    numbers moved.
    """
    assignments = [_assignment(f"alone_{i}", "full", True, region_index=i) for i in range(17)]
    # 45 blocks packed two-per-region (22 pairs + 1 triple = 45): use region
    # indices offset past the 17 "alone" ones so they don't collide.
    shared_region = 100
    for i in range(21):
        assignments.append(
            _assignment(f"shared_pair_{i}_0", "partial", True, region_index=shared_region + i)
        )
        assignments.append(
            _assignment(f"shared_pair_{i}_1", "partial", True, region_index=shared_region + i)
        )
    assignments.append(_assignment("shared_triple_0", "heavy_occlusion", True, region_index=999))
    assignments.append(_assignment("shared_triple_1", "heavy_occlusion", True, region_index=999))
    assignments.append(_assignment("shared_triple_2", "heavy_occlusion", True, region_index=999))

    result = purity_counts(assignments)
    assert result["overall"]["alone"] == 17
    assert result["overall"]["shared"] == 45
    assert result["overall"]["covered"] == 62


def test_alone_is_gameable_by_shattering_but_mean_coverage_exposes_it() -> None:
    """Why `coverage_sum` exists: "alone" alone would score a shredder as a winner.

    Two blocks in one merged region (both "shared") versus the same two blocks
    each chopped into fragments, where each block's best fragment barely clears
    the 30% coverage bar. The second case looks like a perfect instance split
    by the "alone" count and is in fact strictly worse -- the regions no longer
    contain the blocks. Mean coverage (`coverage_sum / covered`) is what tells
    them apart, and this is not hypothetical: it is the real difference between
    the baseline (mean 1.000) and watershed (mean 0.780) in
    `experiments/region_instance_split_probe.md`.
    """
    merged = purity_counts(
        [
            _assignment("a", "full", True, region_index=0, coverage=1.0),
            _assignment("b", "full", True, region_index=0, coverage=1.0),
        ]
    )
    shattered = purity_counts(
        [
            _assignment("a", "full", True, region_index=0, coverage=0.34),
            _assignment("b", "full", True, region_index=3, coverage=0.31),
        ]
    )

    # By purity alone, shattering looks like a total win.
    assert merged["overall"]["alone"] == 0
    assert shattered["overall"]["alone"] == 2
    # Mean coverage is what reveals it as a loss.
    merged_mean = merged["overall"]["coverage_sum"] / merged["overall"]["covered"]
    shattered_mean = shattered["overall"]["coverage_sum"] / shattered["overall"]["covered"]
    assert merged_mean == pytest.approx(1.0)
    assert shattered_mean == pytest.approx(0.325)
    assert shattered_mean < merged_mean


def test_merge_purity_counts_sums_across_snapshots() -> None:
    per_snapshot = [
        purity_counts([_assignment("a", "full", True, region_index=0)]),
        purity_counts(
            [
                _assignment("b", "full", True, region_index=0),
                _assignment("c", "full", True, region_index=0),
            ]
        ),
    ]
    merged = merge_purity_counts(per_snapshot)
    assert merged["overall"] == {
        "total": 3,
        "covered": 3,
        "uncovered": 0,
        "alone": 1,
        "shared": 2,
        "coverage_sum": 3.0,
    }


def test_purity_counts_region_index_is_snapshot_scoped_not_global() -> None:
    """Regression guard: two DIFFERENT snapshots' regions must never be conflated

    just because they reuse the same small integer index. `purity_counts` must
    only ever be called with assignments from one snapshot at a time (see
    `evaluate_strategies`); this test locks in that a naive global merge would
    be wrong, by checking that merging per-snapshot results (each internally
    consistent) does NOT accidentally count cross-snapshot blocks as sharing.
    """
    snapshot_1 = purity_counts([_assignment("a", "full", True, region_index=0)])
    snapshot_2 = purity_counts([_assignment("b", "full", True, region_index=0)])
    merged = merge_purity_counts([snapshot_1, snapshot_2])
    # Both are "alone" within their own snapshot; summing per-snapshot alone
    # counts (1 + 1 = 2) is correct. A buggy global-index merge would instead
    # have called them "shared" because both used region_index=0.
    assert merged["overall"]["alone"] == 2
    assert merged["overall"]["shared"] == 0


# --------------------------------------------------------------------------
# threshold_regions (thin wrapper over extract_region_proposals)
# --------------------------------------------------------------------------


def test_threshold_regions_splits_at_higher_threshold() -> None:
    # A saddle-shaped probability map: two 0.9 blobs joined by a 0.6 bridge.
    probability_map = np.zeros((30, 60), dtype=np.float64)
    probability_map[5:20, 5:20] = 0.9
    probability_map[5:20, 25:40] = 0.9
    probability_map[10:15, 18:27] = 0.6  # bridge, only above threshold 0.5

    low = threshold_regions(probability_map, threshold=0.5, min_component_area_px=10)
    high = threshold_regions(probability_map, threshold=0.7, min_component_area_px=10)

    assert len(low) == 1  # bridge merges the two blobs into one component
    assert len(high) == 2  # raising the threshold above the bridge's value splits them


# --------------------------------------------------------------------------
# evaluate_strategies (no model: it takes an already-computed probability map)
# --------------------------------------------------------------------------


def test_evaluate_strategies_counts_regions_that_cover_no_block() -> None:
    """The empty-region counter is the other half of the anti-gaming pair.

    Two separate 0.9 blobs, but only one of them is a real block: the second
    is a false positive the strategy emitted. It must be counted as a region
    covering no block, since a splitter that manufactures regions faster than
    it separates blocks is over-segmenting, not splitting.
    """
    probability_map = np.zeros((40, 80), dtype=np.float64)
    probability_map[5:20, 5:20] = 0.9  # real block
    probability_map[5:20, 50:65] = 0.9  # spurious region, no block behind it

    blocks = SnapshotBlocks(
        masks={"block_a": _square_mask((40, 80), 5, 5, 15)},
        visibility={"block_a": "full"},
        dropped=[],
    )
    config = DinoRegionProposerConfig(foreground_threshold=0.5, min_component_area_px=10)

    outcomes = evaluate_strategies(
        probability_map,
        blocks,
        shipped_config=config,
        thresholds=(0.5,),
        coverage_threshold=0.30,
        watershed_min_peak_distance_px=10,
        watershed_min_peak_height_px=2.0,
    )

    baseline = outcomes["baseline (shipped config)"]
    assert baseline.region_count == 2
    assert baseline.regions_covering_no_block == 1
    assert baseline.purity["overall"]["alone"] == 1
    assert baseline.purity["overall"]["coverage_sum"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# watershed_regions
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_CV2, reason="requires opencv-python")
def test_watershed_splits_a_dumbbell_shaped_blob() -> None:
    # Two 14x14 squares joined by a thin 2-px-tall neck: a single connected
    # component that distance-transform watershed should still split into two,
    # since it has two well-separated local maxima and the neck's own distance
    # transform (~1 px) stays below `min_peak_height_px` so it spawns no
    # spurious third marker.
    probability_map = np.zeros((30, 60), dtype=np.float64)
    probability_map[8:22, 5:19] = 0.9
    probability_map[8:22, 41:55] = 0.9
    probability_map[14:16, 19:41] = 0.9  # connecting neck

    # Sanity check: this really is one connected component at threshold 0.5.
    assert len(threshold_regions(probability_map, threshold=0.5, min_component_area_px=10)) == 1

    regions = watershed_regions(
        probability_map,
        foreground_threshold=0.5,
        min_component_area_px=10,
        min_peak_distance_px=10,
        min_peak_height_px=2.0,
    )
    assert len(regions) == 2


@pytest.mark.skipif(not _HAVE_CV2, reason="requires opencv-python")
def test_watershed_on_empty_foreground_returns_no_regions() -> None:
    probability_map = np.zeros((20, 20), dtype=np.float64)
    assert watershed_regions(probability_map, foreground_threshold=0.5) == ()


@pytest.mark.skipif(not _HAVE_CV2, reason="requires opencv-python")
def test_watershed_on_single_round_blob_does_not_over_split() -> None:
    probability_map = np.zeros((30, 30), dtype=np.float64)
    probability_map[10:20, 10:20] = 0.9  # one compact blob, one local maximum

    regions = watershed_regions(
        probability_map,
        foreground_threshold=0.5,
        min_component_area_px=10,
        min_peak_distance_px=10,
        min_peak_height_px=2.0,
    )
    assert len(regions) == 1
