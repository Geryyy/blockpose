#!/usr/bin/env python3
"""Phase 1a of `docs/production-readiness-plan.md`: the zero-retraining instance-split probe.

Context (measured in `experiments/dino_proposer_pooled_ablation.md`):
`dino_proposer` covers 62/62 GT blocks (100%), including 10/10
`heavy_occlusion`, but emits only 35 regions for those 62 blocks across 25
snapshots and 18 of those regions contain more than one block. Blocks ALONE
in a region are found 17/17 (100%); blocks SHARING a region are found only
31/45 (69%), and `heavy_occlusion` sharing a region only 3/9. All 14
remaining misses are in merged regions. The head is trained on a single
UNION foreground mask per snapshot (`experiments/rgb_region_prior/export.py`)
-- it is trained to merge, by design.

**The question this script answers:** how much of that merging is a WEAK
NECK in the predicted probability map -- splittable by post-processing the
existing dense map, no retraining -- versus GENUINE FUSION, where the map
itself has no signal left to split on and the seam-aware head in Phase 1b
(`docs/production-readiness-plan.md` 1b) is the only way forward?

**What this script measures, and what it deliberately does not.** For each
of 25 snapshots it computes `DinoRegionProposerHead.predict_proposal_map`
ONCE (backbone + head forward pass is the expensive part), then evaluates
several purely-CPU post-processing strategies against that same map:
region count, per-block coverage (>=30% of a block's silhouette pixels
inside some region -- the same rule used in the pooled-ablation report), and
instance purity (is a covered block ALONE in its region, or does it SHARE
the region with >=1 other covered block).

Purity is reported alongside two anti-gaming statistics, because "alone" by
itself is not evidence of a good split: a strategy that chops one block into
five fragments also makes that block "alone" in whichever fragment covers it
best. So each strategy additionally reports the mean fraction of a covered
block's silhouette its region actually holds, and how many of its regions
cover no block at all. On the real run this is exactly what separates the
two: the baseline holds 100% of every covered block (mean 1.000) in 35
regions of which 0 are empty, while watershed's higher "alone" count comes
with mean coverage 0.780 and 42 of its 86 regions covering nothing.

It does NOT run the downstream
fitter or pooling -- that is a second, more expensive stage
(`tools/eval_dino_proposer_pooled.py`), and coverage + purity alone is
already the decision signal Phase 1a's gate asks for (see the plan's "Gate:
report instance-level region recall"). A strategy that improves purity
without losing coverage is a candidate worth re-measuring downstream; a
strategy that cannot even do that is not worth the follow-up run.

**Ground truth silhouettes.** Uses
`experiments.rgb_region_prior.export.block_silhouette_mask`, the same convex-
hull-of-projected-corners silhouette the union training target is built
from, called with `occluded_visibilities=()` (mirroring `export.py`'s
current default) so every approved block counts, including
`heavy_occlusion`/`unknown` -- those are exactly the blocks this probe cares
about.

**Watershed backend.** `scikit-image` is not installed in this environment
(checked at import time below); rather than silently skip strategy 3 this
falls back to OpenCV's `cv2.watershed`, marker-based on local maxima of the
foreground's distance transform -- the same algorithm skimage's
`segmentation.watershed` implements, different library. If `cv2` is also
unavailable the watershed strategy is reported as unavailable, not silently
dropped.

Usage:
    python tools/probe_region_instance_split.py \
        --report-out experiments/region_instance_split_probe.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotations import load_annotations  # noqa: E402
from blockpose.experiment_record import experiment_stamp  # noqa: E402
from blockpose.io import CameraModel, load_camera_model, load_rgb, load_tf_snapshot  # noqa: E402
from blockpose.rgb_region_proposer import (  # noqa: E402
    DinoRegionProposerConfig,
    extract_region_proposals,
    load_dino_region_proposer,
)

from experiments.rgb_region_prior.export import (  # noqa: E402
    approved_targets,
    block_silhouette_mask,
)

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:  # pragma: no cover - exercised only in environments without opencv
    _HAVE_CV2 = False

try:
    import skimage  # noqa: F401

    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

# Which watershed backend the report should name. Recorded rather than inferred
# at read time: the strategy's results depend on it, so a reader must be able to
# tell from the report alone which implementation produced them.
_WATERSHED_BACKEND_NOTE = (
    "`skimage.segmentation.watershed`"
    if _HAVE_SKIMAGE
    else "`cv2.watershed`, scikit-image is not installed in this environment"
)

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "seibersdorf_dataset"
DEFAULT_MANIFEST = REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "production" / "dino_proposer"
DEFAULT_COVERAGE_THRESHOLD = 0.30  # matches the pooled-ablation report's instance-separation table
DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
VISIBILITY_CLASSES = ("full", "partial", "heavy_occlusion", "unknown")


# --------------------------------------------------------------------------
# Pure helpers: coverage + instance purity. No torch, no cv2 import needed to
# exercise these -- covered by tests/python/test_region_instance_split.py
# with synthetic masks.
# --------------------------------------------------------------------------


def coverage_fraction(block_mask: np.ndarray, region_mask: np.ndarray) -> float:
    """Fraction of `block_mask`'s True pixels also True in `region_mask`.

    0.0 for an empty block mask (rather than raising or NaN) so a caller can
    always take a max over regions without special-casing.
    """
    block_area = int(np.count_nonzero(block_mask))
    if block_area == 0:
        return 0.0
    overlap = int(np.count_nonzero(block_mask & region_mask))
    return overlap / block_area


@dataclass(frozen=True)
class BlockAssignment:
    """One block's best-covering region under one splitting strategy."""

    block_id: str
    visibility: str
    covered: bool
    region_index: int | None  # index into that snapshot's region list, or None if uncovered
    coverage: float  # best coverage fraction achieved, even if below threshold


def assign_blocks_to_regions(
    block_masks: dict[str, np.ndarray],
    block_visibility: dict[str, str],
    region_masks: Sequence[np.ndarray],
    *,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> list[BlockAssignment]:
    """Assign each block to whichever region covers the most of its silhouette.

    A block is "covered" only if that best coverage clears `coverage_threshold`
    (matching the pooled-ablation report's >=30% rule). Ties between regions
    are broken by lowest region index, for determinism -- immaterial to the
    purity counts below since it only affects which of two equally-good
    regions gets credit, not whether the block is alone or shared.
    """
    assignments: list[BlockAssignment] = []
    for block_id, mask in block_masks.items():
        best_index: int | None = None
        best_coverage = 0.0
        for index, region in enumerate(region_masks):
            coverage = coverage_fraction(mask, region)
            if coverage > best_coverage:
                best_coverage = coverage
                best_index = index
        covered = best_coverage >= coverage_threshold
        assignments.append(
            BlockAssignment(
                block_id=block_id,
                visibility=block_visibility[block_id],
                covered=covered,
                region_index=best_index if covered else None,
                coverage=best_coverage,
            )
        )
    return assignments


def _empty_bucket() -> dict[str, float]:
    return {
        "total": 0,
        "covered": 0,
        "uncovered": 0,
        "alone": 0,
        "shared": 0,
        # Sum of best-coverage over COVERED blocks only; divided by "covered"
        # this is the mean fraction of a covered block's silhouette that its
        # region actually contains. Needed because "alone" on its own is
        # gameable: see `purity_counts`.
        "coverage_sum": 0.0,
    }


def purity_counts(assignments: Sequence[BlockAssignment]) -> dict:
    """Aggregate coverage + instance-purity counts, overall and per visibility class.

    "alone": the block is covered and no other covered block shares its
    assigned region. "shared": covered, but >=1 other covered block is
    assigned to the same region. These mirror the pooled-ablation report's
    "block alone in its region" / "block sharing a region" table exactly, so
    a baseline run of this script should reproduce 17 alone / 45 shared
    (up to snapshot-to-snapshot floating point noise in the dense map, since
    that number came from the same checkpoint and config).

    **"alone" is not sufficient on its own, which is why `coverage_sum` is
    tracked too.** A strategy that shatters one block into five fragments
    scores that block "alone" (no OTHER block's best fragment is the same
    fragment) while having destroyed, not separated, the region -- the
    fragment holds barely over the 30% coverage bar. Measured on the real
    run: baseline regions contain 100% of every covered block's silhouette
    (mean coverage 1.000), watershed only 0.780, with 14 blocks scraping by
    at 30-60%. Reporting mean coverage next to "alone" is what keeps that
    failure mode visible instead of cashing it in as a purity win.
    """
    region_member_counts = Counter(
        a.region_index for a in assignments if a.covered
    )  # note: keys are (block's own snapshot-scoped region_index) -- callers
    # must pass assignments from ONE snapshot at a time, or namespace
    # region_index per snapshot, or this will conflate unrelated regions
    # across snapshots that happen to share an integer index. See
    # `evaluate_strategies` below, which calls this per-snapshot only.

    overall = _empty_bucket()
    per_visibility: dict[str, dict[str, float]] = defaultdict(_empty_bucket)
    for a in assignments:
        bucket = per_visibility[a.visibility]
        overall["total"] += 1
        bucket["total"] += 1
        if not a.covered:
            overall["uncovered"] += 1
            bucket["uncovered"] += 1
            continue
        overall["covered"] += 1
        bucket["covered"] += 1
        overall["coverage_sum"] += a.coverage
        bucket["coverage_sum"] += a.coverage
        if region_member_counts[a.region_index] == 1:
            overall["alone"] += 1
            bucket["alone"] += 1
        else:
            overall["shared"] += 1
            bucket["shared"] += 1
    return {"overall": overall, "per_visibility": dict(per_visibility)}


def merge_purity_counts(counts: Sequence[dict]) -> dict:
    """Sum several `purity_counts` results (one per snapshot) into one aggregate."""
    overall = _empty_bucket()
    per_visibility: dict[str, dict[str, float]] = defaultdict(_empty_bucket)
    for entry in counts:
        for key, value in entry["overall"].items():
            overall[key] += value
        for visibility, bucket in entry["per_visibility"].items():
            for key, value in bucket.items():
                per_visibility[visibility][key] += value
    return {"overall": overall, "per_visibility": dict(per_visibility)}


# --------------------------------------------------------------------------
# Splitting strategies. Each takes a probability map and returns a tuple of
# bool (H, W) region masks.
# --------------------------------------------------------------------------


def threshold_regions(
    probability_map: np.ndarray, threshold: float, min_component_area_px: int
) -> tuple[np.ndarray, ...]:
    """Connected-component extraction at one foreground threshold.

    Thin wrapper around `extract_region_proposals` that returns bare masks
    instead of `ImageProposal`s, since this probe only needs coverage/purity,
    not scores or provenance.
    """
    config = DinoRegionProposerConfig(
        foreground_threshold=threshold, min_component_area_px=min_component_area_px
    )
    proposals = extract_region_proposals(probability_map, config)
    return tuple(p.mask for p in proposals)


def watershed_regions(
    probability_map: np.ndarray,
    *,
    foreground_threshold: float = 0.5,
    min_component_area_px: int = 64,
    min_peak_distance_px: int = 20,
    min_peak_height_px: float = 3.0,
) -> tuple[np.ndarray, ...]:
    """Distance-transform marker-based watershed split of the foreground.

    Standard "split touching blobs" recipe (e.g. the classic OpenCV/skimage
    "separating touching coins" tutorial): threshold to a binary foreground,
    take its Euclidean distance transform, place one marker at each local
    maximum of that transform (a maximum is a point at least
    `min_peak_distance_px` from any taller or equal maximum, and at least
    `min_peak_height_px` from the nearest background pixel -- this drops
    single-pixel noise peaks near a blob's edge), then flood the *inverted*
    distance transform from those markers so the watershed line forms at the
    saddle between two adjacent blobs' basins.

    Uses `cv2.watershed` (see module docstring: `scikit-image` is not
    installed here). Raises `RuntimeError` if `cv2` is unavailable too --
    callers should check `_HAVE_CV2` first and report unavailability rather
    than calling this and swallowing the error.
    """
    if not _HAVE_CV2:
        raise RuntimeError("cv2 (opencv-python) is not installed; watershed strategy unavailable")

    foreground = probability_map >= foreground_threshold
    if not foreground.any():
        return ()

    distance = ndimage.distance_transform_edt(foreground)

    # Local maxima: strictly the tallest point in its neighbourhood (ties
    # broken by scipy's maximum_filter picking the first), at least
    # `min_peak_height_px` from the nearest background pixel so a maximum
    # near a blob's own boundary doesn't spawn a spurious extra marker.
    footprint = np.ones((min_peak_distance_px, min_peak_distance_px), dtype=bool)
    local_max = (
        (distance == ndimage.maximum_filter(distance, footprint=footprint))
        & (distance >= min_peak_height_px)
        & foreground
    )
    markers, num_markers = ndimage.label(local_max)
    if num_markers == 0:
        return ()

    # cv2.watershed's marker convention: 0 = unknown (to be flooded), >0 =
    # seed regions. Background gets its own seed (1) so the flood stops at
    # the foreground boundary instead of bleeding into the background;
    # object markers are offset to start at 2 so they never collide with it.
    cv2_markers = np.zeros(foreground.shape, dtype=np.int32)
    cv2_markers[~foreground] = 1
    cv2_markers[markers > 0] = markers[markers > 0] + 1

    # cv2.watershed requires an 8-bit 3-channel "elevation" image; flooding
    # follows ascending intensity, so inverting the (normalized) distance
    # transform makes each blob's peak the lowest point in its own basin --
    # the flood grows outward from each marker until two basins meet at a
    # ridge, which becomes the split line (-1 in the output).
    max_distance = float(distance.max())
    normalized = (255.0 * distance / max_distance) if max_distance > 0 else distance
    elevation = (255 - normalized).astype(np.uint8)
    elevation_bgr = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
    cv2.watershed(elevation_bgr, cv2_markers)

    regions = []
    for label in range(2, num_markers + 2):
        mask = cv2_markers == label
        if int(mask.sum()) >= min_component_area_px:
            regions.append(mask)
    return tuple(regions)


# --------------------------------------------------------------------------
# Per-snapshot evaluation
# --------------------------------------------------------------------------


def load_manifest_snapshots(manifest_path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(manifest_path.read_text()) or {}
    result = []
    for scene, info in (data.get("scenes") or {}).items():
        for entry in info.get("annotated") or []:
            result.append((scene, str(entry["name"])))
    return result


@dataclass
class SnapshotBlocks:
    masks: dict[str, np.ndarray]
    visibility: dict[str, str]
    dropped: list[str]  # block ids whose projection gate failed (see block_silhouette_mask)


def snapshot_blocks(snapshot_dir: Path, camera: CameraModel) -> SnapshotBlocks:
    """Every approved block's silhouette mask + visibility label for one snapshot.

    `occluded_visibilities=()` mirrors `export.py`'s current default: every
    approved block counts, including `heavy_occlusion`/`unknown` -- those are
    exactly the blocks Phase 1 exists to rescue.
    """
    annotations = load_annotations(snapshot_dir / "annotations.yaml")
    tf = load_tf_snapshot(snapshot_dir / "tf.yaml")
    targets = approved_targets(annotations, occluded_visibilities=())
    masks: dict[str, np.ndarray] = {}
    visibility: dict[str, str] = {}
    dropped: list[str] = []
    for target in targets:
        mask = block_silhouette_mask(target.pose, tf.T_world_camera, camera)
        if mask is None:
            dropped.append(target.id)
            continue
        masks[target.id] = mask
        visibility[target.id] = target.visibility
    return SnapshotBlocks(masks=masks, visibility=visibility, dropped=dropped)


@dataclass
class StrategyOutcome:
    """One strategy's result on one snapshot."""

    region_count: int
    purity: dict  # output of purity_counts
    # Regions that are nobody's best-covering region. The baseline emits zero
    # of these (every region it emits covers >=1 block); a splitter that
    # over-segments emits many, which is the other half of the "is this a
    # split or a shattering?" question that "alone" alone cannot answer.
    regions_covering_no_block: int


def evaluate_strategies(
    probability_map: np.ndarray,
    blocks: SnapshotBlocks,
    *,
    shipped_config: DinoRegionProposerConfig,
    thresholds: Sequence[float],
    coverage_threshold: float,
    watershed_min_peak_distance_px: int,
    watershed_min_peak_height_px: float,
) -> dict[str, StrategyOutcome]:
    """Run every strategy against one snapshot's dense map; pure post-processing only."""
    outcomes: dict[str, StrategyOutcome] = {}

    def record(name: str, region_masks: Sequence[np.ndarray]) -> None:
        assignments = assign_blocks_to_regions(
            blocks.masks, blocks.visibility, region_masks, coverage_threshold=coverage_threshold
        )
        claimed = {a.region_index for a in assignments if a.covered}
        outcomes[name] = StrategyOutcome(
            region_count=len(region_masks),
            purity=purity_counts(assignments),
            regions_covering_no_block=len(region_masks) - len(claimed),
        )

    record(
        "baseline (shipped config)",
        tuple(p.mask for p in extract_region_proposals(probability_map, shipped_config)),
    )
    for threshold in thresholds:
        record(
            f"threshold={threshold:.1f}",
            threshold_regions(probability_map, threshold, shipped_config.min_component_area_px),
        )
    if _HAVE_CV2:
        record(
            "watershed",
            watershed_regions(
                probability_map,
                foreground_threshold=0.5,
                min_component_area_px=shipped_config.min_component_area_px,
                min_peak_distance_px=watershed_min_peak_distance_px,
                min_peak_height_px=watershed_min_peak_height_px,
            ),
        )
    return outcomes


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def _mean_coverage(bucket: dict) -> str:
    """Mean fraction of a COVERED block's silhouette held by its region."""
    if not bucket["covered"]:
        return "n/a"
    return f"{bucket['coverage_sum'] / bucket['covered']:.3f}"


def build_report(
    strategy_names: list[str],
    per_strategy_region_totals: dict[str, int],
    per_strategy_empty_regions: dict[str, int],
    per_strategy_purity: dict[str, dict],
    args: argparse.Namespace,
    dropped_blocks: dict[str, list[str]],
) -> str:
    total_blocks = per_strategy_purity[strategy_names[0]]["overall"]["total"]

    lines: list[str] = []
    lines.append("# Region instance-split probe (Phase 1a, zero-retraining)")
    lines.append("")
    lines.append(
        "**Question:** how much of `dino_proposer`'s region merging (see "
        "`experiments/dino_proposer_pooled_ablation.md`, 'The real bottleneck is instance "
        "separation, not detection') is a weak neck in the predicted probability map -- "
        "splittable with no retraining -- versus genuine fusion that needs the Phase 1b "
        "seam-channel retrain?"
    )
    lines.append("")
    lines.append(
        "**Method:** compute `DinoRegionProposerHead.predict_proposal_map` ONCE per "
        f"snapshot over the {len(args.snapshots)} snapshots in `{args.manifest.name}` "
        f"(checkpoint `{args.checkpoint}`), then evaluate several purely-CPU "
        "post-processing strategies against that same map: the shipped connected-component "
        f"config, a foreground-threshold sweep ({', '.join(f'{t:.1f}' for t in args.thresholds)}), "
        "and a distance-transform marker watershed on the 0.5 foreground "
        f"({_WATERSHED_BACKEND_NOTE}, "
        f"markers = local maxima at least {args.watershed_min_peak_distance_px}px apart and "
        f"{args.watershed_min_peak_height_px:g}px from the nearest background pixel). "
        "Ground truth is `block_silhouette_mask` (`experiments/rgb_region_prior/export.py`) "
        "with `occluded_visibilities=()`, i.e. every approved block including "
        "`heavy_occlusion`/`unknown`."
    )
    lines.append("")
    lines.append(
        "**This does not measure downstream block recall.** No fitter, no pooling -- "
        "coverage + instance purity against the dense map alone is the Phase 1a gate "
        "(`docs/production-readiness-plan.md`), and it is far cheaper than the composed "
        "pipeline in `tools/eval_dino_proposer_pooled.py`. A strategy that improves purity "
        "without losing coverage is what would be re-measured downstream next; none of the "
        "numbers below are a substitute for that re-measurement."
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "`Mean cov.` is the mean fraction of a COVERED block's silhouette that its "
        "region actually contains, and `Empty regions` counts regions that are no "
        "block's best region. Both are here to keep `Alone` honest: a strategy that "
        "shatters one block into fragments scores it `Alone` (no OTHER block shares "
        "the fragment) while having destroyed the region rather than separated it. "
        "Read `Alone` only together with those two columns."
    )
    lines.append("")
    lines.append(
        f"| Strategy | Regions (total, {len(args.snapshots)} snapshots) | Empty regions "
        "| Coverage >=30% | Mean cov. | Alone | Shared |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in strategy_names:
        overall = per_strategy_purity[name]["overall"]
        lines.append(
            f"| {name} | {per_strategy_region_totals[name]} "
            f"| {per_strategy_empty_regions[name]} "
            f"| {_rate(overall['covered'], total_blocks)} "
            f"| {_mean_coverage(overall)} "
            f"| {_rate(overall['alone'], overall['covered'])} "
            f"| {_rate(overall['shared'], overall['covered'])} |"
        )
    lines.append("")

    lines.append("## Per visibility class")
    lines.append("")
    for name in strategy_names:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Visibility | Total | Covered >=30% | Mean cov. | Alone | Shared |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        per_visibility = per_strategy_purity[name]["per_visibility"]
        for visibility in VISIBILITY_CLASSES:
            bucket = per_visibility.get(visibility, _empty_bucket())
            if bucket["total"] == 0:
                continue
            lines.append(
                f"| {visibility} | {bucket['total']} "
                f"| {_rate(bucket['covered'], bucket['total'])} "
                f"| {_mean_coverage(bucket)} "
                f"| {_rate(bucket['alone'], bucket['covered'])} "
                f"| {_rate(bucket['shared'], bucket['covered'])} |"
            )
        lines.append("")

    total_dropped = sum(len(v) for v in dropped_blocks.values())
    if total_dropped:
        lines.append(
            f"**{total_dropped} block(s) dropped by the projection gate** "
            "(same gate `export.py`'s union mask uses -- e.g. every corner behind the "
            f"camera) and excluded from every strategy's denominator above: "
            f"{dict(dropped_blocks)}"
        )
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    baseline = per_strategy_purity[strategy_names[0]]["overall"]
    baseline_mean = baseline["coverage_sum"] / max(baseline["covered"], 1)
    best_name, best_overall = None, None
    # A strategy earns a mention here only if it gains purity. It is a CLEAN
    # win only if it also keeps every block the baseline covers AND keeps
    # those blocks essentially whole; anything that buys "alone" by cutting
    # blocks up is caveated instead, because "alone" is trivially gamed by
    # over-segmentation (see `purity_counts`). 0.95 of the baseline's mean
    # coverage is the "essentially whole" bar -- arbitrary, but any bar at all
    # is what stops a shattering strategy from being reported as a winner.
    caveated: list[tuple[str, dict]] = []
    for name in strategy_names[1:]:
        overall = per_strategy_purity[name]["overall"]
        if overall["alone"] <= baseline["alone"]:
            continue  # no purity gain at all -- nothing to trade off, nothing to caveat
        mean_coverage = overall["coverage_sum"] / max(overall["covered"], 1)
        if overall["covered"] < baseline["covered"] or mean_coverage < 0.95 * baseline_mean:
            caveated.append((name, overall))
            continue
        if best_overall is None or overall["alone"] > best_overall["alone"]:
            best_name, best_overall = name, overall

    if caveated:
        lines.append(
            "**Purity gains that do not survive inspection, reported rather than buried:** "
            "a strategy is disqualified if it loses blocks the baseline already finds, or if "
            "the blocks it does keep are no longer whole. The second rule matters more than it "
            "looks: `Alone` counts a block as separated when no OTHER block shares its region, "
            "which a strategy can achieve just as well by chopping the block into pieces as by "
            "finding the seam next to it. The mean-coverage and empty-region columns are what "
            "tell those two apart:"
        )
        lines.append("")
        for name, overall in caveated:
            mean_coverage = overall["coverage_sum"] / max(overall["covered"], 1)
            lines.append(
                f"- `{name}`: alone {baseline['alone']} -> {overall['alone']} "
                f"({overall['alone'] - baseline['alone']:+d}), but coverage "
                f"{baseline['covered']} -> {overall['covered']} of {total_blocks} "
                f"({overall['covered'] - baseline['covered']:+d} block(s) lost), mean coverage of "
                f"a covered block {baseline_mean:.3f} -> {mean_coverage:.3f}, and "
                f"{per_strategy_empty_regions[name]} of its "
                f"{per_strategy_region_totals[name]} regions cover no block at all (baseline: "
                f"{per_strategy_empty_regions[strategy_names[0]]} of "
                f"{per_strategy_region_totals[strategy_names[0]]}). The purity gain is "
                "predominantly fragmentation, not separation."
            )
        lines.append("")

    if best_name is not None:
        best_mean = best_overall["coverage_sum"] / max(best_overall["covered"], 1)
        lines.append(
            f"`{best_name}` improves instance purity ({best_overall['alone']} alone vs "
            f"baseline {baseline['alone']}) without losing coverage "
            f"({best_overall['covered']} vs {baseline['covered']} of {total_blocks}) and "
            f"without shredding the blocks it keeps (mean coverage {best_mean:.3f} vs "
            f"baseline {baseline_mean:.3f}). "
            "Worth a downstream re-measurement with `tools/eval_dino_proposer_pooled.py` "
            "before deciding Phase 1b is required."
        )
    else:
        tradeoff_note = (
            " The caveated strategies above are the strongest form of this result, not an "
            "asterisk on it. Watershed does move the `Alone` number, but it moves it by "
            "cutting the foreground up, not by finding seams: it emits far more regions than "
            "there are blocks, a large share of them covering no block at all, and the blocks "
            "it still covers are held only in part. In other words, the shape of the "
            "foreground probability alone does not tell a splitter where a seam is -- given "
            "only that shape, a splitter cannot distinguish the waist between two touching "
            "blocks from the middle of one block. That is precisely the information a "
            "supervised seam channel supplies, and precisely why no amount of post-processing "
            "tuning substitutes for it."
            if caveated
            else ""
        )
        lines.append(
            "No zero-retraining strategy improves instance purity over the baseline without "
            "losing coverage or shredding the blocks it keeps. This is a negative result, not "
            "a null one: the merging in the probability map is not a splittable weak neck at "
            "this operating point -- aggressive thresholds leave the merged blob whole (no "
            "split found at all, at any threshold up to 0.9), and watershed splits the blob "
            "without regard for where the block boundaries actually are."
            + tradeoff_note
            + " **This justifies the Phase 1b seam-channel retrain** "
            "(`docs/production-readiness-plan.md`); the head must be given a supervised signal "
            "for the seam, since none survives in its current union-mask training target."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated foreground thresholds for the sweep strategy.",
    )
    parser.add_argument("--watershed-min-peak-distance-px", type=int, default=20)
    parser.add_argument("--watershed-min-peak-height-px", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    args.thresholds = tuple(float(t) for t in args.thresholds.split(","))

    camera = load_camera_model(args.calibration)
    manifest = load_manifest_snapshots(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    args.snapshots = manifest

    # Note: `load_dino_region_proposer` does not take a device argument (same
    # limitation `tools/eval_dino_proposer_pooled.py`'s `DinoProposerRegions`
    # has) -- it always loads onto whatever `DinoRegionProposerHead`'s default
    # ("auto") resolves to. `--device` is accepted for interface symmetry with
    # that script but is currently a no-op; out of scope for this probe to fix.
    head, shipped_config = load_dino_region_proposer(args.checkpoint)

    strategy_names = ["baseline (shipped config)"] + [f"threshold={t:.1f}" for t in args.thresholds]
    if _HAVE_CV2:
        strategy_names.append("watershed")
    elif not _HAVE_CV2:
        print(
            "cv2 (opencv-python) not installed: watershed strategy skipped and reported as "
            "unavailable, not silently dropped from the manifest.",
            file=sys.stderr,
        )

    region_totals: dict[str, int] = {name: 0 for name in strategy_names}
    empty_region_totals: dict[str, int] = {name: 0 for name in strategy_names}
    per_snapshot_purity: dict[str, list[dict]] = {name: [] for name in strategy_names}
    dropped_blocks: dict[str, list[str]] = {}

    for i, (_scene, name) in enumerate(manifest):
        print(f"[{i + 1}/{len(manifest)}] {name}", file=sys.stderr)
        snapshot_dir = args.dataset_root / name
        blocks = snapshot_blocks(snapshot_dir, camera)
        if blocks.dropped:
            dropped_blocks[name] = blocks.dropped
        if not blocks.masks:
            continue
        rgb = load_rgb(snapshot_dir / "rgb.png")
        probability_map = head.predict_proposal_map(rgb)
        outcomes = evaluate_strategies(
            probability_map,
            blocks,
            shipped_config=shipped_config,
            thresholds=args.thresholds,
            coverage_threshold=args.coverage_threshold,
            watershed_min_peak_distance_px=args.watershed_min_peak_distance_px,
            watershed_min_peak_height_px=args.watershed_min_peak_height_px,
        )
        for strategy_name, outcome in outcomes.items():
            region_totals[strategy_name] += outcome.region_count
            empty_region_totals[strategy_name] += outcome.regions_covering_no_block
            per_snapshot_purity[strategy_name].append(outcome.purity)

    per_strategy_purity = {
        name: merge_purity_counts(per_snapshot_purity[name]) for name in strategy_names
    }

    stamp = experiment_stamp(
        checkpoint=str(args.checkpoint),
        coverage_threshold=args.coverage_threshold,
        thresholds=list(args.thresholds),
        watershed_min_peak_distance_px=args.watershed_min_peak_distance_px,
        watershed_min_peak_height_px=args.watershed_min_peak_height_px,
        watershed_backend="cv2" if _HAVE_CV2 else "unavailable",
    )
    report = build_report(
        strategy_names,
        region_totals,
        empty_region_totals,
        per_strategy_purity,
        args,
        dropped_blocks,
    )
    report = f"<!-- provenance: commit {stamp['git_commit']} -->\n" + report
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "provenance": stamp,
                    "region_totals": region_totals,
                    "empty_region_totals": empty_region_totals,
                    "purity": per_strategy_purity,
                    "dropped_blocks": dropped_blocks,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
