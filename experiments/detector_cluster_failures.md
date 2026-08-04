# Merged clusters: why the PCL detector drops wall and stack blocks

Measured against the 62 approved blocks in `config/annotated_snapshots_seibersdorf.yaml`
(25 snapshots, `data/seibersdorf_dataset`). A block counts as detected when an accepted
candidate's centre lies within 0.30 m of the GT centre. Baseline: **recall 39/62 (62.9%),
precision 39/52 (75.0%)**.

Raw evidence: `experiments/_sweep_results.json`. Reproduce with
`python tools/sweep_detector_params.py`.

## The hypothesis held

`DetectionParameters.cluster_max_extent_xy` is 1.5 m. `cpp/include/blockpose/core/pipeline.hpp`
drops any unsplit connected component wider than that in X or Y *before a single plane is fit to
it*. Blocks touching in a wall merge into one component and are discarded wholesale.

Contingency table over all 62 GT blocks, clustering after the detector's own ground removal:

| | detected | missed |
| --- | --- | --- |
| component **over** the 1.5 m gate | 1 | 10 |
| component under the gate | 38 | 13 |

A block in an over-gate component is missed **91%** of the time, against 25% otherwise.

**But merging explains at most 10 of the 23 misses.** The other 13 are missed while sitting in a
perfectly ordinary component and are a different failure mode entirely — out of scope here. Nothing
in this document should be read as addressing them.

### Reconciling two counts

An earlier hand analysis found 9 over-gate blocks; this tool finds 11. Both are right for their
method. The earlier one clustered the output of `annotation_preprocess.remove_ground`; this tool
reproduces the detector's own sequence (voxel downsample at `voxel_size` first, then ground removal,
then DBSCAN), which is what the gate actually sees. **The 11 figure is the one used throughout this
report**, because it is measured against the pipeline being diagnosed.

## The gate is simply set too low

Grid over `cluster_max_extent_xy` x `dbscan_eps`, scored on all 62 blocks. Recall alone is a
misleading objective here: a looser gate that fits one box across two touching blocks would raise
recall while making the detector worse, so precision is reported at every point.

| max_extent_xy | dbscan_eps | recall | precision |
| --- | --- | --- | --- |
| 1.50 (baseline) | 0.07 | 62.9% | 75.0% |
| 1.75 | 0.07 | 71.0% | 77.2% |
| 2.00 | 0.07 | 71.0% | 77.2% |
| **2.25** | **0.07** | **74.2%** | **78.0%** |
| 2.50 – 3.50 | 0.07 | 74.2% | 78.0% |
| 2.25 | 0.08 | 77.4% | 64.0% |
| 1.50 | 0.10 | 66.1% | 87.2% |

Nine settings beat the baseline on **both** metrics. The recommended one is a strict Pareto
improvement from changing a single parameter:

> **`cluster_max_extent_xy: 1.5 -> 2.25`** — recall 62.9% -> **74.2%** (+11.3 pp),
> precision 75.0% -> **78.0%** (+3.0 pp). `dbscan_eps` stays at 0.07.

Two details make this more than a lucky cell:

- **It plateaus.** 2.25, 2.50, 2.75, 3.00 and 3.50 at `eps=0.07` all give identical 74.2%/78.0%.
  Raising the gate past 2.25 changes nothing, which means the gate was rejecting valid clusters
  rather than protecting against over-merged ones. 2.25 is the knee, not a tuned optimum.
- **Precision rises.** The feared failure — one box spanning two touching blocks — does not
  materialise at `eps=0.07`. It does at `eps=0.08`, which buys 3 pp of recall for an 11 pp precision
  collapse; that setting should not be used, and an earlier draft of this report recommending it was
  wrong.

## Effect at the recommended setting

**Zero previously-detected blocks were lost.** 7 of the 11 over-gate blocks were recovered.

| scene | baseline recall | at (2.25, 0.07) |
| --- | --- | --- |
| 2026_07_07_wall | 0/2 | **2/2** |
| 2026_07_07_wall3 | 1/2 | **2/2** |
| 2026_07_07_wall2 | 1/3 | **2/3** |
| 2026_07_07_partially_wet | 0/3 | **2/3** |
| 2026_07_15_shadow | 1/3 | **2/3** |
| all other scenes | unchanged | unchanged |

Still missed despite the looser gate: `partially_wet/manual_2`, `wall2/manual_1`,
`shadow/manual_1`. Their components exceed even 3.5 m or fail for the second, unrelated reason.

## What this sizes for an RGB-guided splitter

Raising the gate lets a merged component through, but the fitter must then find several blocks
inside one component. Eight components each contain more than one GT block:

| snapshot | GT blocks | component extent | pairwise centre distance |
| --- | --- | --- | --- |
| partially_wet | 3 | 2.11 m | 1.38 / 0.92 / 0.70 m |
| wall2 | 2 | 1.88 m | 0.91 m |
| shadow | 2 | 1.81 m | 0.91 m |
| wall | 2 | 1.79 m | 0.93 m |
| wall3 | 2 | 1.56 m | 0.86 m |
| pickup_and_stack_rotated2 | 2 | 1.29 m | 0.63 m |
| misaligned_stack | 2 | 1.18 m | 0.72 m |
| pickup_and_stack_rotated | 2 | 0.93 m | 0.59 m |

Centres sit **0.59–1.38 m** apart — comfortably above the 0.30 m match radius, so a splitter has
real signal to work with, and well below the 0.9 m block length, so the blocks genuinely touch.
Purely geometric splitting is hard at this spacing; RGB (mortar lines, edges, colour discontinuity)
separates these visually even where the point cloud is one continuous surface.

## Applied

`cluster_max_extent_xy` is now 2.25 in both `python/blockpose/core_types.py` and
`cpp/include/blockpose/core/proposals.hpp`, guarded by a test that fails if either drifts.
Re-measured after the change: recall 46/62 (74%), precision 46/59 (78%).

## Which fitter, and where the remaining gap is

A follow-up disambiguated two things that were being conflated. On the SAME perfect (GT-box) crop
of each missed block:

| regions | fitter | recovers |
| --- | --- | --- |
| perfect crop | `snap_block_to_cloud` (seeded) | 17/23 |
| perfect crop | `_run_detection` (full pipeline) | 11/23 |
| SAM grid-6 mask | `snap_block_to_cloud` | 7/23 |
| SAM grid-8 mask | `_run_detection` | 6/23 |

Two conclusions, one of which corrects an earlier reading of the mask-funnel numbers:

- The seeded fitter is the better one for region-based fitting, but only by ~1 block once the
  regions come from SAM rather than ground truth.
- **Region quality dominates.** Going from SAM regions to perfect regions is worth 10 blocks;
  swapping the fitter is worth 1. An earlier note claiming "the bottleneck is the fitter's
  strictness, not proposal quality" was wrong, and is retracted here.

Notably the detector produces ZERO raw hypotheses on 12 of 23 perfect single-block crops, despite
a median 617 points and 76% of them surviving the ground stage. Cropping is not the problem
(`cluster_min_size` is never breached) and adding a 1.6 m skirt of context makes it worse, 7/23.
Whatever suppresses the proposal stage on an isolated block is unexplained and worth its own study.

## Combined configuration

| configuration | recall | precision |
| --- | --- | --- |
| original baseline | 63% | 75% |
| gate 2.25 (shipped) | 74% | 78% |
| gate 2.25 + SAM regions -> seeded fitter | 79% | 70% |

The gate alone is the only change that improves both metrics, which is why it is the one applied.
Pooling SAM regions buys 3 more blocks for 8 points of precision; worth enabling when recall
matters more, but it is not a default.

## What was not tested

- **No held-out split.** The grid was scored on the same 62 blocks it is recommended from. With one
  annotated snapshot per scene there is no split available. Treat 2.25 as "the knee of a plateau on
  this data", not a validated hyperparameter — the plateau is the reason to trust it, not the peak.
- **Only two parameters.** `cluster_min_extent_xy`, `cluster_max_extent_z`, `region_max_extent_xy`
  and the RANSAC gates were left at defaults; `region_max_extent_xy` (3.0) in particular interacts
  with this change and was not swept.
- **The splitter itself was not built or evaluated** — only sized.
- **The 13 non-merged misses were not investigated here.**
