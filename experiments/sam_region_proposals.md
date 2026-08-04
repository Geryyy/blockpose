# SAM2 region proposals: does zero-shot segmentation close the recall gap?

No training, no fitter changes. SAM2 (`facebook/sam2.1-hiera-small`) is prompted with a 8x8 point grid per image; surviving masks are projected to 3-D with `blockpose.rgb_annotation` (the same code the interactive annotator uses) and run through the existing geometric fitter's internal entry point (`annotation_candidates._run_detection`) exactly like a DINO proposal region. SAM-sourced accepted hypotheses are pooled with the whole-cloud PCL candidates and de-duplicated by centre distance (0.25 m). A block counts as detected when an accepted candidate's centre is within 0.30 m of the GT centre (Hungarian matching, same rule as the established baseline).

## Headline

| | Recall | Precision |
|---|---|---|
| Baseline (whole-cloud PCL) | 39/62 (63%) | 39/52 (75%) |
| SAM2 region proposals + PCL, pooled | 45/62 (73%) | 45/69 (65%) |
| Oracle ceiling (crop to GT box, same fitter) | 90% recall | -- |

The oracle row is not reproduced by this script; it is the given premise from the earlier
crop-to-GT-box experiment (median 617 points/box, 17/23 misses recovered at 2.6 cm median error).

**Verdict, stated plainly:** at the specified default grid density (8x8), pooling SAM2 region
proposals with the whole-cloud PCL candidates recovers **6 of the 23** blocks the baseline misses
(+10 recall points, 63% -> 73%) but does so by adding **13 new false positives** against those 6
true recoveries -- more than two new false detections for every one true one -- which drags
precision down 10 points (75% -> 65%). A naive union-and-de-duplicate policy with no reranking or
verification is **not a free win**: it trades precision for recall roughly 2:1 against the gain,
and closes only 6/23 = 26% of the gap the oracle says is recoverable (which would be full recovery
of 17/23, reaching ~90% recall). SAM region proposals clearly carry *some* real signal -- they do
recover real misses the whole-cloud detector cannot see, including 4 of the 16 `rgb_mask` blocks
and one previously-missed `auto_fit` block -- but a naive pool is a net precision loss for a modest
recall gain, not an unambiguous improvement. See "Grid density sensitivity" below: a sparser 6x6
grid gets a much better trade (5 recovered / 4 new FPs) than the specified 8x8 default, so the
verdict is sensitive to a hyperparameter this script did not tune systematically.

## Miss recovery

Of 23 GT blocks missed by the whole-cloud PCL baseline, SAM2 region proposals recovered 6 once
pooled and de-duplicated (grid=8, the tool's specified default). SAM-sourced candidates that ended
up unmatched to any GT block (new false positives beyond the baseline's own): 13.

Caveat on that false-positive count: it is computed as "unmatched pooled detections whose surviving
de-dup source tag is `sam`" -- i.e. whenever a SAM candidate and a PCL candidate describe the same
physical block within the 0.25 m de-dup radius, the de-dup keeps the higher-scoring one and that one
"owns" the slot. If SAM's version of an already-baseline-matched block happens to be the one kept
and it drifts (rarely, given the 0.25 m radius) just outside the 0.30 m GT match radius, this method
would count it as a "new" FP even though PCL alone would also have covered that block. We did not
observe this happening in a spot check of the recovered/FP lists, but the count is a reasonable
upper bound on "false positives attributable to SAM," not a certified lower bound.

Recovered blocks:
- 2026_07_07_grip_at_bottom_no_sides_high__1783428305_919404288 / manual_0 (provenance: rgb_mask_pca_fallback)
- 2026_07_07_misaligned_stack__1783428546_619315968 / manual_0 (provenance: rgb_mask)
- 2026_07_07_partially_wet__1783431057_617312512 / manual_0 (provenance: rgb_mask)
- 2026_07_07_pickup_and_stack_rotated2__1783429084_722002176 / auto_0 (provenance: auto_fit)
- 2026_07_07_wall__1783429196_422551296 / manual_0 (provenance: rgb_mask)
- 2026_07_07_wall3__1783429356_122876928 / manual_1 (provenance: rgb_mask)

## Per-scene breakdown

| Scene | GT | Baseline recall | Baseline precision | Pooled recall | Pooled precision |
|---|---|---|---|---|---|
| 2026_07_07_grip_at_bottom | 9 | 6/9 (67%) | 6/7 (86%) | 6/9 (67%) | 6/7 (86%) |
| 2026_07_07_grip_at_bottom_no_sides | 3 | 3/3 (100%) | 3/6 (50%) | 3/3 (100%) | 3/7 (43%) |
| 2026_07_07_grip_at_bottom_no_sides_high | 3 | 2/3 (67%) | 2/2 (100%) | 3/3 (100%) | 3/6 (50%) |
| 2026_07_07_grip_at_bottom_rotated | 3 | 2/3 (67%) | 2/2 (100%) | 2/3 (67%) | 2/2 (100%) |
| 2026_07_07_grip_at_top | 3 | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) |
| 2026_07_07_grip_at_top_rotated | 3 | 1/3 (33%) | 1/2 (50%) | 1/3 (33%) | 1/3 (33%) |
| 2026_07_07_misaligned_stack | 3 | 2/3 (67%) | 2/2 (100%) | 3/3 (100%) | 3/4 (75%) |
| 2026_07_07_partially_wet | 3 | 0/3 (0%) | 0/1 (0%) | 1/3 (33%) | 1/3 (33%) |
| 2026_07_07_pickup | 1 | 1/1 (100%) | 1/2 (50%) | 1/1 (100%) | 1/3 (33%) |
| 2026_07_07_pickup_and_stack_rotated | 7 | 6/7 (86%) | 6/8 (75%) | 6/7 (86%) | 6/8 (75%) |
| 2026_07_07_pickup_and_stack_rotated2 | 7 | 5/7 (71%) | 5/6 (83%) | 6/7 (86%) | 6/9 (67%) |
| 2026_07_07_pickup_and_stack_rotated3 | 2 | 1/2 (50%) | 1/1 (100%) | 1/2 (50%) | 1/1 (100%) |
| 2026_07_07_static | 3 | 3/3 (100%) | 3/4 (75%) | 3/3 (100%) | 3/5 (60%) |
| 2026_07_07_wall | 2 | 0/2 (0%) | n/a | 1/2 (50%) | 1/1 (100%) |
| 2026_07_07_wall2 | 3 | 1/3 (33%) | 1/1 (100%) | 1/3 (33%) | 1/1 (100%) |
| 2026_07_07_wall3 | 2 | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) | 2/3 (67%) |
| 2026_07_15_shadow | 3 | 1/3 (33%) | 1/1 (100%) | 1/3 (33%) | 1/1 (100%) |
| 2026_07_15_shadow2 | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| 2026_07_15_shadowpartial | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |

## Per-GT-provenance breakdown

`rgb_mask`/`rgb_mask_pca_fallback` are the human-annotated provenances for blocks the detector historically misses (see `config/annotated_snapshots_seibersdorf.yaml`); `auto_fit` is a block the detector's own geometric fit already found.

| Provenance | GT count | Baseline recall | Pooled recall |
|---|---|---|---|
| auto_fit | 34 | 33/34 (97%) | 34/34 (100%) |
| rgb_mask | 16 | 3/16 (19%) | 7/16 (44%) |
| rgb_mask_pca_fallback | 12 | 3/12 (25%) | 4/12 (33%) |

## Wall clock

Grid density: 8x8 (64 prompts/snapshot) -- the tool's specified default, used for every table above.
SAM2-region pipeline (grid inference + mask filtering + per-mask projection + fitter) per snapshot:
mean 2.17 s, median 1.63 s, min 1.26 s, max 8.16 s, over 25 snapshots on an RTX 5070 Ti
(`facebook/sam2.1-hiera-small`, `cuda`, auto-selected). The 8.16 s max is the first snapshot, which
pays one-time model-load cost (~7 s); every subsequent snapshot is close to the 1.63 s median. Total
wall clock for the full 25-snapshot sweep at grid=8 was under a minute.

Mask funnel across all snapshots (grid=8): 1600 masks generated -> 1064 survived the pixel-area
filter (`--mask-min-area-px 400`, `--mask-max-area-frac 0.35`) -> 853 survived
`select_points_from_mask` projection -> 31 produced an accepted (`fate == "final"`) fitter
hypothesis. Most surviving masks (853) never produce a hypothesis at all: the existing fitter is
strict, which is exactly why pooling doesn't produce runaway false positives at grid=8 -- but also
why it recovers only 6 of 23 misses rather than closing more of the oracle's 17/23 gap.

## Grid density sensitivity

The `--grid` default of 8x8 is the one specified for this tool and is what every table above uses.
For context, an informal (not systematically tuned, no held-out split) sweep over 6x6, 8x8, and
12x12 on this same 25-snapshot set gave:

| Grid | Recall | Precision | Misses recovered (of 23) | New FPs | Mean wall/snapshot |
|---|---|---|---|---|---|
| 6x6 (36 prompts) | 44/62 (71%) | 44/61 (72%) | 5 | 4 | ~1.1 s |
| 8x8 (64 prompts, default) | 45/62 (73%) | 45/69 (65%) | 6 | 13 | 2.17 s |
| 12x12 (144 prompts) | 42/62 (68%) | 42/71 (59%) | 3 | 18 | 4.61 s |

A sparser 6x6 grid has a noticeably better precision/recall trade than the 8x8 default: nearly the
same recall gain (5 vs 6 misses) for roughly a third of the new false positives (4 vs 13). Denser
12x12 is worse on every axis here, not better -- more grid points does not mean more useful regions,
it means more redundant/spurious masks competing for the same strict fitter, at higher cost. This is
a real, if narrow, finding: naive grid density is not "more is better," and the specified 8x8
default is not the best point in this small sweep. We did not chase this further (no 2-10 grid
sweep, no per-scene-optimal grid, no combination with mask-filter threshold tuning) -- see below.

## What we could not test

- **No held-out split.** All 25 snapshots are both the tuning signal (grid-density sweep above) and
  the reported result. With only 25 snapshots / 62 GT blocks there is no train/val/test separation
  available in this dataset for that purpose; the 8x8-vs-6x6 comparison should be read as "this
  dataset, these settings," not as a validated hyperparameter choice.
- **No reranking/verification stage.** The task scope is explicitly "no training, no change to the
  fitter" -- we also did not add any lightweight reranker/verifier on top of the pooled candidates
  (e.g. the repo's own `rgb_edge_prior` or DINO verifier machinery, scoring SAM-sourced hypotheses
  the same way it already scores PCL ones). Given that most of the SAM-sourced false positives are
  presumably geometrically weaker than the true recoveries, such a filter could plausibly keep more
  of the recall gain while cutting the false-positive cost -- but that is a hypothesis, not something
  measured here.
- **No mask-filter threshold sweep.** `--mask-min-area-px` (400) and `--mask-max-area-frac` (0.35)
  were set from a single manual inspection of one snapshot's mask-area histogram, not tuned.
- **Only one SAM2 checkpoint.** Only `facebook/sam2.1-hiera-small` was tried, since it's the one
  confirmed cached and working; the larger SAM2.1 variants (base/large) were not evaluated for
  either mask quality or the wall-clock cost they'd add.
- **Only one prompting strategy.** A single positive point per mask (grid cell) was used, matching
  the annotator's own interactive click-to-segment mode. Box prompts, multi-point prompts, or
  automatic mask generation with SAM2's own proposal-density heuristics were not tried.
- **Single-GPU timing only.** Wall-clock numbers are from one RTX 5070 Ti; no CPU or multi-GPU
  measurement was attempted.
- **The oracle figure (90%, 17/23 at 2.6 cm) is cited, not reproduced.** It comes from the earlier
  crop-to-GT-box experiment referenced in the task; this script does not recompute it, so it should
  be read as the stated ceiling this experiment is measuring progress against, not as independently
  re-verified here.
