<!-- provenance: commit b9c1b25, cluster_max_extent_xy=2.25, dbscan_eps=0.07, score_context=full -->
# SAM2 region proposals: does zero-shot segmentation close the recall gap?

> **Stale numbers — read this first.** Every figure in the *original* version of this report was
> measured with `DetectionParameters.cluster_max_extent_xy = 1.5`, which was the default at the
> time. That default now ships at **2.25** (see `experiments/detector_cluster_failures.md`), and
> the command quoted here does **not** freeze detector parameters, so re-running it with today's
> defaults does not reproduce the original tables (39/62, 45/62 etc. below no longer occur). An
> independent audit's rerun at `cluster_max_extent_xy=2.25` gave baseline 46/62, 46/59 and pooled
> 49/62, 49/74 -- recovering 3 of the 16 remaining misses with 14 new unmatched detections, rather
> than 6 of 23 with 13. The *relative* conclusions of the original report still hold; its absolute
> counts describe the old (1.5) configuration.
>
> **Second correction, layered on top of the first (Phase 0.4): the numbers this banner itself
> quoted above (46/62, 49/62, 49/74) were ALSO measured with a scoring defect that inflates
> region-sourced scores.** That rerun predates `--score-context` (added below); it implicitly used
> `score-context=crop`, the mode now known invalid. The centre recall/precision quoted in this
> banner happen to reproduce exactly under the corrected `full` default (49/62, 49/74 -- see
> "Headline" below), but strict recall does not: 43/62 under the old crop-context scoring vs 44/62
> corrected. See "## Corrections: the score-context bug (Phase 0.4)" below for the full comparison
> and the mechanism. **Every table below this point uses the corrected `full` context (now the
> tool's default) at `cluster_max_extent_xy=2.25`.** Any future rerun must pin both the detector
> parameters and `--score-context` to reproduce a given table.

No training, no fitter changes. SAM2 (`facebook/sam2.1-hiera-small`) is prompted with a 8x8 point grid per image; surviving masks are projected to 3-D with `blockpose.rgb_annotation` (the same code the interactive annotator uses) and run through the existing geometric fitter's internal entry point (`annotation_candidates._run_detection`), scored against the whole-cloud `SensorContext` (`--score-context full`, the default -- see Corrections below), exactly like a DINO proposal region. SAM-sourced accepted hypotheses are pooled with the whole-cloud PCL candidates and de-duplicated by centre distance (0.25 m), keeping the higher-scoring candidate of any pair closer than that radius. A block counts as detected when an accepted candidate's centre is within 0.30 m of the GT centre (Hungarian matching, same rule as the established baseline).

**Pooling caveat, stated once up front:** the pooling policy measured throughout this report is a naive union-and-dedup -- every SAM-sourced candidate competes with every PCL candidate for every block, including blocks PCL already found correctly. `experiments/dino_proposer_pooled_ablation.md` measures a **gap-only** alternative for the DINO-proposer region source (region candidates may only fill blocks geometry found nothing for, never contest a block PCL already claimed) and finds it net-positive with zero regressions (45/62 strict, +1 over its baseline, vs this report's union-pool managing +0 on strict recall at the corrected default -- 44/62 pooled against 44/62 baseline -- and *-1* under the invalid crop scoring; the pre-2.25 write-up reported no strict column at all, so no comparison to it is possible on this axis). Gap-only pooling was not implemented or measured for the SAM2 source in this script; it is the same architectural fix that would also make the score-context bug below structurally impossible (a gap-only pool never has two sources contesting the same slot, so there is nothing for an inflated score to unfairly win).

## Headline

| | Recall (centre) | Precision (centre) | Recall (strict) |
|---|---|---|---|
| Baseline (whole-cloud PCL) | 46/62 (74%) | 46/59 (78%) | 44/62 (71%) |
| SAM2 region proposals + PCL, pooled | 49/62 (79%) | 49/74 (66%) | 44/62 (71%) |
| Oracle ceiling (crop to GT box, same fitter) | 88.7% recall | -- | -- |

The oracle row is not reproduced by this script; it is the given premise from the earlier
crop-to-GT-box experiment (median 617 points/box, **16**/23 misses recovered at 2.6 cm median
error). That count used to be quoted as 17 here; `experiments/detector_cluster_failures.md`
corrects it -- 17 is the number of *fitter successes*, but one of them lands 0.3287 m from ground
truth and so fails the same 0.30 m rule this report scores by, giving (39+16)/62 = 88.7%. The 88.7%
in the table above and the "16" here are the same corrected figure; do not re-pair 88.7% with 17.

**Strict** additionally requires symmetry-aware rotation error <= 30 deg and oriented-box IoU >=
0.25. Quote it, not the centre-only column: centre distance alone counts a box 88 degrees out of
alignment as a hit. A large centre/strict gap means boxes are landing near blocks rather than on
them.

**Verdict, stated plainly (current default: `cluster_max_extent_xy=2.25`, `--score-context
full`):** pooling SAM2 region proposals with the whole-cloud PCL candidates recovers **3 of the 16**
blocks the baseline misses (+5 centre-recall points, 74% -> 79%) at the cost of **13 new false
positives** against those 3 true recoveries -- more than four new false detections for every one
true one -- which drags centre precision down 12 points (78% -> 66%). On the metric that actually
matters for whether a pose is usable (**strict**), the pool is a wash: 44/62 baseline, 44/62 pooled,
exactly the same count -- the 3 centre-recovered blocks either fail the rotation/IoU gate themselves
or are offset by blocks the pool's contested dedup causes to regress elsewhere (see Corrections
below for one concrete such regression, visible under the old crop-context scoring). A naive
union-and-dedup policy is not a free win on this dataset at this configuration: it trades precision
for a centre-only recall gain and produces **no net improvement at all** on strict recall, versus
the gap-only policy measured net-positive for the DINO-proposer source (see Pooling caveat above).

## Corrections: the score-context bug (Phase 0.4)

`sam_candidates_for_snapshot` used to call `_run_detection(points[selection.indices], sensor_origin,
detection_params)` -- building the `SensorContext` each SAM-sourced fit is *scored* against from
just that mask's selected points, while `evaluate_snapshot`'s baseline builds its `SensorContext`
from the whole cloud, and `dedupe_candidates` pools the two and keeps "highest raw score wins"
(`tools/eval_sam_region_proposals.py::dedupe_candidates`, sorted by `-c.score`). Those two scores are
not comparable: 60% of the detector's score is ray-based evidence
(`evidence.hpp::normalized_evidence_features` f[2]/f[3]/f[4] -- covered/expected faces, supported
rays vs free-space violations, supported/incident rays), computed against whatever `SensorContext`
it is handed. A region crop of a few hundred points cannot accrue the free-space violations the
~114k-point whole cloud would charge the same hypothesis, so a SAM-sourced candidate's score comes
out systematically **higher** than a whole-cloud PCL candidate's for reasons that have nothing to do
with which is the better fit. Measured directly on one snapshot (`fit_region_scored_against`'s
docstring, identical crop, identical fit, identical pose): score against its own crop context =
0.7714, against the full cloud = 0.7205 (+0.0509). This is the exact same defect documented and
fixed for the DINO-proposer pooling tool in `experiments/dino_proposer_pooled_ablation.md`'s "Bug
1"; this script had the identical uncorrected call site.

Fixed via `--score-context full` (now the default), which scores every SAM-sourced fit against the
whole-cloud context via the new `fit_region_scored_against` helper (ported from
`tools/eval_dino_proposer_pooled.py`, same function). `--score-context crop` reproduces the
original, invalid behaviour.

Measured side by side, both at the current `cluster_max_extent_xy=2.25` (25 snapshots, 62 GT
blocks, grid=8, dedup radius 0.25 m -- only `--score-context` differs between the two rows):

| configuration | Recall (centre) | Precision (centre) | Recall (strict) | new FPs |
| --- | --- | --- | --- | --- |
| `crop` context (original, invalid) | 49/62 (79%) | 49/74 (66%) | 43/62 (69%) | 14 |
| **`full` context (corrected, current default)** | **49/62 (79%)** | **49/74 (66%)** | **44/62 (71%)** | **13** |

The centre-only columns happen to be identical here -- on this dataset, at this grid density, the
inflation was not large enough to flip which source wins a contested block for centre-distance
matching purposes in aggregate. It DID flip the outcome for at least one block: strict recall drops
from 44/62 to 43/62 and one more SAM-sourced candidate goes unmatched (14 vs 13 new FPs) under the
invalid `crop` scoring, meaning in that one case the inflated region score won a slot over the
better-fitting PCL (or a better-fitting region) candidate. That the aggregate centre numbers moved
so little here, versus the DINO-proposer tool's `dino_proposer_pooled_ablation.md` Corrections
section (where the same class of bug moved pooled strict recall from 36/62 to 43/62, a 7-block
swing), is itself informative: **the size of the score-inflation effect is pool-composition-
dependent** -- how often a contested block has a region candidate and a PCL candidate close in score
determines how much a fixed +0.05-ish inflation matters. It is not something to assume is small just
because it measured small once. Any report built on this pooling methodology must pin
`--score-context` and should not assume its effect transfers across datasets/configurations.

## Miss recovery

Of 16 GT blocks missed by the whole-cloud PCL baseline, SAM2 region proposals recovered 3 once
pooled and de-duplicated (grid=8, `--score-context full`, the tool's current defaults). SAM-sourced
candidates that ended up unmatched to any GT block (new false positives beyond the baseline's own):
13.

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
- 2026_07_07_pickup_and_stack_rotated2__1783429084_722002176 / auto_0 (provenance: auto_fit)

## Per-scene breakdown

| Scene | GT | Baseline recall | Baseline precision | Pooled recall | Pooled precision |
|---|---|---|---|---|---|
| 2026_07_07_grip_at_bottom | 9 | 6/9 (67%) | 6/7 (86%) | 6/9 (67%) | 6/7 (86%) |
| 2026_07_07_grip_at_bottom_no_sides | 3 | 3/3 (100%) | 3/6 (50%) | 3/3 (100%) | 3/7 (43%) |
| 2026_07_07_grip_at_bottom_no_sides_high | 3 | 2/3 (67%) | 2/2 (100%) | 3/3 (100%) | 3/6 (50%) |
| 2026_07_07_grip_at_bottom_rotated | 3 | 2/3 (67%) | 2/2 (100%) | 2/3 (67%) | 2/2 (100%) |
| 2026_07_07_grip_at_top | 3 | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) |
| 2026_07_07_grip_at_top_rotated | 3 | 1/3 (33%) | 1/2 (50%) | 1/3 (33%) | 1/4 (25%) |
| 2026_07_07_misaligned_stack | 3 | 2/3 (67%) | 2/2 (100%) | 3/3 (100%) | 3/4 (75%) |
| 2026_07_07_partially_wet | 3 | 2/3 (67%) | 2/3 (67%) | 2/3 (67%) | 2/4 (50%) |
| 2026_07_07_pickup | 1 | 1/1 (100%) | 1/2 (50%) | 1/1 (100%) | 1/3 (33%) |
| 2026_07_07_pickup_and_stack_rotated | 7 | 6/7 (86%) | 6/8 (75%) | 6/7 (86%) | 6/8 (75%) |
| 2026_07_07_pickup_and_stack_rotated2 | 7 | 5/7 (71%) | 5/6 (83%) | 6/7 (86%) | 6/9 (67%) |
| 2026_07_07_pickup_and_stack_rotated3 | 2 | 1/2 (50%) | 1/1 (100%) | 1/2 (50%) | 1/1 (100%) |
| 2026_07_07_static | 3 | 3/3 (100%) | 3/4 (75%) | 3/3 (100%) | 3/5 (60%) |
| 2026_07_07_wall | 2 | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| 2026_07_07_wall2 | 3 | 2/3 (67%) | 2/2 (100%) | 2/3 (67%) | 2/2 (100%) |
| 2026_07_07_wall3 | 2 | 2/2 (100%) | 2/3 (67%) | 2/2 (100%) | 2/3 (67%) |
| 2026_07_15_shadow | 3 | 2/3 (67%) | 2/2 (100%) | 2/3 (67%) | 2/2 (100%) |
| 2026_07_15_shadow2 | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| 2026_07_15_shadowpartial | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |

## Per-GT-provenance breakdown

`rgb_mask`/`rgb_mask_pca_fallback` are the human-annotated provenances for blocks the detector historically misses (see `config/annotated_snapshots_seibersdorf.yaml`); `auto_fit` is a block the detector's own geometric fit already found.

| Provenance | GT count | Baseline recall | Pooled recall |
|---|---|---|---|
| auto_fit | 34 | 33/34 (97%) | 34/34 (100%) |
| rgb_mask | 16 | 10/16 (62%) | 11/16 (69%) |
| rgb_mask_pca_fallback | 12 | 3/12 (25%) | 4/12 (33%) |

## Wall clock

Grid density: 8x8 (64 prompts/snapshot) -- the tool's specified default, used for every table above.
SAM2-region pipeline (grid inference + mask filtering + per-mask projection + fitter) per snapshot:
mean 2.36 s, median 1.89 s, min 1.32 s, max 8.78 s, over 25 snapshots on an RTX 5070 Ti
(`facebook/sam2.1-hiera-small`, `cuda`, auto-selected). `--score-context` does not materially affect
wall clock (the `crop`-context rerun measured mean 2.41 s / median 1.93 s over the same 25
snapshots) -- both modes run the identical fitter call, differing only in which points populate the
`SensorContext` passed alongside it.

Mask funnel across all snapshots (grid=8): 1600 masks generated -> 1064 survived the pixel-area
filter (`--mask-min-area-px 400`, `--mask-max-area-frac 0.35`) -> 853 survived
`select_points_from_mask` projection -> 33 produced an accepted (`fate == "final"`) fitter
hypothesis.

## Grid density sensitivity

**Not re-measured for this pass -- read before quoting.** The table below is carried over unchanged
from the original (pre-2.25, pre-`--score-context`) report: it was measured at
`cluster_max_extent_xy=1.5` AND with the now-known-invalid `crop` scoring, i.e. it is stale on both
axes the corrections above and the earlier staleness banner separately address. It is kept only as a
directional pointer ("denser is not automatically better") pending a proper rerun at the current
defaults; treat every number in this subsection as unverified against the current configuration.

The `--grid` default of 8x8 is the one specified for this tool. For context, an informal (not
systematically tuned, no held-out split) sweep over 6x6, 8x8, and 12x12 on this same 25-snapshot set
gave, under the stale (1.5, crop-context) configuration:

| Grid | Recall | Precision | Misses recovered (of 23) | New FPs | Mean wall/snapshot |
|---|---|---|---|---|---|
| 6x6 (36 prompts) | 44/62 (71%) | 44/61 (72%) | 5 | 4 | ~1.1 s |
| 8x8 (64 prompts, default) | 45/62 (73%) | 45/69 (65%) | 6 | 13 | 2.17 s |
| 12x12 (144 prompts) | 42/62 (68%) | 42/71 (59%) | 3 | 18 | 4.61 s |

A sparser 6x6 grid had a noticeably better precision/recall trade than the 8x8 default under that
stale configuration: nearly the same recall gain (5 vs 6 misses) for roughly a third of the new false
positives (4 vs 13). Whether this still holds at `cluster_max_extent_xy=2.25` with corrected
`--score-context full` scoring is unverified.

## What we could not test

- **No held-out split.** All 25 snapshots are both the tuning signal (grid-density sweep above) and
  the reported result. With only 25 snapshots / 62 GT blocks there is no train/val/test separation
  available in this dataset for that purpose.
- **No reranking/verification stage, no gap-only pooling.** The task scope is explicitly "no
  training, no change to the fitter" -- we also did not add any lightweight reranker/verifier on top
  of the pooled candidates, nor implement the gap-only pooling policy that
  `dino_proposer_pooled_ablation.md` measures net-positive for the DINO-proposer source (see Pooling
  caveat above). Given that most of the SAM-sourced false positives are presumably geometrically
  weaker than the true recoveries, either could plausibly keep more of the recall gain while cutting
  the false-positive cost -- but that is a hypothesis, not something measured here.
- **No grid-density sweep at the corrected configuration.** See "Grid density sensitivity" above --
  the only sweep on record predates both the 2.25 cluster-extent default and the `--score-context`
  fix.
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
- **The oracle figure (88.7%, 16 of 23 at 2.6 cm median error) is cited, not reproduced.** It
  comes from the earlier crop-to-GT-box experiment referenced in the task; this script does not
  recompute it, so it should be read as the stated ceiling this experiment is measuring progress
  against, not as independently re-verified here.
