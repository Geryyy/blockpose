<!-- provenance: commit 67a496d -->
# Composed pipeline ablation: geometry -> +dino_proposer -> +dino_verifier

**In-sample numbers** (see module docstring): the loaded checkpoints are trained on the union of all 25 annotated snapshots this reports on, unless `--proposer-checkpoint`/`--verifier-checkpoint` point at a held-out-fold pair instead.

| Stage | Recall (centre) | Precision (centre) | Recall (strict) |
|---|---|---|---|
| geometry (whole-cloud PCL) | 46/62 (74.2%) | 46/59 (78.0%) | 44/62 (71.0%) |
| + dino_proposer (pooled) | 48/62 (77.4%) | 48/63 (76.2%) | 43/62 (69.4%) |
| + dino_proposer + dino_verifier (reranked) | 48/62 (77.4%) | 48/63 (76.2%) | 43/62 (69.4%) |

**Strict** additionally requires symmetry-aware rotation error <= 30 deg and oriented-box IoU >= 0.25.

## Miss recovery

Of 16 GT blocks missed by whole-cloud PCL alone: pooling `dino_proposer` regions recovers 2; adding `dino_verifier` reranking recovers 2.

## Per-scene breakdown

| Scene | GT | geometry | + dino_proposer | + dino_proposer + verifier |
|---|---|---|---|---|
| 2026_07_07_grip_at_bottom | 9 | 6/9 | 6/9 | 6/9 |
| 2026_07_07_grip_at_bottom_no_sides | 3 | 3/3 | 3/3 | 3/3 |
| 2026_07_07_grip_at_bottom_no_sides_high | 3 | 2/3 | 3/3 | 3/3 |
| 2026_07_07_grip_at_bottom_rotated | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_07_grip_at_top | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_07_grip_at_top_rotated | 3 | 1/3 | 1/3 | 1/3 |
| 2026_07_07_misaligned_stack | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_07_partially_wet | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_07_pickup | 1 | 1/1 | 1/1 | 1/1 |
| 2026_07_07_pickup_and_stack_rotated | 7 | 6/7 | 6/7 | 6/7 |
| 2026_07_07_pickup_and_stack_rotated2 | 7 | 5/7 | 6/7 | 6/7 |
| 2026_07_07_pickup_and_stack_rotated3 | 2 | 1/2 | 1/2 | 1/2 |
| 2026_07_07_static | 3 | 3/3 | 3/3 | 3/3 |
| 2026_07_07_wall | 2 | 2/2 | 2/2 | 2/2 |
| 2026_07_07_wall2 | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_07_wall3 | 2 | 2/2 | 2/2 | 2/2 |
| 2026_07_15_shadow | 3 | 2/3 | 2/3 | 2/3 |
| 2026_07_15_shadow2 | 1 | 1/1 | 1/1 | 1/1 |
| 2026_07_15_shadowpartial | 1 | 1/1 | 1/1 | 1/1 |

## dino_proposer region funnel

Across all snapshots: 35 regions generated -> 35 survived `select_points_from_mask` projection -> 33 produced an accepted (`fate == "final"`) fitter hypothesis.

## Corrections: two bugs invalidated the first version of this report

The first committed version of this table reported pooled strict recall
**36/62 (58.1%)** and concluded pooling was catastrophic. Both numbers were
artifacts. Recorded here rather than quietly amended.

**Bug 1 -- scores were not comparable across sources.** 60% of the detector's
score is ray evidence (`evidence.hpp::normalized_evidence_features` f[2]/f[3]/f[4]:
covered/expected faces, supported rays vs free-space violations, supported/incident
rays), computed against whatever `SensorContext` it is handed.
`annotation_candidates._run_detection` builds that context from *whatever points it
is given*, so a region crop (~1,400 points) could not accrue the free-space
violations the whole cloud (~114,000 points) would have charged it. Measured
directly -- identical crop, identical fit, identical pose, only the context differs:

    score(own crop context) = 0.7714    score(full cloud context) = 0.7205    +0.0509

The 9 contested blocks were won by region candidates by margins of 0.005-0.09, the
same magnitude. Fixed via `--score-context full` (now the default); the invalid
behaviour is reproducible with `--score-context crop`.

**Bug 2 -- the rerank stage was an arithmetic no-op.** `dedupe_candidates` returns a
list whose members are pairwise separated by more than `radius_m` (its loop
invariant). The rerank pass fed that already-separated list back into the same
function at the same radius, so every candidate trivially passed the separation
test regardless of score: the reranked row was *mathematically identical* to the
pooled row no matter what `dino_verifier` output. Fixed by scoring and
de-duplicating the **pre-dedup** pool in one pass.

Effect of each fix on pooled strict recall:

| configuration | pooled strict | reranked strict |
| --- | --- | --- |
| crop context, no-op rerank (as first published) | 36/62 (58.1%) | 36/62 (58.1%) |
| crop context, rerank fixed | 36/62 (58.1%) | 37/62 (59.7%) |
| **full context, rerank fixed (current default)** | **43/62 (69.4%)** | **43/62 (69.4%)** |

Baseline geometry is 44/62 (71.0%), so score-arbitrated pooling is still net
negative by one block even fully fixed.

## dino_verifier cannot work as an additive scalar, but ranking is intact

`calibration_temperature` in `models/production/dino_verifier/config.json` is
**12.497**. `_calibrate_score` is `sigmoid(logit(raw)/T)`, so at that temperature
the entire raw range 0.01-0.99 maps to a contribution span of just **0.073**
against a `max_score_contribution` cap of +/-0.20:

| raw | calibrated | contribution |
| ---: | ---: | ---: |
| 0.01 | 0.40910 | -0.03636 |
| 0.50 | 0.50000 | +0.00000 |
| 0.99 | 0.59090 | +0.03636 |

Contested geometry margins here are 0.005-0.09, so the verifier frequently
*cannot* overcome them even in principle. A temperature of 12.5 on a sigmoid head
is itself a red flag that the logits were near-degenerate when it was fit; it
should be re-fit on a held-out split with its NLL/ECE recorded.

**But the transform is strictly monotonic in `raw`.** Ranking is preserved exactly.
So the verifier is unusable as an additive score contribution, and unaffected by
this bug when used as an **argmax over pose hypotheses for one block** -- which is
what its training data actually describes: positives are GT +/-5 cm / +/-8 deg,
negatives are yaw 90/270 deg, position 0.35/0.6 m, tilt 25/45 deg
(`manifest.json::candidate_params`). It was trained to answer "does this projected
outline align with the block in RGB", not "is candidate A better than candidate B".

## The real bottleneck is instance separation, not detection

Measured per block: does any predicted region cover its silhouette (>=30%), and is
that region shared with another block?

| | coverage | found |
| --- | --- | --- |
| every GT block | **62/62 (100%)**, incl. **10/10 `heavy_occlusion`** | -- |
| block alone in its region | 17 | **17/17 (100%)** |
| block sharing a region | 45 | 31/45 (69%) |

By visibility, for shared regions: `full` 11/11, `partial` 13/20,
**`heavy_occlusion` 3/9**. All 14 still-missed blocks are in merged regions; no
isolated block is ever missed.

So `dino_proposer` already detects everything, including the occluded blocks it was
built for. The loss is entirely downstream: 35 regions cover 62 blocks and 18 of
those regions contain more than one block, so the fitter receives a merged
multi-block point set and emits one box. The head is *trained* to merge --
`rgb_region_prior/export.py` builds a single **union** foreground mask per snapshot
by design ("is there a block anywhere in this region", not "which specific block"),
on the v1 assumption that geometry would separate instances downstream. Geometry
cannot separate touching blocks; that is the problem the RGB path exists to solve.

**Next step, in order:** (1) make the proposer instance-aware -- add a seam/boundary
output channel (labels are free, `block_silhouette_mask` already renders per-block
silhouettes) and split the foreground by seams; cheap zero-retraining probe first:
raise the threshold or watershed the existing probability map to see how much
merging is a weak neck. (2) Use `dino_verifier` as an argmax pose disambiguator over
a per-candidate hypothesis set, not as a scalar. (3) Pool gap-only -- region
candidates may only fill blocks geometry found nothing for; this also makes Bug 1
structurally impossible. Measured gap-only: 48/62 centre, **45/62 strict (72.6%)**,
+1 over baseline with zero regressions.
