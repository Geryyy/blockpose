<!-- provenance: commit a9715cc -->
# Composed pipeline ablation: geometry -> +dino_proposer -> +dino_verifier

**In-sample numbers** (see module docstring): the loaded checkpoints are trained on the union of all 25 annotated snapshots this reports on, unless `--proposer-checkpoint`/`--verifier-checkpoint` point at a held-out-fold pair instead.

| Stage | Recall (centre) | Precision (centre) | Recall (strict) |
|---|---|---|---|
| geometry (whole-cloud PCL) | 46/62 (74.2%) | 46/59 (78.0%) | 44/62 (71.0%) |
| + dino_proposer (pooled) | 48/62 (77.4%) | 48/63 (76.2%) | 36/62 (58.1%) |
| + dino_proposer + dino_verifier (reranked) | 48/62 (77.4%) | 48/63 (76.2%) | 36/62 (58.1%) |

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

## Why dino_verifier made zero difference here

Not a bug -- checked directly. Sampling `verifier_adjusted_scores` on the
first 5 snapshots, every pooled candidate's `geometry_score +
verifier_contribution` shifted by a roughly **uniform** -0.05 to -0.07,
regardless of which candidate it was or how good a fit it was. A uniform
per-scene shift can never change which of two competing candidates wins a
dedup tie-break, so the reranked row is identical to the pooled row not
because the verifier abstained (it didn't -- `score_candidate_with_dino`
returned real, non-abstained scores throughout) but because it isn't
discriminating between candidates in this setting at all.

The likely cause is a **crop-distribution mismatch** between how
`dino_verifier` was trained and how this script uses it.
`experiments/rgb_candidate_prior/export.py` builds every training sample
from a `union_crop_bbox` shared across all candidate siblings for one
annotated target (deliberately, so ranking can't leak pose information
through different crop framing -- see `render.py`'s module docstring). This
script instead renders each pooled candidate its own individual, tightly
fit crop (`render.crop_bbox` on that candidate alone), since there is no
"one target, several siblings" grouping once candidates come from two
independent, unrelated sources (whole-cloud PCL vs `dino_proposer` regions).
That is a real, plausible distribution shift the standalone verifier
benchmark (93.6% held-out top-1, `models/production/README.md`) does not
exercise, because its evaluation set is built the same shared-crop way
training is.

**Before ever wiring `dino_verifier` into a real rerank role**: re-run this
script's candidates through the shared-crop rendering path instead (crop
per *scene*, or per group of spatially-close competing candidates, rather
than per candidate), and check whether the verifier starts discriminating
once its input distribution matches training again. Until then, treat the
verifier's rerank contribution here as unproven, not merely small.
