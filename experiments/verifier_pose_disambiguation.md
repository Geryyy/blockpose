<!-- provenance: commit b9c1b25 -->
# `dino_verifier` within-block pose disambiguation

Tests `dino_verifier` in the role its own training data (`experiments/rgb_candidate_prior/candidates.py`) actually built it for: picking the right pose among yaw siblings of the SAME block, not comparing different blocks against each other. See module docstring of `tools/eval_verifier_pose_disambiguation.py` for the full method.

## Method

1. Run the whole-cloud PCL detector (`annotation_candidates._run_detection`) on each of the 25 annotated snapshots; keep `fate == "final"` candidates.
2. For each accepted candidate, build a pose hypothesis set: the candidate as-is plus yaw rotations of 90, 180, 270 deg about the candidate's OWN up axis (its body axis nearest world +Z) through its own position, mirroring `candidates.py`'s hard-negative construction. A rotation that is a genuine symmetry of the block's own extents (`blockpose.pose.symmetry_equivalents`) is dropped as not distinguishable -- for the canonical 0.9x0.6x0.6 laid block this collapses the set to `{0 deg, 90 deg}` (180 deg is always a symmetry of a rectangular cuboid about its own vertical axis, and 270 deg is then always equivalent to 90 deg). Yawing about world +Z instead would leave both duplicates in the set for any candidate tilted more than ~0.5 deg -- 95% of this dataset's -- silently turning the intended head-to-head argmax into a best-of-two-per-side one; see the tool's module docstring.
3. Render every hypothesis' RGB crop + conditioning image (`experiments.rgb_candidate_prior.render`), with ALL hypotheses for one block sharing ONE crop bbox (`render.union_crop_bbox`) so ranking cannot leak pose through crop framing.
4. Score every hypothesis with `dino_verifier` (`score_candidate_with_dino`), take the argmax by calibrated score, and use that hypothesis's pose as the corrected pose. An all-abstained hypothesis set (no usable crop, high uncertainty) keeps the original pose.
5. Score centre recall (`blockpose.evaluate.match_detections`, 0.30 m) and strict recall (`blockpose.experiment_record.strict_match`) before and after correction, per block.

**In-sample numbers**: the loaded checkpoint (`/workspace/models/production/dino_verifier`) is trained on the union of all 25 annotated snapshots this reports on, unless it points at a held-out-fold checkpoint instead.

| Metric | Before | After |
|---|---|---|
| Centre recall (<= 0.30 m) | 46/62 | 46/62 |
| Strict recall | 44/62 | 44/62 |

**Strict** additionally requires symmetry-aware rotation error <= 30 deg and oriented-box IoU >= 0.25.

## Addressable pool

Pose disambiguation can only ever change blocks that are already found at centre level but fail strict -- it cannot manufacture a centre-level match `match_detections` never made, and it cannot fix a candidate that fails strict on IoU alone (a yaw rotation does not change fit quality). That pool has **2** block(s) out of 62 GT across all 25 snapshots. Of those, 2 were fixed (centre-match retained, strict now passes) and 0 were not.

**2 block(s) that were strict-passing BEFORE correction broke AFTER it** -- correction moved a candidate that was already right onto a wrong hypothesis. This is a real regression, not noise, and is itemised in the per-block table below.

**Read this in context**: 2 out of 62 GT blocks is a small pool. A result of +1 or +2 strict recoveries here is inside this dataset's noise floor and should not be reported as a generalizable improvement without a held-out check across more scenes.

## Per-block outcomes

Only blocks in the addressable pool (centre-matched, strict-failing before correction) or that regressed are listed individually; every other block's strict status is unchanged by construction (see Addressable pool above).

| Scene | Snapshot | GT id | Centre before/after | Strict before/after | Chosen yaw (deg) | Usable/total hypotheses | Outcome |
|---|---|---|---|---|---|---|---|
| 2026_07_07_grip_at_bottom_no_sides | 2026_07_07_grip_at_bottom_no_sides__1783428274_923417088 | manual_0 | Y/Y | N/Y | 90 | 2/2 | GAINED |
| 2026_07_07_partially_wet | 2026_07_07_partially_wet__1783431057_617312512 | manual_1 | Y/Y | Y/N | 90 | 2/2 | BROKEN |
| 2026_07_07_pickup | 2026_07_07_pickup__1783429278_921587712 | auto_0 | Y/Y | N/Y | 90 | 2/2 | GAINED |
| 2026_07_07_wall | 2026_07_07_wall__1783429196_422551296 | manual_1 | Y/Y | Y/N | 90 | 2/2 | BROKEN |

## Per-scene aggregate

| Scene | GT | Strict before | Strict after |
|---|---|---|---|
| 2026_07_07_grip_at_bottom | 9 | 6/9 | 6/9 |
| 2026_07_07_grip_at_bottom_no_sides | 3 | 2/3 | 3/3 |
| 2026_07_07_grip_at_bottom_no_sides_high | 3 | 2/3 | 2/3 |
| 2026_07_07_grip_at_bottom_rotated | 3 | 2/3 | 2/3 |
| 2026_07_07_grip_at_top | 3 | 2/3 | 2/3 |
| 2026_07_07_grip_at_top_rotated | 3 | 1/3 | 1/3 |
| 2026_07_07_misaligned_stack | 3 | 2/3 | 2/3 |
| 2026_07_07_partially_wet | 3 | 2/3 | 1/3 |
| 2026_07_07_pickup | 1 | 0/1 | 1/1 |
| 2026_07_07_pickup_and_stack_rotated | 7 | 6/7 | 6/7 |
| 2026_07_07_pickup_and_stack_rotated2 | 7 | 5/7 | 5/7 |
| 2026_07_07_pickup_and_stack_rotated3 | 2 | 1/2 | 1/2 |
| 2026_07_07_static | 3 | 3/3 | 3/3 |
| 2026_07_07_wall | 2 | 2/2 | 1/2 |
| 2026_07_07_wall2 | 3 | 2/3 | 2/3 |
| 2026_07_07_wall3 | 2 | 2/2 | 2/2 |
| 2026_07_15_shadow | 3 | 2/3 | 2/3 |
| 2026_07_15_shadow2 | 1 | 1/1 | 1/1 |
| 2026_07_15_shadowpartial | 1 | 1/1 | 1/1 |

## Verdict

2 block(s) gained and 2 block(s) broken -- they exactly cancel, so the aggregate strict recall is unchanged (44/62 both before and after) even though 4 individual blocks' poses actually changed. This is precisely the failure mode the audit that motivated this script was worried about: an unchanged (or improved) aggregate number can hide real per-block regressions. On this dataset, within-block disambiguation via argmax over a {90 deg, base} hypothesis set is evidence-neutral at best -- it does not net-improve strict recall, and it is not free of regressions. It should not be described as a working correction mechanism without a larger, held-out evaluation.
