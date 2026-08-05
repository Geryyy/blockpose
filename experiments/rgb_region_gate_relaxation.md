# Region-scoped gate relaxation: negative, and it exposed the real mechanism

**Result: no effect. But the diagnostics that explain why change the picture.**

## The idea

8 of 10 `heavy_occlusion` blocks were recorded as rejected at the proposal
gate on `extent_z` (median 0.19 m against `cluster_min_extent_z = 0.30`), and a
*global* relaxation of that gate was already falsified (recall unchanged,
precision down). So: relax it **only inside a region `dino_proposer` asserts
contains a block**, keeping it strict everywhere else. The RGB region is the
evidence that a thin band is a block rather than clutter.

## Measured — gap-only pooling, relaxation swept

| `cluster_min_extent_xy/z` | centre | strict | detections | `heavy_occlusion` |
| --- | ---: | ---: | ---: | ---: |
| 0.30 (control, no relaxation) | 48/62 | 45/62 | 63 | 4/10 |
| 0.20 | 48/62 | 45/62 | 63 | 4/10 |
| 0.15 | 48/62 | 45/62 | 63 | 4/10 |
| 0.10 | 48/62 | 45/62 | 63 | 4/10 |

Identical at every level. Relaxing the gate to a third of its shipped value
changes nothing.

**A first run appeared to show +2/+1 and was wrong**: its control skipped
adding region candidates entirely, so it compared *bare baseline* against
*gap-only pooling with relaxation* and credited the relaxation with the pooling
gain we already knew about. Recorded because the confound is easy to repeat.

## Why — the blocks are not rejected, they are mis-fitted

Per-stage diagnostics on the 7 still-missed `heavy_occlusion` blocks, region-
cropped with the gate relaxed to 0.10: **all 7 produce a hypothesis.** Not one
fails at proposal, plane fitting, or top/side pairing. Their fates:

| | |
| --- | --- |
| nearest hypothesis to the true block | **median 1.12 m** (min 0.098, only **1 of 7** within 0.30 m) |
| fate | 6 `final` (accepted), 1 `pre_refinement_selection_rejected` |
| score of the nearest | 0.65 – 0.88 |

So the pipeline confidently accepts a well-scored box a metre from the block.
The region crop contains the gripper and neighbouring structure, the occluded
block contributes only a ~0.18 m band, and the fitter locks onto the dominant
geometry instead. It scores well because the score measures how well *that box*
explains the points it chose — the same per-hypothesis blindness that made
`multi_block_hypotheses` fail.

The gate was never the binding constraint. Admitting a component does not help
when the component being fitted is not the block.

## What this points at next, and it is cheap

**The evaluation path never removes the gripper.** `filter_pzs100_gripper`
(the shipped PZS100 self-filter, in the C++ core and used by the ROS adapter
and the annotation tool) is called by `annotation_candidates.py`,
`annotation_preprocess.py` and `tools/annotate_snapshot.py` — but **not** by
`tools/evaluate_detector_snapshots.py` or `tools/eval_dino_proposer_pooled.py`.
Both feed the detector a raw cloud that still contains gripper returns.

Two consequences, both worth testing before anything else:

1. The occluded-block failure diagnosed above is *exactly* what removing the
   gripper should fix — it is the gripper's own returns that the fit is locking
   onto. RGB says where the block is; FK says which points are the gripper.
   Neither is doing the job alone, and they have never been combined here.
2. Every number in this repo's evaluation may be measured **without** a filter
   production applies, which would make the recorded baseline a lower bound
   rather than an estimate. This needs checking before the field-test numbers
   are compared against anything.
