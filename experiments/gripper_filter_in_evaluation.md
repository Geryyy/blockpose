# The evaluation was measuring the detector without the gripper self-filter

**Result: precision was understated by ~9 points, and it changes the verdict on
the RGB proposer.**

`filter_pzs100_gripper` is shipped in the C++ core and applied by the ROS
adapter and `tools/annotate_snapshot.py` -- but **not** by
`tools/evaluate_detector_snapshots.py` or `tools/eval_dino_proposer_pooled.py`.
Every recorded number in this repo therefore describes a detector fed a cloud
that still contains the robot's own gripper returns.

## Measured, 25 snapshots / 62 approved blocks

| gripper | RGB pooling | centre | strict | detections | precision | `heavy_occlusion` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | off | 46/62 | 44/62 | 59 | 78.0% | 3/10 |
| raw | on (gap-only) | 48/62 | 45/62 | 63 | 76.2% | 4/10 |
| **filtered** | off | 46/62 | 44/62 | 53 | **86.8%** | 3/10 |
| **filtered** | on (gap-only) | 48/62 | 45/62 | 55 | **87.3%** | 4/10 |

Filtering removed 42,176 points across the corpus and eliminated 6 detections
(8 with RGB pooling) **at zero recall cost** -- every detection it removed was a
false positive. Recall is byte-identical in all four rows; only precision moves.

One snapshot failed to filter (missing rail TF) and was scored unfiltered.

## Two consequences

**1. The shipped baseline understates production precision.** Production
applies this filter; the evaluation did not. The honest baseline for the
geometry detector is **46/62 centre, 44/62 strict, 86.8% precision**, not the
78.0% recorded elsewhere in this repo. Field-test results should be compared
against the filtered number.

**2. It changes the verdict on `dino_proposer`.** Unfiltered, gap-only pooling
looked like a marginal trade: +2 centre, +1 strict, precision 78.0% -> 76.2%,
i.e. paying ~2 points of precision for 2 blocks. Filtered, the same pooling
gives +2 centre, +1 strict with precision going **86.8% -> 87.3%** -- slightly
*up*. The precision cost was never the RGB regions; it was gripper returns
being fitted as blocks, which the RGB path happened to surface more of.

So the RGB proposer, used gap-only on a properly filtered cloud, is a small but
**clean** win rather than a trade.

## What it does not fix

`heavy_occlusion` recall is unchanged (3/10, and 4/10 with RGB). The hypothesis
that motivated this test -- that the mis-fitted hypotheses documented in
`rgb_region_gate_relaxation.md` were locking onto the gripper's own returns, so
removing them would recover the blocks -- is **falsified**. The fits still land
a median ~1.1 m away. Whatever they are locking onto, it is not the gripper.
