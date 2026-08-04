# Ground-truth orientation, and why partial-observation placement stalls

Two results from the same measurement pass, at the shipped detector defaults
(`cluster_max_extent_xy=2.25`, `dbscan_eps=0.07`). Scored with
`blockpose.experiment_record.strict_match`: centre <= 0.30 m, symmetry-aware
rotation <= 30 deg, oriented-box IoU >= 0.25.

## 11 of 62 ground-truth poses have the wrong long axis

Re-fitting every approved pose from the points inside its own box:

| how the pose was created | n | refit succeeded | median position delta | median rotation delta |
| --- | ---: | ---: | ---: | ---: |
| `auto_fit` (detector) | 13 | 13 | 0.025 m | **2.0 deg** |
| `rgb_mask` (fitted from an RGB selection) | 6 | 6 | 0.023 m | **1.9 deg** |
| `rgb_mask_pca_fallback` (hand-placed) | 12 | 6 | 0.168 m | **88.4 deg** |

The comparison is symmetry-aware, so a 90 deg disagreement is not a face
relabelling — it means the recorded long axis points the wrong way. Poses the
fitter produced agree with their own geometry to about 2 degrees, so the
disagreement is specific to the fallback rather than a property of the metric.

`tools/audit_annotation_poses.py` lists 11 suspect blocks in total (a few
`rgb_mask` and one `auto_fit` pose are also flagged near 85-90 deg). It only
reports: an automatic refit is the same mechanism that produced the suspect
poses, so it is not trustworthy as a replacement. These need a human in
`tools/annotate_snapshot.py`.

**This contaminates strict-scored evaluation.** A detection can be correct and
still score as a miss when the reference orientation is 90 deg out.

## Partial-observation placement does not clear the bar

Three ways of placing a block from a perfect (GT-box) crop of each of the 16
blocks the shipped detector still misses:

| method | centre-only | strict | median rotation error |
| --- | ---: | ---: | ---: |
| single dominant face + gravity prior | 11/16 | **0/16** | — |
| PCA fallback (`pca_oriented_block_pose`) | 11/16 | **2/16** | 41.3 deg |
| seeded fitter (`snap_block_to_cloud`) | 9/16 | **5/16** | 44.1 deg |

Centre-only scoring would have reported the first row as 11/16 and a success.
It places boxes near blocks, not on them. This is the case strict scoring was
added for, and it caught it on first use.

But the strict column is itself depressed by the bad ground truth above.
Excluding blocks whose GT came from the PCA fallback:

| GT subset | seeded fitter, strict |
| --- | ---: |
| all 16 missed blocks | 5/16 = 31% |
| excluding `pca_fallback` GT | **4/7 = 57%** |

So the honest reading is: the method is roughly twice as good as the headline
suggests, and the headline is dragged down by references that are themselves
wrong. **Neither number is a basis for shipping.** Fix the ground truth first;
re-measure after.

## Consequence for sequencing

Correcting the 11 suspect poses is now a prerequisite, not a tidy-up. Until it
is done:

- strict-scored recall is understated by an unknown amount,
- any claim about orientation accuracy is unsafe,
- and a partial-observation placement stage cannot be evaluated, so it should
  not be built.

The detector changes already shipped (`cluster_max_extent_xy` 2.25, gripper
`inner_clearance_m` 0.08) are unaffected: both were established on centre
distance and independently re-verified under strict scoring by the audit, and
all seven gate recoveries are `rgb_mask` blocks, none of which are suspect.
