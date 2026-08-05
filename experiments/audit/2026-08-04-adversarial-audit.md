# Adversarial audit of detector/SAM empirical claims

Audited read-only at branch `feat/annotation-tool-hardening`, commit `b024eb355db14b2c1bb07f2786969817938fcfaa`. The repository remained clean. Scratch scripts and results are under `/tmp/blockpose-audit` only. Native core: `0.1.0`, `vision_enabled=True`; CUDA: RTX 5070 Ti.

## Findings, ordered by severity

### HIGH — CONFIRMED: the documented SAM reproduction command no longer reproduces the report

Affected: `experiments/sam_region_proposals.md:3,8-28,32-34,81-85`; `experiments/detector_cluster_failures.md:122-144`.

Running the report's own grid-8 command at the audited commit produced baseline **46/62, 46/59** and pooled **49/62, 49/74**, recovering 3 of 16 current misses with 14 new unmatched SAM detections. The report says baseline **39/62, 39/52** and pooled **45/62, 45/69**, recovering 6 of 23 with 13 new FPs. The cause is configuration drift: `_run_detection` now uses the shipped 2.25-m extent default, while the recorded SAM report used the old 1.5-m baseline. The report command does not freeze detector parameters.

The historical numbers may describe an earlier configuration, but they are not reproducible from the cited command at the claimed commit. `/tmp/blockpose-audit/sam8.{md,json}` contains the rerun.

### HIGH — CONFIRMED: “17/23 recovered” is 17 fitter successes, but only 16 detections under the declared 0.30-m rule

Affected: `experiments/detector_cluster_failures.md:115-131`; `experiments/sam_region_proposals.md:11-14,146-149`.

Using each missed GT pose as the seed, its oriented box plus 5 cm as the crop, and `SnapParams(reject_ambiguous=False)` reproduced 17 successful snap calls. One result was **0.3287 m** from GT, however, so only **16/23** meets the reports' 0.30-m detection rule. Median error among valid matches was 0.0250 m. With default ambiguity rejection, the result is 10/23. The report must state the non-default ambiguity setting and distinguish API success from evaluation recovery.

Therefore the 90% oracle ceiling (`39 + 17 = 56/62`) is overstated by one under the declared metric; the matched ceiling is 55/62 = 88.7%.

### HIGH — CONFIRMED: the “unexplained” zero-hypothesis crop anomaly has ordinary deterministic gates

Affected: `experiments/detector_cluster_failures.md:133-136`.

The exact anomaly reproduced: 23 baseline misses, perfect crops, **11 detector matches and 12 zero-raw cases**. Native stage instrumentation classified the 12:

- **7 proposal-gate failures**: no component passed the requirement that both world-axis XY extents be at least 0.30 m and Z extent be at least 0.30 m (`proposals.hpp:271-283`). Three failed XY only, three Z only, one both. Main components had ample points; none failed the size gate.
- **2 plane-classification failures**: RANSAC found side planes but no plane within the top-angle gate, so no pair/hypothesis was synthesized.
- **2 top-height failures**: top support height was below `0.70 * 0.60 = 0.42 m` (`pipeline.hpp:288-303`).
- **1 support-consistency failure**: a valid top/side pair existed, but support-height disagreement exceeded 0.135 m (`pipeline.hpp:288-303`).

This is not starvation and not a C++ defect in executing its configured rules. It is a **misapplied observability assumption**: global minimum AABB extents are used as if every isolated/occluded block must expose 0.30 m along all world axes. The sentence “cropping is not the problem” is too strong: the cropped observations are specifically rejected for partial extents.

A naive global fix was falsified. Lowering min XY/Z extents from 0.30/0.30 to 0.20/0.20 or 0.10/0.10 kept recall at 46/62 and reduced precision from 46/59 (78.0%) to 46/62 (74.2%) or 46/61 (75.4%). The plausible fix is a separate observation-aware region/crop path or seeded fitting, not globally weakening proposal gates.

### MEDIUM — CONFIRMED: the 1.75-m sweep row is wrong

Affected: `experiments/detector_cluster_failures.md:44-52`; possibly “nine settings” at line 54.

At `(cluster_max_extent_xy=1.75, eps=.07)`, the independent rerun gives **40/62 recall and 40/53 precision** (64.5%, 75.5%), not 44/62 and 44/57 (71.0%, 77.2%). The 2.0 row does reproduce as 44/62, 44/57. The main 2.25 result and the 2.25–3.5 plateau reproduce exactly.

The report's own cited `experiments/_sweep_results.json` also contains 40/62 and 40/53 at 1.75, so this is a transcription error in the Markdown rather than run-to-run variation.

### MEDIUM — CONFIRMED: centre-only scoring inflates absolute pose-quality results, but does not explain the 2.25 gain

Affected methodology: both reports' centre-only evaluation.

With symmetry-aware orientation <=30 degrees plus exact oriented-box IoU >=0.25:

| setting | centre-only R/P | strict-pose R/P |
|---|---:|---:|
| 1.5 / .07 | 39/62, 39/52 | 37/62, 37/52 |
| 2.25 / .07 | 46/62, 46/59 | 44/62, 44/59 |
| 2.25 / .08 | 48/62, 48/75 | 43/62, 43/75 |

The same two centre matches at .07 have ~86–88 degree symmetry-aware orientation error. Crucially, all seven 2.25-only recoveries are genuine: 0.006–0.035 m translation error, 1.2–2.9 degree rotation error, IoU 0.87–0.97, and zero baseline hits are lost. Thus C2 survives the attack. C4 becomes stronger: eps .08 has *lower* strict recall than 2.25/.07 as well as much worse precision.

### MEDIUM — CONFIRMED: gripper-clearance numbers are historical 1.5-gate results, not current-tool results

Affected: `config/pzs100_gripper_filter.yaml:17-23`.

Twenty-four snapshots / 59 blocks have usable rail TF; `wall2`'s three blocks are excluded because its captured TF lacks K9/K11 rail frames. At the historical 1.5 detector gate, the comments reproduce:

- clearance 0.00: 34/59 recall, 34/42 precision; 6.296% of GT-box points removed, worst block 55.37%.
- clearance 0.08: 37/59 recall, 37/44 precision; 0.066% removed (rounds to 0.1%).
- clearance 0.10: identical detection counts, zero GT-box points removed.

At the shipped 2.25 gate, the corresponding counts are 41/59 and 44/59, not 34/59 and 37/59. The comments need to freeze `cluster_max_extent_xy=1.5` or be updated.

The “knee” interpretation is weak. Extending clearance beyond the stated .10 range is non-monotonic: at .12/.14 the 1.5-gate detector reaches 40/59, at .24 it reaches 41/59, and other values fall back to 39/59. Once GT-point deletion reaches zero, admitted robot returns change clustering/hypotheses unpredictably. This is an in-sample rig-specific trade-off, not a validated geometric knee.

### MEDIUM — CONFIRMED: fallback-derived GT strongly affects the oracle conclusion

Affected methodology: M4 and C6.

Of 23 baseline misses, provenance is 13 `rgb_mask`, 9 `rgb_mask_pca_fallback`, and 1 `auto_fit`. Perfect-crop detector recovery is only **2/9** for fallback GT versus **9/14** for the rest. Seeded snap success is 3/9 fallback, but only 2/9 satisfy 0.30 m, versus 14/14 success/match for non-fallback misses. Hence the oracle is materially conditioned on approximate GT pose/crop orientation. By contrast, all seven gate-2.25 recoveries are `rgb_mask`, so C2 does not depend on fallback GT.

### LOW — SUSPECTED: precision is annotation-relative, not physically adjudicated

Affected: all precision claims, especially `sam_region_proposals.md:18-21`.

No `unmatched_detection_reviews` entries exist for the annotated snapshots, and the reports do not show an RGB/cloud review of unmatched detections. I did not certify whether any are real but unlabelled blocks. Therefore the numerical precision against YAML is reproduced where stated, but “false positive” as a physical claim remains unverified. Best/worst relabelling bounds require manual visual adjudication.

### LOW — CONFIRMED limitation: there is no independent held-out validation

Affected: `detector_cluster_failures.md:150-159`; `sam_region_proposals.md:123-128`.

The reports disclose this correctly. `config/rgb_ablation_splits_seibersdorf.yaml` explicitly has `selection_eligible:false`, `final_enabled:false`, and grouping based on scenario directory prefixes rather than validated run IDs. Leave-one-scenario-out would be a sensitivity analysis, not independent generalisation evidence.

## Claim disposition

- **C1 baseline 39/62, 39/52:** CONFIRMED with explicit 1.5/.07 parameters.
- **C2 2.25 gives 46/62, 46/59; zero lost:** CONFIRMED, including strict pose quality.
- **C3 flat 2.25–3.5 at eps .07:** CONFIRMED. The separate 1.75 row is false.
- **C4 eps .08 gives 48/62, 48/75 (64%):** CONFIRMED centre-only; strict pose quality is 43/62, 43/75.
- **C5 clearance 0 versus .08:** CONFIRMED only under unstated historical 1.5 gate; stale for shipped defaults and not robust beyond .10.
- **C6 perfect crops:** detector 11/23 and zero-raw 12/23 CONFIRMED; snap 17 successes but only 16/23 matches, so recovery/oracle claim REFUTED as written. SAM subclaims are not reproducible from the documented command after detector-default drift.

## Remaining 13 non-merged baseline misses

Using the report artifact's own component assignment to identify the 13 (that assignment is a Python proxy, so this classification is SUSPECTED at the membership boundary), exact perfect-crop traces show: 7 become detections when isolated (whole-cloud interference/context), 4 still fail minimum extent gates, and 2 fit only side planes with no top candidate. This directly contradicts the report's statement that all 13 were uninvestigated, but does not replace a full per-component whole-cloud native lineage audit.

## Reproduction artifacts

- `crop-stages.json`, `crop-stages.log`: exact crop stage traces.
- `strict-metrics.json`: centre/orientation/IoU metrics and seven recoveries.
- `gripper-sensitivity-cme1.5*.json`: clearance sweep.
- `sam8.md`, `sam8.json`: current documented-command SAM rerun.
- `snap-oracle.json`: seeded oracle results.
- `plateau.json`, `min-extent-sweep.json` where completed; console values were also captured during the audit.

Unable to verify: the historical grid-6 SAM seeded-snap 7/23 result (no frozen masks or exact seed/ambiguity convention), physical status of unmatched candidates, and true held-out generalisation.
