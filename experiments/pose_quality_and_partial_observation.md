# Occlusion, not annotation error: why the last blocks resist geometry

Superseded conclusion first, because the earlier version of this file was
wrong. It claimed 11 of the 62 approved poses had a wrong long axis, inferred
from re-fitting each pose from its own points and finding ~88 degrees of
symmetry-aware disagreement on blocks placed with the annotator's PCA fallback.

**The ground truth is correct.** The operator checked the flagged snapshots.
The disagreement measures the REFIT failing, not the label. The confound: the
PCA fallback is used precisely BECAUSE a block is too occluded for the fitter,
so `pca_fallback` is a proxy for occlusion and I read it as a cause of error.

## What the disagreement actually tracks

Refit-vs-recorded rotation, grouped by the annotator's own visibility label:

| visibility | n | median rotation delta | over 30 deg |
| --- | ---: | ---: | ---: |
| `full` | 23 | **1.2 deg** | 0 |
| `partial` | 20 | 2.1 deg | 3 |
| `heavy_occlusion` | 8 | **87.8 deg** | 7 |
| `unknown` | 5 | 2.0 deg | 1 |

Every fully visible block agrees to about a degree. The metric is a measure of
**observability**, and `tools/audit_annotation_poses.py` now reports it as such.

## The occluder is the gripper, and FK locates it exactly

Ray-casting each block's surface from the sensor against the FK gripper boxes
and against every other annotated block:

| visibility | n | surface hidden by the gripper | by other blocks | median rotation delta |
| --- | ---: | ---: | ---: | ---: |
| `full` | 23 | 0% | 0% | 1.2 deg |
| `partial` | 24 | 29% | 8% | 2.1 deg |
| `heavy_occlusion` | 10 | **76%** | 0% | **87.8 deg** |

Two things follow. Occlusion is caused by the gripper, essentially never by
neighbouring blocks; and it is computable **before fitting**, from FK alone,
with no inference. A detector can know which hypotheses it is unable to
constrain rather than discovering it by failing.

## Where this points, and what it rules out

`evidence.hpp` already computes `VisibilityEvidence` -- `incident_rays`,
`supported_rays`, `expected_faces`, `covered_faces` -- from sensor geometry.
It has no notion of a known occluder. A ray blocked by the gripper stops short,
so its range never matches the hypothesis surface: it is counted as incident
but unsupported, and the hypothesis is penalised for geometry the gripper is
standing in front of. Excluding occluder-blocked rays from `incident_rays` is a
bounded change in the core, and it is the change these measurements support.

**A second camera viewpoint does not help.** The camera sits 0.17 m from the
LiDAR, so it shares the shadow almost exactly:

| viewpoint | median surface hidden by gripper |
| --- | ---: |
| LiDAR | 76% |
| camera | 74% |
| hidden from BOTH | 74% |

In 0 of 10 heavily occluded blocks does the camera see more than 20 percentage
points beyond what the LiDAR sees. RGB cannot recover the hidden geometry of
THESE blocks by parallax -- it would have to infer the pose from the visible
quarter, which is a different and harder claim than "RGB adds information".
That is not an argument against RGB in general: RGB measurably helps
*finding* blocks the geometry stage never proposes (recall on `rgb_mask` blocks
rose 19% -> 44% with SAM regions). It is an argument that the occluded-pose
problem specifically will not be solved by adding a camera at this baseline.

## Partial-observation placement, re-read

Three placements on perfect crops of the 16 blocks the shipped detector misses:

| method | centre-only | strict |
| --- | ---: | ---: |
| single dominant face + gravity prior | 11/16 | **0/16** |
| PCA fallback | 11/16 | **2/16** |
| seeded fitter | 9/16 | **5/16** |

The first row is the case strict scoring was added for: centre-only would have
called it 11/16 and a success, and it is placing boxes near blocks rather than
on them.

Given the corrected reading above, the strict column is not depressed by bad
labels -- it is the honest score against correct labels on blocks that are 76%
hidden. Orientation is genuinely underdetermined there. Better placement
heuristics will not fix it; only supplying the missing constraint will, which
is what occlusion-aware evidence is for.

## Occluder-aware evidence: implemented, measured, and it does not help

`SensorContext` now carries `occluders`, and `visibility_evidence` drops rays a
known occluder stopped rather than counting them as incident-but-unsupported
(commit b83c70d). The reasoning holds and the unit tests confirm the mechanism.
Supplying the FK gripper boxes across the annotated set:

| gripper occluders | recall (centre) | recall (strict) | precision |
| --- | ---: | ---: | ---: |
| off | 44/59 = 75% | 42/59 = **71%** | 44/57 = 77% |
| on | 44/59 = 75% | 38/59 = **64%** | 44/57 = 77% |
| heavy_occlusion blocks only | 3/10 | 3/10 | — |

No gain, and strict recall drops 7 points. Dropping occluded rays removes their
penalty but also their support, so score denominators shrink and hypotheses
become less separable — a worse-oriented one can now score comparably.

**The diagnosis is that this targets the wrong stage.** For the heavily
occluded blocks:

    a raw hypothesis exists near the block : 3/10
    one survives to "final"                : 3/10

Seven of ten are lost with **no hypothesis ever generated**. Evidence scoring
runs on hypotheses; where none exists it cannot help however well it reasons.
This matches the earlier stage instrumentation, where 7 of 12 zero-hypothesis
crops failed the 0.30 m minimum XY/Z extent gate in `proposals.hpp` — before
any evidence is computed.

The occluder primitive is kept because it is correct and inert by default
(empty occluders reproduce previous behaviour exactly, and the ROS consumer
builds against it), but **it should not be enabled**: on this data it is a
small regression.

### Where the ray-cast idea should act instead

Hypothesis GENERATION, not scoring. A component is currently rejected for being
under 0.30 m in XY or Z. When that shortfall is explained by the gripper's
shadow — which FK gives us before fitting — the gate is measuring the occluder,
not the block, and rejecting on it is the error. Relaxing the gate *only where
the missing extent falls inside a known shadow* is narrow, keeps the gate's
protection everywhere else, and is the thing these measurements actually
support. A global relaxation was already falsified: recall unchanged, precision
down.

## Two mechanisms built, measured, and reverted

Both targeted a stage that is not where these blocks fail. Recording them so
the next attempt does not repeat either.

**Occluder-aware visibility evidence.** `SensorContext` carried known occluder
volumes and `visibility_evidence` dropped rays a known occluder had stopped,
instead of counting them incident-but-unsupported. Correct in principle and
unit-tested. Supplying the FK gripper boxes across the annotated set:

| gripper occluders | recall (centre) | recall (strict) | precision |
| --- | ---: | ---: | ---: |
| off | 44/59 = 75% | 42/59 = **71%** | 77% |
| on | 44/59 = 75% | 38/59 = **64%** | 77% |

No gain and a 7-point strict regression: dropping occluded rays removes their
penalty but also their support, shrinking score denominators and making
hypotheses less separable.

**Shadow-aware relaxation of the minimum-extent gate.** A component under the
0.30 m minimum was to be admitted when the missing extent lay inside a known
shadow. Swept at 0.5/0.7/0.9 hidden-fraction thresholds: recall 44/59 at every
one, identical to off. It never fires.

### Why neither worked: the diagnosis was an artifact

The evidence for "these blocks fail the extent gate" came from re-running the
detector on clouds CROPPED to a single block. Cropping naturally produces short
extents, so the gate failures were an artifact of the probe, not pipeline
behaviour. On the whole cloud:

| component containing a heavy_occlusion block | ext X | ext Y | ext Z | points |
| --- | ---: | ---: | ---: | ---: |
| median of 10 | 0.88 | 0.62 | 0.76 | 328 |

**0 of 10 fail any proposal gate.** They pass comfortably, so a waiver on that
gate can never fire, and evidence scoring cannot help either because 7 of the 10
never produce a hypothesis at all.

The failure is therefore downstream of clustering and upstream of scoring --
region splitting, plane fitting, or top/side pairing. The audit's crop-level
instrumentation is consistent with that (2 plane-classification, 2 top-height,
1 support-consistency failures), and a top-plane requirement is exactly what a
gripper standing over a block would defeat. But that is now a hypothesis, and
the last two were also hypotheses that measured well on a probe and did nothing
in the pipeline.

### What the next attempt needs first

`ProposalComponentDiagnostics` and the plane/pairing stage outcomes exist in
C++ but are not exposed through the bindings, which is why every diagnosis here
had to be reconstructed from an external probe -- and why two of them were
wrong. Exposing the per-stage outcome for a whole-cloud run, so a block can be
traced from component to rejection reason without a stand-in, is the
prerequisite for targeting anything. It is also cheap next to building a third
mechanism on a guess.

## The stage diagnostics settle it, and retract the section above

`ProposalComponentDiagnostics`, wide-proposal skips, plane-region, plane-patch
and pairing records were already computed by the detector and simply never
crossed the binding. They are now on `InferenceTrace`. No detector logic
changed.

Tracing each heavily occluded block through the REAL pipeline:

| where the block is lost | count |
| --- | ---: |
| component rejected at the gate -- `extent_z` | 5 |
| component rejected at the gate -- `size`, `extent_xy`, `extent_z` | 2 |
| component rejected at the gate -- `extent_xy`, `extent_z` | 1 |
| gate passed, but no plane patches fitted | 2 |

**8 of 10 fail at the proposal gate, dominated by the Z extent.** The section
above concluded "0 of 10 fail any proposal gate" from an sklearn DBSCAN
stand-in; that stand-in does not reproduce the pipeline's own clustering, and
the conclusion was wrong. It is retracted here.

The nearest component to a heavily occluded block measures a median **0.19 m**
in Z against a 0.30 m gate -- typically 0.59 x 0.41 x 0.18 with ~110 points.
Partial and full blocks sit at 0.63 m and 0.52 m and pass. The gripper leaves a
thin visible band of a 0.6 m block, and the gate reads that band as too small
an object.

### What this does and does not license

It does NOT revive the shadow-aware waiver as written. That was measured and
changed recall at no threshold, and the reason is now visible: even admitted, a
0.18 m band is unlikely to yield a correct 0.6 m pose, because the top face the
fitter pairs against is exactly what the gripper hides. This is consistent with
the strict-scoring result -- orientation is underdetermined at 76% occlusion no
matter which stage lets the component through.

The honest position is that three probes have now disagreed with each other,
and only the last used the detector's own records. Any further change here
should be justified from those records, and measured with strict scoring,
before it is written.

## The missing constraint is the gripper pose, and it was there all along

The operator's objection was that seeing the RGB made the annotation obvious,
so this must be a method bottleneck rather than a sensing limit. That is
correct, and it identifies a prior the detector never had: **a block in this
workcell is aligned with the gripper that placed or holds it.** FK gives that
for free.

Orientation error against ground truth, all 59 blocks with rail TF, comparing
the gripper's tool-centre-point orientation against a world-aligned control:

| visibility | n | TCP orientation | world-aligned control |
| --- | ---: | ---: | ---: |
| `full` | 21 | **3.7 deg** | 23.3 deg |
| `partial` | 23 | **5.5 deg** | 86.9 deg |
| `heavy_occlusion` | 10 | **6.5 deg** | 88.7 deg |
| `unknown` | 5 | 28.4 deg | 83.8 deg |

| | n | TCP orientation error |
| --- | ---: | ---: |
| blocks at the grasp point (<0.6 m) | 14 | **2.3 deg** |
| blocks merely behind the gripper | 45 | 6.4 deg |

The control matters: a fixed world-aligned guess scores 23-89 deg, so the scene
is not trivially axis-aligned and the TCP is genuinely carrying the
information. On exactly the blocks where the visible geometry disagrees with
itself by 88 deg, the gripper pose is right to 6.5 deg.

This reframes everything above. The earlier conclusion -- "orientation is
underdetermined at 76% occlusion" -- is true of the CLOUD IN ISOLATION and
false of the problem. A 0.18 m visible band plus a known orientation is a
solvable pose: the band fixes position, the prior fixes rotation, and the block
model supplies the extent the gripper hides. That is the same reasoning the
operator applies when the RGB makes it obvious.

### Why non-gripped blocks align too, and what that costs

Blocks merely behind the gripper still track it to 6.4 deg because this robot
placed them; they inherit its yaw. That is a real regularity of the workcell,
not a law. It is also in-sample: 59 blocks from 24 scenes of one cell, and the
`unknown`-visibility blocks at 28.4 deg are a reminder that it is not
universal. A detector using it should treat the gripper pose as a PRIOR to be
confirmed by whatever geometry is visible, not as an answer -- otherwise it
will confidently mis-pose the first block a human puts down by hand.

### What to build

Hypothesis generation seeded by the gripper prior, for components the gate
currently rejects: take the visible band's position, take orientation from the
TCP, instantiate a canonical block, and score it with the existing evidence.
That needs no new sensing, no training, and no relaxation of the gates for
everything else -- the prior is what tells you a thin band is a block rather
than clutter. Measure it strict-scored, and against the `unknown`-visibility
blocks specifically, since those are where the prior is weakest.

## Correction: the gripper prior targets blocks that are already found

The section above proposed building the gripped-block case first, on the
grounds that the prior there is causal (2.3 deg) rather than statistical. Two
measurements retire that plan.

**The gripped blocks are already detected.** Of the 14 blocks within 0.6 m of
the FK grasp point, the current detector already finds **12**. The prior has at
most two blocks to contribute, and both are cases where a pose already exists.

**The blocks that fail are hidden by the gripper, not held by it.** The ten
`heavy_occlusion` blocks sit a median **1.23 m** from the grasp point, and only
3 are within 0.6 m. FK therefore gives no handle on their position: they are
elsewhere in the scene and the gripper merely stands in front of them.

Prompting SAM2 in RGB at the projected FK grasp point confirms it directly:

| prompt | usable masks | median points | median % of points on the GT block |
| --- | ---: | ---: | ---: |
| FK grasp point | 10/10 | 833 | **0%** |
| GT block centre (upper bound) | 10/10 | 402 | **33%** |

The FK prompt segments the gripper, which is what is actually at that pixel.
And even a perfect prompt yields a mask only a third of whose projected points
land on the block -- the mask bleeds onto whatever surrounds it.

So the remaining problem is unchanged and correctly stated as: **locate a block
that nothing in the kinematics points at, from a 0.18 m visible band, well
enough to pose it.** That is the region-proposal problem, measured earlier at
7 of 23 recovered with SAM grid regions. The gripper prior does not shortcut
it. The orientation correlation is still real and may still be useful as a
tie-breaker once a candidate exists, but it cannot generate the candidate.

## The occluded-visibilities choice, resolved: include them

The previous commit made `--occluded-visibilities` a real flag on
`export.py` but did not run training -- the GPU was unavailable mid-session.
It now is. Both arms (default: exclude `heavy_occlusion`/`unknown`; `none`:
include every approved block) were trained on the identical 19-fold
leave-one-scenario-out split (folds 12/13/14/16/20 have no exported snapshot
at all -- annotation-incomplete, unrelated to occlusion -- and are excluded
from both arms identically), pooled to one held-out prediction per block
across all folds.

**A measurement bug made the first pass of this comparison meaningless, and
is worth recording.** `report.py` computed each snapshot's recall denominator
via `approved_targets(annotations)` with no override, so it silently used
`export.py`'s own default (exclude `heavy_occlusion`/`unknown`) regardless of
which arm's checkpoint was being scored. Both arms therefore first measured
identical recall (0.85 vs 0.83) on the same 47 easy blocks -- the 15 blocks
either arm was actually trying to move never entered the denominator. Fixed
by giving `report.py` the same `--occluded-visibilities` flag as `export.py`,
threaded into `_block_masks_for_sample`, plus a `recall_by_visibility`
breakdown in the report so this class of result is visible directly instead
of requiring a second bug to notice its absence.

Rescored both checkpoints against the full 62-block set:

| visibility | blocks | excluded-arm recall | inclusive-arm recall |
| --- | ---: | ---: | ---: |
| full | 23 | 22/23 (0.96) | 22/23 (0.96) |
| partial | 24 | 18/24 (0.75) | 17/24 (0.71) |
| heavy_occlusion | 10 | 2/10 (0.20) | **4/10 (0.40)** |
| unknown | 5 | 4/5 (0.80) | 3/5 (0.60) |
| **overall** | 62 | 46/62 (0.742) | 46/62 (0.742) |
| false positives (pooled) | -- | 8 | **2** |

Overall recall is identical -- both arms recover 46 of 62 -- because the
inclusive arm's gain on `heavy_occlusion` (+2 blocks) is offset by a loss of
one `partial` and one `unknown` block. But `heavy_occlusion` is exactly the
class the region head exists to rescue (the geometric detector's own gate
already handles `full`/`partial` reasonably), and there recall doubles. False
positives fall from 8 to 2 across the same 19 folds, a larger and cleaner
effect than the recall swing.

Both counts are small (10 and 5 blocks) and this is one dataset of 24 scenes,
so treat the exact ratios as noisy. But the direction is consistent with the
hypothesis that motivated the flag: excluding `heavy_occlusion`/`unknown` from
the union target means the head never sees a positive example of the one
class it is meant to rescue. **Decision: switch the default to
`--occluded-visibilities none`** and re-export/retrain the deployed checkpoint
on that basis; keep the old default available for anyone re-running this
comparison.
