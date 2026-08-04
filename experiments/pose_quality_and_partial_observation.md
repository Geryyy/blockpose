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
