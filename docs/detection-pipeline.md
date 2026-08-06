# The geometry detection pipeline — point cloud in, block poses out

What `blockpose::core::detect()` does to a point cloud, stage by stage, and
what it hands back. This is the algorithm `concrete_block_detector` runs; the
ROS package is a thin adapter that converts a `PointCloud2` into `Points`,
supplies TF-derived context, and converts the result into ROS messages. None of
that adapter work is described here — this is the portable core only.

Everything below lives in `cpp/include/blockpose/core/`. There is **no PCL and
no ROS dependency** anywhere in it: the geometry is plain Eigen, the RANSAC and
DBSCAN implementations are the package's own, and every stage is deterministic
for a fixed seed and a fixed input ordering.

## Entry points

```cpp
#include <blockpose/core/blockpose_core.hpp>

DetectionResult detect(
  const Points & input,                       // world-frame XYZ, metres
  const DetectionParameters & params = {},
  const SensorContext * sensor_context = nullptr,
  const PosePriors * priors = nullptr,
  const VisualScorer * visual_scorer = nullptr);

DetectionResult detect_without_refinement(...);  // same, stops before SDF refinement
```

`detect_without_refinement()` is the ablation entry point. It runs stages 1–11
below and returns; `detect()` calls it and then adds stages 12–16. Both are
`inline` in headers — there is no separate compiled detection unit.

### What the four inputs mean

| Input | Required | Effect if absent |
| --- | --- | --- |
| `input` (`std::vector<Eigen::Vector3d>`) | yes | Fewer than 3 points returns an empty result immediately. Points must be finite or `voxel_downsample` throws. |
| `params` | no | Defaults in `DetectionParameters` (`proposals.hpp`) are the shipped contract, asserted by `cpp/tests/consumer/`. |
| `sensor_context` | no, but **material** | Rays/misses from the sensor origin. Without it, three of the five evidence features are structurally zero, so every hypothesis scores at most 0.40 instead of 1.00. See "Evidence score". |
| `priors` | no | No FK-seeded recovery, and `prior_match.score` contributes 0 to selection. |
| `visual_scorer` | no | `visual_evidence` stays unavailable and contributes 0. This is the hook the optional RGB path would use; it is a `Pose -> score` callback, not a region proposer. |

`SensorContext` is validated on every use (`validate_sensor_context`): ray
directions are normalised in place, `ray_origins`/`miss_origins` default to the
single `origin` when empty, and any non-finite or zero-length ray throws.

## The stages

```
input Points
  │
  ├─ 1  voxel_downsample (0.04 m)
  ├─ 2  remove_ground ──────────────────► LocalGroundModel  (plane + 0.5 m cell map)
  │                                       above_support_points  (clearance ≥ 0.05 m)
  ├─ 3  proposal seed filter (height ≥ 0.15 m)
  ├─ 4  dbscan_proposals ───────────────► ≤ 8 ranked components
  │        └─ per component:
  ├─ 5      density_scaled_parameters (range-compensated gates)
  ├─ 6      split_connected_row (density-valley tiling)
  ├─ 7      wide-proposal skip
  ├─ 8      fit_planes  (≤ 5 sequential RANSAC patches, then in-plane splitting)
  ├─ 9      candidate_plane_sets  (top ⨯ best side, per dims variant)
  ├─ 10     synthesize_pose + make_hypothesis  → evidence score
  ├─ 11  score threshold → conflict alternatives → ground bounds
  │                                    [detect_without_refinement returns here]
  ├─ 12  FK prior seeding
  ├─ 13  refine_poses  (Levenberg–Marquardt on the box SDF, 2 passes)
  ├─ 14  rescore + ground bounds
  ├─ 15  select_hypotheses  (greedy OBB non-maximum suppression)
  └─ 16  sort by centre z, descending
       │
       ▼
  DetectionResult
```

### 1. Voxel downsample

`voxel_downsample(input, voxel_size)` — 4 cm cubes by default, floor-indexed,
each occupied voxel replaced by the **mean** of its points (not a representative
point). Input is canonically sorted first and output sorted again, so the result
does not depend on the order points arrived in. Fewer than 3 surviving points
ends the run.

### 2. Ground removal and the local ground model

This produces both the ground model and the cloud everything downstream works
on. Two clouds are involved:

- a coarse 10 cm downsample of the 4 cm cloud, used **only** as the RANSAC
  fitting set — plane fitting on the dense cloud is wasted work;
- the 4 cm cloud, which is what actually gets classified.

RANSAC (`ground_ransac_distance` 0.05 m, 500 iterations, `ransac_seed`) fits a
dominant plane. If its normal is more tilted than `ground_normal_min_z` (|n·z| <
0.7), the fit is rejected as not-a-ground-plane and the code falls back to a
horizontal plane at the 5th percentile of z, tagged `source = "percentile"`.
A RANSAC exception falls back the same way. On success the normal is flipped
upward and tagged `source = "ransac"`.

`make_local_ground_model` then adds terrain on top of that plane. Points are
bucketed into 0.5 m XY cells (`local_ground_cell_size`); a cell with at least 3
points takes the **10th percentile** of its points' residual above the plane,
and if that is within `ground_thickness` (0.15 m) the cell stores
`plane_z(cell centre) + residual`. Cells that are too sparse or sit too far
above the plane store nothing and fall back to the bare plane.

So `LocalGroundModel::support_z(p)` is: plane height at `p`, plus the offset of
the nearest occupied cell centre within `1.75 × cell_size`, or just the plane
height when no cell is that close. `height(p)` is `p.z - support_z(p)`.

`above_ground` keeps points with `height ≥ local_ground_clearance` (0.05 m) —
note this is the *clearance*, not the thickness. On the percentile fallback path
the filter uses `thickness` (0.15 m) instead, so the fallback is strictly more
aggressive. This cloud is kept as `above_support_points` and is what refinement
fits against later.

### 3. Proposal seed filter

DBSCAN does **not** run on `above_ground`. It runs on the subset with
`height ≥ ground_thickness` (0.15 m). The lower band is retained for refinement
but excluded from clustering, because ground-adjacent returns bridge neighbouring
blocks into one component and make a cluttered block pick the wrong side plane.

### 4. Proposals

`dbscan_proposals` clusters with `dbscan_eps` 0.07 m and `dbscan_min_points` 5,
using a spatial hash grid for neighbour queries. Each raw component then passes
four gates:

| Gate | Bounds | Parameter |
| --- | --- | --- |
| point count | 30 … 50 000 | `cluster_min_size` (density-scaled), `cluster_max_size` |
| XY extent, both axes | 0.3 … 3.0 m | `cluster_min_extent_xy`, `region_max_extent_xy` |
| Z extent | 0.3 … 3.5 m | `cluster_min_extent_z`, `cluster_max_extent_z` |
| centroid height above ground | ≤ 3.0 m | `cluster_max_center_z` |

Survivors are ranked by **point count, descending**. The block-multiple
"compatibility" measure — how close the component's extents are to integer
multiples of `block_dims` — only breaks ties between components of identical
size, with the canonical centroid ordering as the final tiebreak. The ranked
list is then truncated to `proposal_max_components` (8).

### 5. Density scaling

`density_scaled_parameters` compensates for range: it takes the median distance
from `sensor_context->origin` to the component's points and scales the
*count*-based gates by `clamp((5 / median_range)², 0.25, 1.0)`, flooring at 12.
Only `min_inliers` and `cluster_min_size` are scaled. Without a `SensorContext`
this is the identity, so a distant sparse block that would pass on hardware can
fail in a fixture that omits the context.

The rounding is written out longhand as ties-to-even to match NumPy rather than
`std::round`, so the ambient floating-point rounding mode cannot move the
detector contract.

### 6. Row splitting

`split_connected_row` handles the case where several abutting blocks form one
connected component. Along the longer horizontal axis it computes
`count = round(extent / block width)` and only proceeds if `count` is 2…8 and
the extent is within 0.12 m of `count × width`.

It then requires *observable* boundaries: at each interior boundary it counts
points in a narrow valley band versus the shoulders either side, and demands the
valley be sparser than 45% of what the shoulder density predicts. If any
boundary fails, the component is returned whole. This is deliberately
conservative — a seamless long object must not get tiled into imaginary blocks.

### 7. Wide-proposal skip

If splitting produced a single region and either horizontal extent exceeds
`cluster_max_extent_xy` (2.25 m), the component is dropped without ever reaching
plane fitting, unless `fit_wide_proposals` is set. It is recorded in
`wide_proposal_skips` — otherwise it would vanish leaving no trace in any other
diagnostic, since the lineage record starts later.

### 8. Plane fitting

`fit_planes` extracts up to `max_planes` (5) planes from a region by sequential
RANSAC, removing inliers each round and stopping when the remainder falls below
`min_inliers` (40, density-scaled).

Each RANSAC pass (`ransac_distance` 0.02 m, 1000 iterations, seed
`ransac_seed + plane_index`) samples from at most `ransac_search_max_points`
(2048) uniformly strided points, adapting its iteration budget to the best
inlier ratio so far. The winning plane is then re-fitted by least squares — the
smallest-eigenvalue eigenvector of the inlier covariance — and the refit is kept
only if it explains at least as many points.

`split_plane_patch` then handles coplanar-but-disjoint surfaces: the tops of two
separate blocks at the same height fit one plane. It finds connected components
*within* the patch by union-find on the in-plane projection, and if at least two
groups reach `min_inliers` it emits the original patch **plus** each group as
separate patches. Both survive to pairing, so a merged-top interpretation and a
split-top interpretation compete on evidence.

### 9. Dimension variants and top/side pairing

`candidate_dims` yields the configured `block_dims` and its XY swap — by default
(0.9, 0.6, 0.6) and (0.6, 0.9, 0.6). Every region is paired independently under
each variant, so both yaw interpretations of a square-ish footprint compete.

`candidate_plane_sets` classifies each patch by the angle of its normal to the
ground normal: **top** if within `top_plane_angle_deg` (30°) of vertical, **side**
if within `side_plane_angle_deg` (20°) of horizontal. A patch can be neither;
with a 30°/20° split it cannot be both.

Tops are processed highest-first. For each top, every side is tested against
four gates, and the best survivor is chosen by `gap + |drop - height/2|`:

| Gate | Condition |
| --- | --- |
| drop range | 0.15 m < (top height − side centroid height) < `dims[2]` − 0.1 m |
| support band | side spans the band from 0.30 m to 0.15 m below the top |
| tangential overlap | ≥ 0.15 m of shared extent along the top×side tangent |
| gap | top points lie within 0.1 m of the side plane |

A top with no acceptable side still emits a pair with a null side. Top-only
hypotheses are a normal output, not a failure path — they just carry
`confidence = 0.5` and one observed face instead of two.

### 10. Pose synthesis and evidence

`synthesize_pose` builds the frame directly:

- local **Z** is the top patch normal, flipped upward;
- local **X** is the side normal projected into the top plane, signed to point
  away from the side patch. With no side, it is the dominant PCA axis of the
  cluster's world-XY footprint — taken in XY rather than in the top plane
  because on a real stacked cluster those two differed by 34° of yaw;
- local **Y** completes the right-handed frame;
- position starts at `top.centroid - Z × dims[2]/2` (i.e. the top face defines
  the centre height), then, when a side exists, slides along X so the side face
  sits exactly `dims[0]/2` from the centre.

`canonicalize_pose` rotates the (0.6, 0.9, 0.6) variant 90° about local Z so
every emitted pose carries the canonical 0.9 × 0.6 × 0.6 dims. **Poses are
cuboid centres, and `rotation` is `R_world_block`.**

`make_hypothesis` then scores it (see below) and records
`support_height_m = ground.height(bottom face centre)` — signed, so negative
means the box is sunk into the terrain.

Two geometric acceptance conditions decide whether the hypothesis becomes a raw
candidate at all:

- the top patch must be at least `0.70 × dims[2]` above the ground, and
- a box whose top is below `1.1 × dims[2]` must have `|support_height_m| ≤ 0.135`
  — i.e. a block that claims to be on the ground must actually sit on it.

Rejects are recorded in `raw_lineage` with `fate = "geometric_rejected"`.

### 11. Thresholding and pre-refinement selection

Surviving hypotheses are filtered by `evidence.score ≥ min_score`. **`min_score`
defaults to 0.0, so this gate is a no-op unless configured** — worth knowing
before concluding it is doing work.

The visual scorer, if any, runs here. Then `select_conflict_alternatives` groups
hypotheses by oriented-box overlap (each half-extent shrunk 0.08 m) and keeps up
to `conflict_alternatives` (default 1) *diverse* members per group, where
diverse means more than 0.02 m or 2° apart. With the default limit this is
greedy suppression by selection score; raising it lets refinement choose between
competing closed-form fits instead of committing early.

Finally, centres outside `0 ≤ height ≤ cluster_max_center_z` (3.0 m) are
dropped. `detect_without_refinement` returns here.

### 12. FK prior seeding (`detect()` only)

A prior with `source == "fk"`, positive weight and matching dims can seed a
hypothesis that no DBSCAN component ever produced — the gripped or heavily
occluded block case. The seed is refined against the points its own surface owns
(within `refine_band`), needs `refine_min_points` (20) of them, must clear
`min_score`, and — critically — must beat any hypothesis it overlaps by
`fk_seed_min_evidence_gain` (0.15) in *geometric* evidence. A prior cannot win
on its prior bonus alone; it is recovery for a missing detection, not a
competitor.

Seed refinement uses the prior's own tolerances rather than the tighter generic
guard, because grasp offsets legitimately exceed the latter.

### 13. Refinement

`refine_poses` runs `refine_iterations` (2) passes. Each pass re-assigns points
to poses — nearest by absolute box SDF, accepted within `refine_band` (0.10 m) —
then fits each pose independently.

`refine_pose` is Levenberg–Marquardt with Marquardt diagonal damping, minimising
a Huber cost (`refine_huber_scale` 0.05) over the box signed-distance function.
The Jacobian is analytic, chain-ruled through the SO(3) right Jacobian because
the rotation parameter is a finite right-multiplied rotation vector rather than
an infinitesimal perturbation.

A **safety guard** discards the entire correction — returning the unrefined pose
— if it exceeds `refine_max_translation` (0.15 m) or `refine_max_rotation_deg`
(20°). Refinement can decline; it cannot run away.

`refine_preserve_top_axis_if_gravity_worsens` (default off) additionally rebuilds
the rotation around the original top axis whenever refinement tilts the box
further from the ground normal.

### 14–16. Rescore, suppress, sort

Refined poses are rescored from scratch by `make_hypothesis` — the score always
describes the pose actually being returned — re-checked against the ground
bounds, then reduced by `select_hypotheses`: greedy non-maximum suppression that
walks candidates in descending selection score and drops any that overlaps an
already-accepted box.

```
selection score = evidence.score + prior_match.score + visual_evidence.score
```

Note that only NMS ordering uses the prior and visual terms. The FK seed gate in
stage 12 deliberately compares bare `evidence.score`.

Output is sorted by centre z, descending — highest block first.

## The evidence score

`normalized_evidence_score` is a fixed weighted sum of five features in [0, 1]:

| # | Feature | Definition | Weight |
| --- | --- | --- | --- |
| 0 | support | `min(support_points / 400, 1)` | 0.25 |
| 1 | height | `max(0, 1 − top_height_error / 0.2)` | 0.15 |
| 2 | face coverage | `covered_faces / expected_faces` | 0.20 |
| 3 | free space | `supported_rays / (supported_rays + violations)` | 0.20 |
| 4 | surface | `supported_rays / incident_rays` | 0.20 |

Features 2–4 come entirely from `visibility_evidence`, which needs a
`SensorContext`. **Without one they are all zero and the maximum achievable
score is 0.40.** Any threshold tuned against deployed behaviour will therefore
behave differently in a fixture that omits sensor context.

`visibility_evidence` casts the relevant subset of rays into the box frame: for
each of the six faces it records whether any ray *should* have hit it
(`expected`) and whether one actually terminated there within 6 cm (`covered`).
Separately it counts free-space violations — rays that passed clean through a
shrunken box, plus miss rays that should have been blocked by it.

`observability` is derived from the same numbers: `full` needs ≥3 observed faces
and ≥0.98 clean rays, `partial` needs ≥2 and ≥0.90, otherwise `limited`.

## The output

`DetectionResult` carries the result and a complete audit trail.

### Result

| Field | Meaning |
| --- | --- |
| `poses` | Final poses, sorted by z descending. Parallel to `hypotheses`. |
| `hypotheses` | `CuboidHypothesis` each: pose, evidence, visual evidence, `support_height_m`, `prior_match`, `trace_id`, `source` (`plane_fit` or `<prior>_seed`). |
| `diagnostics` | Per final pose: the five raw features, `observability`, `support_height_m`. |
| `ground` | The `LocalGroundModel` — plane plus cell map. Evaluate `support_z(x, y)` for terrain height anywhere in the scene. |
| `above_support_points` | The post-ground-removal cloud refinement fitted against. |
| `counts`, `timings` | Per-stage point/candidate counts and per-stage milliseconds. |

### Audit trail

| Field | Granularity |
| --- | --- |
| `proposal_diagnostics` | One per raw DBSCAN label: extent, centroid, and which of the four gates it failed. |
| `wide_proposal_skips` | Components dropped at stage 7. |
| `plane_region_diagnostics` | Per region: plane count, leftover points, why `fit_planes` stopped. |
| `plane_patch_diagnostics` | Every patch produced, whether or not it was ever classified top/side. |
| `pairing_diagnostics` | Per top plane: every side attempted, each gate's value and verdict, which was chosen. |
| `raw_lineage` | Per synthesized hypothesis: its component/region/dims/top/side provenance, the synthesized pose, and its `fate`. |
| `refined_candidate_trace` | Per candidate entering the final selection, with `selection_score`, `stage`, and `fate`. |

`fate` is the single field to read when asking "why is this block missing":

`geometric_rejected` → `score_rejected` → `pre_refinement_selection_rejected` →
`pre_refinement_bounds_rejected` → `post_refinement_nms_rejected` → `final`.

Every diagnostic field is written *beside* a decision the pipeline already made,
never by recomputing it. They can be ignored at zero cost to behaviour.

## What the core deliberately does not do

- **No tracking, no world state, no IDs.** Each call is independent and
  anonymous; identity and persistence belong to the world model.
- **No TF.** Points must already be in the frame the caller wants poses in —
  in this stack, `world`.
- **No RGB in the default path.** `VisualScorer` is a pose-scoring hook only.
  Turning a 2-D region into points to fit is not implemented; see
  `docs/production-readiness-plan.md`.
- **No self-filtering of the gripper.** That is `gripper_self_filter.hpp`, which
  the adapter applies to the cloud *before* calling `detect()`.
- **No search over block size.** Only `block_dims` and its XY swap are tried.

## Where to look in the source

| Stage | File |
| --- | --- |
| Composition, stage order, timings | `pipeline.hpp` |
| Ground model, RANSAC, percentiles | `geometry.hpp` |
| Parameters, DBSCAN, plane pairing, synthesis, NMS | `proposals.hpp` |
| Visibility rays and the score | `evidence.hpp` |
| SDF refinement | `refine.hpp` |
| Prior matching | `priors.hpp` |

`proposals.hpp` and `pipeline.hpp` are written in a very dense one-statement-
per-line style. The stage boundaries above are the ones the code actually has —
read them alongside `detect_without_refinement`, which is the spine.
