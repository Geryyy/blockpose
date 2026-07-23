# Python reference detector plan

This documents the **Python reference track** and its accumulated evidence.
The active takeover sequence is now in `docs/next-steps.md`: start the C++20
core and ROS 2 evaluation/replay node immediately, port `SCENE_DISCOVERY`
first against complete rosbags, and require parity with this Python reference
before adding the prior-conditioned modes. The node is not yet a production
replacement. `docs/implementation-plan.md` (v4) remains the long-term
architectural target.

## Problem shape

- Blocks are exact 0.6 × 0.6 × 0.9 m concrete boxes. `config/wall_plans.yaml` is
  the authority and **every plan block is laid down** with the 0.9 m axis
  horizontal, dims `(0.9, 0.6, 0.6)`, course centres at z = 0.3 with a 0.6 m
  rise. An earlier version of this plan claimed the long axis was roughly
  vertical; that was wrong for the target task. Blocks may still be *staged*
  upright, so the detector must cope with both, but the wall itself is laid.
- **Plan heights include a terrain allowance.** `basic_pick_and_place` sits at
  z = 0.45, which is the same laid block with 0.15 m of clearance because the
  real ground is not at z = 0 and the terrain is uneven. Block heights are
  therefore **not** exact multiples of the block height above a single plane —
  an assumption the detector currently makes.
- They can **tilt a few degrees** in pitch/roll (uneven ground) → estimate full
  **6-DOF** pose; gravity is a strong *prior / initialization*, not a hard
  constraint.
- The symmetry group **depends on the orientation** and is now derived from the
  extents in `pose.symmetry_equivalents`: order 8 for either square prism, but
  about a *different* axis. Assuming a fixed 4-fold group about local Z scored a
  90° yaw error on a laid block as zero, so any metric on wall geometry was
  meaningless before this was fixed. Pass `dims` to `evaluate.rotation_error_deg`
  whenever blocks are not the upright default.
- The Seyond LiDAR sees only faces toward the sensor → partial visibility is the
  norm; the detector must work from a top + one/two side faces.

## Measured sensor model (2026-07)

The generator's parameters are no longer guesses. `tools/measure_sensor_model.py`
derives them from the three real snapshots; `blockpose/synth.py` records them as
`MEASURED_*` constants and `seyond_measured()`.

| quantity | measured | generator default | verdict |
|---|---|---|---|
| range noise (near field) | ~0.005 m | 0.010 m | ~2x pessimistic |
| range noise (10-15 m) | ~0.010-0.015 m | 0.010 m + slope | matched |
| incidence dependence | none detectable (0.42-0.55 cm across 26-72° off-normal) | isotropic | matched |
| areal angular pitch | 0.239° | 0.25° | matched (within 5%) |
| nearest-neighbour gap | 0.133° | n/a | *within-scanline only* |
| field of view | ~118° x 72° | 70° x 55° | narrower |

The two angular numbers differ by ~2x because the pattern is non-repetitive
(dense along scan lines, sparse across). **The areal figure governs how many
points land on a block face** — matching the NN gap would overstate on-face
density by ~3.5x and make the suite optimistic. The default suite is therefore
realistic-to-conservative, not flattering.

## Model-based refinement (implemented)

`blockpose/refine.py` fits the exact cuboid to every assigned return by
minimizing the analytic box SDF over the full 6-DOF pose, initialized from the
closed-form plane-fit guess. Multi-block scenes get point assignment first
(nearest surface within a band) so a stack's two hypotheses stop contaminating
each other. Measured over the full scenario matrix, 3 seeds:

| family | trans p95 before → after | rot p95 before → after |
|---|---|---|
| overall | 0.0444 → **0.0058** m | 1.796 → **0.527°** |
| noise | 0.0425 → 0.0050 m | 2.268 → **0.505°** |
| stack | 0.0108 → 0.0071 m | 3.175 → **0.613°** |
| multi_block | 0.0163 → 0.0006 m | 0.680 → 0.157° |

Recall and false positives are unchanged everywhere; cost is ~84 → ~110 ms per
scene. Refinement also **removes the run-to-run pose nondeterminism**: repeated
detection of one scene varied by 2.85 mm before and 0.000 mm after, because the
fit converges to the same optimum regardless of which RANSAC plane seeded it.

Gate thresholds were ratcheted accordingly: translation p95 0.15 → 0.025 m,
rotation p95 2.0 → 1.0°.

## Reproducibility (resolved)

`blockpose/ransac.py` replaces Open3D 0.19's `segment_plane`, which takes no seed
and returns different planes on identical input (`o3d.utility.random.seed` does
not reach it). A second, subtler source was `_remove_ground`'s coarse voxel pass:
Open3D does not guarantee downsample output ordering, and a seeded sampler draws
by index, so the cloud is now sorted there as it already was elsewhere.

Detection is byte-identical across repeated runs on all 25 scenarios. Cost is
~110 → ~122 ms per scene; accuracy is unchanged (overall translation p95 0.0058 →
0.0057 m, rotation p95 0.527 → 0.534°). Adaptive early termination keeps a
pure-numpy fitter competitive: a dominant plane at an 80% inlier ratio needs ~13
trials rather than the nominal 1000.

## Orientation handling (partly implemented)

`BlockPose` now carries `dims`, and `DetectionParams.candidate_dims` lets the
detector hypothesise several orientations, each expanded with its yaw-swapped
twin (the closed-form pose constrains the centre with `dims[0]/2` along the
observed side normal, which silently assumes a square footprint; for a laid
0.9 x 0.6 block the visible side may be either face, and guessing wrong puts the
centre 0.15 m out at 90 degrees of yaw).

Selection moved from a fixed 0.2 m centre radius to an exact separating-axis
overlap test. The two readings of one block sit ~0.21 m apart -- just outside the
old radius -- so both used to survive as a false positive. Overlap is also the
right criterion for walls: 2 cm-spaced and even touching courses do not
interpenetrate, so neighbours never suppress each other.

`evaluate.rotation_error_deg` compares poses via `pose.frame_alignments`, which
re-expresses a detection in the ground truth's axis ordering before measuring.
Without it a correct laid-block detection labelled (0.6, 0.9, 0.6) scored 90
degrees against ground truth labelled (0.9, 0.6, 0.6) -- the same block.

**Multi-orientation is off by default.** With both candidates enabled the
evidence model cannot reliably rank them: `height_consistency` is blind wherever
a top height is a common multiple of 0.9 and 0.6 (1.8 m, i.e. any two-course
upright stack), and without that tie-breaker a laid hypothesis wins on upright
geometry -- measured as a 0.20 m centre error plus false positives on clutter.
Ranking needs a feature penalising a box that protrudes into space the returns
show is empty. Until then, enable `candidate_dims` explicitly when the
orientation is known from a plan.

Measured effect on the plans: `basic_pick_and_place` 0/1 -> 1/1 once the laid
candidate is enabled. The walls stay at 0 -- orientation was not their blocker.

## Known envelope and open weaknesses

Measured on the stratified matrix (`diagnostics_stratified`), none gated yet:

- **Standoff cliff at ~15-20 m.** Recall is 1.0 at 10 m, ~0.5 by 20 m, 0 at 25 m.
  Real snapshots contain returns to ~31 m. Cause is point starvation against
  fixed `cluster_min_size` / `min_inliers`, not pose quality — errors stay small
  right up to the point of disappearing.
- **Blind detection cannot separate a laid wall course** — and the reason is now
  measured precisely, not guessed. The plans specify **2 cm** joints (0.92 m
  centres, 0.9 m blocks). In the *raw* cloud that gap is observable to ~6 m
  (23–37 returns fall into each 2 cm slot, set back ~11 cm behind the faces; 4
  by 8 m). But those slot returns are **stripped by ground removal and
  clustering** before any splitter runs — the block-surface cluster that
  survives (top + front faces) is coplanar and continuous across the joint, with
  no set-back signal even at zero voxel size (measured: near-boundary set-back
  max 2 cm, versus 11 cm in the raw cloud). So the gap is not below sensor
  resolution; it is below the resolution of the *processed* cluster, and the
  earlier "raise `cluster_max_extent_xy`" result (0→1 of 3) was one big-box
  hypothesis landing on one block by luck, not a real split. A set-back splitter
  was prototyped and reverted: it either missed the gap or tiled the elongated
  hard negative, because on the processed cluster the two are indistinguishable.

  **This is the case the plan prior is for** (`blockpose/plan_prior.py`). When
  the wall is being built the poses are known — the robot placed the blocks — so
  the joints do not need to be discovered. `detect_with_plan` assigns returns to
  each expected block, refines with the same SDF fit, and scores with the same
  evidence model. Result on the plans (priors jittered ±4 cm to make refinement
  work): `row3_middle_last` 3/3 at ≤18 mm, `example_wall` 3 of 5 cleanly (the
  two-course cases lose the mutually occluded blocks, correctly flagged by a low
  score). A block present in the plan but not yet placed scores 0.38–0.41 versus
  0.76–0.96 for placed blocks, a clean verify threshold. Blind separation of
  2 cm joints from a single scan remains out of reach and is not the priority,
  since the target task always has the plan.
- **Touching rows are silently dropped** (recall 0). A flush 3-block row exceeds
  `cluster_max_extent_xy` with no density valley to split on, so the whole
  cluster is discarded to protect against the elongated-object hard negative.
- **Structured gaps fragment DBSCAN.** A contiguous dropout stripe splits a
  cluster into sub-threshold pieces even when ~140 returns survive; uniform
  thinning to 60% is handled fine. Fixed-eps density clustering is the cause.
- **Uneven ground costs recall** (~0.8 on the `uneven_ground` family) — the
  single-plane RANSAC ground model plus a 0.15 m removal band is the suspect.

## Why synthetic first

Real GT is currently untrustworthy (static `T_world_seyond` misplaces the cloud
per snapshot; needs TF re-export). Synthetic gives **exact** GT today so the
`evaluate.py` metrics mean something, reproduces the block geometry faithfully
(it is an exact box), and lets us generate the hard cases (stacks, occlusion,
tilt, clutter) that 3 real frames never will.

Scope guardrails: **Open3D raycasting into box meshes**, not a physics engine or
Isaac Sim. Reuse real Seyond ray directions and add realistic degradations
(range noise, dropout, uneven-ish ground) to keep the sim-to-real gap small.
Validate on real frames as soon as they land; never let synthetic be the only
judge. Isaac Sim stays gated behind a *measured* gap (plan v4 §8).

## Phases and gates

**Phase A — Synthetic harness + regression gate (current).**
- `blockpose/synth.py`: raycast N upright-ish boxes (position + yaw + small
  pitch/roll tilt) from a sensor pose → `(N,3)` cloud + exact `BlockPose` GT.
  Deterministic seeds; optional ground, range noise, dropout.
- `tools/eval_detector.py`: run `detect_blocks` over a generated suite, emit
  `metrics.json` via `evaluate.py`, exit non-zero on threshold miss.
- *Gate:* clean single-block scenes — translation ≤ a few cm, symmetry-aware
  rotation ≤ ~2°; recall = 1 on isolated blocks, 0 false positives on empty ground.

**Phase B — Robustness levers on the existing pipeline.**
- Gravity prior: fit ground once, initialize Z-up, but keep roll/pitch free.
- Known dimensions as a prior: gate clusters by expected extent; recover center
  from a single visible face + gravity + dims.
- Point-to-plane / box ICP refine after the closed-form guess (symmetry-aware).
- Confidence from surface coverage + fit residual (not a hardcoded 1.0).
- *Gate:* error held under a ramp of noise / dropout / yaw / tilt on synthetic.

**Phase C — Multiple blocks, stacks, clutter.**
- Greedy accept + non-overlap; split stacked blocks by height layers.
- *Gate:* precision/recall targets on multi-block scenes; zero FP on empty scans.

**Phase D — Real validation (after TF re-export, in progress).**
- Same detector + evaluator on real snapshots; measure the sim-to-real gap; add
  LiDAR intensity as a feature only if it earns its keep.
- TF re-export landed: snapshots now ship a `tf.yaml` (the crane's actual joint
  state at capture time; `blockpose.io.load_tf_snapshot`). The static
  `transforms_pzs100_initial_pose_1.yaml` config was misplacing the cloud by
  several metres relative to recorded ground truth for at least one snapshot —
  large enough that 0 of 3 blocks matched until `tools/detect_blocks.py` was
  switched to prefer `tf.yaml` when present.
- `tools/eval_real_snapshots.py` is the real-data counterpart to
  `eval_detector.py`: no fixed gate yet (too few real snapshots to fit
  thresholds to signal rather than noise), but it discovers every `tf.yaml`
  snapshot under `data/registration_snapshots/` and reports the same shape of
  metrics, so a regression is visible run over run.
- **Ground truth here is not exact.** `registrations.yaml` is itself the
  output of the upstream RGB-based registration pipeline, which can fail to
  register a block it detected (`registration_ok: false`) — observed on 2 of 4
  snapshots at time of writing, both reporting zero registered blocks despite
  having candidate detections. An unmatched detection in a low/zero-GT
  snapshot may be a real, un-registered block, not a false positive; see the
  harness's module docstring and `precision_grounded_snapshots_only` metric,
  which excludes zero-GT snapshots rather than silently scoring them as pure
  false-positive noise.
- **Real-data orientation disambiguation needed visibility evidence that real
  callers weren't providing.** `detection.py`'s free-space/visibility scoring
  (the "ranking needs a feature that penalises a box protruding into empty
  space" gap noted above) is only computed when a `SensorContext` is passed —
  synthetic callers always provide one (from the raycaster), but
  `tools/detect_blocks.py` never constructed one for real point clouds, so
  that scoring was silently inactive on every real snapshot. Real point clouds
  can't recover true sensor misses, but `SensorContext.from_points(origin,
  points)` derives ray directions/ranges from the returns themselves, which is
  enough: on one real snapshot this dropped mean rotation error on 2 of 3
  blocks from ~87° (wrong long/short face) to ~1.5°. It also changed which
  block was recovered — one, sitting close to the crane/gripper arm, dropped
  out of the accepted set — a real, unresolved recall/orientation-accuracy
  trade-off, not yet safe to call solved on one example.
- **No hypothesis had a minimum quality bar.** `_select_hypotheses` did greedy
  non-overlap NMS with no absolute score floor, so any cluster that produced a
  plausible top/side plane pair was reported regardless of how weak its
  evidence was. The crane/gripper hardware directly above a block produced
  exactly such a hypothesis. A geometric exclusion zone (tried first, using
  the crane-side frames in `tf.yaml`) turned out not to work: the gripper's
  position is fixed relative to the crane's *own* joint state, not to the
  sensor, and `tf.yaml` does not export a tool/gripper frame — so the same
  clutter can appear anywhere in sensor-relative space depending on the arm's
  current pose. Measured instead: real spurious hypotheses score under ~0.2
  while every confirmed real block scores over ~0.4 (see `_SCORE_WEIGHTS`).
  `DetectionParams.min_score` (new, default `0.0` — a no-op, so every
  synthetic scenario and gate is unaffected) lets real-data callers apply that
  cut; `tools/detect_blocks.py` and `tools/eval_real_snapshots.py` default it
  to `0.4`. This is a first, deliberately conservative threshold from four
  snapshots, not a tuned classifier — revisit as more real data lands.
