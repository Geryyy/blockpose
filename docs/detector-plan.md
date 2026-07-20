# Python-first block detector plan

This is the working plan for the **Python prototype track**: develop a simple,
robust point-cloud block detector against synthetic data now, validate on real
frames once they are re-exported with per-snapshot TF, then port the frozen
algorithm to the C++ core. `docs/implementation-plan.md` (v4) remains the
long-term architectural target.

## Problem shape

- Blocks are exact 0.6 × 0.6 × 0.9 m concrete boxes, resting on the 0.6 × 0.6
  base so the 0.9 m long axis is roughly vertical (world Z).
- They can **tilt a few degrees** in pitch/roll (uneven ground) → estimate full
  **6-DOF** pose; gravity is a strong *prior / initialization*, not a hard
  constraint.
- Square cross-section → 4-fold yaw symmetry (already handled in
  `pose.symmetry_equivalents` and `evaluate.rotation_error_deg`).
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

## Known envelope and open weaknesses

Measured on the stratified matrix (`diagnostics_stratified`), none gated yet:

- **Standoff cliff at ~15-20 m.** Recall is 1.0 at 10 m, ~0.5 by 20 m, 0 at 25 m.
  Real snapshots contain returns to ~31 m. Cause is point starvation against
  fixed `cluster_min_size` / `min_inliers`, not pose quality — errors stay small
  right up to the point of disappearing.
- **Blocks in contact are not detected at all** — the largest gap, and it is in
  the target application. A wall course scores 0 recall while a row with 10 cm
  gaps scores 1.0 (`assembly` family); `config/scenes/wall.yaml` recovers 1 of
  16. This is the `touching_row` problem at working scale. It is genuinely
  ambiguous against `elongated_full_height_negative` — three flush blocks and
  one 1.8 m solid produce the same measurement — so it needs scored hypotheses
  plus a scene prior, not better geometry.
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

**Phase D — Real validation (after TF re-export).**
- Same detector + evaluator on real snapshots; measure the sim-to-real gap; add
  LiDAR intensity as a feature only if it earns its keep.
