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
