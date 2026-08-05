# Status and handoff — picking this up in the ROS 2 environment

Written at the end of the `feat/annotation-tool-hardening` work, as the
pick-up point for integrating the detector into the wall-assembly framework.
Everything here is measured, not estimated; where something is unproven or
unknown it says so.

Deeper reading: `docs/ros2-integration.md` (how to consume and evaluate),
`docs/production-readiness-plan.md` (the plan and its kill criteria),
`docs/cpp-nanobind-consolidation.md` (what is and is not bound to Python).

## What ships

**`blockpose_core` v0.2.0** — the portable C++ detector core. ROS-free,
GPU-free, no RGB dependency. A ROS 2 node consumes it with
`find_package(blockpose_core)` + `blockpose::core`;
`cpp/tests/consumer/` is a working reference consumer.

```bash
make ship-check      # builds, installs, and runs a ROS-style consumer against it
```

That gate asserts the shipped parameter contract and **fails on drift** — it
is not a smoke test. Verified by building a drifted copy: exit 1.

**Build dependencies** (a clean clone cannot build without these):
`libeigen3-dev`, `libopencv-dev`, and `libgtest-dev` for the unit tests.
All three were missing from `docker/Dockerfile.core` and are now added. Two of
them had been hiding real problems — see "What bit us" below.

## Measured performance

On 25 hand-annotated snapshots / 62 approved blocks (`data/seibersdorf_dataset`):

| metric | value |
| --- | --- |
| recall, centre (0.30 m Hungarian match) | 46/62 (74.2%) |
| **recall, strict** (adds symmetry-aware rotation ≤ 30° and oriented-box IoU ≥ 0.25) | **44/62 (71.0%)** |
| precision (centre) | 46/59 (78.0%) |

**Quote strict, not centre.** Centre-only counts a box 88° out of alignment as
a hit; strict is the number that says whether a pose is usable by a gripper.

Shipped detector parameters, which the behaviour depends on:
`cluster_max_extent_xy = 2.25`, `dbscan_eps = 0.07`. Record them with any
result — a report that does not pin them stops reproducing silently, which has
already happened once in this repo.

### By scene class — where it works and where it does not

| scene class | blocks | centre | strict | notes |
| --- | ---: | ---: | ---: | --- |
| static baseline | 3 | 100% | **100%** | control |
| wall row (blocks touching) | 7 | 86% | **86%** | the old merged-cluster failure, fixed by the 2.25 m gate |
| pick-and-place / staging | 17 | 76% | 71% | mostly isolated blocks |
| lighting / wet surface | 8 | 75% | 75% | |
| stack | 3 | 67% | 67% | n=3, effectively unmeasured |
| **gripper-held** | **24** | **67%** | **62%** | worst class; holds all 10 `heavy_occlusion` blocks |

Gripper scenes are 39% of the dataset and the dominant failure mode. The
blocks that fail there are **not** the block in the gripper — they sit a
median 1.5 m from the tool centre, occluded *by* the gripper. FK and the
gripper self-filter therefore do not address them (measured: 0 of 8 missed
blocks are within 0.6 m of the TCP).

**Five scene classes have no annotated snapshot at all** — `stacking`, `wet`,
`pre_stack`, `stacked`, `static_occluded`. Nothing is known about them.

## What is deliberately NOT in the ROS path

- **`dino_verifier` — dropped.** Measured three independent ways (scalar
  rerank, refit calibration, within-block argmax); no benefit any of them.
  Checkpoint and reports stay as a recorded negative result. **Its
  `ScoreRgbCandidates` service in `docs/dino-rgb-prior-ros-integration.md` is
  obsolete — do not implement it.**
- **`dino_proposer` — kept on the roadmap, not wired in.** It detects 62/62
  blocks including 10/10 `heavy_occlusion`, the only mechanism that sees what
  geometry misses. But it merges touching blocks into one region (it is
  *trained* to — the target is a union mask), and the fitter emits one box per
  region. Blocks alone in a region are found 17/17; blocks sharing one, 31/45.
  Unblocking it needs a seam-channel retrain **and** C++ 2-D region ingestion,
  which does not exist (`VisualScorer` is a `Pose → score` hook, not a
  region → points → fit path).
- **ONNX — deferred.** A Python node running the PyTorch checkpoint is the
  cheaper v1; revisit only if latency fails. Latency has never been measured.

## Evaluating on new bags

The full loop now has tooling at every stage (`docs/ros2-integration.md` has
the detail):

```
record_reference_bag.sh → export_reference_bag.py → init_snapshot_annotations.py
   → annotate_snapshot.py (manual)  → build_snapshot_manifest.py
   → evaluate_detector_snapshots.py                    # scored report out
```

`evaluate_detector_snapshots.py` works with **no manifest**
(`--manifest ""` discovers annotated snapshots directly), which is what a
fresh bag export needs. It emits per-scene and per-visibility tables, pins the
parameters, and names every skipped or failed snapshot so nothing leaves the
denominator silently.

**You still need hand annotation for evaluation.** Weak/self-supervised labels
derived from the detector cannot score the detector — that is circular. Use
`build_promotion_splits.py` to reserve whole recording sessions that are never
trained or tuned on.

### On unlabelled bags

Self-supervision was already tried here and **backfired**: mining 1,473 weak
anchors from 286 unlabelled captures took recall from 36/43 to 17/43 (first
attempt) and 29/43 (second, after fixing weak data drowning out real data).
The reason is structural — anchors come from the geometry detector, so they
teach the model to reproduce what geometry already finds, not to cover its
blind spots. Do not simply add more unlabelled data.

The untested idea worth trying, because it targets the real bottleneck: a
confident geometry detection is *one block with a known 3-D box*, so
projecting it gives a **per-instance** outline rather than a merged blob —
i.e. the seam supervision the retrain needs, for free. Continuous bag video
also allows reprojecting a block observed clearly at one moment into a frame
where it is occluded, which is the one mechanism that escapes the
circularity.

## What bit us — worth knowing before you trust a number

- **Scores are not comparable across `SensorContext`s.** 60% of the detector
  score is ray evidence computed against whatever context it is handed, so a
  cropped point set cannot accrue the free-space violations the whole cloud
  would charge it, and scores out systematically higher. This silently
  invalidated a published ablation and was live in the production
  `engine.py` ROI path (fixed). Never compare a score from a crop against one
  from the whole cloud.
- **Missing build deps hid real failures.** `libeigen3-dev` made the ship
  gate unrunnable, which in turn hid 3 failing C++ tests for three commits
  (they were stale, asserting behaviour a later commit deliberately removed —
  now fixed, 43/43 pass). `libopencv-dev` was missing for the compiled Python
  extension. Wire `make ship-check` into CI when there is one — **there is no
  CI runner in this repo today**.
- **Aggregates hide regressions.** An unchanged headline concealed 9 blocks
  being broken while 1 was gained. Always report gained/broken per block.
- **This dataset cannot resolve ±2 blocks.** 62 blocks, one annotated
  snapshot per scene, `selection_eligible: false`. Any small improvement is
  inside the noise. This is the binding constraint on everything, and more
  annotated recordings are the only fix.

## Test status

633 Python tests, 43 C++ tests, ship-check gate — all green at the tip of
this branch.

## Suggested first steps in the ROS 2 environment

1. Build and install `blockpose_core` v0.2.0; run `make ship-check` in the
   target environment to confirm the surface before wiring anything.
2. Wire the geometry detector into the wall-assembly framework. Expect ~71%
   strict recall, and expect gripper-occluded blocks to be the failure mode.
3. Record and hand-annotate a few whole sessions; reserve them with
   `build_promotion_splits.py` and never tune on them. This unblocks every
   remaining decision.
4. Only then revisit `dino_proposer` (seam retrain + C++ region ingestion),
   with the kill criterion already agreed: if it does not beat geometry alone
   on strict recall by more than the noise floor, keep the geometry-only
   detector.
