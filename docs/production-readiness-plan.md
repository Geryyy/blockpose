# RGB priors: production-readiness plan (v2)

**v2 supersedes v1.** v1's Phases 0, 1a, 2a and 2b have been executed; their
results are recorded below and they changed the plan. The headline changes:
`dino_verifier` is **dropped from the runtime path**, ONNX is **deferred**, and
the priority is now shipping the working geometry detector into ROS 2 so
performance can be verified against real rosbags — because the 62-block dataset
cannot resolve the effects we are chasing, and bag data is the thing that
unblocks everything else.

## What v1 established

| phase | outcome |
| --- | --- |
| 0 — correctness | **Done.** Production `engine.py` no longer scores ROI queries against an inflated crop context; the annotation GUI no longer adopts axis-swapped twins into ground truth; `eval_sam_region_proposals.py` / `ablate_rgb_proposals.py` corrected. 586 tests pass. |
| 1a — zero-retrain split probe | **Negative, decisive.** No post-processing splits merged regions: thresholds to 0.9 never split the blob; watershed's purity gain is fragmentation (mean coverage of a covered block 1.000 → 0.789, 42 of 86 regions covering nothing). The foreground probability shape alone cannot tell the waist between two touching blocks from the middle of one. **Justifies the seam-channel retrain.** |
| 2a — verifier calibration refit | **Negative.** Refitting makes it strictly worse (contribution span 0.073 → ~0.00002). `T=12.497` is not the bug; single-scalar temperature scaling is the wrong family for a ranking-trained head. Nothing shipped. |
| 2b — verifier as within-block argmax | **Neutral.** 2 blocks gained, 2 broken, exactly cancelling; strict recall 44/62 before and after. Its 93.6% standalone accuracy is on picking ground truth out of *synthetic* perturbations and does not transfer to the real candidate distribution. |

## Decisions taken

**1. `dino_verifier` is dropped from the runtime path.** Three independent
measurements (scalar rerank, refit calibration, within-block argmax) produced
no benefit. The checkpoint, training code and reports stay in the repo as a
recorded negative result; nothing in the runtime depends on them.
Consequence: **`ScoreRgbCandidates` is removed from the ROS contract** — see
`docs/dino-rgb-prior-ros-integration.md`, which still specifies it.

**2. `dino_proposer` is kept, and is the only RGB module on the roadmap.** It
detects 62/62 blocks including 10/10 `heavy_occlusion` — the only mechanism in
the system that sees the blocks geometry misses. Its benefit is currently
unrealised (+2 centre / +1 strict, inside noise) because merged regions destroy
it downstream. That is one fixable problem, not a dead end.

**3. ONNX is deferred.** `docs/dino-rgb-prior-ros-integration.md` already
specifies the deployment adapter as a *separate optional node*, explicitly so
that "GPU/ONNX-runtime deployment [is] optional". A Python ROS 2 node running
the PyTorch checkpoint satisfies v1 at a fraction of the cost. ONNX becomes
worthwhile only if that node misses its latency budget.

**4. Validation moves to rosbags.** 62 blocks over 25 snapshots cannot resolve
a ±2-block effect, and five scene classes have no annotated snapshot at all.
Every remaining accuracy claim is gated on bag data, not on this corpus.

## Cost / benefit of everything still on the table

| work | cost | measured or expected benefit | verdict |
| --- | --- | --- | --- |
| Ship geometry core to ROS 2 + bag eval loop | **low** — the C++ core already builds, installs and has a consumer check (`tools/check_cpp_consumer.sh`) | unblocks all validation; 44/62 strict, 86.8% precision available today | **do first** |
| Gap-only pooling policy | **~zero** — policy change, already implemented in the eval tool | +1 strict, 0 regressions; makes the score-comparability defect structurally impossible | **do, free** |
| Seam-channel retrain (1b) | **medium** — one extra conv channel, labels free from `block_silhouette_mask`, same data, same frozen backbone | all 14 misses are in merged regions; isolated blocks are found 17/17 | **do second, validate on bags** |
| C++ 2-D region ingestion | **medium-high** — does not exist; `VisualScorer` is a `Pose → score` hook, not a region → points → fit path | required for the proposer to reach the detector at all | **required for 1b to ship** |
| FK orientation prior as tie-breaker | low-medium | ~2 blocks on this data (centre→strict gap); real signal (6.5° vs 88.7°) but small prize | **defer** |
| `plan_prior` C++ port | high — deleted in `d949647`, needs a Stage-3 port | walls already 86% strict; needs a real plan source that does not exist yet | **defer** |
| ONNX export + parity | medium | latency only | **defer until measured** |
| `dino_verifier` anything | any | none measured, three times | **drop** |

## Track A — ship what works, and close the evidence loop (priority)

The geometry detector *is* the product today: 46/62 centre, **44/62 strict**,
86.8% precision, with the 2.25 m cluster gate, the PZS100 gripper self-filter and
the ROI-context fix all in the C++ core. It has no RGB dependency and no GPU
dependency. Getting it into ROS 2 lets real bags produce the evidence every
other decision is waiting on.

- **A1. Version and release the C++ core.** Run `tools/check_cpp_consumer.sh`
  as the gate — it already verifies a ROS-style consumer can find, link and run
  against the installed package. Tag a version; the detector's behaviour is
  parameterised by `DetectionParameters`, so record the shipped values
  (`cluster_max_extent_xy=2.25`, `dbscan_eps=0.07`) with the tag.
- **A2. Bag → snapshot → annotate → evaluate loop. DONE.**
  `tools/evaluate_detector_snapshots.py` is the "snapshots in, scored report
  out" tool (per-scene and per-visibility, pinned parameters, honest health
  section; manifest optional so a fresh bag export works). It reproduces the
  recorded baseline exactly, and now applies the gripper self-filter the ROS
  adapter uses: 46/62 centre, 44/62 strict, 46/53 (86.8%) precision.
  `tools/build_snapshot_manifest.py` generates the manifest the corpus file
  claimed was regenerable but was not, reproducing the committed selection
  exactly. `tools/record_reference_bag.sh` is no longer pinned to one
  workspace. Original description follows.

  **A2 (original).** The pieces exist and were
  just repaired: capture/replay, the annotation tool (now de-duplicating
  properly, so it no longer writes axis-swapped twins into ground truth), and
  the scoring rules (`match_detections` at 0.30 m centre, `strict_match` for
  rotation + oriented-box IoU). Wire them into one documented path so a bag in
  gives a scored report out.
- **A3. Reserve a promotion-gate set. TOOLING DONE, DATA OUTSTANDING.**
  `tools/build_promotion_splits.py` builds a splits manifest from an
  operator-supplied session mapping instead of a directory-name prefix, so a
  reserved set is genuinely held back; it refuses to emit a split that cannot
  gate (session in both halves, or an empty reserve) and its output is
  consumed unmodified by `experiments/fold_safety.py`. What remains is not
  code: independent recording sessions must actually be captured and
  annotated. Original description follows.

  **A3 (original).** Nominate whole recording sessions that
  are annotated and then *never* used for tuning or selection. Without this
  there is no honest way to accept or reject anything later.

*Exit gate:* the operator can take a rosbag and get recall/precision (centre and
strict) for the geometry detector, with per-scene breakdown, without touching
this repo's Python.

## Track B — make `dino_proposer` actually usable

Runs in parallel with A, but its validation depends on A2 producing bag data.

- **B1. Seam-channel retrain.** Add a second output channel predicting
  inter-block seams; split the foreground by seams. Labels are free —
  `block_silhouette_mask` already renders per-block silhouettes, so seams are
  the boundaries between adjacent instance masks. Same frozen backbone, same
  training data, no new annotation.
  *Gate:* instance purity (blocks alone in their region; baseline 17/62) **and**
  mean coverage of a covered block (baseline 1.000) — 1a proved purity alone is
  gameable by fragmentation, so both must be reported.
- **B2. C++ 2-D region ingestion.** The detector must accept regions, project
  its already-filtered cloud into them, and run the ordinary local fit. This is
  the missing structural piece and it does not exist today. Keep the geometric
  authority boundary from the integration doc: the proposer never emits a pose.
- **B3. Optional Python `dino_rgb_prior_node`.** Serves `ProposeRgbRegions`
  only. Bounded timeout; unavailable model / failed TF / stale timestamp is
  reported as *RGB unavailable* and falls back to the geometry-only path. RGB
  absence is never negative evidence.
- **B4. Pooling policy: gap-only.** Region candidates may only fill blocks
  geometry found nothing for; they never contest a block geometry already
  claimed. This is the only configuration measured net-positive, and it makes
  cross-source score comparison — the defect behind the whole v1 audit —
  structurally impossible.

*Exit gate:* on bag data with a reserved promotion set, geometry + proposer
beats geometry alone on **strict** recall by more than the noise floor, without
an unacceptable precision or latency regression. If it does not, keep the
geometry-only detector and say so.

## Explicitly not doing, and why

- **`dino_verifier` in the runtime.** Measured neutral or worse three ways.
- **ONNX.** A Python node meets v1; revisit only if latency fails.
- **FK-seeded candidate generation.** Falsified: of the blocks near the grasp
  point the detector already finds 12 of 14, and the blocks that actually fail
  sit a median 1.5 m from the tool centre — they are occluded *by* the gripper,
  not held *by* it, so FK gives no handle on their position.
- **Global proposal-gate relaxation.** Falsified: lowering minimum XY/Z extents
  to 0.20 or 0.10 leaves recall at 46/62 and costs precision.

## Sequence

```
A1 release core ──┬─> A2 bag eval loop ──> A3 reserve promotion set
                  │         │
                  │         └────────────────┐
B1 seam retrain ──┴─> B2 C++ region path ──> B4 gap-only ──> validate on bags
                          B3 python node ────┘
```

Track A is the critical path: it is cheap, it ships something that already
works, and it produces the data without which Track B cannot be judged.
