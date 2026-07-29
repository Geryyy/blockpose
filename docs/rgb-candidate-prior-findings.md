# RGB candidate-prior — findings so far

Running log of what we built and learned probing whether RGB is a viable prior
for the PCL-based block detector. Design: [rgb-candidate-prior.md](rgb-candidate-prior.md).
Plan: [rgb-candidate-prior-plan.md](rgb-candidate-prior-plan.md). Code:
`experiments/rgb_candidate_prior/`.

## Question

Does RGB (a frozen DINOv2 backbone or similar) give the geometry detector a
useful prior — i.e. can it tell a correct cuboid candidate from an incorrect one
that geometry alone leaves ambiguous — **on held-out recording sessions**?

## What we built (offline, CPU, no ROS)

A renderer-agnostic offline pipeline under `experiments/rgb_candidate_prior/`:

- `gates.py` — non-negotiable input gates (RGB/cloud sync, calibration/TF
  validity, per-projection finiteness/in-bounds). A failed gate drops the
  sample; it is never guessed.
- `candidates.py` — per annotated block, a labelled candidate set: GT + in-tol
  positives, real PCL siblings, controlled yaw/position/tilt hard negatives, and
  `ignore` for `heavy_occlusion`. Orientation labelling is symmetry-aware
  (`rotation_error_deg`), so symmetry-equivalent poses are never mislabelled.
- `render.py` — RGB crop, conditioning image (visible-edge wireframe + silhouette
  channels; depth/occlusion channel reserved), and an audit overlay.
- `export.py` — one JSONL record per candidate with full provenance (checksums,
  sync delta, GT provenance) + a health report; per-snapshot errors are isolated.
- `sessions.py` — repeatable snapshot→session assignment (timestamp-gap
  clustering + manual overrides).
- `dataset.py` — JSONL loader + strict session-level splitter (never by
  frame/crop/candidate).
- `probe.py` — the viability probe: frozen DINOv2 embeddings + leave-one-session-out
  logistic regression.
- `pipeline.py` — one command: sessions → export → probe.

Contract: everything keys off the JSONL record `(rgb, conditioning,
candidate_pose_world, label, source)`. Any future data producer (a copy-paste
renderer, NVIDIA Isaac) plugs into the same interface.

## Data

23 registration snapshots, ~61 approved blocks. No real session metadata yet, so
sessions are **provisional**: timestamp gaps > 120 s → 6 clusters. Full export
yields 684 candidate samples. GT provenance breaks down as 28 `manual`, 21
`upstream_registration`, 12 `rgb_candidate_fit`; the last group is **excluded**
from the probe because training an RGB scorer on RGB-seeded GT is circular.

## Result — RGB is a viable prior (with a data caveat)

Leave-one-session-out linear probe, frozen DINOv2-small, RGB crop only (no
fine-tuning, no conditioning image), circular GT excluded. Chance = 0.50.

| Setup | Samples | Sessions | Pooled held-out AUROC | Mean session AUROC |
|---|---|---|---|---|
| All candidates | 536 | 6 | **0.745** | 0.739 (all folds 0.60–0.91) |
| Real PCL siblings only | 146 | 5 | **0.719** | 0.674 |

**Read:** frozen RGB features separate correct-from-incorrect candidates well
above chance on held-out sessions, on both the full and the least-gameable
real-only set. This is a *floor* — RGB-only, linear, frozen — so the full
two-stream head with the conditioning image and light head training should do
better.

**Caveat (data, not method):** per-session variance is high. The largest real
held-out fold (n=42) sits at 0.54, near chance; small folds swing to 0.25/1.00
on 4–20 samples. 6 provisional sessions / ~49 clean blocks is too thin to call
this deployment-ready. Closing the gap is a **data-quantity** problem — more
independent, real recording sessions — which is exactly the fork the design doc
said this baseline exists to resolve.

## Measurement caveats

- Sessions are timestamp-clustered proxies, not verified recording runs.
- AUROC measures candidate separability, not end-to-end detector improvement.
- This is a linear probe, not the fused calibrated scorer.

## Compute

Everything above ran on **CPU** (no GPU). The approach is frozen-backbone +
tiny-head, so the only heavy step is feature extraction, done once and cached to
`.npz`. Head training on cached 384-d features is instant on CPU. A GPU is only
needed to (a) fine-tune/unfreeze the backbone at scale or (b) generate photoreal
synthetic (Isaac). Development and deployment of the frozen prior are feasible
CPU-only; for tight ROS 2 latency, export to ONNX Runtime (+ int8) or use a
lighter backbone (ConvNeXt-Tiny / MobileNet).

## On NVIDIA Isaac (synthetic data)

Decoupled from the viability question, which real data already answered. Isaac
Sim + Replicator is a strong choice **later** for photoreal synthetic with free
3D ground truth (pose, segmentation, depth, occlusion) — useful for hard cases
(gripped-block occlusion, extreme yaw) and to fill the reserved depth/occlusion
conditioning channel. It is an *investment* (RTX GPU, faithful rig incl. camera
distortion), not a shortcut, and it cannot substitute for real held-out data
used to calibrate deployment probabilities. Test synthetic value cheaply first
(copy-paste renderer), and adopt Isaac only once confirmed training-data-limited.

## Recommended next steps

1. **Ingest more real data** (in progress) — directly fixes the measured
   variance problem; the pipeline runs it in one command.
2. Build the full frozen-backbone + conditioning-image head; fine-tune the head
   and temperature-calibrate on a real held-out session.
3. Wire the gripper/occluder mask (gripped blocks are in the promotion criteria).
4. Add unit tests before any promotion into `python/blockpose/`.
5. Only then: end-to-end RGB-reranked vs geometry-only top-1 on held-out sessions.
