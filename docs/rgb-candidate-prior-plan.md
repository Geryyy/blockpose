# RGB candidate-prior — implementation plan

Concrete, phased plan for the approach in [rgb-candidate-prior.md](rgb-candidate-prior.md).
It maps that decision onto this repository and scopes the **first offline
baseline experiment**, which is the only thing that should be built before we
know whether RGB helps.

All experimental code lives under `experiments/rgb_candidate_prior/`, outside
`python/blockpose/`, per the development contract in the design doc. It is
promoted into `python/blockpose/` only after the promotion criteria pass.

## What already exists (reuse, do not reinvent)

- `blockpose.rgb_edge_prior` — the shipped **classical** candidate scorer
  (edge-distance likelihood, neutral-on-uncertain, gripper-mask exclusion,
  cuboid self-occlusion handling in `_sample_projected_edges`). The learned
  prior is its successor and reuses its projection/visibility logic.
- `tools/ablate_rgb_edge_prior.py` — the offline geometry-vs-RGB rerank harness.
  Copy its candidate generation (`detect_blocks_detailed(collect_trace=True)
  .trace.raw_hypotheses`), its correctness tolerances (0.25 m / 30°), and its
  explicit "not selection-eligible until sessions are independent" reporting.
- `blockpose.projection` (`world_to_camera`, `project_points`), `blockpose.viz`
  (`box_corners`, `_EDGES`). Note: `blockpose.synth` is **LiDAR-only** (raycast
  point clouds) — it does not render RGB, so synthetic RGB is new work.
- `blockpose.pose` — `symmetry_equivalents`, `frame_alignments`, and
  `blockpose.evaluate.rotation_error_deg` already implement **symmetry-aware**
  orientation comparison; yaw equivalence must go through these.
- `blockpose.io` — `load_snapshot`, `load_tf_snapshot` (`T_world_camera`,
  `T_world_cloud`), `load_rgb`, `load_camera_model` (`CameraModel` with K/D).
- `blockpose.annotations` — `load_annotations`, `annotation_complete`,
  `AnnotationTarget.is_evaluation_target`, per-block `pose`/`dims`/`visibility`.
- The `sam` optional extra (torch/torchvision/transformers) is the opt-in model
  runtime the doc's `make python-sync` (`uv sync --all-extras`) installs.
- Data: 23 `data/registration_snapshots/*` (~61 block labels), each with
  `rgb.png`, `cloud.pcd`, `tf.yaml`, `metadata.yaml`, `annotations.yaml`.

## Phase 0 — Scaffold & input gates (CPU, no torch)

- `experiments/rgb_candidate_prior/README.md` restating the contract and
  promotion criteria; a `__init__.py` so the package imports without torch.
- `gates.py`: the non-negotiable input gates from the design doc.
  - RGB/cloud sync: `|image.stamp - cloud.stamp|` from each `metadata.yaml`
    against a recorded tolerance; export the measured delta on every sample.
  - Calibration/TF validity: intrinsics, distortion length, image resolution
    matches `CameraModel`, and a finite `world -> camera` at the candidate stamp.
  - Per-projection: finite pixels, positive depth, in-image-bounds.
  - Failed gate ⇒ **no sample** (never a guessed one) + a recorded reason.

## Phase 1 — Exporter (CPU, no torch)

`export.py` writes one JSONL record per candidate, paths relative to the export
root, in the exact schema from the design doc.

- Per snapshot: load points/tf/rgb/camera; generate candidates via
  `detect_blocks_detailed(..., collect_trace=True)`.
- Per approved evaluation target, build a **candidate set** with configurable
  position/orientation tolerances:
  - `positive`: GT pose + small acceptable perturbations;
  - hard negatives: real `raw_hypotheses` siblings within a radius, yaw siblings
    (via `symmetry_equivalents`, so true symmetries are *not* emitted as
    negatives), and controlled position/roll/pitch perturbations;
  - `ignore`: heavily occluded / not RGB-judgeable (`visibility`-driven).
- Per candidate, render the **conditioning image** (projected visible cuboid
  edges/silhouette, reusing the edge-visibility rule from `rgb_edge_prior`) and
  a fixed-margin RGB crop; also save a human-audit overlay.
- Record `sample_id`, `split_group` (session/bag id), `rgb`, `conditioning`,
  `candidate_pose_world`, `label`, and `source{snapshot, image_stamp_ns,
  cloud_stamp_ns, sync_delta_s, calibration_checksum, tf_checksum}` plus image
  checksum and label provenance.

## Phase 2 — Dataset & session splitting (CPU, no torch)

- `dataset.py`: load the JSONL, group by `split_group`, expose candidate sets.
- Session-level hold-out splitter (never by frame/crop/candidate). Hold out
  complete sessions before claiming RGB helps.
- Symmetry-aware positive/negative resolution shared with the exporter.

## Training-data generation (the critical input)

Two sources; the real one is already implemented by `export.py`.

**A. Real — primary, used for fine-tune + calibration.** Each annotated block
yields a candidate set of ~13 samples: 1 GT positive, up to 4 in-tolerance
positive perturbations, the real PCL siblings within `candidate_radius_m`
(labelled by tolerance), and controlled yaw / position / tilt hard negatives.
Measured yield: 114 samples from 9 blocks over 3 snapshots (~55/59 pos/neg), so
**~700–900 samples across all 23 snapshots**. Correlated by scene, so they are
split by session, never by frame/crop/candidate. This is enough to train a
frozen-backbone + tiny-head model on its own.

**B. Synthetic — deferred, and it needs NEW code.** `blockpose.synth` raycasts
box meshes into **point clouds only**; it does **not** render RGB. Synthetic
pretraining therefore requires a new RGB renderer, not a reuse:

- cheap first cut (copy-paste): render a textured cuboid at a known pose through
  the **real camera model** onto real-image backgrounds, add the gripper mesh
  and domain randomisation (illumination, concrete texture, dirt, background,
  blur, exposure, occlusion). Reuses the exporter schema and the conditioning
  renderer unchanged; only the RGB pixels are synthetic.
- faithful fallback: full PBR render (Blender / pyrender) with a concrete
  material and outdoor HDRI lighting.

Run the baseline on real data **first**. Whether synthetic rendering or more
annotations is the better next investment is exactly what the baseline exists to
decide, so synthetic is built only if the real-only baseline is data-starved.

## Phase 3 — Model & calibration (GPU, `sam` extra)

Two-stream, because the conditioning image is a first-class input — a bare
embedding cannot say whether *this* box matches the pixels:

```
RGB crop 224x224          -> frozen backbone         -> f_rgb   (D)
conditioning 224x224 (2ch)-> small trainable convstem -> f_cond (64)
concat[f_rgb, f_cond]     -> 2-layer MLP head         -> (logit, log-variance)
```

- **Backbone (frozen):** ConvNeXt-Tiny (timm), D=768, CPU-deployable, default.
  Alternative for the small-data regime: DINOv2 ViT-S/14 (D=384, strong frozen
  features). Frozen ⇒ no backbone training cost and no overfit on ~800 samples.
- **Conditioning stem (trainable):** 3–4 conv layers (~50k params), kept
  separate so the frozen 3-channel backbone is untouched.
- **Head (trainable):** concat → MLP → 2 outputs. `logit` → `rgb_probability`;
  `log-variance` → aleatoric `uncertainty` (the abstention signal). Total
  trainable ≈ **250k params**, matched to the dataset size.
- **Calibration:** temperature scaling on a held-out session (reliability curve
  + Brier/ECE); optional MC-dropout for epistemic uncertainty.
- **Escalation if under-performing (cheapest first):** unfreeze late backbone
  blocks → LoRA/adapters → then invest in synthetic data.
- Predictions JSONL keyed by `sample_id`: `rgb_probability`, `uncertainty`,
  `model_id`, `calibration_version`.

## Phase 4 — Held-out report (GPU, `sam` extra)

`report.py` compares against geometry-only on held-out sessions (incl. gripped
blocks): top-1 correct-candidate rate, yaw/roll error counts, reliability curve
+ Brier/ECE, post-selection pose errors + abstention rate, CPU/GPU latency, and
deterministic offline replay. Reuse the honesty of the edge-prior ablation:
report `selection_eligible: false` until sessions are demonstrably independent.

## Promotion gate

Only after Phase 4 passes on held-out sessions: export to ONNX and add as an
**optional** detector-side scorer that fuses calibrated RGB log-likelihood with
the existing geometric/prior scores. The first experiment's purpose is to decide
whether more synthetic rendering or more annotations is the better next
investment — not to ship a model.

## Parallel track — RGB seeding → PCL validation

The scorer above only re-ranks candidates geometry already produced; it cannot
solve a block for which **no** candidate exists. A complementary track uses RGB
to *seed* candidates that the existing PCL registration pipeline then validates.

- This is a distinct workstream, **sequenced after** the offline scorer
  baseline. It shares this plan's gates, exporter, and session-split infra, so
  the scorer work is cheap groundwork for it.
- Keep the existing contract from `blockpose.rgb_proposals`: RGB is *proposal
  evidence only* — it may select/weight cloud points or seed an ROI for the
  geometric fitter, but it **never** produces a metric pose, covariance,
  verification state, or a `Missing` decision. PCL registration remains the sole
  source of the 6-D pose and the accept/reject.
- This bounds monocular scale/depth/calibration/domain-gap risk: RGB only says
  "a block is probably here"; registration snaps it to geometry or rejects it.
- Track new metrics it introduces: seed precision, added false-positive rate,
  and recovery rate on blocks with no geometric candidate. A seeded region still
  needs enough LiDAR support to register — RGB cannot rescue an empty cloud.
- Same non-negotiable input gates apply (sync tolerance, valid TF/intrinsics,
  projection sanity), since it also crosses the RGB↔cloud boundary.

## Status

- [ ] Phase 0 — gates
- [ ] Phase 1 — exporter + audit overlays
- [ ] Phase 2 — dataset + session splitter
- [ ] Phase 3 — model / synthetic pretrain / calibration (GPU)
- [ ] Phase 4 — held-out report (GPU)
