# experiments/rgb_candidate_prior

Offline data/export baseline for the DINO RGB **proposer and verifier** in
[../../docs/dino-rgb-prior-ros-integration.md](../../docs/dino-rgb-prior-ros-integration.md).

The current exporter covers candidate-conditioned visual likelihoods. It is
also the data/provenance basis for the future region proposer. RGB does not
regress a metric pose; captured snapshots and annotations are external,
read-only inputs.

## Contract

- Prototype in the **core devcontainer**, not the ROS image or the detector
  package. On the GPU machine run `make python-sync` (`uv sync --all-extras`)
  to install the opt-in model deps declared by the `sam` extra.
- The exporter writes one JSONL record per candidate, paths relative to the
  export root (schema in the design doc). Failed input gates produce **no
  sample** rather than a guessed one.
- Predictions are JSONL keyed by `sample_id` with at least `rgb_probability`,
  `uncertainty`, `model_id`, `calibration_version`. ROS consumes only
  evidence/regions after held-out promotion tests pass.

## Layout

- `gates.py` — non-negotiable input gates (sync, calibration/TF, projection).
- `export.py` — Python-detector candidate-set baseline + JSONL exporter + audit overlays.
- `cpp_export.py` — production C++ post-refinement candidate-trace exporter. It
  never fabricates GT or perturbation candidates, so it is the input for an
  end-to-end C++ geometry-vs-RGB ranking ablation.
- `dataset.py` — JSONL loader + session-level hold-out splitter *(Phase 2)*.
- `model.py`, `train.py`, `report.py` — model / calibration / held-out report
  *(Phases 3–4, GPU, `sam` extra)*.

## Running it

One command (sessions → export → viability probe):

```bash
python -m experiments.rgb_candidate_prior.pipeline \
    --snapshots-dir data/registration_snapshots \
    --export-root out/rgb_candidate_prior
```

Or each stage by hand: `sessions` (write `sessions.yaml`), `export` (JSONL +
crops + overlays + health report), `probe` (frozen DINOv2 leave-one-session-out).

For the production path, first produce a detector trace with the same C++
commit and parameter files being evaluated. The trace must contain a
post-refinement, pre-NMS `candidate_trace` for every snapshot; `raw_lineage` is
not accepted because it is pre-refinement. Then export only those real candidates:

**`--detector-trace` is *not* `detector_candidates.yaml`.** The
`scene_discovery_capture/*/detector_candidates.yaml` files written during
capture are a different, schema_version 1 artifact (YAML, a flat `candidates`
list, `dimensions` alongside `pose`, `orientation_xyzw`). `cpp_export.py`
consumes the schema_version 2 JSON trace (a `snapshots` list, per-candidate
`pose["dims"]`) emitted **on demand** by the C++
`concrete_block_detector_snapshot_runner` in its schema-v2 candidate-trace
mode — run that against the capture to produce the JSON file this command
needs.

```bash
python -m experiments.rgb_candidate_prior.cpp_export \
  --snapshots-dir /path/to/scene_discovery_capture \
  --detector-trace /path/to/cpp_detector_trace.json \
  --sessions /path/to/recording_sessions.yaml \
  --export-root out/cpp_candidate_prior
```

`--sync-tolerance-s` defaults to the live 0.10 s gate. For a reviewed offline
capture whose RGB/cloud pairing has been explicitly verified, pass its audited
tolerance (for example `--sync-tolerance-s 0.35`); it is retained in the
manifest and is not a claim that this looser gate is safe for live replay.

All candidates competing for one annotated target share the exact same RGB crop
(the union of their valid projections and the target projection). Only their
conditioning render and overlay vary by candidate; this prevents a
pose-dependent crop from becoming a ranking shortcut.

## Adding new data

New captures follow the same `registration_snapshots/<name>/` layout (`rgb.png`,
`cloud.pcd`, `tf.yaml`, `metadata.yaml`, `annotations.yaml`). Drop them in and
re-run the pipeline over the **whole** set so sessions re-cluster deterministically.
Check `manifest.json → health`: `snapshot_status`, `gate_failures`, per-session
label balance, and `warnings` surface bad or thin new data instead of hiding it.
Once real session labels exist, pass them via `sessions --overrides` to replace
the timestamp-gap proxy without changing anything downstream.

## Compute

Runs CPU-only. The backbone is frozen; features are extracted once and cached to
`.npz`, and head training on cached features is instant. A GPU is only needed to
fine-tune the backbone at scale or to render photoreal synthetic (Isaac).

`Pillow` (crop/render I/O) and `scikit-learn` (linear-probe baseline) are core
declared dependencies. DINO itself remains opt-in through the existing model
extra, so importing either exporter does not load Torch or a model.

## Status

The existing frozen-DINO candidate ranking probe is feasibility evidence only.
It does not establish end-to-end recovery of missed blocks; use the architecture
document's ablation gate before promoting either module into ROS.
