# experiments/rgb_candidate_prior

Offline baseline for the RGB **candidate-conditioned visual likelihood**
described in [../../docs/rgb-candidate-prior.md](../../docs/rgb-candidate-prior.md).
Phasing and rationale: [../../docs/rgb-candidate-prior-plan.md](../../docs/rgb-candidate-prior-plan.md).

RGB **scores** candidates that the geometry pipeline already produced (and may
abstain). It does not detect blocks or regress a metric pose. Captured
snapshots and annotations are external, read-only inputs.

## Contract

- Prototype in the **core devcontainer**, not the ROS image or the detector
  package. On the GPU machine run `make python-sync` (`uv sync --all-extras`)
  to install the opt-in model deps declared by the `sam` extra.
- The exporter writes one JSONL record per candidate, paths relative to the
  export root (schema in the design doc). Failed input gates produce **no
  sample** rather than a guessed one.
- Predictions are JSONL keyed by `sample_id` with at least `rgb_probability`,
  `uncertainty`, `model_id`, `calibration_version`. ROS integration consumes
  only scored candidates, and only after the promotion criteria pass.

## Layout

- `gates.py` — non-negotiable input gates (sync, calibration/TF, projection).
- `export.py` — candidate-set builder + JSONL exporter + audit overlays *(Phase 1)*.
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

## Findings

See [../../docs/rgb-candidate-prior-findings.md](../../docs/rgb-candidate-prior-findings.md):
frozen DINOv2 gives ~0.72–0.75 held-out AUROC — a viable prior, currently limited
by held-out data quantity, not model capacity.
