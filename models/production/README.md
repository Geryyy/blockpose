# Production model artifacts

Trained heads plus the frozen backbone they run on, vendored so a deploy
target needs neither a training environment nor network access to HuggingFace
Hub. **These are standalone, measured artifacts — neither head is wired into
the C++ detector's candidate pool yet.** See
`docs/dino-rgb-prior-ros-integration.md` for the target integration and
`notebooks/blockpose_detector_showcase.ipynb` for the full evidence-stage
comparison these numbers come from.

## Contents

```
models/production/
  dino_proposer/{config.json, dense_head.pt}   -- region-proposal head
  dino_verifier/{config.json, mlp_head.pt}     -- candidate re-ranking head
  hf_cache/hub/models--facebook--dinov2-small/ -- frozen backbone, HF cache layout
```

Both heads sit on the same frozen `facebook/dinov2-small` backbone; only the
small heads are trained (`dense_head.pt` ~1.7 MB, `mlp_head.pt` ~0.8 MB).

## Provenance

Both heads are trained on `data/seibersdorf_dataset` (25 annotated snapshots,
`config/rgb_ablation_splits_seibersdorf.yaml`'s 24-fold leave-one-scenario-out
grouping), at commit `ccfd428` plus this directory's own commit.

**`dino_proposer`** — `experiments/rgb_region_prior`, `--occluded-visibilities
none` (the measured-better default, see
`experiments/pose_quality_and_partial_observation.md`):

```bash
python -m experiments.rgb_region_prior.export --export-root out/rgb_region_prior_seibersdorf_all \
  --snapshots-dir data/seibersdorf_dataset --sessions config/seibersdorf_sessions.yaml --occluded-visibilities none
python -m experiments.rgb_region_prior.train --export-root out/rgb_region_prior_seibersdorf_all \
  --splits config/rgb_ablation_splits_seibersdorf.yaml --checkpoint-out <this>/dino_proposer
```

Measured (19-fold held-out CV, pooled): recall 46/62 (74.2%) overall,
**40% on `heavy_occlusion`** (up from 20% excluding those blocks from
training) with pooled false positives 8 → 2.

**`dino_verifier`** — `experiments/rgb_candidate_prior`, same dataset/splits:

```bash
python -m experiments.rgb_candidate_prior.export --export-root out/rgb_candidate_prior_seibersdorf \
  --snapshots-dir data/seibersdorf_dataset --sessions config/seibersdorf_sessions.yaml
python -m experiments.rgb_candidate_prior.train --export-root out/rgb_candidate_prior_seibersdorf \
  --splits config/rgb_ablation_splits_seibersdorf.yaml --checkpoint-out <this>/dino_verifier
```

Measured top-1 candidate-ranking accuracy: **93.6% (44/47)**, genuinely
held-out (pooled across the same 19-fold CV, each prediction from a head that
never trained on its fold) — this replaces an earlier in-sample-only estimate
(89.4%, 42/47 union-trained) on a different, older dataset
(`data/registration_snapshots`). The report's geometry-ordering baseline
comparison is not well-formed on this export (0 candidate sets have both a
scored pos/neg pair and a labelled `raw_hypothesis`, so no head-to-head number
is reported here — see `experiments.rgb_candidate_prior.report`'s own
caveat).

This export required two fixes, both applied upstream before this directory
was built: `experiments/rgb_candidate_prior/export.py`'s `_raw_candidates`
was requiring the legacy `registrations.yaml` format Seibersdorf doesn't have
(fixed: `require_registrations=False`, ground truth already comes from
`annotations.yaml`), and the C++ detector core (`blockpose._core`) was not
importable in a bare shell missing `libopencv-calib3d406t64` (fixed locally
by installing it; **the production Dockerfile should be checked for the same
gap** — `docker/Dockerfile.core` installs generic X11/OpenGL runtime libs but
no explicit `libopencv-*` runtime package, relying on `pip install
opencv-python` alone, which does not provide the system `.so` the compiled
extension links against).

## Using the vendored backbone offline

`hf_cache/` is a standard HuggingFace Hub cache directory. Point `HF_HOME` at
it (or merge it into the existing `docker-compose.yml` `model-cache` volume)
and set `HF_HUB_OFFLINE=1` to load `facebook/dinov2-small` with zero network
access — no code changes needed, since
`blockpose._dino_backbone.load_frozen_backbone` already calls
`AutoModel.from_pretrained("facebook/dinov2-small")` and transformers checks
the local cache first:

```bash
export HF_HOME=/path/to/models/production/hf_cache
export HF_HUB_OFFLINE=1
```

## Loading the heads

```python
from blockpose.rgb_region_proposer import load_dino_region_proposer
from blockpose.rgb_verifier import load_dino_verifier

proposer_head, proposer_config = load_dino_region_proposer("models/production/dino_proposer")
verifier_head, verifier_config, calibration_temperature = load_dino_verifier("models/production/dino_verifier")
```
