# RGB candidate-prior approach

## Decision

Use RGB as a **candidate-conditioned visual likelihood**, not as raw
embeddings, monocular 6-D pose regression, or a general-purpose YOLO detector.
The detector continues to generate feasible cuboid candidates from PCL, FK, and
eventually the wall plan / registered blocks. RGB only scores those candidates
and may abstain when it is uncertain. The final selection fuses calibrated RGB
log-likelihood with the existing geometric and optional prior scores.

This is the smallest learning problem that can correct ambiguous yaw/roll and
occlusion cases without discarding the reliable geometry pipeline. It does not
solve a block for which *no* candidate exists; FK or wall priors must seed such
a candidate first.

## Model

For each image/candidate pair:

1. Project the candidate cuboid with the recorded camera calibration and make
   a spatial conditioning image (visible edges/faces, silhouette, and optional
   depth/occlusion or gripper/occluder mask).
2. Crop the image around that projection with a fixed context margin.
3. Feed RGB through a pretrained visual backbone (initially frozen) and fuse
   its spatial features with the conditioning image in a small classification
   head.
4. Output `P(candidate is correct | RGB, projected candidate)` plus uncertainty.

A pretrained CNN/ConvNeXt-style backbone or a pretrained ViT is appropriate.
Start with the lightest backbone that meets latency; train only the head (and,
if needed, adapters or late backbone layers). An embedding without the
candidate render is not a pose prior: it cannot say whether *this* box matches
the pixels.

## Data and labels

Current labelled material is about **23 snapshots / 61 block labels**. Nearby
captures, crops, and candidates from the same scene are correlated; they must
not be treated as independent train/test samples.

V1 is a candidate-set ranking task: exactly one candidate is positive for a
labelled block, using explicit configurable position/orientation tolerances.
Yaw equivalence must respect the block's geometric symmetries rather than
penalising symmetry-equivalent orientations. Create those candidate sets from
each annotated block:

- positive: the ground-truth pose and small acceptable perturbations;
- hard negatives: real PCL alternatives, yaw siblings, and controlled
  position/roll/pitch perturbations;
- ignore/uncertain: heavily occluded or ambiguous cases where the label cannot
  be judged in RGB.

Pretrain the small head with synthetic renders of the actual block dimensions,
camera model, gripper, and candidate errors. Randomise illumination, concrete
texture, dirt, background, blur, exposure, and occlusion. Then fine-tune and
calibrate on real annotations. Synthetic data is valuable coverage, but is not
sufficient for deployment probabilities because outdoor imagery, concrete
appearance, gripper occlusion, timing, and calibration create a large domain
gap.

Split by recording session/rosbag/day, never by frame, crop, or candidate.
Hold out complete sessions before deciding that RGB helps.

## Non-negotiable input gates

- RGB and cloud/candidate stamps must be within a recorded, justified sync
  tolerance; export the measured delta for every sample.
- Camera intrinsics, distortion convention, image resolution, and the
  `world -> camera` transform must be present and valid at the candidate stamp.
- Every projection must be checked for finite values, positive depth, and image
  bounds. Keep a rendered overlay for manual audit.
- Store image identity/checksum, source snapshot, calibration and TF
  identity/checksums, TF stamps, and label provenance in the JSONL record.
  Failed gates produce no sample rather than a guessed one.

## Development environment and contract

Prototype offline inside this repository's **core devcontainer**, not in the
ROS image and not in the detector package. On the GPU machine, open the
Blockpose devcontainer and run `make python-sync`; it installs the opt-in model
dependencies already declared by Blockpose. Put experimental code under
`experiments/rgb_candidate_prior/` only when the exporter is started. Keep it
outside `python/blockpose/` until it passes the promotion criteria below.

Captured snapshots and annotations are external read-only inputs. An exporter
writes one JSONL record per candidate, with paths relative to the export root:

```json
{
  "sample_id": "bagA/snapshot_001/block_0/candidate_03",
  "split_group": "bagA",
  "rgb": "images/...png",
  "conditioning": "renders/...png",
  "candidate_pose_world": {"translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
  "label": "positive",
  "source": {
    "snapshot": "...", "image_stamp_ns": 0, "cloud_stamp_ns": 0,
    "sync_delta_s": 0.0, "calibration_checksum": "...", "tf_checksum": "..."
  }
}
```

Predictions use JSONL keyed by `sample_id` and contain at least
`rgb_probability`, `uncertainty`, `model_id`, and `calibration_version`. The
ROS integration consumes only these scored candidates after offline validation.

## Promotion criteria

Compare against geometry-only on held-out sessions, including gripped blocks:

- higher top-1 correct-candidate rate and fewer yaw/roll errors;
- no unacceptable recall or false-positive regression;
- calibrated likelihood (reliability curve plus Brier score/ECE);
- reported pose errors after candidate selection and an explicit abstention
  rate;
- acceptable CPU/GPU latency and deterministic offline replay.

Only after these pass should the model be exported to ONNX and added as an
optional detector-side scorer. The first experiment is an offline baseline:
frozen pretrained backbone, small head, real candidate hard negatives, and a
session-held-out report. It will establish whether extra synthetic rendering or
more annotations are the best next investment.
