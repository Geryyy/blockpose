# DINO RGB priors: training and ROS integration

## Decision

RGB is optional evidence for the PCL-based scene-discovery detector. It has two
strictly bounded jobs:

- **`dino_proposer`** recovers a missed or merged block by predicting a
  high-recall 2-D region. The detector selects already PZS100-filtered LiDAR
  points projected into that region and runs its ordinary local cuboid fit.
- **`dino_verifier`** scores fitted 3-D candidates using their projected image
  evidence. It may make a bounded reranking contribution or abstain.

Neither module emits a metric pose or accepts a block. A final block must pass
the existing local-fit, PCL-support, free-space, and ray-evidence checks. A
fully unobservable block remains a problem for FK or a future wall-plan prior.

Use one frozen DINOv2 backbone and train only the compact heads on real data:

- a dense 2-D region head for `dino_proposer`;
- a candidate-conditioned MLP for `dino_verifier`.

Do not add Mask2Former, SAM, or Grounded-SAM to the runtime now. They are valid
offline annotation and ablation tools. Introduce a mask decoder only if a
measured experiment proves that DINO regions retain too much clutter *and*
pixel-accurate masks improve PCL-validated 3-D recovery.

## Runtime ownership

`world_model_node` remains the orchestrator and owner of persistent blocks,
association, planning-scene objects, and final RGB pose overlays. The C++
`concrete_block_detector` remains the owner of all candidates and metric poses.
The DINO model is evidence-only and never publishes a `BlockArray`.

```text
world_model
  └─ /concrete_block_detector/discover_blocks
       ├─ PZS100 self-filter -> geometry candidates
       ├─ FK prior -> FK-seeded local fits
       ├─ dino_proposer -> 2-D regions -> filtered-PCL selection
       │                    -> local fit -> rgb_lifted candidates
       ├─ unified pool: plane_fit | fk_seed | rgb_lifted | wall_plan_seed
       ├─ dino_verifier -> bounded RGB evidence per fitted candidate
       └─ geometry/ray validation + NMS -> BlockArray -> world_model
```

The preferred deployment adapter is a separate optional
`dino_rgb_prior_node`. This makes GPU/ONNX-runtime deployment optional and
keeps model dependencies out of the PCL core. The detector invokes it once,
synchronously, for each `discover_blocks` request. Its call has a bounded
timeout; unavailable model, invalid TF, failed timestamp gate, or timeout is
reported as **RGB unavailable** and takes the existing geometry-only path.

The PCL core must not depend on ROS or DINO. The detector-node adapter supplies
RGB regions before local fitting and verifier evidence after fitting but before
NMS. A future in-process ONNX implementation may implement the same interface
without changing candidate selection.

## ROS contracts

Define two small versioned interfaces. They can start as services in a common
perception-interface package; do not use a continuous detector topic because
scene discovery is an on-demand, atomic request.

```text
ProposeRgbRegions
  request:  image, CameraInfo, cloud_header, world_T_camera
  response: model_id, model_version, gate_status,
            regions[{bbox_px, confidence, uncertainty}]

ScoreRgbCandidates
  request:  same RGB context,
            candidates[{stable_id, pose_world, dimensions, source}]
  response: model_id, model_version, gate_status,
            evidence[{stable_id, score, uncertainty, abstained}]
```

The detector selects a camera image and CameraInfo using the cloud timestamp,
then validates intrinsics, image dimensions, and `world -> camera` TF. The
request and response carry that cloud stamp. Results for another stamp, model
version mismatch, invalid projection, or a late response are discarded. RGB
absence is not negative evidence.

`dino_proposer` receives no raw/unfiltered cloud. The detector alone projects
its filtered cloud into returned regions, applies a configurable pixel margin,
and feeds the retained points through the normal fitter. This protects the
geometric authority boundary and lets the same local fitter evaluate every
candidate source.

`dino_verifier` runs one shared DINO feature extraction per selected image and
scores all pre-NMS fitted candidates. Its head pools features under each
candidate's projected interior, visible edge band, exterior, and known gripper
occlusion. Its contribution is capped by configuration and cannot rescue a
candidate that fails geometry/ray checks.

## Configuration and diagnostics

Keep the existing FK and classical edge prior independently switchable for
ablations. Add DINO modules in the detector scene-discovery configuration:

```yaml
modules:
  priors:
    fk: {enabled: true}
    rgb_edge: {enabled: true}       # classical baseline
    dino_proposer:
      enabled: false
      service: /dino_proposer/propose_regions
      timeout_s: 0.20
      min_confidence: 0.55
      point_selection_margin_px: 12
    dino_verifier:
      enabled: false
      service: /dino_verifier/score_candidates
      timeout_s: 0.20
      max_score_contribution: 0.20
```

Publish enough evidence to audit every request:

- `/cbp/debug/scene_discovery/dino_proposer/overlay` — regions on the selected
  RGB image;
- `/cbp/debug/scene_discovery/dino_proposer/selected_cloud` — points retained
  by each region;
- `/cbp/debug/scene_discovery/dino_proposer/lifted_candidates`;
- `/cbp/debug/scene_discovery/dino_verifier/overlay` — projected cuboids and
  score/uncertainty;
- the existing diagnostics payload — timestamp/TF gates, model version,
  regions, retained point counts, candidate source, score contribution,
  abstention, and geometry-only fallback reason.

The final `/cbp/debug/scene_discovery_pose_overlay` remains the world model's
accepted-pose view; the new outlets explain how a candidate arrived there.

## Offline data and weak supervision

All training belongs in Blockpose on the GPU development machine. ROS only
loads a versioned exported model artifact (initially ONNX) and performs
inference.

```text
rosbags / scene-discovery captures
  -> annotation + detector/FK trace export
  -> track and multi-view association using TF
  -> weak labels + explicit human corrections
  -> frozen DINO features + train proposer/verifier heads
  -> held-out bag evaluation
  -> versioned ONNX artifact -> dino_rgb_prior_node
```

Use high-confidence PCL/FK tracks as weak positive anchors, then reproject a
track into neighbouring video frames through the recorded camera TF. This
turns a confident observation in one frame into supervision for another frame
where the geometry detector missed it. Never treat an absent detector result
as a negative label: it is unknown. Keep only anchors with valid sync/TF,
multi-frame or multi-view agreement, strong PCL/ray evidence, and no ambiguous
gripper occlusion.

An EMA teacher is appropriate for the two heads: it receives weakly augmented
frames and updates from the student weights, while the student sees strong
augmentation. The DINO backbone stays frozen initially. Train the proposer on
track/augmentation/temporal consistency and the verifier on ranking real
candidate siblings plus temporal consistency. Human review is reserved for
untracked DINO proposals, detector/DINO disagreements, and uncertain or
persistently missed tracks.

Video cannot supervise a block that is never confidently visible in any frame;
that case needs an FK/wall prior or a sparse manual correction.

## Evaluation and promotion gate

Evaluate complete bags/capture sessions, never random frames or candidates.
Report these ablations using the identical input snapshots and geometry gates:

```text
geometry
geometry + FK
geometry + FK + classical RGB edge prior
geometry + FK + dino_proposer
geometry + FK + dino_proposer + dino_verifier
```

For each variant report recall of manually verified blocks, false positives,
translation/orientation errors, recovery of prior no-candidate cases, and
latency/fallback rate. A DINO module is promoted only if it improves
PCL-validated 3-D recall or selection on held-out bags without an unacceptable
false-positive or latency regression.
