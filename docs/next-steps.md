# Detector closeout: measured next steps

The frozen reviewed snapshots are a development diagnostic, not a production
acceptance set.  The checked-in notebook contains the most recent rendered
values.  Rerun
`uv run python tools/eval_detector_modes.py --timeout-ms 5000 --output out/frozen-mode-benchmark.json`
for exact revision- and machine-specific values.  In particular,
`SCENE_DISCOVERY` deadline outcomes can vary around the soft 500 ms budget, so
one run's p95 must never be treated as a stable timing claim. The 5 s offline
run is for accuracy only; report it separately from the 500 ms operational
deadline experiment. Its operational latency/stability gate is currently
**failed**: accuracy applies only to completed queries and must be reported
separately from deadline compliance.
The report records worktree provenance; a dirty-tree result is not
revision-identical to its Git SHA.

Current qualitative state: blind discovery still misses annotated blocks and
needs timeout profiling; `REFINE_BLOCK` returns present-target verification
results from LOSO plan priors; and `REFINE_GRASPED` remains withheld because
legacy snapshots lack an independent FK/grasp pose, covariance, and gripper
trajectory.

## Current decision

Start the C++20 core and a thin ROS 2 **evaluation node now**. Its immediate
purpose is to replay complete rosbags, expose data/TF assumptions, and score
results against reviewed snapshots. It is not a production replacement and must
not be connected to safety- or autonomy-critical behavior until the gates below
are met. Python remains the behavioral reference during the port.

## Ordered takeover work

1. **Create the replayable skeleton.** Add the C++20 library target and ROS 2
   evaluation-node package, starting with one-shot `SCENE_DISCOVERY` request
   validation and diagnostic reporting. For this mode the required recorded
   inputs are a timestamped point cloud and the `/tf` plus `/tf_static` chain
   needed to reach the configured
   query frame; workspace/query configuration comes from the caller. Joint/FK
   and gripper metadata may be recorded and diagnosed but are optional for
   `SCENE_DISCOVERY`. Keep ROS types outside the core. Do not implement the
   other modes yet.

2. **Make bag ingestion explicit and testable.** Resolve every cloud into the
   configured world/query frame at its acquisition time. Preserve per-point
   time, intensity, `scan_id`/`scan_idx`, beam/return flags, and sensor
   trajectory when present; interpolate available transforms over the scan
   interval rather than using the bag-start or latest transform. Record which
   optional fields were available in diagnostics. Time-aligned joint/FK,
   gripper trajectory, and grasp covariance become required only for the later
   `REFINE_GRASPED` mode. A caller-supplied physical workspace or the
   construction/world model defines the inference ROI; annotation ROIs are for
   scoring only.

3. **Port `SCENE_DISCOVERY` first.** Implement preprocessing, workspace crop,
   proposal generation, cuboid refinement, scoring, covariance, observability,
   and cooperative deadline handling in small tested slices. Profile deadline
   failures and recovery on full clouds rather than tuning only on legacy
   snapshots.

4. **Maintain reviewed snapshot evaluation.** Freeze representative queries
   from full bags with annotations, structured detector outputs, overlays, and
   revision provenance. Score pose, count, and false-positive metrics from
   reviewed labels; document every intentional behavior change.

5. **Add modes in evidence order.** After `SCENE_DISCOVERY` meets its replay and
   reviewed-snapshot gate, port `REFINE_BLOCK` using plan/placement priors. Then port
   `REFINE_GRASPED` using time-aligned gripper FK, grasp covariance, gripper
   trajectory, and occlusion/filtering inputs. Geometry owns final pose,
   covariance, and absence/observability status; RGB/SAM may remain an optional
   proposal source only.

6. **Build the real evaluation set.** Record by session, not adjacent frames.
   Keep a development split and an untouched held-out session split. Include
   independently labelled absent planned blocks, deviations up to 10 cm,
   seamless walls, gripper occlusion, empty/machinery scenes, and laid and
   upright blocks. Promotion beyond an evaluation node requires this set and
   the production acceptance conditions below.

## Input and failure contract

- Missing a required cloud, timestamp, frame ID, or transform chain is a
  structured input error. Do not substitute identity, latest TF, or a static
  transform silently.
- Per-point timing is required when motion compensation is requested. If it is
  absent, either reject the query or run an explicitly configured
  `UNDESKEWED` evaluation path and mark the result degraded.
- Missing optional intensity/beam/return metadata disables only the dependent
  evidence and is reported in diagnostics; it must not change array alignment.
- A valid query with no supported block returns success with an empty detection
  list, not an input error. A deadline returns a typed timeout with partial-work
  diagnostics, never a fabricated `Missing` result.
- `Missing` means sufficient expected-visible evidence supports absence;
  `NotObservable` means the data cannot distinguish absence from occlusion or
  inadequate coverage. `REFINE_GRASPED` must not run without time-aligned FK
  and its declared grasp uncertainty.

## Deliverables and gates

| Deliverable | Gate |
| --- | --- |
| Replay skeleton | A parameterized command replays one complete external bag query deterministically; the diagnostic executable validates required topics/transforms, reports data availability and timing, and does not claim detector-result compatibility yet |
| `SCENE_DISCOVERY` port | Reviewed snapshot score, missing-TF behavior, and deadline fixtures pass |
| `REFINE_BLOCK` port | Plan-prior fixtures cover `Confirmed`, `Deviated`, `Missing`, and `NotObservable` |
| `REFINE_GRASPED` port | Full-bag fixtures include scan-time TF/FK, covariance, and gripper trajectory; occlusion and missing-data behavior pass |
| Runtime promotion | Development performance gates and held-out production acceptance conditions below pass; ROS wrapper output equals standalone core output |

## First task for the next agent

Inventory the workspace build and message conventions, then commit the smallest
buildable skeleton: a C++20 core library plus a thin ROS 2 evaluation executable
that validates a `SCENE_DISCOVERY` request and reports diagnostic input/topic
availability without emitting a detector result. Add smoke tests for valid and
missing-required-input behavior. Document a parameterized replay command such
as `ros2 bag play "${BLOCKPOSE_BAG}" --topics /seyond/points /tf /tf_static`,
including the actual configured point topic and node invocation once those are
known. State that `BLOCKPOSE_BAG` must name a complete external bag supplied by
the operator: no replay bag is checked into this repository. Do not port
detector math or invent a new result status/schema in that first change.

## Gates

Development gates guide iteration; they are not production evidence:

| Area | Development gate |
| --- | --- |
| Free isolated synthetic | recall and precision at least 0.98; center p95 at most 20 mm; rotation p95 at most 1° |
| Staging rows/stacks | recall and precision at least 0.95 for sufficiently visible blocks |
| Empty/machinery | at most 0.05 false positives per scan |
| Placed verification | at least 0.95 `Confirmed`/`Deviated`/`Missing` status accuracy; score `NotObservable` separately |
| Latency | end-to-end query p95 at most 500 ms on the target machine |
| Stability | repeat runs and point permutations produce stable structured results; currently failed for operational deadline behavior |

Production acceptance additionally requires a held-out, session-separated real
set, no threshold fitting on it, confidence calibration with uncertainty, and
no regression in safety-critical scenario subsets.  The current snapshots do
not satisfy those conditions and must not be used to claim generalization.
