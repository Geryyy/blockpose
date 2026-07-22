# Detector closeout: measured next steps

The frozen reviewed snapshots are a development diagnostic, not a production
acceptance set.  The checked-in notebook contains the most recent rendered
values.  Rerun
`uv run python tools/eval_detector_modes.py --output out/frozen-mode-benchmark.json`
for exact revision- and machine-specific values.  In particular,
`SCENE_DISCOVERY` deadline outcomes can vary around the soft 500 ms budget, so
one run's p95 must never be treated as a stable timing claim.  Its operational
latency/stability gate is currently **failed**: accuracy applies only to
completed queries and must be reported separately from deadline compliance.
The report records worktree provenance; a dirty-tree result is not
revision-identical to its Git SHA.

Current qualitative state: blind discovery still misses annotated blocks and
needs timeout profiling; `REFINE_BLOCK` returns present-target verification
results from LOSO plan priors; and `REFINE_GRASPED` remains withheld because
legacy snapshots lack an independent FK/grasp pose, covariance, and gripper
trajectory.

## Ordered work

1. Improve blind discovery before adding complexity. Profile all timeout and
   deadline failures, enforce cooperative timeout behavior, and raise recovery
   on the existing full-cloud benchmark. Define a non-ground-truth workspace ROI from the
   construction/world model (or a caller-supplied physical workspace), then
   test it separately from the annotation scoring ROI. Do not tune against
   annotation ROI or use it in inference.

2. Record rich, synchronized ROS inputs for each new query: per-point time,
   intensity, `scan_id`/`scan_idx`, return flags, sensor/TF trajectory,
   no-return rays when available, and gripper/FK pose plus covariance. This
   unlocks deskew, ray-consistent free-space evidence, gripper line filtering,
   and an honest `REFINE_GRASPED` evaluation. RGB/SAM remains a proposal aid;
   geometry owns the final pose, covariance, and Missing/NotObservable result.

3. Freeze a new real dataset by recording session, not adjacent frames. Keep a
   development split for design choices and an untouched held-out session split
   for the final report. Add independently labelled absent planned blocks for
   `Missing`, placement deviations up to 10 cm, seamless walls, gripper
   occlusion, empty/machinery scenes, and both laid and upright blocks.

4. Freeze portable Python artifacts once the development behavior is selected:
   versioned mode JSON reports, golden frame inputs and expected structured
   results, overlays, parameters, calibration data, and a clean-Git revision.
   Keep the notebook as the human-readable report; keep the JSON as the exact
   machine-readable evidence.

5. Port only the frozen core to C++20/ROS 2: frame/ray preprocessing,
   workspace crop, proposals, plan-conditioned refinement, scoring,
   observability, covariance, and structured failures. Make the current
   orchestrator's three one-shot modes adapters to that core; retain the
   SegmentImage/mask-cutout path as an optional proposal source. Require
   Python/C++ parity on golden inputs before switching runtime.

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
