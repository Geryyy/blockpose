# Current detector pipeline

This document is the canonical description of the **as-built Python detector**
as of 2026-07-22. It describes what callers get now, not the intended C++/ROS
architecture. Status labels are deliberate:

- **Default** — selected by the public `HybridBlockDetector` without flags.
- **Opt-in** — implemented, but requires an explicit flag or richer input.
- **Planned** — not implemented; do not rely on it.

The long-term design is in [implementation-plan.md](implementation-plan.md).
The historical experimental detail remains in [detector-plan.md](detector-plan.md)
and [verify-placed.md](verify-placed.md).

## Inputs and common behavior

**Default:** `HybridBlockDetector` accepts a world-frame XYZ array. Its result
has pose, dimensions, raw score/features, an observability label, runtime, and
unavailable-evidence fields. A 500 ms post-hoc soft deadline discards results
if processing already exceeded the deadline; it does not interrupt Python work.

**Opt-in:** `PointFrame` supports per-point relative time, a sensor trajectory,
no-return rays, and a gripper trajectory. Points are deskewed into world frame
when a sensor trajectory and relative times are supplied. `FrameBurst` fuses
deskewed points, but the engine currently drops per-point/per-frame ray origins
and no-return rays on that path; fused blind and verification queries therefore
have no `SensorContext`. A single `PointFrame` builds one context using the
sensor pose at its frame timestamp, so its ray calculations still use one origin
after deskewing rather than a per-point origin.

**Planned:** scan-consistent ray evidence with per-point origins, retained
no-return rays through fusion, transform uncertainty, and time-interpolated
gripper geometry.

## DetectFree — geometry-only discovery

**Default:** `detect_free` calls `detect_blocks_detailed` with the configured
laid-block dimensions `(0.9, 0.6, 0.6)`, 4 cm voxel downsampling, ground
removal, density proposals/clustering, deterministic plane fitting, cuboid
hypotheses, evidence scoring, non-overlap selection, and SDF-based 6-DoF
refinement. Its runtime `confidence` and `covariance` fields are currently
always `None`. Offline synthetic calibration utilities exist, but no confidence
model is loaded or injected into the detector runtime.

The detector can use a single-origin `SensorContext` to score returned-ray and
no-return free-space consistency. Raw XYZ input has no sensor context and
reports missing evidence channels instead. The current public default does not
enable multi-orientation candidates (`candidate_dims=[]`) or multiscale
proposals, and retains no additional conflict alternatives
(`conflict_alternatives=1`).

**Known limits:** blind clustering cannot reliably split touching or seamless
wall blocks; it also loses recall at long range and on some stacked/assembly
scenes. It is appropriate for isolated/staged geometry, not as the identity
authority for a planned wall.

**Opt-in:** callers may provide candidate dimensions when orientation is known.
This changes hypothesis ranking and is not enabled by the default API.

## VerifyPlaced — plan-conditioned verification

Caller-supplied `expected_blocks` are the current identity input for placed
blocks. `set_world_state()` stores values but the facade does not currently
consume them during verification; construction-plan/world-state integration is
therefore not wired end to end. The two implemented verifier paths have
different contracts.

**Default (legacy plan prior):** `verify_placed` calls
`plan_prior.detect_with_plan`. It removes ground, assigns each remaining return
exclusively to the nearest expected cuboid surface within a 12 cm band, applies
the same bounded SDF refinement/evidence model as blind detection, then maps
results to statuses:

- `Confirmed` when refined, sufficiently supported/scored, and within 5 cm of plan.
- `Deviated` when the same evidence is present but translation exceeds 5 cm.
- `Missing` only with a sensor context, zero support, incident rays, and
  free-space violations.
- `NotObservable` otherwise.

This default path returns no covariance and does not use course constraints.

**Opt-in (v1):** constructing `HybridBlockDetector(verify_placed_v1=True)`
calls `verify_placed_blocks`. It performs visibility-aware expected-face
assignment, robust bounded cuboid fitting (10 cm / 10° trust bounds), Jacobian
rank/weak-direction analysis, and geometric covariance when the fit has enough
constraint. `Confirmed` means fit residual is acceptable and deviation is at
most 3 cm and 2°; a supported in-bound fit outside that tolerance is `Deviated`.
`Missing` additionally requires at least three returned/no-return free-space
rays through a visible planned volume; raw silence remains `NotObservable`.

V1 covariance is a fit-curvature estimate, not a calibrated end-to-end
uncertainty: it excludes sensor-pose error and lacks empirical coverage gates.
The default façade passes plain `BlockPose` values, so its v1 route does not
provide structured `PlacedBlockPrior` or course constraints.

**Opt-in (direct v1 API):** `verify_placed_blocks` accepts `PlacedBlockPrior`
and `CourseConstraint`. Course layout may fill data-null translation directions
only and reports `LayoutInferred`; it does not change independent presence
status or measured deviation. Shared-course yaw refinement is separately gated
by `VerifyPlacedParams(enable_course_yaw_refinement=True)`, requires a named
plan and at least three compatible members, and preserves independent output if
its evidence/timeout checks fail. Current real snapshot topology declares no
courses, so this path has synthetic/unit coverage but no real wall acceptance.

## EstimateGripped

**Default:** `estimate_gripped` is a compatibility wrapper around
`verify_placed` with one grasp prior. It relabels a supported result as
`Confirmed`/`Deviated`, or otherwise `NotObservable`, and marks its source
`grasp_prior`.

**Planned:** gripper-link filtering, gripper occlusion/shadow reasoning, and
FK/robot-state covariance. The present wrapper must not be interpreted as a
dedicated gripped-object estimator.

## Reproducible evaluation snapshot

The following figures were generated on 2026-07-22 at revision
`ff82dfc067f55ef26037bcf025197fc3238492c6` with:

```bash
uv run python tools/eval_real_snapshots.py --metrics-out /tmp/blockpose-real-metrics.json
uv run python tools/eval_verify_placed_snapshots.py --output /tmp/blockpose-verify-metrics.json
uv run python tools/eval_detector.py --metrics-out /tmp/blockpose-synth-metrics.json
```

The real blind-detection run scored 23 complete annotated snapshots (61
reference targets, after the evaluator's three configured snapshot exclusions):
40 matches, 65.6% recall, 85.1% precision, 7 false positives, 4.02 cm
translation p95, 5.14° rotation p95, and approximately 0.4 s runtime p95.
Runtime is machine- and load-sensitive. These are diagnostics, not release
gates: annotations include manual/RGB-derived reference poses and the command's
matching policy; rerun it when annotations, data, parameters, revision, or
machine change.

The synthetic Phase-A gate passed (8/8 isolated scenes, zero empty-scene false
positives; 6.21 mm translation p95 and 0.249° rotation p95). This is an
evaluator-configured upright-orientation test: it sets `(0.6, 0.6, 0.9)` per
scene and does not validate `HybridBlockDetector()`'s public laid-block default.
It is intentionally narrower than the deployment problem: the same diagnostic
matrix has zero blind recall on its touching and standoff families and 22.2%
recall for its wall-plan family. Run `uv run python tools/eval_detector.py` to
reproduce the gate and diagnostics.

The v1 diagnostic evaluates 15 leakage-safe cross-snapshot priors only. It is
explicitly marked `acceptance_eligible: false`; it does not establish real
wall-course behavior or tune thresholds. Run the second command above to
inspect its per-target results.

## What must change before production acceptance

**Planned:** consolidate legacy and v1 into one documented verification
contract; preserve per-point scan geometry and no-return evidence end to end;
implement gripper-aware estimation; and collect held-out, plan-identified real
wall captures with course membership recorded at capture time. Acceptance should
then gate status correctness (especially false `Missing`), identity, pose by
observability, covariance coverage, and latency on held-out recording sessions.
