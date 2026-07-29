# Implementation Plan v4: Reliable Block Pose Estimation (Standalone, In-Wall, Gripped)

## 0. Goal and reframing

**Primary goal:** reliable 6-DoF pose estimation of 0.6 × 0.6 × 0.9 m concrete
blocks in three situations:

1. **gripped** — block held by the PZS100, partially occluded by it;
2. **placed / in-wall** — block at a pose known from the construction plan and
   placement history, possibly with invisible seams;
3. **free** — block in the staging/pickup area with no reliable prior.

**Operating model:** on-demand. The behavior tree issues a *query*; the engine
answers with poses, covariances, and confidences for one captured frame (or a
short burst of frames). There is no continuous tracking loop. Latency budget:
**≤ 500 ms per query** on the target machine (soft; a BT tick can wait).

**Central design decision (changed from v3):** this is primarily a
**verification/refinement problem, not a detection problem.** For gripped and
placed blocks, a strong pose prior always exists (gripper FK; construction
plan + placement history). The main pipeline is therefore
*prior-conditioned pose refinement*: start from the expected pose, register
the expected-visible model against the measured cloud, and report the refined
pose, deviation from prior, and confidence — or a clear "not where expected"
result. From-scratch detection is needed **only** for free blocks in the
staging area, which is the geometrically easy case (isolated or loosely
grouped blocks on the ground).

Consequences:

- the hard v3 cases (seamless walls, rear blocks, heavy occlusion) become
  near-trivial because those blocks never need to be *found*, only confirmed;
- no candidate-hypothesis graph, no global candidate selection, no wall
  grammar in the core path;
- a learned detector is **deferred behind an explicit evidence gate**
  (Section 10) and is expected never to be needed.

RGB is dropped from the estimation path entirely. LiDAR **intensity** replaces
it as the cheap appearance cue (Section 6.2): concrete vs. painted steel vs.
ground reflectivity is illumination-invariant, which is exactly where RGB
failed.

---

## 1. Query API (what the behavior tree calls)

Three query types, one engine:

```text
EstimateGripped   FK pose prior from TF, grasp-uncertainty model,
                  PZS100 occlusion handling mandatory.

VerifyPlaced      one or more block IDs (or "all in view"), priors from
                  the world model / construction plan. Returns per block:
                  refined pose, deviation from expected, status
                  {Confirmed, Deviated, Missing, NotObservable}.

DetectFree        workspace region; from-scratch geometry detection,
                  returns zero or more blocks with poses.
```

All three share preprocessing, visibility modeling, registration, covariance,
and evaluation. The engine holds persistent construction state (block priors,
planned wall) updated by the BT via explicit calls — the engine does not
subscribe to anything itself.

```cpp
namespace block_perception {

class BlockPerceptionEngine {
public:
    static std::unique_ptr<BlockPerceptionEngine> create(const Config& config);
    static std::unique_ptr<BlockPerceptionEngine>
        create_from_yaml(const std::filesystem::path& path);

    virtual ~BlockPerceptionEngine() = default;

    virtual GrippedResult  estimate_gripped(const FrameInput&, const GrippedQuery&)  = 0;
    virtual PlacedResult   verify_placed  (const FrameInput&, const PlacedQuery&)   = 0;
    virtual DetectResult   detect_free    (const FrameInput&, const DetectQuery&)   = 0;

    // Persistent world state, pushed by the caller.
    virtual void set_block_priors(std::vector<BlockPrior>) = 0;
    virtual void set_planned_wall(std::optional<PlannedWall>) = 0;
    virtual void notify_placement(const PlacementEvent&) = 0;

    virtual void reset() = 0;
    virtual std::string implementation_version() const = 0;
};

} // namespace block_perception
```

Note both `create(const Config&)` **and** `create_from_yaml`: parameter sweeps
from Python must not require writing temp YAML files.

---

## 2. Architecture and language decisions

### 2.1 Single C++20 core, pybind11 bindings, thin ROS 2 wrapper

Unchanged in spirit from v3: one C++ implementation used both online and in
Python experiments. ROS 2 restricted to recording, live I/O, and publishing.
The core must not publicly expose `rclcpp`, `sensor_msgs`, TF2, rosbag2, or
MoveIt types.

### 2.2 Eigen IS a public vocabulary type (changed from v3)

The "no third-party types in the public API" rule applies to Open3D, PCL,
Torch, ONNX Runtime, and ROS — **not** to Eigen. Rationale:

- pybind11 has first-class Eigen support (`pybind11/eigen.h`): zero-copy
  NumPy interop for free. Custom `Vec3f`/`Matrix6d` PODs would force
  hand-written buffer protocols and copies at exactly the boundary the
  Python workflow (sweeps, notebooks) exercises most.
- Eigen is header-only and already a hard dependency of Open3D, Ceres, and
  every ROS conversion.

Public types:

```cpp
namespace block_perception {

using TimestampNs = std::int64_t;

struct Pose3d {
    Eigen::Isometry3d T{Eigen::Isometry3d::Identity()};
};

using Covariance6d = Eigen::Matrix<double, 6, 6>;

struct PointCloud {
    // Row-major (N,3) float32; zero-copy NumPy mapping.
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> xyz;

    // Empty (0 rows) when not recorded. Same N otherwise; validate() checks.
    Eigen::VectorXf intensity;
    Eigen::VectorXf relative_time_s;   // REQUIRED for live crane data (deskew)
    Eigen::Matrix<std::uint16_t, Eigen::Dynamic, 1> beam_id;

    void validate() const;
};

} // namespace block_perception
```

### 2.3 Precision and Open3D interop — decide now, not in profiling

Core point storage is **float32**. Internally use the **Open3D tensor API
(`open3d::t`)**, which supports float32 with contiguous buffers, instead of
the legacy `open3d::geometry` API (double-precision `Eigen::Vector3d` AoS,
which would force a full-cloud convert-and-copy on every call). If a needed
operation exists only in the legacy API, wrap the conversion in one place and
count it in `RuntimeStats`.

### 2.4 Monorepo until the API stabilizes (changed from v3)

One repository. The ROS 2 package lives in a subdirectory that colcon builds
only when a ROS environment is sourced; plain CMake builds skip it. Split into
two repos only after the public API has survived several phases unchanged.

```text
block_perception/
├── CMakeLists.txt
├── include/block_perception/        # public API (Eigen-based)
├── src/
│   ├── preprocessing/               # crop, deskew, gripper filter, normals
│   ├── registration/                # plane fit, closed-form pose, Ceres refine
│   ├── visibility/                  # raycast expected-visible model
│   ├── queries/                     # gripped / placed / free pipelines
│   ├── covariance/
│   └── io/                          # canonical sample reader/writer
├── python/                          # pybind11 module + pure-Python tools
│   ├── bindings/
│   └── block_perception/            # dataset, synth, analysis, rerun logging
├── ros2/block_perception_ros2/      # built only under colcon
├── apps/                            # block_query, block_evaluate, block_view
├── tests/                           # unit + regression (see Section 11)
├── config/
└── docs/
```

Core dependencies: C++20, Eigen, Open3D (tensor API), **Ceres (required — see
6.5)**, yaml-cpp, nlohmann_json, CLI11, pybind11. Optional: Rerun C++ SDK
(tools only).

### 2.5 Visualization: Rerun only (changed from v3)

One viewer. Rerun covers sequence scrubbing, per-entity toggling, per-point
inspection, scalar timelines, and side-by-side GT/prediction. Keep the debug
sink abstraction but implement only:

```text
NullDebugSink      (benchmarks, CI)
RerunDebugSink
```

Add Polyscope later **only if** a concrete interactive need (manual pose
nudging, live threshold sliders) actually blocks debugging. Do not build it
speculatively. Open3D viewer calls may appear in throwaway scripts but never
in the library.

Rerun entity layout as in v3 (`/world/...`, `/metrics/...`, `/diagnostics`).

---

## 3. Data model

### 3.1 Canonical sample (unchanged in spirit)

```text
sample_000123/
├── points.npz          # xyz f32, intensity, relative_time_s, beam_id
├── transforms.json     # T_world_sensor(t), T_world_gripper(t)  ← trajectories, not single poses
├── labels.json         # GT/pseudo-label block poses + symmetry set + visibility
├── occluders.json      # PZS100 mesh id/hash + pose trajectory
└── metadata.json       # scene category, source bag+timestamp, schema versions
```

**Change from v3:** `transforms.json` stores short **pose trajectories**
covering the scan interval (sampled TF), not single snapshots — required for
deskewing and for time-interpolated gripper filtering (Section 5).

Schema versioning: one version field per file kind; hard-fail on unknown
versions. (Keep this; drop the separate "model I/O schema" until a model
exists.)

### 3.2 Bags

- New recordings: rosbag2/MCAP with per-point time, intensity, beam id,
  `/tf`, `/tf_static`, joint states, PZS100 state, placement events.
- Direct MCAP Python reading for standard messages, no ROS env.
- Dedicated ROS exporter container for legacy `.db3` bags / custom messages.

---

## 4. Phase P — throwaway Python spike (new, before any API freeze)

> **2026-07 execution note:** the repository now has a Python reference and
> measured snapshots, so Phase P is historical rather than the next task. The
> active sequence is `docs/next-steps.md`: build the C++20 + ROS 2 evaluation
> replay skeleton now, port `SCENE_DISCOVERY` first using complete rosbags, and
> preserve the reviewed-snapshot evaluation contract. Later phases below describe the target capability,
> not permission to claim production readiness.

**Duration: 2–3 days. Explicitly throwaway code.**

Export 10–20 real Seyond frames (any quick path). In a Python notebook with
Open3D:

1. crop workspace, estimate normals, extract planes;
2. check: are top+side planes of blocks reliably recoverable at real ranges,
   noise, and crane vibration? At what voxel size?
3. plot intensity histograms for block faces vs. gripper vs. ground — does
   intensity separate them?
4. inspect per-point timestamps: how large is intra-scan motion at typical
   crane speeds (project TF over the scan interval)? Is deskew ~cm-level or
   ~mm-level?

**Purpose:** the project risk is algorithmic, not performance. These four
answers determine API details (feature fields, deskew necessity, plane-fit
parameters) *before* the C++ types are frozen. Findings go into
`docs/spike_findings.md`; the notebook is then deleted or archived, never
extended into a detector.

---

## 5. Preprocessing (shared by all queries)

Order:

1. **Deskew / motion compensation (new, mandatory for live data).**
   Per-point transform into the world frame using `relative_time_s` and the
   interpolated `T_world_sensor(t)` trajectory. On a crane, sensor and grasped
   block move during the sweep; skipping this biases every downstream stage.
   Synthetic data must be able to simulate this (Section 8).
2. Workspace crop (query-specific region).
3. **Gripper filtering with time-interpolated gripper pose:**
   a. surface removal: points within `surface_padding` (start 10–30 mm) of the
      simplified PZS100 collision mesh at `T_world_gripper(t_point)`;
   b. shadow test: ray from sensor origin; if the mesh intersects before the
      measured range → invalid shadow return. Bounded region only, BVH-backed.
4. Optional ground removal (staging queries).
5. Voxel downsample (size from spike findings).
6. Normal estimation.
7. **Intensity features:** per-point intensity retained through downsampling
   (mean); used as a plane-membership and block-vs-clutter cue if the spike
   confirms separability.

**Sensor-pose uncertainty (new):** `T_world_sensor` from crane FK carries
boom-deflection error. Treat it as uncertain:

- carry a configurable prior covariance on the sensor pose;
- optionally refine the sensor pose per query against the ground plane and/or
  known static structure (a one-shot plane alignment), when such references
  are in view;
- propagate remaining sensor-pose uncertainty into block covariances instead
  of pretending FK is exact.

---

## 6. Pose estimation core (the actual product)

### 6.1 Expected-visible model

For a candidate/prior pose: raycast the block model and the PZS100 mesh from
the sensor origin (Open3D raycasting) → per-face visible fraction, visible
model points, and classification of missing data as
{negative evidence, gripper-hidden, block-hidden, out-of-view}. Model surfaces
hidden by the gripper are excluded from correspondences, coverage, and
free-space penalties.

### 6.2 Segmentation-light front end

Plane extraction on the (deskewed, filtered) cloud; planes classified by
normal vs. gravity and by intensity statistics. This is the entire "detector
front end" for the free-block case and the sanity check for the prior cases.

### 6.3 Closed-form pose from planes, then refinement (changed from v3)

Point-to-plane ICP on a cuboid has flat cost valleys: translation parallel to
a visible face is unconstrained by that face; with top+one-side visibility one
translation direction is pinned only by edges or priors. Therefore:

1. **Closed-form initialization:** with known dimensions and gravity, 2–3
   fitted plane equations give the pose (up to the symmetry set) directly —
   cheap, no basin-of-attraction issues.
2. **Edge/boundary correspondences as a first-class residual type:** boundary
   points of visible faces constrain the sliding directions. Extract them
   explicitly (normal-discontinuity / boundary estimation) and weight them
   separately.
3. **Joint refinement in Ceres (single nonlinear least squares), residuals:**
   - point-to-plane residuals on expected-visible faces;
   - edge residuals;
   - **prior residual** `|| log(T_prior⁻¹ · T) ||²_Σprior` where the prior is
     FK/grasp (gripped), plan/history (placed), or absent (free);
   - optional sensor-pose block if refining `T_world_sensor` jointly.

This replaces "ICP then fuse prior": the v3 grasp-prior equation (its §9.4)
already *was* a joint NLLS — so Ceres is a **required** dependency, decided
now.

### 6.4 Symmetry and observability

Square cross-section → 4-fold symmetry about the long axis plus 180° flips.
Return the canonical pose plus the symmetry-equivalent set. Classify
observability per block:

```text
WellConstrained | PartiallyConstrained (sliding mode present)
| SymmetryAmbiguous | LayoutInferred | Unobservable
```

`PartiallyConstrained` must name the unconstrained direction(s) in the result
(from the Ceres Jacobian spectrum) so the BT can decide whether that direction
matters for the current action.

### 6.5 Covariance — calibrated, not raw

`(JᵀJ)⁻¹` from ICP-style problems is known to be optimistic. Ship:

- Jacobian-based covariance scaled by a **per-scenario calibration factor**
  estimated on synthetic GT and pseudo-labeled real data (compare predicted
  vs. empirical error distributions; store factors in config);
- a hard floor from sensor-pose uncertainty (Section 5).

Document clearly which frame and convention the 6×6 covariance uses.

### 6.6 Axis convention — one definition (fix from v3)

Define once in `docs/conventions.md` and one header: block frame = geometric
center, **x along the 0.9 m axis**, z up in the canonical resting pose;
dimensions constant `{0.9, 0.6, 0.6}` everywhere (v3 stated 0.6×0.6×0.9 in
one place and {0.9,0.6,0.6} in another). Also record the casting tolerance of
real blocks; if faces deviate > ~5 mm from nominal, allow a small per-face
inflation in correspondences rather than pretending blocks are exact.

---

## 7. The three query pipelines

### 7.1 EstimateGripped

```text
FK prior (T_world_gripper · T_gripper_block_nominal, wide Σ)
→ preprocessing with gripper filter at interpolated poses
→ expected-visible model under PZS100 occlusion
→ closed-form check → Ceres joint refine with grasp prior
→ pose, covariance, deviation-from-FK, observability
```

Failure semantics: if visible block surface fraction < threshold →
`NotObservable`, return the FK prior with inflated covariance, flagged
`source = TrackedPrior`. Never silently return an unrefined prior as if
measured.

### 7.2 VerifyPlaced

Per queried block ID (or all in view):

```text
prior pose from world model / plan
→ expected-visible model (occlusion by neighboring planned blocks AND gripper)
→ local crop around expected pose
→ Ceres refine with plan prior
→ status: Confirmed (residual + deviation small)
        | Deviated (converged, pose offset > tolerance → report offset)
        | Missing  (expected-visible surface absent, free space observed)
        | NotObservable (insufficient visible surface)
```

Wall handling collapses into this: a wall is verified block-by-block against
the plan; blocks whose faces are flush/seamless are constrained by the shared
wall surface plus course/lattice residuals *as additional residuals in the
same refinement*, and reported `LayoutInferred` when individually
unobservable. No standalone wall-grammar module; a wall-level pose summary
(dominant surface plane + course heights) is computed as a byproduct for
diagnostics.

### 7.3 DetectFree

Scoped to the staging area:

```text
preprocessing → plane extraction → top+side (later: two-side, top-only)
hypotheses with gravity + fixed dimensions
→ per-hypothesis expected-visible scoring
   S = w_s·surface + w_n·normal + w_c·coverage − w_f·free − w_o·overlap
→ greedy non-overlap selection with visibility recomputation
→ Ceres refine each accepted block (no prior residual)
```

This is v3's geometry detector, but scoped: no rear-block reasoning, no wall
lattice, no candidate graph optimization — those cases are handled by
VerifyPlaced. Stacks/rows in staging: keep the greedy accept-recompute loop;
a global optimizer only if greedy demonstrably fails on the eval set.

---

## 8. Synthetic data

Python orchestration + Open3D raycasting, canonical sample writer,
deterministic seeds and manifests. Changes from v3:

1. **Ray templates from real Seyond frames** (unchanged, good idea): normalize
   real returns to unit directions per sensor mode, replay into mesh scenes.
2. **Motion simulation (new, required):** synthesize sensor/gripper pose
   trajectories over the scan interval and assign per-point timestamps, so the
   deskew stage and time-interpolated gripper filter have ground truth to be
   tested against. Without this, Phase-level acceptance of Section 5 is
   impossible.
3. **Intensity model (simple):** per-material constant + noise (concrete,
   steel, ground), enough to exercise the intensity features; refine only if
   the feature earns its keep on real data.
4. Scene progression: single block → block+ground → block+PZS100 (gripped
   geometry) → row → stack → partial wall → full wall (seams closed) →
   misplaced-block wall → staging clutter.
5. GT per sample: pose + symmetry set, per-point instance ID, visible-face
   fractions, occluder identity, PZS100-occluded fraction, support relations.

Isaac Sim: only after a *measured* synthetic-to-real gap on the frozen real
eval set attributable to material response / dropout / clutter. Same gate as
v3, kept.

---

## 9. Real data

- Export **20–50 representative frames** early (Phase 1), covering: gripped
  (multiple grasp heights/swing states), placed against wall, seamless wall,
  staging singles and stacks. Manually checked pseudo-label poses via a small
  Rerun/CLI adjust tool.
- Freeze a real **test set** as soon as it exists; never tune on it. Split
  rules unchanged from v3: by recording day / wall / construction run / sensor
  mounting; never adjacent frames across train/test.

---

## 10. Learned components — evidence-gated (heavily demoted from v3)

**Gate:** build any learned component only if, on the frozen real eval set,
the geometry system misses a recall/precision target that parameter tuning
and the priors cannot close, in a scenario that matters operationally.

**If the gate triggers, the first learned component is point-wise
segmentation** (block / gripper / ground / clutter), *not* an amodal 3D
detector:

- one known rigid class with exact dimensions → amodal box regression solves
  a harder problem than exists; segmentation feeding the existing plane/pose
  pipeline is less data-hungry and slots in as a preprocessing stage;
- **deployment constraint decided up front:** use only export-friendly ops
  (dense voxel/pillar convs or vanilla point MLPs). VoteNet/FCAF3D/TR3D are
  ruled out for deployment: sparse-conv and PointNet++ custom CUDA ops have no
  practical ONNX path. If a chosen architecture can't export, the deployment
  path is **libtorch/TorchScript in-process** (first choice) or a Python
  sidecar process (second); this is decided per-model *before* training.
- No MMDetection3D. A small custom training script on the canonical dataset.

Everything downstream (visibility, refinement, covariance) is unchanged —
the segmentation output only replaces/augments Section 6.2.

---

## 11. Testing, evaluation, and agent workflow support

This plan will be executed by an agentic build/test/refine workflow. Every
phase therefore ends in **machine-checkable gates**: CLI commands that exit
non-zero on failure and emit metrics as JSON.

### 11.1 Test layers

1. **Unit tests** (ctest/GTest): transforms, symmetry set, plane fitting,
   closed-form pose, deskew on analytic trajectories, gripper filter on
   synthetic meshes, covariance frame conventions.
2. **Regression on synthetic GT:** `block_evaluate --suite synthetic` runs the
   full engine on generated scenes and asserts metric thresholds
   (translation/rotation error, status correctness for Deviated/Missing
   scenarios, deskew residual). Deterministic seeds; thresholds in a checked-in
   `expected_metrics.yaml`; the agent tightens thresholds as quality improves,
   never loosens without a human-visible diff.
3. **Frozen real eval:** `block_evaluate --suite real_frozen` — reported, and
   gated once pseudo-labels are trusted.
4. **Golden-output snapshots** for a handful of samples (serialized
   FrameResult JSON, tolerance-compared) to catch silent behavior drift.

### 11.2 Metrics (per scenario subset)

```text
translation error (split: constrained vs. sliding directions)
symmetry-aware rotation error
status accuracy for VerifyPlaced (Confirmed/Deviated/Missing/NotObservable)
covariance calibration (empirical vs. predicted error ratio)
false positives per DetectFree scan; recall vs. visible fraction
runtime per stage vs. 500 ms budget
```

Scenario subsets: gripped-low-occlusion, gripped-heavy-occlusion, placed-open,
placed-seamless-wall, misplaced-block, staging-isolated, staging-stack.

### 11.3 Agent-facing conventions

- All apps take `--config`, `--sample/--suite`, `--metrics-out metrics.json`,
  `--rerun-out file.rrd`; no interactive prompts.
- `block_query --mode {gripped|placed|free}` runs one query on one sample —
  the primary debugging entry point.
- Every failure mode returns a structured `Diagnostic{code,message}`;
  diagnostic codes are enumerated in one header and stable.
- Synthetic generation is fully reproducible from a manifest
  (`generate --manifest scenes.yaml --seed N`).
- CI: build (plain CMake, no ROS) + unit + synthetic regression on every
  change; ROS package build in a separate optional job.

---

## 12. Phase plan with acceptance gates

**Phase P — Python spike (2–3 days).** Section 4.
*Gate:* `docs/spike_findings.md` answers plane-recoverability, voxel size,
intensity separability, intra-scan motion magnitude.

**Phase 0 — Skeleton.** Repo, public API (Eigen types), Config + YAML,
canonical sample reader/writer, pybind11 module with zero-copy NumPy checks,
Rerun sink, `block_query` app, unit tests for transforms/symmetry.
*Gate:* one synthetic and one real exported sample load and round-trip through
C++ and Python; `pytest` proves NumPy zero-copy; CI green.

**Phase 1 — Data loop.** 20–50 real frames exported with TF trajectories +
PZS100 mesh/hash; pseudo-label tool; synthetic generator with motion
simulation and single-block/gripped scenes; evaluator skeleton emitting JSON.
*Gate:* `block_evaluate --suite synthetic` runs end-to-end (engine may return
priors only); real and synthetic samples processed by the same binary.

**Phase 2 — Core estimation, gripped + isolated.** Preprocessing incl. deskew
and time-interpolated gripper filter; expected-visible raycasting; plane
extraction; closed-form pose; Ceres joint refinement with edge residuals and
grasp prior; symmetry + observability; covariance (uncalibrated).
*Gate (synthetic):* gripped and isolated scenes — translation ≤ 10 mm RMS in
constrained directions, rotation ≤ 1°, sliding directions correctly flagged
`PartiallyConstrained`; deskew reduces error on moving-sensor scenes by a
measured margin. *Gate (real):* stable poses on the gripped and isolated
frames, judged against pseudo-labels.

**Phase 3 — VerifyPlaced + wall.** Plan priors, neighbor-occlusion in the
visibility model, status classification, shared-surface/course residuals,
`LayoutInferred` reporting. Misplaced-block synthetic scenarios.
*Gate:* status accuracy ≥ 95 % on synthetic Deviated/Missing/Confirmed suite;
seamless-wall verification does not regress pose error vs. open placement.

**Phase 4 — DetectFree.** Top+side hypotheses, scoring, greedy selection,
stacks/rows in staging.
*Gate:* precision/recall targets on staging subsets; zero false positives on
empty-workspace scans.

**Phase 5 — Covariance calibration + sensor-pose refinement.** Per-scenario
calibration factors; optional per-query sensor pose refinement; propagate.
*Gate:* covariance calibration ratio within [0.5, 2] per scenario subset.

**Phase 6 — ROS 2 wrapper + BT integration.** Service/action interface
mapping the three queries; PointCloud2 conversion preserving per-point time
and intensity; TF trajectory sampling over the scan interval; priors pushed
from world model; Rerun recording toggle; runtime monitoring.
*Gate:* on-demand query from the BT on live/replayed data within the 500 ms
budget; integration test: wrapper output == standalone output on the same bag
frame.

**Phase 7 (conditional) — learned segmentation.** Only if the Section 10 gate
triggers on the frozen real eval set.

Explicitly postponed indefinitely: amodal 3D detectors, MMDetection3D,
Isaac Sim, global integer optimization, factor-graph tracking, Polyscope,
CVAT/FiftyOne, embedded Python in ROS, wall-grammar module, second repo.

---

## 13. Recommended components

1. Open3D **tensor** API — float32 clouds, raycasting, registration kernels.
2. Eigen — public vocabulary types; pybind11 `eigen.h` for zero-copy NumPy.
3. Ceres — joint pose refinement with priors (required).
4. pybind11 — bindings.
5. Rerun (C++ & Python SDKs) — sole visualization stack.
6. MCAP Python — direct bag reading without ROS.
7. rosbag2/MCAP — recording; exporter container for legacy bags.
8. robot_body_filter / robot_self_filter — reference implementations for
   mesh surface + shadow filtering (adapt the ray-shadow test, add per-point
   time interpolation).
9. Seyond official ROS driver — acquisition; confirm it publishes per-point
   time and intensity fields.
10. libtorch — contingent deployment path for Phase 7 if it ever happens.

---

## 14. Summary of changes vs. v3

| v3 | v4 | Why |
|---|---|---|
| Detection-first, priors as score bonus | Verification-first; detection only for staging | Plan + placement history make most blocks a refinement problem; kills the hard cases structurally |
| Continuous-pipeline framing | Three on-demand queries matching the BT | Matches actual operation; sets a concrete 500 ms budget |
| Custom Vec3f/Matrix6d PODs | Eigen public types | Zero-copy NumPy via pybind11; Eigen unavoidable anyway |
| Open3D unspecified | Open3D tensor API, float32 committed | Legacy API is double AoS → hidden full-cloud copies |
| No motion handling | Mandatory deskew + time-interpolated gripper filter; TF trajectories in samples | Crane motion during scan biases everything |
| FK treated as exact input | Sensor-pose uncertainty modeled, optional per-query refinement | Boom deflection |
| ICP-centric refinement | Closed-form plane pose + edge residuals + single Ceres NLLS with priors | Cuboid sliding modes; §9.4 grasp prior was already a joint NLLS |
| Ceres optional | Ceres required | Follows from the above |
| VoteNet/FCAF3D/TR3D → ONNX | Ruled out (no ONNX path); gated point-wise segmentation, libtorch fallback | Sparse-conv/PointNet++ ops don't export |
| MMDetection3D adapter | Dropped | Heavy dependency for one class |
| Rerun + Polyscope + Open3D viewers | Rerun only | Maintenance tax |
| Two repos | Monorepo, ROS subpackage | API churn phase |
| YAML-only engine construction | Config struct + YAML | Python sweeps without temp files |
| Intensity recorded, unused | Intensity as segmentation feature | Illumination-invariant cue where RGB failed |
| No prototype allowed | 2–3 day throwaway spike before API freeze | Risk is algorithmic, not performance |
| Prose success criteria | Machine-checkable gates, JSON metrics, CLI conventions | Plan is executed by an agentic workflow |
