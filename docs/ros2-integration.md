# Getting the geometry detector into a ROS 2 node, and verifying it on your bags

This is the operator map for Track A of `docs/production-readiness-plan.md`
(v2): "integrate into a ROS 2 node and verify detector performance with my
rosbags." It is a map, not a promise -- every step below says whether the
tooling exists, partly exists, or does not exist, based on what is actually in
this tree today. Where a step has no tooling, that gap is named rather than
glossed over.

## 1. Consuming the core from a ROS 2 node

**Exists and is gated.** `cpp/` builds an INTERFACE CMake package,
`blockpose_core` (currently version `0.2.0` -- see `cpp/CMakeLists.txt`'s
version-bump comment for why: the cluster gate moved 1.5 -> 2.25 and the
gripper self-filter, `annotation_snap` and per-stage diagnostics were added
since the `0.1.0` tag was implied). A downstream ROS 2 package consumes it the
ordinary ament/catkin way:

```cmake
find_package(blockpose_core REQUIRED)
target_link_libraries(concrete_block_detector PRIVATE blockpose::core)
```

`blockpose::core` is header-only and pulls in only Eigen3 (`>= 3.3`, found via
`find_package(Eigen3 3.3 REQUIRED NO_MODULE)`) -- no ROS, no Python, no GPU.
There is a second, optional target, `blockpose::vision`, built only if OpenCV
is found at configure time; the geometry core itself stays dependency-free.

**Build dependency the consuming environment must supply:** Eigen3.
`cpp/CMakeLists.txt` hard-requires it at configure time and nothing in `cpp/`
vendors it -- there is no `FetchContent` fallback. `docker/Dockerfile.core`
installs `libeigen3-dev`, but only as of Track A1: the image previously had no
`eigen` entry at all, so `tools/check_cpp_consumer.sh` could not run at all in
a container built from it (`docs/ENVIRONMENT.md` is where the core image's
system-level C++ build dependencies are listed). Any environment *not* built
from that image -- your ROS 2 workspace, a bare host, CI -- must still supply it
(`apt-get install libeigen3-dev` on Debian/Ubuntu). Without it, `cmake -S cpp
-B build` fails at the `find_package(Eigen3 3.3 REQUIRED NO_MODULE)` line
before anything else runs -- which also means a fresh clone cannot build
`python/blockpose/_core*.so`, since that extension links the same
`blockpose::core` target and `*.so` is gitignored (`.gitignore:21`), so it is
never present pre-built.

**The pre-ship gate:** `tools/check_cpp_consumer.sh`. It configures the core
with `-DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_CORE_BUILD_PYTHON=OFF`
(GTest is deliberately not required -- the script's own comment: "GTest is not
installed in every dev container... the ROS node does not need it"), builds
it, installs it to a scratch prefix, then configures/builds/runs
`cpp/tests/consumer` against that *installed* package via
`-DCMAKE_PREFIX_PATH`. Run it before shipping anything the ROS node links:

```bash
tools/check_cpp_consumer.sh   # or: make ship-check
```

`make ship-check` is deliberately *not* part of `make check` (which is
`pytest lint` only): the ROS-facing surface costs a from-scratch CMake
configure + build and needs a C++ toolchain plus Eigen3, which a Python-only
contributor's environment has no other reason to have. The consequence for an
operator: passing `make check` says nothing about whether the ROS node still
builds.

`cpp/tests/consumer/` is the reference example of what a downstream consumer
looks like -- its own top comment says so ("Stands in for the ROS node"). Its
`CMakeLists.txt` is the minimal `find_package(blockpose_core REQUIRED)` +
`target_link_libraries(... blockpose::core)` pattern above; its `main.cpp`
includes the three representative headers (`pipeline.hpp`, `proposals.hpp` --
which is where `DetectionParameters` and its defaults live --, and
`gripper_self_filter.hpp`'s PZS100 jaw-box builder) and checks them against
measured values, e.g. `cluster_max_extent_xy == 2.25` and jaw-box thickness
`0.30 - 0.08 = 0.22`. Those checks use the file's own `REQUIRE`, deliberately
*not* `assert()`: the gate builds `-DCMAKE_BUILD_TYPE=Release`, which defines
`NDEBUG` and would compile an `assert()` away, so the gate would print the
numbers and exit 0 whatever they were. Model a real node's CMake and headers
on this directory.

Verified in this container, which was built before `Dockerfile.core` gained
`libeigen3-dev`, so Eigen3 was installed by hand first:

```
$ apt-cache policy libeigen3-dev
libeigen3-dev:
  Installed: 3.4.0-4build0.1
$ tools/check_cpp_consumer.sh
cluster_max_extent_xy = 2.25
dbscan_eps = 0.07
jaw box thickness = 0.22 (expect 0.22 = 0.30 - 0.08)
ROS-style consumer OK
```

## 2. What the shipped detector is, and what it is measured at

Geometry-only: point-cloud clustering, plane fitting and gating in
`cpp/include/blockpose/core/`. No RGB, no GPU, no learned model in the
default runtime path (see section 5 for what is explicitly excluded and why).

| Metric | Definition | Measured value |
| --- | --- | --- |
| Centre recall | Hungarian-matched to ground truth within 0.30 m (`blockpose.evaluate.match_detections`, `max_dist`) | 46/62 |
| Strict recall | Centre match **and** symmetry-aware rotation error <= 30 deg **and** oriented-box IoU >= 0.25 (`blockpose.experiment_record.strict_match`, `DEFAULT_MAX_ROTATION_DEG=30.0`, `DEFAULT_MIN_IOU=0.25`) | 44/62 |
| Precision | matched detections / total detections | 78% |

Measured on the 25-snapshot, 62-block Seibersdorf set
(`config/annotated_snapshots_seibersdorf.yaml`).

**Strict is the number that matters for whether a pose is usable.** Centre
match alone accepts a box that is in the right place but facing the wrong
way -- `experiment_record.py`'s own module docstring: "Centre-only scoring
counts a box 0.29 m away with 88 degrees of orientation error as a hit;
measured over the annotated set it inflated recall by two blocks at every
setting." A gripper planning off a centre-only match can approach a block
90 degrees wrong.

Shipped parameters (`blockpose::core::DetectionParameters`, defaulted in
`cpp/include/blockpose/core/proposals.hpp`):

| Parameter | Value | Why |
| --- | --- | --- |
| `cluster_max_extent_xy` | 2.25 m | raised from 1.5 m; see `experiments/detector_cluster_failures.md`. A Python test guards against this drifting back. |
| `dbscan_eps` | 0.07 m | clustering radius for the DBSCAN proposal stage |

Both values are asserted directly in `cpp/tests/consumer/main.cpp`, so
`tools/check_cpp_consumer.sh` re-verifies them on every run against the
*installed* package, not just the in-tree build.

## 3. The bag -> snapshot -> annotate -> evaluate loop

This is the loop that turns a rosbag into a scored report. **Every stage now
has tooling**; the wiring gap this section previously named was closed by
`tools/evaluate_detector_snapshots.py`, and the manifest/promotion-split
generators below.

| Stage | Tooling | Status |
| --- | --- | --- |
| Record a bag | `tools/record_reference_bag.sh` | **Exists, portable.** The ROS setup path and topic list are now overridable (env/flags) with the original values as defaults, so the previous invocation is unchanged. Preflights that the setup file exists and `ros2` is on `PATH`, failing with an actionable message. |
| Export bag -> snapshot | `tools/export_reference_bag.py` | **Exists.** Reads a bag (`sqlite3` or `mcap`, auto-detected), pulls RGB images, point clouds and TF into a snapshot directory: `python3 tools/export_reference_bag.py <bag_dir> --output <dir> --point-topic ... --image-topic ... --camera-info-topic ...`. Run inside a sourced ROS 2 workspace (needs `rosbag2_py`, `rclpy`, `rosidl_runtime_py`). |
| Initialize a manifest | `tools/init_snapshot_annotations.py` | **Exists.** Creates provisional `annotations.yaml` per snapshot (auto-fit registration/RGB targets, dedup by distance). Defaults to `data/registration_snapshots`; point it at your exported snapshots. |
| Hand-annotate / correct | `tools/annotate_snapshot.py`, documented in `docs/annotation-tool.md` | **Exists, interactive.** Open3D GUI: mask the gripper, run auto-fit, add/adjust boxes, mark visibility, save. Requires manual human review per snapshot -- there is no unattended annotation path, by design (auto-fit is a starting point, not ground truth). |
| Score detections against ground truth | `blockpose.evaluate.match_detections` / `evaluate`, `blockpose.experiment_record.strict_match` | **Exists as a library, not a bag-facing CLI.** `evaluate()` returns precision/recall/translation/rotation stats for one snapshot's detections vs. its ground truth. Existing callers (`tools/sweep_detector_params.py`, `tools/ablate_rgb_proposals.py`) invoke the C++ detector through the **Python nanobind binding** (`blockpose.core_bridge.detect_blocks_detailed_cpp`) over the whole annotated corpus for parameter sweeps and ablations -- not over bag-derived snapshots, and not as a per-scene report generator. |
| Build the manifest | `tools/build_snapshot_manifest.py` | **Exists.** Derives the `config/annotated_snapshots_seibersdorf.yaml` shape from a snapshot directory (a snapshot qualifies on `annotation_complete: true` plus at least one approved block). Verified to reproduce the committed manifest's scene/snapshot selection exactly -- 19 scenes, 25 snapshots. Previously the manifest said "regenerate rather than edit by hand" while no generator existed. |
| Produce a scored report | `tools/evaluate_detector_snapshots.py` | **Exists -- this was the gap.** Snapshots in, scored report out: runs the C++ detector with pinned `DetectionParameters`, scores centre (Hungarian, 0.30 m) and strict (adds symmetry-aware rotation and oriented-box IoU), and emits markdown + JSON with per-scene and per-visibility breakdowns, an `experiment_stamp`, and an honest health section listing every skipped/failed snapshot. The manifest is **optional** -- with `--manifest ""` it discovers annotated snapshots directly, which is what a fresh bag export needs. Reproduces the recorded baseline exactly: 46/62 centre, 44/62 strict, 46/59 precision. |
| Reserve a promotion gate | `tools/build_promotion_splits.py` | **Exists.** Generates a splits manifest from an operator-supplied session mapping (snapshot -> recording session) rather than a directory-name prefix, so reserved sessions are a genuine held-back set. Emits `grouping_evidence: operator_supplied_session_mapping`, and refuses to emit an unusable split (a session in both development and final, or a reserved set with no approved blocks). Output is consumed unmodified by `experiments/fold_safety.py`; verified that reserved snapshots never appear in a development fold. |

The manifest/splits format that the scoring functions and the annotated
corpus share:

- `config/annotated_snapshots_seibersdorf.yaml` -- derived (not hand-edited)
  index of which snapshots are annotation-complete, grouped by `scene` (the
  capture-directory prefix before `__`). Totals as of this commit: 25
  annotated snapshots, 62 approved blocks, 19 of 24 distinct scenes
  represented (5 without annotation -- see section 4).
- `config/rgb_ablation_splits_seibersdorf.yaml` -- leave-one-scenario-out
  development folds over the same scene grouping, used for RGB proposer/
  verifier ablations. See section 4 for why this is *not* a promotion gate.

## 4. Reserving a promotion-gate set (plan v2's A3)

**The splits mechanism exists; a real promotion gate does not yet, and the
existing file says so explicitly.** `config/rgb_ablation_splits_seibersdorf.yaml`
sets:

```yaml
selection_eligible: false
final_enabled: false
grouping_evidence: scenario_label_directory_prefix_not_validated_run_ids
```

Its own header comment explains why: the "scenario" groups are parsed from
capture-directory name prefixes (`grip_at_bottom`, `wall`, `wet`, `shadow`,
...) -- "a real experimental-condition boundary, but still not a verified
independent robot run/bag ID." Every scenario is currently a *development*
fold (leave-one-scenario-out), used for tuning and ablation comparisons. None
of it is marked `final`, and the file is explicit that promoting specific
scenarios to a held-out gate requires "real run_ids," not directory-name
parsing.

**What a real promotion-gate set requires**, per `docs/production-readiness-plan.md`'s
A3: whole independent recording sessions, annotated, and then *never* used
for tuning or selection -- not folds carved out of the same corpus that
parameter sweeps already touch.

**Why this matters, with the measured reason:** 62 blocks over 25 snapshots,
spread across only 19 annotated scenes (16 scenes contribute a single
annotated snapshot, three contribute three -- consecutive captures within a
scene are near-duplicate frames and are deliberately left unannotated). A +-2
block swing is inside the noise floor
of a set this small -- the production-readiness plan states this directly
("62 blocks over 25 snapshots cannot resolve a +-2-block effect"), and it is
the same magnitude as the two centre-only spurious matches the strict metric
was built to exclude (section 2) and the dino_proposer's currently
unrealised "+2 centre / +1 strict, inside noise" benefit (plan v2, decision
2). A held-out set built from this same 25-snapshot corpus could not tell a
real +2-block improvement from noise, whichever direction it points.

Five scene classes have **no annotated snapshot at all** and so cannot be
represented in any gate built from this corpus today, per
`config/annotated_snapshots_seibersdorf.yaml`'s `scenes_without_annotation`:

| Scene | Covered by (nearest annotated scene) | RGB similarity |
| --- | --- | --- |
| `stacking` | `misaligned_stack` | 0.85 |
| `pre_stack` | `misaligned_stack` | 0.86 |
| `stacked` | `misaligned_stack` | 0.83 |
| `static_occluded` | `pickup` | 0.76 (loosest match in the set) |
| `wet` | `partially_wet` | 0.95 (near-duplicate) |

A promotion gate is therefore not a config flip on the existing file -- it
needs new, independently-recorded, independently-annotated sessions (ideally
covering the five gap scenes above), explicitly walled off from anything used
for parameter tuning.

## 5. Deliberately not in the ROS path yet, and why

Per `docs/production-readiness-plan.md` (v2):

| Component | Status | Why |
| --- | --- | --- |
| `dino_verifier` | **Dropped from the runtime path.** | Three independent measurements found no benefit: scalar rerank (neutral), calibration refit (strictly worse: contribution span 0.073 -> ~0.00002), within-block argmax (2 gained / 2 broken, net zero, strict recall unchanged at 44/62). Checkpoint and code stay in-repo as a recorded negative result; nothing in the runtime depends on them. Consequence: `ScoreRgbCandidates` is removed from the ROS contract even though `docs/dino-rgb-prior-ros-integration.md` still specifies it -- that integration doc is stale on this point. |
| `dino_proposer` | **Kept as the only RGB module on the roadmap, but not wired in.** Needs C++ 2-D region ingestion, which **does not exist**. | It detects 62/62 blocks including 10/10 heavy-occlusion cases -- the only mechanism in the system that sees blocks geometry misses. `VisualScorer` in `cpp/include/blockpose/core/proposals.hpp` (confirmed by reading it: `using VisualScorer = std::function<VisualEvidence(const Pose &)>;`) is a **Pose -> score** hook, not a **region -> points -> fit** path. Region ingestion is plan v2's Track B2, explicitly "the missing structural piece" and "required for 1b to ship." |
| ONNX | **Deferred.** | `docs/dino-rgb-prior-ros-integration.md` already specifies the RGB adapter as a separate optional node so GPU/ONNX deployment is optional. A Python ROS 2 node running the PyTorch checkpoint meets v1 at a fraction of the engineering cost; ONNX is revisited only if that node misses its latency budget. |

The geometry core (section 1-2) has no RGB dependency and no GPU dependency,
by construction -- it is the only part of the system ready to ship into the
ROS node today.
