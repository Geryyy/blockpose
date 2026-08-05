# C++-as-truth / nanobind consolidation — context and tasks

This is a handoff brief for an agent picking up where a prior session left
off. Read this whole file before starting; it captures the target
architecture, exactly what's already done, what's still wrong, and the
constraints that apply to every stage below.

> **Status update — C++-only cutover (commit `d949647`, "refactor: make C++
> core the detector truth"):** The Python business-logic modules named below
> (`detection.py`, `refine.py`, `plan_prior.py`, `verify_placed.py`,
> `wall_plans.py`, `confidence.py`, `ransac.py`, `mode_benchmark.py`, and the
> Python RGB edge prior `rgb_edge_prior.py`) — and their dedicated test files
> (`test_detection.py`, `test_refine.py`, `test_plan_prior.py`,
> `test_verify_placed.py`, `test_wall_plans.py`, `test_confidence.py`,
> `test_ransac.py`, `test_mode_benchmark.py`, `test_rgb_edge_prior.py`, and
> the parity harness `test_core_bridge_parity.py`) — were **deleted outright**
> in that commit, not merely superseded. They no longer exist anywhere in the
> working tree. Free detection now uses only the nanobind C++ bridge. Placed,
> plan, and gripped queries intentionally return an unsupported-mode result
> until their native APIs are implemented. Everything below describing those
> modules in the present tense is describing the **historical baseline
> recorded at `d949647~1`** — recover the source with e.g.
> `git show d949647~1:python/blockpose/verify_placed.py` before resuming any
> port; do not assume the described behavior still runs.
>
> This also means Stage 2 below (the refine.py under-constrained-axis
> anomaly) has no live reproduction anymore — its evidence
> (`test_core_bridge_parity.py`'s `_KNOWN_REFINE_ANOMALY_SCENARIOS`) was
> deleted along with `refine.py` itself. Resuming that investigation means
> reconstructing the parity harness from git history first, not picking up
> where it left off.

## Goal

- **All detector *business logic* lives in C++** (`cpp/core`, and new C++
  modules as needed). New algorithm changes land there first.
- **Python's role is (a) a thin nanobind interface onto that C++ code**, and
  **(b) prototyping/tooling that has no business being in C++**:
  annotation tooling, DINO/RGB-prior training (needs PyTorch), and synthetic
  scene generation / eval / benchmark harnesses used to test the system.
- Once a piece of business logic has a validated C++ port + a passing parity
  test + every real caller migrated to the nanobind bridge, **the Python
  duplicate gets deleted** — it does not linger as a "reference
  implementation" forever. Two long-lived parallel implementations of the
  same logic is exactly the problem this consolidation is fixing. (As of
  `d949647`, several duplicates were deleted before a native replacement
  existed at all — see the status update above. That trade-off — no native
  placed/plan/gripped verification at all, rather than a stale Python one —
  was a deliberate call, not this document's original intent.)

This is not a new direction invented for this doc — `README.md`'s "Portable
C++ core" section already says the nanobind bridge "is intentionally
separate from the pure Python annotation/training package while packaging is
being consolidated." This file is the concrete plan for finishing that.

## What's already true (do not redo)

A prior session did real, committed work here. Don't repeat it, but do build
on it:

- **Build environment.** `cpp/CMakeLists.txt` had a real bug —
  `find_package(nanobind CONFIG REQUIRED)` was never preceded by
  `find_package(Python COMPONENTS Interpreter Development REQUIRED)`, so
  `BLOCKPOSE_CORE_BUILD_PYTHON=ON` had apparently never been built
  successfully before. Fixed. `nanobind` is now a `dev` extra in
  `pyproject.toml`/`uv.lock`. To rebuild the extension after a `cpp/` change:
  ```bash
  cmake -S cpp -B build/blockpose-python -DBLOCKPOSE_CORE_BUILD_PYTHON=ON \
    -DBLOCKPOSE_CORE_BUILD_TESTING=OFF -DBLOCKPOSE_VISION_BUILD=OFF \
    -Dnanobind_DIR="$(/opt/venv/bin/python3 -m nanobind --cmake_dir)" \
    -DPython_EXECUTABLE=/opt/venv/bin/python3
  cmake --build build/blockpose-python --parallel
  cp build/blockpose-python/_core.cpython-312-x86_64-linux-gnu.so python/blockpose/
  ```
  (The `.so` must be copied into `python/blockpose/` directly, not installed
  via `cmake --install` into `site-packages/blockpose/` — this repo's
  `blockpose` package is an editable install that resolves via
  `_editable_impl_blockpose.pth` straight to `python/blockpose/`, so anything
  installed into `site-packages/blockpose/` is dead and never imported.) The
  compiled extension also links against the **system** OpenCV runtime
  (`libopencv-dev`, not pip's `opencv-python`) — if `import blockpose._core`
  fails with a missing `libopencv_*.so`, see `docker/Dockerfile.core` and
  `models/production/README.md`.
- **`cpp/bindings/blockpose_core_module.cpp`** (`blockpose._core`, nanobind
  module) is widened beyond the original `poses`/`count`/basic-`counts`
  stub: it now also returns `refined_candidate_trace` (a list of dicts —
  `id`, `source`, `lineage_id`, `pose`, `score`, `support_points`,
  `selection_score`, `stage`, `fate` — mirrored from C++'s
  `RefinedCandidateTrace`, `pipeline.hpp`) and the full `PipelineCounts`
  (`plane_regions`, `plane_fit_calls`, `plane_search_points`,
  `plane_full_points_scored`, `plane_trials_evaluated`, `plane_valid_trials`,
  `refinement_candidates`, on top of the original 6 count fields). Still not
  bound: `raw_lineage`, `prior_match`, `visual_evidence`, custom
  `DetectionParameters` (discover() only runs with C++ defaults today —
  no way to pass parameters from Python yet).
- **`python/blockpose/core_bridge.py`** wraps `blockpose._core.discover()`
  and reshapes its dict into `TraceHypothesis`/`blockpose.pose.BlockPose`
  dataclasses (`core_types.py`) so a caller gets a typed result rather than a
  raw dict. `diagnostics` and `stage_timings_ms` are always empty/`{}` — the
  C++ core doesn't compute per-hypothesis `Observability` classification or
  per-stage timing yet (see Stage 1 below). Uses
  `importlib.import_module("blockpose._core")` rather than
  `import blockpose._core as _core` — the latter resolves via a cached
  attribute on the `blockpose` package object once any successful import has
  happened, silently defeating `sys.modules`-based test mocking. If you add
  more lazy-imports of `blockpose._core` anywhere, use the same
  `importlib.import_module` pattern.
- **`tests/python/test_core_bridge.py`** covers the current bound surface
  (`refined_candidate_trace`, `PipelineCounts`, the dataclass reshaping).
  Full suite as of this doc's last update:
  `python -m pytest tests/python/ -q` → 526 passed, 1 skipped (run this
  yourself before trusting the number — it will drift).

## What's NOT yet true — the actual gap

Nothing has switched over to the C++ bridge for anything beyond free-block
discovery. Placed/plan/gripped verification and the classical RGB edge prior
have **no implementation at all** right now — not "Python only," genuinely
absent (see the status update at the top):

| Business logic | Historical Python implementation | Bound via nanobind? |
| --- | --- | --- |
| Free-block geometric detection (detect→evidence→select→refine) | **Removed** (`detection.py` + `refine.py`, deleted in `d949647`); C++ equivalent: `cpp/include/blockpose/core/*.hpp` | Only the default-params, no-custom-config path (`discover()`) |
| Placed-block verification | **Removed** (`verify_placed.py`, 1471 lines: course-aware joint fitting, face-boundary reasoning, `PlacedObservability`) | **No C++ equivalent at all** |
| "Verify every expected pose" plan-prior query | **Removed** (`plan_prior.py`) | **No** — C++'s `priors.hpp` has a narrower FK-seed-and-compete mechanism (`pipeline.hpp`'s `fk_seed_min_evidence_gain` path), not this always-verify query |
| Wall-plan loading | **Removed** (`wall_plans.py`, `config/wall_plans.yaml`) | **No** |
| Confidence calibration | **Removed** (`confidence.py`) | **No** |
| Free/placed/gripped facade | `python/blockpose/engine.py` (`HybridBlockDetector`) — still present | **No** |
| Classical RGB edge prior | **Removed** (`rgb_edge_prior.py`); C++ equivalent exists: `cpp/src/rgb_edge_prior.cpp` (`blockpose::vision`, built behind `BLOCKPOSE_VISION_BUILD`) | **No** — `blockpose::vision` isn't in the nanobind bindings at all |
| `Observability`/`calibrated_confidence` classification | **Removed** (was `detection.py::_diagnostics`, `confidence.py`) | **No** |
| Per-stage timing instrumentation | **Removed** (was `detection.py::_detect_blocks_impl`, 11 named stages) | **No** — `pipeline.hpp` has zero timing code |
| PZS100 gripper self-filter | `gripper_self_filter.hpp`: rail endpoint poses → two outward-facing boxes → exclusion; used by the ROS adapter and snapshot runner | `filter_pzs100_gripper()` accepts world-frame FK poses; frame lookup remains adapter-specific |

**Explicitly out of scope, correctly Python-only, no action needed:**
`rgb_verifier.py`, `rgb_region_proposer.py`, `_dino_backbone.py`,
`_head_checkpoint.py`, `experiments/rgb_candidate_prior/`,
`experiments/rgb_region_prior/` (DINO training, needs PyTorch — see
`docs/dino-rgb-prior-ros-integration.md` for how these plug into the C++
detector without becoming C++ themselves);
`annotation_tool.py`, `annotation_candidates.py`, `annotation_snap.py`,
`annotation_batch.py`, `annotation_roi.py`, `annotations.py`,
`tools/annotate_snapshot.py` (annotation GUI); `synth.py`, `scenes.py`,
`scenario_matrix.py` (synthetic scene generation for tests). The PZS100
self-filter is intentionally not a TF reader: its portable API consumes
world-frame parent/rail poses, while ROS and snapshot adapters resolve the
configured frame names. This keeps the articulated robot model out of the
detector core and makes the same filter available to standalone evaluation.

**Eval/benchmark harnesses:** `tools/eval_detector_modes.py` now fails
explicitly rather than importing deleted Python detector code; other
evaluation tools must call the C++ discovery bridge. These harnesses can
reasonably stay Python-authored — but once a piece of business logic has a
C++ port, they should call *that* (via the nanobind bridge) rather than
staying frozen on a Python duplicate, or the whole point of "C++ is truth" is
undermined by the thing that's supposed to be grading it running old code.

## Constraints and principles (apply to every stage)

- **Ask before scope-expanding, destructive, or hard-to-reverse actions** —
  same standing git-safety rules as the rest of this repo (no force-push, no
  `--no-verify`, confirm before deleting/reverting real code).
- **Full suite must stay green after every stage**:
  `python -m pytest tests/python/ -q`. New, expected gaps get a documented
  `xfail(strict=True, reason=...)`, never a silently loosened assertion and
  never a deleted test.
- **Reuse existing structures instead of reinventing.** In particular:
  - `cpp/include/blockpose/core/priors.hpp`'s `PosePrior`/`PriorMatch`/
    `best_prior_match` substrate is explicitly designed to carry FK, RGB, or
    wall-plan priors through the same scoring path — a plan-prior port
    (Stage 3) should feed it wall-plan priors rather than building a parallel
    mechanism.
  - `cpp/include/blockpose/core/proposals.hpp`'s `VisualScorer`/
    `VisualEvidence` hook already exists for pluggable RGB evidence in the
    C++ scoring/selection loop — an RGB edge-prior port (Stage 4) should use
    this, not a bolt-on side channel. `dino_proposer`/`dino_verifier`
    (`docs/dino-rgb-prior-ros-integration.md`) are a separate, already-decided
    evidence path and do not need this hook.
  - `contracts.py`'s `DetectorRequestContract`/`DetectorResponseContract`
    already define an intended language-neutral wire schema for "the
    Python-to-C++ ROS 2 boundary" — worth reading before designing new
    nanobind dict shapes, even though it isn't implemented in C++ yet.
- **Report negative/uncertain results plainly.** If a port doesn't match
  historical Python behavior, or a design choice has a real tradeoff, say so
  in the handoff/PR description — don't spin it as done when it isn't.
- **When a design choice isn't obvious, ask rather than assume** — e.g.
  whether wall-plan YAML parsing should live in C++ or stay a thin Python
  adapter that constructs `PosePrior`s and passes them across the bridge;
  whether confidence calibration can legitimately stay as Python glue
  operating on nanobind-returned numbers, or should be ported too.

## Staged tasks, if this work is resumed

Nothing below is in progress. Stage 1 is the only piece with no
prerequisite; it's also a direct prerequisite for the ROS integration of
`dino_proposer`/`dino_verifier` (`docs/dino-rgb-prior-ros-integration.md`),
since that work also needs custom `DetectionParameters` reachable from
Python. Stages 2 and onward each start from a **deleted** Python
implementation — recover the relevant file from `d949647~1` before resuming,
and treat it as historical reference for the intended behavior, not working
code.

### Stage 1 — widen the nanobind surface for what already exists in C++

- Expose `DetectionParameters` from Python: either a nanobind-bound class
  mirroring the C++ struct's fields, or a simpler `**kwargs`/dict-based
  override applied before calling `detect()`. This is a hard prerequisite for
  any real caller that needs non-default parameters — every real caller
  today constructs its own params with non-default fields in at least some
  code paths.
- Bind `raw_lineage` (`RawHypothesisLineage`, already computed, just not
  bound — same pattern as the `refined_candidate_trace` work already done)
  if a caller needs it.
- Decide and implement (or explicitly decide to drop) per-stage timing
  instrumentation in `pipeline.hpp`.
- Decide and implement (or explicitly decide to drop/relocate)
  `Observability`/`calibrated_confidence` classification — this might
  legitimately stay a thin Python post-processing step over nanobind-returned
  numbers.
- Bind `blockpose::vision`'s RGB edge-prior scorer (`cpp/src/rgb_edge_prior.cpp`)
  through nanobind — it's built (behind `BLOCKPOSE_VISION_BUILD`, default
  ON) but has zero Python exposure today.

### Stage 2 — placed/plan/gripped verification: rebuild before resuming

There is no live implementation, historical or native, to build on here
today. Before doing anything else in this area: recover `refine.py` and
`test_core_bridge_parity.py` from `d949647~1`, re-establish the parity
harness, and re-characterize whether the documented under-constrained-axis
anomaly (blocks whose end-caps are never observed drifting ~0.13–0.15m in
the old SciPy refiner) still matters for the C++ Levenberg-Marquardt
refiner, which did not reproduce it in the one case tested historically.

### Stage 3 — port plan-prior verification to C++

Historical behavior (`plan_prior.py::detect_with_plan`, now deleted): always
refine+score every expected pose against exclusively-assigned nearby points,
independent of whether blind geometric detection would ever propose it. If
rebuilt, port into a new C++ module (e.g.
`cpp/include/blockpose/core/plan_verify.hpp`), reusing `priors.hpp`'s
`PosePrior` substrate rather than inventing a parallel mechanism. Depends on
Stage 1's params exposure and a resolved Stage 2.

### Stage 4 — port wall-plan priors and the RGB edge-prior scoring path

- Decide where `config/wall_plans.yaml`-equivalent parsing lives (C++ YAML
  parsing via `yaml-cpp`, already a listed dependency per
  `docs/ENVIRONMENT.md`, vs. a thin Python adapter constructing `PosePrior`s
  and passing them across the bridge) — open question.
- Wire the C++ RGB edge-prior scorer into `proposals.hpp`'s existing
  `VisualScorer` hook in the detection/selection loop, rather than as a
  separate bolt-on scoring pass.

### Stage 5 — port placed-block verification (`verify_placed.py`, historical)

The largest single piece (1471 lines, recoverable from `d949647~1`):
course-aware joint fitting (`CourseConstraint`, shared-yaw-per-course
refinement), per-face boundary observation, expected-visibility computation,
bounded joint least-squares fitting, `PlacedObservability`/`PlacedStatus`
classification. Treat this as its own sub-plan (break it into smaller PRs)
rather than one shot — it's bigger than the entire existing C++ core
combined.

### Stage 6 — migrate remaining real callers

Switch `annotation_candidates.py`'s `pcl` source and `engine.py`'s facade to
call the nanobind bridge instead of any remaining pure-Python path, one at a
time, each with its own before/after parity check.

## Verification, every stage

- `python -m pytest tests/python/ -q` — full suite green.
- `python -m ruff check <touched files>`.
- C++ side: `cmake -S cpp -B build/blockpose-core -DBLOCKPOSE_CORE_BUILD_TESTING=ON`
  + `cmake --build` + `ctest --test-dir build/blockpose-core --output-on-failure`
  (existing GoogleTest suite: `geometry`, `proposals`, `pipeline`, `refine`,
  `evidence`, `gripper_self_filter`, plus `rgb_edge_prior` if
  `BLOCKPOSE_VISION_ENABLED`).
- Rebuild + reinstall the nanobind extension (see the build-environment
  command block above) after any C++ change that touches bound behavior.
- A full GUI smoke test of the annotator is not possible in a headless
  sandbox (no GLX context) — rely on the non-GUI direct-script verification
  pattern instead (load a real snapshot, call the function under test
  directly, print/compare results), and say so explicitly rather than
  silently skipping verification.
