# C++-as-truth / nanobind consolidation — context and tasks

This is a handoff brief for an agent picking up where a prior session left
off. Read this whole file before starting; it captures the target
architecture, exactly what's already done, what's still wrong, and the
constraints that apply to every stage below.

> **Status update — C++-only cutover:** The Python business-logic modules
> named below (`detection`, `refine`, `plan_prior`, `verify_placed`,
> `wall_plans`, `confidence`, and the Python RGB edge prior) have been
> removed. Free detection now uses the nanobind C++ bridge. Placed, plan, and
> gripped queries intentionally return an unsupported-mode result until their
> native APIs are implemented. References to those paths in this document
> describe the historical baseline and remaining-port scope, not live files.

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
  same logic is exactly the problem this consolidation is fixing.

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
  installed into `site-packages/blockpose/` is dead and never imported.)
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
- **`python/blockpose/core_bridge.py`** (new) wraps `blockpose._core.discover()`
  and reshapes its dict into `blockpose.detection`'s existing
  `DetectionResult`/`InferenceTrace`/`TraceHypothesis`/`blockpose.pose.BlockPose`
  dataclasses, so a caller can swap in `detect_blocks_detailed_cpp(points)`
  for `detect_blocks_detailed(points, params)` with minimal code change.
  `diagnostics` and `stage_timings_ms` are always empty/`{}` — the C++ core
  doesn't compute per-hypothesis `Observability` classification or per-stage
  timing yet (see Stage 1 below). Uses `importlib.import_module("blockpose._core")`
  rather than `import blockpose._core as _core` — the latter resolves via a
  cached attribute on the `blockpose` package object once any successful
  import has happened, silently defeating `sys.modules`-based test mocking.
  If you add more lazy-imports of `blockpose._core` anywhere, use the same
  `importlib.import_module` pattern.
- **`tests/python/test_core_bridge.py`, `tests/python/test_core_bridge_parity.py`**
  (new). The parity harness compares `blockpose.detection.detect_blocks_detailed`
  against `core_bridge.detect_blocks_detailed_cpp` on synthetic scenarios
  (`blockpose.scenario_matrix`) and real snapshots
  (`data/registration_snapshots`). Confirms the evidence/scoring math
  (`visibility_evidence`, `free_space_violations_from_misses`,
  `_SCORE_WEIGHTS`/`normalized_evidence_score`) is an exact, line-for-line
  port between the two languages (matched-pose score deltas <0.003 in
  every case tested) — this part of the algorithm is genuinely safe to trust
  as already-consolidated, even though the Python copy hasn't been deleted
  yet.
- **A real, unresolved bug was found and deliberately left unfixed** (see
  Stage 2 — this blocks trusting refine output for real decisions until
  resolved properly): for blocks whose end-caps are never observed (a
  partially-observed isolated block, or a wall-row block occluded by its
  neighbours), Python's SciPy SDF refine step
  (`python/blockpose/refine.py::refine_pose_with_diagnostics`) has a residual
  with near-zero sensitivity to sliding the pose along the box's long axis,
  and can drift ~0.13–0.15m — landing just under the `max_translation=0.15`
  guard, which was built to catch a wrong point assignment, not this. A
  Tikhonov-regularization fix was prototyped, confirmed to fix the degenerate
  cases, but **reverted**: even a tiny regularization weight flipped
  `tests/python/test_plan_prior.py::test_recovers_a_wall_row_blind_detection_cannot_separate`
  onto a different, worse local optimum (0.009m → 0.036m error) — a
  discontinuous basin-hopping effect, not a graceful tradeoff, that needs
  much broader validation than a prior session had budget for. The 4 known-
  affected synthetic scenarios (`isolated_near`, `isolated_tilt_7deg`,
  `isolated_noise_3cm`, `three_separated`) are marked `xfail(strict=True)` in
  `test_core_bridge_parity.py` with the full writeup in a code comment there
  — read it before touching `refine.py`.
- Full suite as of this commit: `pytest tests/python/ -q` → 622 passed, 3
  skipped, 5 xfailed (4 from the anomaly above, 1 pre-existing/unrelated).

## What's NOT yet true — the actual gap

Nothing has switched over to the C++ bridge yet. Every real caller in this
repo still runs the pure-Python implementation:

| Business logic | Lives in | Bound via nanobind? |
|---|---|---|
| Free-block geometric detection (detect→evidence→select→refine) | **Both**, independently: `python/blockpose/detection.py` + `refine.py` **and** `cpp/include/blockpose/core/*.hpp` | Only the default-params, no-custom-config path (`discover()`) |
| Placed-block verification | `python/blockpose/verify_placed.py` (1471 lines: course-aware joint fitting, face-boundary reasoning, `PlacedObservability`) | **No C++ equivalent at all** |
| "Verify every expected pose" plan-prior query | `python/blockpose/plan_prior.py` | **No** — C++'s `priors.hpp` has a narrower FK-seed-and-compete mechanism (`pipeline.hpp`'s `fk_seed_min_evidence_gain` path), not this always-verify query |
| Wall-plan loading | `python/blockpose/wall_plans.py` (`config/wall_plans.yaml`) | **No** |
| Confidence calibration | `python/blockpose/confidence.py` | **No** |
| Free/placed/gripped facade | `python/blockpose/engine.py` (`HybridBlockDetector`) | **No** |
| Classical RGB edge prior | **Both**, independently: `python/blockpose/rgb_edge_prior.py` **and** `cpp/src/rgb_edge_prior.cpp` (`blockpose::vision`, built behind `BLOCKPOSE_VISION_BUILD`) | **No** — `blockpose::vision` isn't in the nanobind bindings at all |
| `Observability`/`calibrated_confidence` classification | `python/blockpose/detection.py::_diagnostics`, `confidence.py` | **No** |
| Per-stage timing instrumentation | `python/blockpose/detection.py::_detect_blocks_impl` (11 named stages) | **No** — `pipeline.hpp` has zero timing code |
| `gripper_self_filter.hpp` (`OrientedBox` exclusion) | C++ only, tested, **zero call sites anywhere** | N/A, and out of scope — see below |

**Explicitly out of scope, correctly Python-only, no action needed:**
`rgb_verifier.py`, `rgb_region_proposer.py`, `_dino_backbone.py`,
`_head_checkpoint.py`, `experiments/rgb_candidate_prior/`,
`experiments/rgb_region_prior/` (DINO training, needs PyTorch);
`annotation_tool.py`, `annotation_candidates.py`, `annotation_snap.py`,
`annotation_batch.py`, `annotation_roi.py`, `annotations.py`,
`tools/annotate_snapshot.py` (annotation GUI); `synth.py`, `scenes.py`,
`scenario_matrix.py` (synthetic scene generation for tests). The gripper
self-filter is also out of scope for now: the gripper is articulated
(boom/arm joints downstream of the sensor mount) and real `tf.yaml` snapshots
never capture its pose, so there's no FK data to build a box filter from —
revisit only if per-snapshot joint-state logging is ever added.

**Eval/benchmark harnesses are a partial exception, worth a deliberate
decision, not silent inertia:** `tools/eval_real_snapshots.py`,
The former `blockpose/mode_benchmark.py` all-mode workflow is retired while
placed/gripped verification lacks a native API; `tools/eval_detector_modes.py`
now fails explicitly rather than importing deleted Python detector code. Other
evaluation tools must call the C++ discovery bridge.
infrastructure and can reasonably stay Python-authored — but once a piece of
business logic has a C++ port, these harnesses should be migrated to call
*that* (via the nanobind bridge) rather than staying frozen on the Python
duplicate forever, or the whole point of "C++ is truth" is undermined by the
thing that's supposed to be grading it running the old implementation.

## Constraints and principles (apply to every stage)

- **Ask before scope-expanding, destructive, or hard-to-reverse actions** —
  same standing git-safety rules as the rest of this repo (no force-push, no
  `--no-verify`, confirm before deleting/reverting real code).
- **Full suite must stay green after every stage**:
  `/opt/venv/bin/python3 -m pytest tests/python/ -q`. New, expected gaps get
  a documented `xfail(strict=True, reason=...)`, never a silently loosened
  assertion and never a deleted test.
- **Any change to `refine.py`'s numerics needs broad validation before
  shipping** — the reverted regularization attempt showed this is a
  nonconvex problem with more than one local optimum in some scenes; a fix
  must be checked against many more scenarios than the ~8 synthetic + 3 real
  snapshots the current parity harness covers, ideally reproducing
  `refine.py`'s own documented "12 seeds per noise level" benchmark
  methodology (mentioned in its module docstring, not currently a runnable
  script in this repo — writing that repeatable benchmark is worth doing as
  part of Stage 2).
- **Reuse existing structures instead of reinventing.** In particular:
  - `cpp/include/blockpose/core/priors.hpp`'s `PosePrior`/`PriorMatch`/
    `best_prior_match` substrate is explicitly designed to carry FK, RGB, or
    wall-plan priors through the same scoring path — the plan-prior port
    (Stage 3) should feed it wall-plan priors rather than building a parallel
    mechanism.
  - `cpp/include/blockpose/core/proposals.hpp`'s `VisualScorer`/
    `VisualEvidence` hook already exists for pluggable RGB evidence in the
    C++ scoring/selection loop — the RGB edge-prior port (Stage 4) should
    use this, not a bolt-on side channel.
  - `contracts.py`'s `DetectorRequestContract`/`DetectorResponseContract`
    already define an intended language-neutral wire schema for "the
    Python-to-C++ ROS 2 boundary" — worth reading before designing new
    nanobind dict shapes, even though it isn't implemented in C++ yet.
- **Report negative/uncertain results plainly.** If a port doesn't match
  Python behavior, or a design choice has a real tradeoff, say so in the
  handoff/PR description — don't spin it as done when it isn't.
- **When a design choice isn't obvious, ask rather than assume** — e.g.
  whether wall-plan YAML parsing should live in C++ or stay a thin Python
  adapter that constructs `PosePrior`s and passes them across the bridge;
  whether `confidence.py`'s calibration can legitimately stay as Python glue
  operating on nanobind-returned numbers, or should be ported too. Several
  such choices are flagged inline below — they're open questions, not
  pre-decided.

## Staged tasks

Do these roughly in order; each stage should be independently shippable
(buildable, tested, committed) rather than one giant change.

### Stage 1 — widen the nanobind surface for what already exists in C++

- Expose `DetectionParameters` from Python: either a nanobind-bound class
  mirroring the C++ struct's fields, or a simpler `**kwargs`/dict-based
  override applied before calling `detect()`. This is a hard prerequisite —
  no real caller can migrate until custom parameters (not just C++'s
  defaults) are reachable from Python, since every real caller today
  constructs its own `DetectionParams()` with non-default fields in at least
  some code paths.
- Bind `raw_lineage` (`RawHypothesisLineage`, already computed, just not
  bound — same pattern as the `refined_candidate_trace` work already done)
  if/when Stage 5's migrated callers need it (the annotator's `pcl` source
  and eval tooling may not).
- Decide and implement (or explicitly decide to drop) per-stage timing
  instrumentation in `pipeline.hpp` — Python's `_detect_blocks_impl` times 11
  named stages; C++ currently times nothing. Needed if any migrated caller
  depends on `stage_timings_ms` (check the native discovery benchmark/
  `eval_real_snapshots.py`
  before assuming yes or no).
- Decide and implement (or explicitly decide to drop/relocate)
  `Observability`/`calibrated_confidence` classification — currently pure
  Python (`detection.py::_diagnostics`, `confidence.py`). This might
  legitimately stay a thin Python post-processing step over nanobind-returned
  numbers (it's arguably calibration/analysis, not core detection logic) —
  flag this as an open question rather than assuming either answer.
- Bind `blockpose::vision`'s RGB edge-prior scorer (`cpp/src/rgb_edge_prior.cpp`)
  through nanobind — it's built (behind `BLOCKPOSE_VISION_BUILD`, default
  ON) but has zero Python exposure today.

### Stage 2 — resolve the refine.py under-constrained-axis anomaly properly

Blocking prerequisite before trusting migrated refine output for placed/
gripped decisions (Stage 3+ callers will otherwise inherit an unresolved
correctness question). Read the full writeup in
`tests/python/test_core_bridge_parity.py` (search
`_KNOWN_REFINE_ANOMALY_SCENARIOS`) first. Suggested approach:
1. Write a repeatable benchmark script reproducing `refine.py`'s own
   documented methodology ("12 seeds per noise level", isolated blocks, a
   range of noise levels) plus new cases covering partial visibility
   (missing end-caps) and wall-row occlusion — the two failure modes found
   so far.
2. Characterize the C++ Levenberg-Marquardt refiner's behavior on the same
   cases (it did *not* exhibit the ~0.14m drift in the one scenario tested —
   understand why, e.g. its damping schedule may naturally regularize
   near-null directions, before assuming it's simply "more correct").
3. Only then design a fix (regularization, a different guard, a rank-check
   on the Jacobian, or something else) and validate it against the full
   benchmark from step 1, not just the 4 scenarios already known to fail.

### Stage 3 — port plan-prior verification to C++

`plan_prior.py::detect_with_plan` always refines+scores every expected pose
against exclusively-assigned nearby points, independent of whether blind
geometric detection would ever propose it. Port this into a new C++ module
(e.g. `cpp/include/blockpose/core/plan_verify.hpp`), reusing `priors.hpp`'s
`PosePrior` substrate rather than inventing a parallel mechanism. Depends on
Stage 2 (uses refine internally) and Stage 1's params exposure.

### Stage 4 — port wall-plan priors and the RGB edge-prior scoring path

- Decide where `config/wall_plans.yaml` parsing lives (C++ YAML parsing via
  `yaml-cpp`, already a listed dependency per `docs/ENVIRONMENT.md`, vs. a
  thin Python adapter constructing `PosePrior`s and passing them across the
  bridge) — open question, ask before committing to one.
- Wire the C++ RGB edge-prior scorer into `proposals.hpp`'s existing
  `VisualScorer` hook in the detection/selection loop, rather than as a
  separate bolt-on scoring pass.

### Stage 5 — port `verify_placed.py`

The largest single piece (1471 lines): course-aware joint fitting
(`CourseConstraint`, shared-yaw-per-course refinement), per-face boundary
observation, expected-visibility computation, bounded joint least-squares
fitting, `PlacedObservability`/`PlacedStatus` classification. Treat this as
its own sub-plan (break it into smaller PRs) rather than one shot — it's
bigger than the entire existing C++ core combined.

### Stage 6 — migrate real callers, then retire Python duplicates

- Switch `annotation_candidates.py`'s `pcl` source, `tools/eval_real_snapshots.py`,
  native discovery benchmark and `engine.py`'s facade to call the
  nanobind bridge instead of the pure-Python implementations, one at a time,
  each with its own before/after parity check.
- Only after every real caller has migrated and parity is proven for that
  piece: delete the corresponding Python duplicate
  (`detection.py`'s algorithm internals, `refine.py`'s solver, `plan_prior.py`,
  `verify_placed.py`, `wall_plans.py`, `confidence.py`, as applicable) rather
  than keeping it "just in case."

## Verification, every stage

- `/opt/venv/bin/python3 -m pytest tests/python/ -q` — full suite green
  (documented xfails aside).
- `/opt/venv/bin/python3 -m ruff check <touched files>`.
- C++ side: `cmake -S cpp -B build/blockpose-core -DBLOCKPOSE_CORE_BUILD_TESTING=ON`
  + `cmake --build` + `ctest --test-dir build/blockpose-core --output-on-failure`
  (existing GoogleTest suite: `geometry`, `proposals`, `pipeline`, `refine`,
  `evidence`, `gripper_self_filter`, plus `rgb_edge_prior` if
  `BLOCKPOSE_VISION_ENABLED`).
- Rebuild + reinstall the nanobind extension (see the Stage-0 command block
  above) and re-run the parity harness
  (`pytest tests/python/test_core_bridge_parity.py -q -v`) after any C++
  change that touches bound behavior.
- A full GUI smoke test of the annotator is not possible in this sandbox (no
  GLX context) — rely on the non-GUI direct-script verification pattern used
  throughout this work (load a real snapshot, call the function under test
  directly, print/compare results) instead, and say so explicitly rather
  than silently skipping verification.
