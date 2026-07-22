# Frozen real-snapshot mode benchmark

Run the rerunnable Python diagnostic:

```bash
mkdir -p out
uv run python tools/eval_detector_modes.py --output out/frozen-mode-benchmark.json
```

The JSON is the notebook input: it has aggregate performance, per-target
prediction/ground-truth poses for overlays, observability, statuses, runtime,
and provenance.  It is deliberately a development diagnostic, not a
production gate.

`SCENE_DISCOVERY` first runs `detect_free` on the full LiDAR frame. Only after
the query finishes does it open the annotation manifest and use the complete
ROI to select predictions and targets for scoring. Ground truth and ROI never
crop or otherwise condition inference.

`REFINE_BLOCK` runs the current `verify_placed_v1` facade with only the
existing leave-one-snapshot-out plan fixtures.  Each fixture rejects itself as
a source snapshot.  Annotations are opened only after inference, to join a
result with its target.  There are no labelled absent planned blocks, so the
report does **not** claim Missing-status accuracy.  Its plan fixtures and all
snapshots remain development material.

`REFINE_GRASPED` is explicitly withheld for real data.  The snapshot taxonomy
has four reviewed gripped labels, but XYZ captures lack an independently
recorded FK/grasp prior, covariance, and gripper trajectory.  Deriving an input
prior from annotation pose would be an oracle evaluation.  The artifact
therefore records `NotRun` for those targets rather than fabricating a pose
metric.  Add this mode only when the recorded ROS fields can provide the
required independent inputs.

Two annotations are tagged `approximate`; their flag is retained per target
and must be considered when reading pose errors.  The artifact records the Git
revision, parameters, taxonomy audit, and snapshot exclusions so results are
comparable after changes.

## Current candidate run and retained baseline provenance

The current candidate was run with the command above on the frozen snapshots
at the repository revision recorded in its generated JSON. `REFINE_BLOCK` had
15 LOSO-prior targets (14 precise annotations): its all-target mean translation
and rotation errors were **20.86 mm / 2.094°**; precise-only values were
**18.85 mm / 1.743°**. Its precise-only p95 was **44.46 mm / 3.443°** and
end-to-end p95 query time was **107.12 ms**. These are development diagnostics,
not acceptance numbers.

The pre-change frozen v1 summary is versioned in
`config/verify_placed_v1_baseline_c9f069d.yaml`, sourced from the clean
`c9f069d` diagnostic. It records the same 15 target keys and reports **21.81
mm / 2.452°** all-target and **19.86 mm / 2.126°** precise-only mean
translation/rotation. The unified artifact asserts exact target-key equality
and emits candidate-minus-baseline deltas for both populations. The baseline is
explicitly marked as recorded legacy aggregate evidence and non-reproducible
with current code: there is no versioned legacy-toggle or target-level legacy
error artifact. The candidate is named
`verify_placed_v1_candidate_v2` despite using the existing v1 facade. No delta
is claimed as a topology/course-layout improvement; the current topology has
no eligible same-snapshot courses.
