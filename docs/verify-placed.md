# VerifyPlaced v1

`VerifyPlaced` is the plan-conditioned path for blocks that have already been
placed. It preserves plan identity across touching blocks by assigning each
return exclusively to its nearest expected cuboid surface before fitting.

The v1 trust region is fixed at ±10 cm translation and ±10° rotation. The fit
uses robust cuboid surface residuals and reports covariance and weak directions
from its Jacobian. Planned-neighbor ray occlusion determines which faces are
expected to be visible.

Status semantics are intentionally conservative:

- `Confirmed`: supported geometry stays within 3 cm and 2° of plan.
- `Deviated`: supported geometry fits inside the verification trust region but
  exceeds the confirmation tolerance.
- `Missing`: actual returned or no-return rays pass through the planned volume.
- `NotObservable`: geometry is insufficient or the block is hidden. Raw XYZ
  silence always has this status and never proves that a block is missing.

The previous `blockpose.plan_prior.detect_with_plan` implementation remains the
default while v1 is a measured prototype. Constructing `HybridBlockDetector`
with `verify_placed_v1=True` explicitly selects v1. It will become the default
only after the real plan-conditioned gates pass without tuning.

Run the leakage-safe real-snapshot diagnostic with:

```bash
uv run python tools/eval_verify_placed_snapshots.py
```

The 15 real priors are development diagnostics only. Their ground truth is
loaded after inference, and the command asserts that no evaluated snapshot is
listed as a prior source. It must not be used to tune thresholds or claim
held-out acceptance.

## Experimental course constraints

The prototype also accepts `PlacedBlockPrior` and `CourseConstraint`. Legacy
`BlockPose` lists remain supported and receive deterministic independent IDs.
Course layout may fill only translation directions classified as null by the
data-only fit, and then reports `LayoutInferred` with conservative propagated
reference-plus-offset uncertainty. It never changes the independent presence
status or its measured deviation fields. An out-of-window layout candidate is
discarded while the original independent result is retained with an explicit
`layout_rejected` diagnostic.

Snapshot topology is loaded from `config/snapshot_plan_topology.yaml` before
annotations. Its poses come from leakage-safe LOSO fixtures; malformed windows,
unknown blocks, duplicate courses, and inconsistent reciprocal offsets are
explicit errors. The current snapshot topology deliberately declares no
courses because the available fixtures do not establish same-snapshot course
membership without consulting ground truth.

Face-boundary diagnostics come only from measured occupancy falloff with a
populated face interior. They are disabled for every multi-block or
course-owned fit until unsplit raw-surface extraction exists, so an ownership
partition at a seamless planned joint cannot become boundary evidence. Planned
joints are never synthesized as observations.
Course-aware mode remains experimental and the legacy verifier remains default
until the full touching-row yaw, pose, status, identity, and latency gates pass.

### Opt-in shared-course yaw

`VerifyPlacedParams(enable_course_yaw_refinement=True)` additionally enables a
conservative pooled raw-geometry yaw search for a course. It is disabled by
default and does not alter engine defaults or snapshot topology.

The solve accepts only a named wall plan with at least three level members,
common nominal yaw, connected exact plan offsets, and no snapshot-source
provenance. It uses one fixed raw course crop, exclusive face assignment on a
deterministic 0.5°/0.1° search over ±10°, and gives each block equal robust
point-plane influence. At least three independently supported yaw-sensitive
anchors, an interior optimum, positive curvature, and material absolute and
relative improvement are required. A timeout, non-finite score, missing anchor,
or confidence-gate failure preserves independent outputs.

Course members are canonicalized by stable block ID before scoring, so plan
serialization order cannot affect the solve. Inputs over
`course_yaw_max_input_points` (250,000 by default) are rejected in O(1) before
cropping or sorting; accepted inputs are then capped deterministically and
equally at 2,000 candidate points per member before the grid search.

Only independently `Confirmed` or `Deviated` members can be changed. Missing,
hidden, and unsupported members are never promoted or adjusted. Accepted yaw
updates keep centers fixed, preserve the frozen evidence-derived status and
deviation fields, retain the independent roll/pitch, report `LayoutInferred`,
and conservatively inflate yaw covariance. Diagnostics retain both the data
pose and layout-filled pose. Per-anchor and leave-one-out yaw agreement reject
a locally mis-yawed member rather than averaging the deviation away. The
`course_yaw_refinement` diagnostic records acceptance and all relevant
objective gates.
