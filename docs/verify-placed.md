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
