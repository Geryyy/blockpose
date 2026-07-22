# Python detector closeout contract (schema v1)

`blockpose.contracts` is the language-neutral reference boundary for the future C++ ROS 2
package. The existing NumPy-based `HybridBlockDetector` API remains unchanged; this contract
does not alter detector behavior. Clouds, RGB images, and masks remain native/out-of-band
payloads and are referenced by stable IDs in JSON.

## Mode mapping

| ROS mode | Python reference call | Core precondition |
|---|---|---|
| `SCENE_DISCOVERY` | `detect_free` | zero priors; zero or more optional image proposals |
| `REFINE_BLOCK` | `verify_placed` | exactly one prior whose ID matches `target_block_id` |
| `REFINE_GRASPED` | `estimate_gripped` | exactly one FK/grasp prior |

The ROS orchestrator may resolve an empty `REFINE_GRASPED.target_block_id` to the newest
`TASK_MOVE` block, but it must pass the resolved FK/grasp prior to the detector core. Unknown
modes are rejected; there is no fallback mode.

The current `SegmentImage` service can be wrapped as an `ImageProposalContract` producer for
`SCENE_DISCOVERY`. It is **not** a drop-in replacement for this three-mode contract: it has no
plan/FK prior, verification status, observability, metric pose covariance, or conservative
`Missing` semantics. RGB proposals are optional supporting evidence. Missing RGB or a missing
mask never establishes that a physical block is `missing`.

## Stable wire conventions

- Every request/response uses integer `schema_version=1`, a stable `query_id`, and an explicit
  mode. Timestamps are integer nanoseconds; poses name their frame.
- Frame metadata follows the existing runtime's `unavailable_evidence` polarity and accepts only
  the schema's fixed evidence-channel enum. An absent entry therefore means the channel is
  available; an entry means it was unavailable, not merely empty. The enum includes the derived
  ray channels `per_point_sensor_origins`, `no_return_ray_origins`, and
  `no_return_relative_time`, so deskew/burst evidence is not silently dropped.
- Pose quaternions are normalized XYZW and sign-canonicalized. The only serialized local-axis
  dimensions are `[0.9,0.6,0.6]`; standing/laid state is expressed by pose orientation, never
  by swapping dimensions. Dimensions and translations are metres;
  angular deviations are degrees where the field name says `_deg`.
- Covariance is symmetric positive-semidefinite row-major 6x6 with tangent order
  `[x,y,z,rx,ry,rz]`, translations in metres, rotations in radians, and the named frame. It uses
  a **left perturbation expressed in that frame**: `T_sample = Exp(delta) * T_mean`. A covariance
  moved to another frame is transformed with that frame transform's SE(3) adjoint. Semidefinite
  zero modes are allowed; negative modes beyond the documented numeric tolerance are rejected.
  `unavailable` means absent, never an all-zero
  covariance. The kind distinguishes uncalibrated fit curvature, calibrated covariance, prior,
  and layout propagation.
- Confidence is an object, not a bare score. A calibrated validity requires a calibration ID;
  uncalibrated ranking scores cannot masquerade as probabilities.
- Status and observability are independent. `prior_only` evidence is restricted to
  `not_observable`/`unobservable`; `missing` requires explicit free-space ray evidence.
- Error-code strings are append-only. Human-readable error messages are not stable API.
- Extension keys must be namespaced. Internal detector diagnostics are not automatically part
  of the stable contract.
- Response IDs are unique. Discovery returns only zero or more `detected` blocks. A successful
  refinement returns exactly its named target with a verification status; a failed response has
  no blocks. This makes zero discoveries distinct from query failure.

`canonical_json()` sorts object keys, uses compact separators, preserves declared array order,
and rejects NaN/infinity. Golden files may therefore compare bytes within the Python reference;
C++ implementations should parse numbers and compare their values rather than depend on a
particular language's float spelling.
