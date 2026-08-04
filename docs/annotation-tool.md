# Snapshot annotation tool

Launch from the repository root:

```bash
python tools/annotate_snapshot.py
python tools/annotate_snapshot.py data/registration_snapshots/<snapshot-name>
```

The window shows the interactive point cloud and RGB image side by side. The
right-hand panel is scrollable and follows the annotation workflow from top to
bottom: **1. Mask the gripper**, **2. Auto-fit candidates**, **3. Add and adjust
blocks**, **4. Visibility and details**, and **5. Save snapshot**. Snapshot
navigation stays above those stages, and undo/redo remains available alongside
the block-adjustment controls. The first four stages scroll independently while
the save stage remains pinned at the bottom of the panel.

The panel deliberately has no cloud-filter knobs beyond the gripper mask.
Ground removal, the workspace ROI and the range cutoff were all measured to
make no difference to what the fitter finds (identical hypothesis counts at
every setting), so they were removed rather than left as controls that look
consequential and are not.

RGB point-prompt segmentation with SAM 2.1 is **on by default** and is the
intended way to recover blocks the fitter misses. Install the optional
dependencies once:

```bash
uv sync --extra sam
uv run python tools/annotate_snapshot.py
```

Weights load in the background at startup, so the first click does not stall;
the readiness line under **Auto-fit candidates** reports the model and device,
or the reason it is unavailable. Pass `--no-sam` to turn it off and fall back
to stored candidate masks, `--sam-model` to choose a different checkpoint,
`--sam-device` to pin the device, or `--sam-offline` to forbid network access.
While one prompt runs, further RGB clicks are ignored with a busy status;
switching snapshots or closing the window safely discards its result.

Model weights are cached under `HF_HOME`, which the container image points at
`/opt/model-cache/huggingface`. `docker-compose.yml` mounts a named
`model-cache` volume there, so the download survives `docker compose run
--rm` instead of repeating on every launch.

## Snapshot navigation

The **Snapshot** section has the snapshot combobox plus `< Prev` / `Next >`
buttons. Both route through the same dirty-check-then-load path: navigating
away from a snapshot with unsaved changes prompts to save, discard, or cancel,
exactly as changing the combobox always has. Prev/Next never wrap; they
disable at the first and last snapshot in `--root`.

Interface strings are deliberately ASCII: the font Open3D ships cannot render
an em dash or ellipsis and draws them as `?`.

## Mask the gripper

**Remove PZS100 gripper points** masks the robot's own jaws using the captured
FK rails, and additionally excludes the
region *above* the jaws — the carrier, boom and arm — which the outward-facing
jaw boxes never cover and which the geometric proposer otherwise fits blocks
to. Two keys in `config/pzs100_gripper_filter.yaml` control it:

| Key | Meaning |
| --- | --- |
| `above_exclusion_height_m` | how far above the jaws to exclude (`0.0` disables) |
| `above_exclusion_margin_m` | lateral margin added to the rail-to-rail gap |

The payload hangs *below* the jaws, so the volume only ever extrudes upward
from the top of the jaw boxes and cannot reach the gripped block. "Up" is world
+Z, deliberately not a structural axis of the rig: which side the payload is on
is gravitational rather than structural, so world +Z separates boom from block
under any rig attitude and under either rail convention (rails running along
the jaw or across it). If the two jaws are not horizontally separated the
volume cannot be built; the jaw boxes still apply and the status line names the
reason, because an enhancement that cannot describe a rig must not disable the
core filter. Both the jaw boxes and the above-gripper volume are drawn in 3D as
read-only outlines so you can see what was masked.

This filter needs the gripper's FK frames (`K9`/`K10_left_rail`,
`K11`/`K12_right_rail`) in the snapshot's `tf.yaml`. Captures that record only
`K0_mounting_base` cannot drive it, and the checkbox stays disabled with the
missing frame named in its reason.

## Auto-fit candidates

Press **Compute auto-fit suggestions** to fit the current snapshot. It never
runs automatically on load or Prev/Next, so navigating stays fast. Every
accepted fit becomes a pending, editable block in one undoable action; rejected
near-misses stay on screen as thin wireframes so you can see where the fitter
looked and declined. Already-adopted detector IDs are skipped on recompute, so
recomputing is always safe.

There is no source picker. `pcl`, `rgb_edge_prior` and `dino_verifier_reranked`
all re-score the *same* raw geometric hypotheses, so they are alternatives, not
additions, and the tool simply uses the most refined one available. Measured
over the 23 snapshots in `data/registration_snapshots`, the RGB edge reranker
strictly dominates raw PCL:

| source | accepted | true positives | precision | recall |
| --- | --- | --- | --- | --- |
| `pcl` | 55 | 46 | 84% | 75% |
| `rgb_edge_prior` | 52 | 46 | **88%** | 75% |

Same recall, three fewer false positives, so there was nothing to choose.
`dino_proposer` is the exception: it runs its own detection over RGB region
proposals and can surface blocks geometry never proposed, so its candidates are
*added* to the base source rather than replacing it. The status label reports
per-source warnings (e.g. a DINO source unavailable for want of a checkpoint)
instead of silently dropping them.

Expect roughly three quarters of blocks to be found, each within about 3 cm of
truth -- so corrections on found blocks are a nudge. Budget your time for the
remaining quarter, which the RGB click path recovers.

The DINO sources need model checkpoints:

```bash
python tools/annotate_snapshot.py --dino-proposer-checkpoint PATH --dino-verifier-checkpoint PATH
```

Both flags default to `None` (that source reports itself unavailable and is
simply skipped) and are independent of each other and of the SAM options.

RGB masks are dilated slightly for calibration error, then filtered by camera
depth and 3D connectivity to reject background bleed. Selected LiDAR points are
highlighted magenta and the fitted box is created as **pending**. Correct it with
the existing pose controls and explicitly approve it, or use **Discard draft /
restore prior**. If
SAM is disabled or inference fails, clicking a checked-in candidate mask still
works. A failed selection or fit never changes the annotation document.

The viewer uses Open3D's normal left-drag orbit, middle/right-drag pan, and wheel
zoom controls. A click on a highlighted box selects it; the block list is the
fallback when boxes overlap. `Ctrl`+left-click creates a pending manual block at
the nearest cloud point within a 30 cm radius around the view ray. Exact XYZ
fields are available when a ray pick is unsuitable. For unusually sparse or
dense scenes, change the radius with `--pick-radius METERS`.

Each `Ctrl`+click draws the computed click ray in cyan. The nearest cloud point
is marked green when accepted or red when it lies outside the configured radius.
A double-click intentionally performs the same operation as a single click; the
second Open3D button event is ignored.

To snap a face, select the block, choose **Auto**, **End** or **Long face**,
then activate **Pick support plane in cloud** and click a nearby point on the
support surface. The tool estimates a local plane and translates only the
chosen face onto it while preserving orientation. A snap remains pending and
must still be approved. `Ctrl`+click only creates a manual draft; it never
implicitly snaps.

The face selector offers two classes rather than six signed axes because the
block's cross-section is a **0.6 x 0.6 square**, so it has 4-fold symmetry
about its long axis. Only two faces are physically distinct:

| Choice | Faces | Size |
| --- | --- | --- |
| **End** | the two square ends | 0.6 x 0.6 |
| **Long face** | all four long faces | 0.9 x 0.6 |

"Side" and "top" are the *same* class here, which is why they are not offered
separately. **Auto** (the default) considers all six and picks whichever face
currently points most toward the plane you clicked, so the sign is resolved for
you. Note that **End** legitimately fails against a horizontal support surface:
the square ends face sideways, so that snap has no meaning.

Pending priors are yellow, approved targets green, rejected priors red, and the
selection blue. The cloud point size defaults to 3.5. Edit position and XYZ Euler angles
numerically or with the 1 cm / 1 degree step buttons. `Z` and `Y`, or the
visible buttons, undo and redo.

Every annotation uses one block convention: dimensions `(0.9, 0.6, 0.6)` in
local XYZ, with local X always the physical 0.9 m long axis. Standing versus
laying is encoded solely by the pose rotation, never by permuting dimensions.
The read-only orientation line reports this long axis's world azimuth and
elevation and labels it **Standing** or **Laying**. The editable roll/pitch/yaw
fields are extrinsic-looking XYZ Euler coordinates of the complete canonical
block frame; equivalent box symmetries can therefore have quite different RPY,
so use the long-axis summary when checking standing versus laying.

The **Rotate 90° yaw** button in **3. Add and adjust blocks** applies exactly
+90° about world Z, composing the rotation matrix directly rather than via
Euler angles (an `xyz` round-trip is singular for a standing block, which is
exactly the case that matters). Like any pose edit it goes through the normal
undo-tracked path and leaves review status untouched, so still approve or
reject as usual.

## Creating a block from an RGB selection

Clicking a block in the RGB pane projects the resulting mask onto the LiDAR
cloud, highlights the selected points magenta, and **always creates a pending
block**. The tool tries the native fitter first; when that cannot find a
supported placement it falls back to a canonical block at the selection's
centroid, oriented by the selection's principal axes, for you to nudge.

The fallback is the normal path for a **gripped** block: the gripper occludes
its top face, so the fitter never finds the top/side pair a canonical
placement needs. Clicking a block is an unambiguous statement that there is
one there, so it produces one either way -- an approximate placement you
correct beats no placement at all.

Blocks placed by the fallback carry provenance `rgb_mask_pca_fallback` rather
than `rgb_mask`, so an approximate placement stays distinguishable from a real
fit. Either way it is a single undo step.

**Create block from selection (N points)** in **3. Add and adjust blocks** runs
the same thing on demand against whatever selection is currently highlighted --
useful after undoing, or to retry once you have adjusted the mask. It is
enabled only while a selection exists and its label shows the live count.

**Approve block** is the only action that makes a block evaluation ground truth.
**Exclude (reason required)** records a detected non-target for audit and shows it
in red; it is distinct from **Delete selected block…**, which removes the block
after confirmation. Delete sits directly below the block list and is disabled
until a block is selected. **Discard draft / restore prior** removes a new manual
draft, or restores an imported prior and returns it to pending. **Reset pose to
saved** changes only the selected block's pose back to its last saved/imported
pose; it does not change visibility or review status. An approved block may carry
an exclusion reason, which keeps it out of scored targets. Controls that require
a selected block are disabled when the selection is empty.

The annotation ROI is always the full captured cloud — there is no manual ROI
editor. Mark a snapshot complete only after resolving all pending priors; save
validates that rule, writes schema v2 atomically, and first copies the
previous manifest to a timestamped `annotations.yaml.backup-*` file.
Navigation and window close prompt before discarding unsaved changes.
Approving or editing marks the document unsaved; persist it with `Ctrl`+`S` or
the **Validate and save snapshot** button.
