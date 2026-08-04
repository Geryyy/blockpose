# Snapshot annotation tool

Launch from the repository root:

```bash
python tools/annotate_snapshot.py
python tools/annotate_snapshot.py data/registration_snapshots/<snapshot-name>
```

The window shows the interactive point cloud and RGB image side by side. The
right-hand panel is scrollable and follows the annotation workflow from top to
bottom: **1. Preprocess cloud**, **2. Auto-fit candidates**, **3. Add and adjust
blocks**, **4. Visibility and details**, and **5. Save snapshot**. Snapshot
navigation stays above those stages, and undo/redo remains available alongside
the block-adjustment controls. The first four stages scroll independently while
the save stage remains pinned at the bottom of the panel. Snapshot navigation
and advanced cloud filters start collapsed to keep the primary workflow compact.

Click a block in RGB to use an existing candidate mask as a 3D selection. To
use SAM 2.1 point-prompt segmentation instead, install the optional
dependencies and enable the model explicitly (the first run downloads its
weights):

```bash
uv sync --extra sam
uv run python tools/annotate_snapshot.py --sam-model --sam-device auto
```

Use `--sam-offline` to prohibit network access and load only already cached
model files. While one prompt is running, further RGB clicks are ignored with a
busy status; switching snapshots or closing the window safely discards its
result.

## Snapshot navigation

The **Snapshot** section has the snapshot combobox plus `< Prev` / `Next >`
buttons. Both route through the same dirty-check-then-load path: navigating
away from a snapshot with unsaved changes prompts to save, discard, or cancel,
exactly as changing the combobox always has. Prev/Next never wrap; they
disable at the first and last snapshot in `--root`.

Interface strings are deliberately ASCII: the font Open3D ships cannot render
an em dash or ellipsis and draws them as `?`.

## Preprocess cloud

**Remove background (range cutoff)** is **on by default**, seeded at the 95th
percentile of point-to-sensor distance, so the cloud you first see is already
trimmed of far returns. Turn it off or raise the limit to see the full capture.
A filter combination that would empty the cloud is refused, and the previous
cloud is kept, rather than leaving nothing to annotate.

**Remove PZS100 gripper points** (under **Advanced cloud filters**) masks the
robot's own jaws using the captured FK rails, and additionally excludes the
region *above* the jaws — the carrier, boom and arm — which the outward-facing
jaw boxes never cover and which the geometric proposer otherwise fits blocks
to. Two keys in `config/pzs100_gripper_filter.yaml` control it:

| Key | Meaning |
| --- | --- |
| `above_exclusion_height_m` | how far above the jaws to exclude (`0.0` disables) |
| `above_exclusion_margin_m` | lateral margin added to the rail-to-rail gap |

The payload hangs *below* the jaws, so the volume only ever extrudes upward
from the top of the jaw boxes and cannot reach the gripped block. "Up" is taken
from the first rail's structural cross-axis and checked against world +Z: if
the rig is tilted more than 60° from upright, inverted, or its rails are
supplied in the wrong order, the filter **refuses to build the volume** and
reports itself disabled rather than risk inverting the exclusion onto the
payload. Both the jaw boxes and the above-gripper volume are drawn in 3D as
read-only outlines so you can see what was masked.

This filter needs the gripper's FK frames (`K9`/`K10_left_rail`,
`K11`/`K12_right_rail`) in the snapshot's `tf.yaml`. Captures that record only
`K0_mounting_base` cannot drive it, and the checkbox stays disabled with the
missing frame named in its reason.

## Auto-fit candidates

The **Auto-fit candidates** stage compares up to four independent block
detectors for the current snapshot: **PCL geometry** (raw geometric
hypotheses), **+ RGB edge validator/reranker** (the classical edge-based
re-score of those PCL hypotheses), **+
DINO proposer**, and **+ DINO verifier (reranked)** (a learned reranking of
the geometry hypotheses). Check the sources you want to see, then press
**Compute auto-fit suggestions** — this never runs automatically on snapshot load or
Prev/Next, so navigating stays fast. Each checked source renders as its own
set of thin wireframe boxes in the 3D view (gray/orange/purple/teal
respectively). Every accepted fit is immediately created as a pending,
editable annotation in one undoable action. The new blocks appear in the
editable block list and can be moved, rotated, snapped, approved, excluded,
or deleted there. Their suggestion overlays disappear, preventing them from
being confused with editable blocks. Duplicate detector IDs are listed once,
preferring the refined source, and already represented IDs are skipped on
recompute. The status label reports
per-source warnings (e.g. a DINO source unavailable because no checkpoint was
given) instead of silently dropping them.

The DINO sources need model checkpoints:

```bash
python tools/annotate_snapshot.py --dino-proposer-checkpoint PATH --dino-verifier-checkpoint PATH
```

Both flags default to `None` (that source reports itself unavailable and is
simply skipped) and are independent of each other and of `--sam-model`.

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
cloud and highlights the selected points magenta. The tool first tries the
native fitter; if that cannot find a supported placement — common on a
partial or heavily occluded view — press **Create block from selection (N
points)** in **3. Add and adjust blocks**. It places a canonical block at the
selection's centroid, oriented by the selection's principal axes, as a normal
pending block you then correct and approve.

The button is enabled only while a selection exists, and its label shows the
live point count. Blocks placed by the fallback carry provenance
`rgb_mask_pca_fallback` (rather than `rgb_mask`) so an approximate placement
can be told from a real fit later. Creating a block is a single undo step.

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
