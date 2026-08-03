# Snapshot annotation tool

Launch from the repository root:

```bash
python tools/annotate_snapshot.py
python tools/annotate_snapshot.py data/registration_snapshots/<snapshot-name>
```

The window shows the interactive point cloud and RGB image side by side. The
right-hand panel is grouped into collapsible sections: **Snapshot**,
**Candidate sources**, **Blocks**, **Pose**, **Review**, **Block actions**,
and **History**, followed by the completion checkbox, notes, and Validate &
Save.

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

The **Snapshot** section has the snapshot combobox plus `◀ Prev` / `Next ▶`
buttons. Both route through the same dirty-check-then-load path: navigating
away from a snapshot with unsaved changes prompts to save, discard, or cancel,
exactly as changing the combobox always has. Prev/Next never wrap; they
disable at the first and last snapshot in `--root`.

## Candidate sources

The **Candidate sources** section compares up to four independent block
detectors for the current snapshot: **PCL geometry** (raw geometric
hypotheses), **+ RGB edge prior** (the classical edge-based re-score), **+
DINO proposer**, and **+ DINO verifier (reranked)** (a learned reranking of
the geometry hypotheses). Check the sources you want to see, then press
**Compute candidates** — this never runs automatically on snapshot load or
Prev/Next, so navigating stays fast. Each checked source renders as its own
set of thin wireframe boxes in the 3D view (gray/orange/purple/teal
respectively) and is purely a read-only comparison overlay: clicking one
never creates or selects an editable target. The status label reports
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
the existing pose controls and explicitly approve it, or use **Abort edit**. If
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

After creating a draft, `Ctrl`+click attempts a conservative top-and-side plane
snap. The explicit `Snap selected to cloud` button retries it for the current
box. A snap remains pending and must still be approved. Undo first restores the
pre-snap draft pose; a second undo removes the draft created by the click.

Pending priors are yellow, approved targets green, rejected priors red, and the
selection blue. The cloud point size defaults to 3.5; the points assigned to
the selected block (within a 10 cm surface band, via the same `assign_points`
used elsewhere in the pipeline) are additionally highlighted cyan so the fit
is easy to judge against the raw cloud. Edit position and XYZ Euler angles
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

The block is symmetric under a 90° roll about its long axis and under an
end-for-end flip, so a snapped or manually posed box can sometimes have its
faces labeled the "other" plausible way. The **Rotate 90°** button in the
**Pose** section cycles the selection through those alternative labelings
(reusing `pose.py`'s existing box-symmetry group, not a new hand-rolled
rotation) without moving the box in the world. Like any pose edit it goes
through the normal undo-tracked path — `Undo` steps back one alternative at a
time — and leaves review status untouched, so still approve or reject as usual.

`Approve` is the only action that makes a block evaluation ground truth. `Reject`
requires a reason. `Abort edit` restores the selected prior and returns it to
pending; `Reset` restores its last recorded state without changing review status.
An approved block may carry an ignore reason, which keeps it out of scored targets.
The **Review** section keeps Approve/Reject/Abort/Reset directly above the
visibility and ignore-reason controls; block-editing actions (add manual draft,
delete, snap to cloud) live in their own **Block actions** section below it.

The annotation ROI is always the full captured cloud — there is no manual ROI
editor. Mark a snapshot complete only after resolving all pending priors; save
validates that rule, writes schema v2 atomically, and first copies the
previous manifest to a timestamped `annotations.yaml.backup-*` file.
Navigation and window close prompt before discarding unsaved changes.
Approving or editing marks the document unsaved; persist it with `Ctrl`+`S` or
the **Validate & Save** button.
