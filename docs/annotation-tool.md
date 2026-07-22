# Snapshot annotation tool

Launch from the repository root:

```bash
python tools/annotate_snapshot.py
python tools/annotate_snapshot.py data/registration_snapshots/<snapshot-name>
```

The window shows the interactive point cloud and RGB image side by side. Click a
block in RGB to use an existing candidate mask as a 3D selection. To use SAM 2.1
point-prompt segmentation instead, install the optional dependencies and enable
the model explicitly (the first run downloads its weights):

```bash
uv sync --extra sam
uv run python tools/annotate_snapshot.py --sam-model --sam-device auto
```

Use `--sam-offline` to prohibit network access and load only already cached
model files. While one prompt is running, further RGB clicks are ignored with a
busy status; switching snapshots or closing the window safely discards its
result.

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
selection blue. Edit position and XYZ Euler angles numerically or with the 1 cm /
1 degree step buttons. `Z` and `Y`, or the visible buttons, undo and redo.

Every annotation uses one block convention: dimensions `(0.9, 0.6, 0.6)` in
local XYZ, with local X always the physical 0.9 m long axis. Standing versus
laying is encoded solely by the pose rotation, never by permuting dimensions.
The read-only orientation line reports this long axis's world azimuth and
elevation and labels it **Standing** or **Laying**. The editable roll/pitch/yaw
fields are extrinsic-looking XYZ Euler coordinates of the complete canonical
block frame; equivalent box symmetries can therefore have quite different RPY,
so use the long-axis summary when checking standing versus laying.

`Approve` is the only action that makes a block evaluation ground truth. `Reject`
requires a reason. `Abort edit` restores the selected prior and returns it to
pending; `Reset` restores its last recorded state without changing review status.
An approved block may carry an ignore reason, which keeps it out of scored targets.

Mark a snapshot complete only after setting the world-space ROI and resolving all
pending priors. Save validates these rules, writes schema v2 atomically, and first
copies the previous manifest to a timestamped `annotations.yaml.backup-*` file.
Navigation and window close prompt before discarding unsaved changes.
Approving or editing marks the document unsaved; persist it with `Ctrl`+`S` or
the **Validate & Save** button.
