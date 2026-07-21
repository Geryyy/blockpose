# Snapshot annotation tool

Launch from the repository root:

```bash
python tools/annotate_snapshot.py
python tools/annotate_snapshot.py data/registration_snapshots/<snapshot-name>
```

The viewer uses Open3D's normal left-drag orbit, middle/right-drag pan, and wheel
zoom controls. A click on a highlighted box selects it; the block list is the
fallback when boxes overlap. `Ctrl`+left-click creates a pending manual block at
the cloud point nearest the view ray. Exact XYZ fields are available when a ray
pick is unsuitable.

Pending priors are yellow, approved targets green, rejected priors red, and the
selection blue. Edit position and XYZ Euler angles numerically or with the 1 cm /
1 degree step buttons. `Z` and `Y`, or the visible buttons, undo and redo.

`Approve` is the only action that makes a block evaluation ground truth. `Reject`
requires a reason. `Abort edit` restores the selected prior and returns it to
pending; `Reset` restores its last recorded state without changing review status.
An approved block may carry an ignore reason, which keeps it out of scored targets.

Mark a snapshot complete only after setting the world-space ROI and resolving all
pending priors. Save validates these rules, writes schema v2 atomically, and first
copies the previous manifest to a timestamped `annotations.yaml.backup-*` file.
Navigation and window close prompt before discarding unsaved changes.
