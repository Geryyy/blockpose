#!/usr/bin/env python3
"""Interactive Open3D snapshot annotator.

Run ``python tools/annotate_snapshot.py`` or pass a snapshot directory.  Normal
Open3D mouse controls orbit/pan/zoom.  Left-click a highlighted box to select it;
the list is a reliable fallback. Ctrl-left-click adds a manual draft at the cloud
point nearest the view ray. Numeric fields provide an exact coordinate fallback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from open3d.visualization import gui, rendering
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_preprocess import (  # noqa: E402
    filter_pzs100_gripper,
    load_pzs100_self_filter_config,
    pzs100_self_filter_available,
)
from blockpose.annotation_snap import (  # noqa: E402
    SnapParams,
    snap_block_face_to_plane,
    snap_block_to_cloud,
)
from blockpose.annotation_tool import (  # noqa: E402
    AUTO_FIT_PROVENANCE_PREFIX,
    VISIBILITIES,
    AnnotationController,
    AnnotationDocument,
    adjacent_snapshot,
    camera_ray,
    full_workspace_roi,
    initial_camera_focus,
    nearest_point_to_ray_with_distance,
    orientation_summary,
    pca_oriented_block_pose,
    pick_oriented_box,
    save_annotations_atomic,
    snapshot_selection_index,
)
from blockpose.io import (  # noqa: E402
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import BLOCK_DIMS  # noqa: E402
from blockpose.projection import (  # noqa: E402
    colorize_world_points,
    project_points,
    world_to_camera,
)
from blockpose.rgb_annotation import (  # noqa: E402
    AsyncSegmentationWorker,
    MaskSelectionParams,
    TransformersSam2Backend,
    build_projection_cache,
    candidate_mask_at,
    load_candidate_masks,
    map_widget_to_image,
    select_points_from_mask,
)
from blockpose.snapshot_exclusions import load_snapshot_exclusions  # noqa: E402

try:
    from blockpose.annotation_candidates import compute_candidate_sources  # noqa: E402
except ImportError:
    compute_candidate_sources = None

DEFAULT_ROOT = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
PZS100_SELF_FILTER_CONFIG = REPO_ROOT / "config" / "pzs100_gripper_filter.yaml"
BOX_COLORS = {
    "pending": (1.0, 0.75, 0.0, 1.0),
    "approved": (0.1, 0.9, 0.2, 1.0),
    "rejected": (0.9, 0.15, 0.15, 1.0),
}
# One overlay color per candidate source, distinct from BOX_COLORS and from
# each other: gray for raw geometry, orange for the RGB edge reranker, purple
# for the DINO proposer, teal for the reranked DINO verifier. Which source is
# drawn is decided by `_active_candidate_sources`, not by the operator.
CANDIDATE_SOURCE_COLORS = {
    "pcl": (0.6, 0.6, 0.6, 1.0),
    "rgb_edge_prior": (1.0, 0.55, 0.0, 1.0),
    "dino_proposer": (0.6, 0.2, 0.85, 1.0),
    "dino_verifier_reranked": (0.0, 0.65, 0.65, 1.0),
}
# Centre-distance de-dup radius for `_promote_accepted_candidates`, matching
# the pooling radius already used for the same "same physical block, several
# candidate ids" problem elsewhere in this repo (`ablate_rgb_proposals.py::
# _dedupe_poses`'s `minimum_separation_m`, `eval_sam_region_proposals.py`'s
# and `eval_dino_proposer_pooled.py`'s `--dedup-radius`, both defaulting to
# 0.25). Block extents are >= 0.6 m, so two accepted candidates this close are
# never distinct physical blocks.
CANDIDATE_DEDUP_RADIUS_M = 0.25
WORKFLOW_SECTION_TITLES = (
    "1. Mask the gripper",
    "2. Auto-fit candidates",
    "3. Add and adjust blocks",
    "4. Visibility and details",
    "5. Save snapshot",
)
# Explicit, GUI-owned order for the "Snap face" combobox. This is deliberately
# not derived from `annotation_snap.FACE_CLASSES`' dict order: doing so was
# exactly the bug being fixed elsewhere (`list(FACE_NORMALS)[selected_index]`
# silently depended on another module's dict insertion order).
SNAP_FACE_CHOICES = ("Auto", "End", "Long face")
BASE_POINT_SIZE = 3.5
# The PZS100 self-filter is intentionally visible in the scene.  Keep its
# surfaces faint so they explain which part of the cloud was masked without
# hiding the annotation targets or auto-fit comparison boxes.
PZS_FILTER_OUTLINE_COLOR = (1.0, 0.46, 0.08, 1.0)
PZS_FILTER_RAIL_COLOR = (0.1, 0.82, 1.0, 1.0)
PZS_FILTER_OUTLINE_WIDTH = 2.0
PZS_FILTER_RAIL_WIDTH = 4.0
# The above-gripper "dead zone" is a distinct exclusion volume (points centred
# inside it are masked/rejected even though they're outside the rail boxes),
# so it gets its own colour distinct from both PZS_FILTER_OUTLINE_COLOR and
# PZS_FILTER_RAIL_COLOR.  Deep blue rather than magenta: magenta is already the
# RGB/SAM point selection (see _refresh_scene), and two saturated magentas in
# one view read as one concept when they are unrelated.
PZS_FILTER_ABOVE_GRIPPER_COLOR = (0.15, 0.30, 0.95, 1.0)
PZS_FILTER_ABOVE_GRIPPER_WIDTH = 3.0
INITIAL_EYE = np.array([0.0, 0.0, 0.0])
INITIAL_UP = np.array([0.0, 0.0, 1.0])
INITIAL_DIRECTION = np.array([-1.0, 0.0, 0.0])
DEFAULT_PICK_RADIUS_M = 0.30
DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-small"
BOX_EDGES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if bin(first ^ second).count("1") == 1
)
BOX_SIGNS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def discover_snapshot_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Snapshot root does not exist: {root}")
    excluded = load_snapshot_exclusions(root)
    return sorted(
        item
        for item in root.iterdir()
        if item.is_dir()
        and (item / "cloud.pcd").is_file()
        and (item / "annotations.yaml").is_file()
        and item.name not in excluded
    )


def cuboid_corners(position: np.ndarray, rotation: np.ndarray, dimensions) -> np.ndarray:
    offsets = BOX_SIGNS * (np.asarray(dimensions) / 2.0)
    return np.asarray(position, dtype=float) + offsets @ np.asarray(rotation, dtype=float).T


def target_provenance_label(target) -> str:
    """Short human-readable source tag for the target list, e.g. "region proposal".

    The operator otherwise cannot tell an auto-adopted region-sourced
    (`dino_proposer`) candidate apart from a whole-cloud one: both just show
    up as another pending target. `dino_proposer` candidate ids are
    namespaced `dino_proposer:{proposal_index}:{hypothesis.id}` (see
    `annotation_candidates.py::_dino_proposer_candidates`); every other
    auto-fit source's id is the bare hypothesis id -- see the ambiguity that
    fact caused for de-dup in `AnnotatorWindow._promote_accepted_candidates`,
    which is exactly why a human reviewing pending targets needs this
    visible. A module-level function, not a method: `AnnotatorWindow` is
    exercised in tests via a `SimpleNamespace` stand-in that only binds an
    explicit method list, so a plain function callable without `self` stays
    usable from `_load_document_widgets` without adding to that list.
    """
    if not target.provenance.startswith(AUTO_FIT_PROVENANCE_PREFIX):
        return "manual"
    candidate_id = target.provenance.removeprefix(AUTO_FIT_PROVENANCE_PREFIX)
    if candidate_id.startswith("dino_proposer:"):
        return "region proposal"
    return "whole cloud"


def draw_projected_edges(
    image: np.ndarray,
    corners_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera,
    color: np.ndarray,
) -> None:
    """Draw in-frame edges of a world-frame cuboid without another GUI dependency."""
    camera_points = world_to_camera(corners_world, T_world_camera)
    uv, in_front = project_points(camera_points, camera)
    height, width = image.shape[:2]
    for first, second in BOX_EDGES:
        if not (in_front[first] and in_front[second]):
            continue
        start, end = uv[first], uv[second]
        outside_image = (
            max(start[0], end[0]) < 0
            or min(start[0], end[0]) >= width
            or max(start[1], end[1]) < 0
            or min(start[1], end[1]) >= height
        )
        if outside_image:
            continue
        start = np.clip(np.rint(start).astype(int), [0, 0], [width - 1, height - 1])
        end = np.clip(np.rint(end).astype(int), [0, 0], [width - 1, height - 1])
        steps = max(abs(end - start).max(), 1)
        for point in np.linspace(start, end, int(steps) + 1).astype(int):
            u, v = point
            image[max(0, v - 1) : min(height, v + 2), max(0, u - 1) : min(width, u + 2)] = color


class AnnotatorWindow:
    def __init__(self, args: argparse.Namespace, initial: Path):
        self.args = args
        self.root = args.root
        self.config = args.config
        self.pick_radius = args.pick_radius
        self.snapshots = discover_snapshot_dirs(self.root)
        if not self.snapshots:
            raise ValueError(f"No snapshots containing cloud.pcd under {self.root}")
        if initial not in self.snapshots:
            exclusion = load_snapshot_exclusions(self.root).get(initial.name)
            detail = f": {exclusion.reason}" if exclusion is not None else ""
            raise ValueError(f"Snapshot {initial.name} is not annotatable{detail}")
        self.snapshot = initial
        self._changing_snapshot_combo = False
        self.points = np.empty((0, 3))
        self.raw_points = np.empty((0, 3))
        self.T_world_cloud: np.ndarray | None = None
        self.rgb_colors: np.ndarray | None = None
        self.T_world_camera: np.ndarray | None = None
        self.camera = None
        self.controller: AnnotationController
        self._last_button_down_s = -np.inf
        self.rgb: np.ndarray | None = None
        self.projection_cache = None
        self.candidate_masks: dict[int, np.ndarray] = {}
        self.selected_rgb_points = np.empty((0, 3))
        self.selected_mask: np.ndarray | None = None
        self.candidate_sources_result = None
        self.snap_face = gui.Combobox()
        for choice in SNAP_FACE_CHOICES:
            self.snap_face.add_item(choice)
        self.snap_face.selected_index = 0  # Auto: resolve across all six axes
        self.snap_pick_mode = False
        # SAM is the tool for the blocks auto-fit never proposes -- measured at
        # a quarter of them -- so it is on unless explicitly disabled.
        # Constructing the backend imports nothing heavy; `_load` is lazy.
        self.sam_backend = (
            None
            if args.sam_model is None
            else TransformersSam2Backend(
                args.sam_model, args.sam_device, local_files_only=args.sam_offline
            )
        )
        self.sam_worker = (
            None if self.sam_backend is None else AsyncSegmentationWorker(self.sam_backend)
        )
        self.sam_status = gui.Label(
            "Segmentation: disabled (--no-sam)"
            if self.sam_backend is None
            else f"Segmentation: loading {args.sam_model}..."
        )

        app = gui.Application.instance
        self.window = app.create_window("Block annotation", 1500, 900)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.rgb_view = gui.ImageWidget()
        self.rgb_view.set_on_mouse(self._on_rgb_mouse)
        # The controls are intentionally one scrollable, top-to-bottom workflow.
        # On smaller displays a plain Vert clipped the important delete/save
        # actions below the window edge.
        self.panel = gui.ScrollableVert(6, gui.Margins(8, 8, 8, 8))
        self.save_footer = gui.Vert(4, gui.Margins(8, 4, 8, 8))
        self.status = gui.Label("")
        self.snapshot_combo = gui.Combobox()
        for item in self.snapshots:
            self.snapshot_combo.add_item(item.name)
        self.snapshot_combo.set_on_selection_changed(self._on_snapshot_changed)
        # Open3D's bundled font cannot render the Unicode triangle glyphs
        # these buttons used to use; ASCII arrows keep them legible in the
        # running app instead of showing as "?".
        self.prev_button = self._button("< Prev", lambda: self._navigate(-1))
        self.next_button = self._button("Next >", lambda: self._navigate(1))
        self.target_list = gui.ListView()
        self.target_list.set_max_visible_items(8)
        self.target_list.set_on_selection_changed(self._on_target_selected)
        self.rgb_toggle = gui.Checkbox("RGB-colored cloud")
        self.rgb_toggle.checked = True
        self.rgb_toggle.set_on_checked(lambda _checked: self._refresh_scene())
        self.self_filter_toggle = gui.Checkbox("Remove PZS100 gripper points")
        self.self_filter_toggle.checked = False
        self.self_filter_toggle.enabled = False
        self.self_filter_toggle.set_on_checked(lambda _checked: self._on_point_filters_changed())
        # Rail poses resolved from the snapshot's captured TF; the only input
        # the native filter needs.
        self.pzs100_filter_inputs: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object] | None
        ) = None
        # Populated only by a successful invocation of the native filter.
        # These are its diagnostic volumes/rail segments, in world coordinates,
        # rather than approximations reconstructed by the GUI.
        self.pzs100_filter_diagnostics: dict[str, np.ndarray] | None = None
        self.self_filter_reason = gui.Label("Disabled: loading PZS100 filter profile")
        # Explicit memory of whether `self.status.text` currently holds one of
        # `_apply_point_filters`' own PZS-filter messages, so the next call can
        # tell "stale PZS message from last time" apart from "unrelated status
        # from some other action" without parsing the label's own text.
        self._pzs_status_is_current = False
        self.candidate_status = gui.Label(
            "unavailable: annotation_candidates module not found"
            if compute_candidate_sources is None
            else ""
        )
        self.fields = [self._number() for _ in range(6)]
        self.orientation_label = gui.Label("No block selected")
        for field in self.fields:
            field.set_on_value_changed(lambda _value: self._pose_fields_changed())
        self.visibility = gui.Combobox()
        for value in VISIBILITIES:
            self.visibility.add_item(value)
        self.visibility.set_on_selection_changed(self._metadata_changed)
        self.ignore_reason = gui.TextEdit()
        self.ignore_reason.placeholder_text = "ignore reason (optional)"
        # `set_on_value_changed` (fires once on focus-loss/Enter) rather than
        # `set_on_text_changed` (fires on every keystroke): each commit does a
        # full `copy.deepcopy` of the document onto the undo stack, so typing
        # a reason used to push one undo entry per character. See
        # `_commit_pending_text_edits` for how a not-yet-blurred edit is still
        # guaranteed to reach the document before Save/navigate/close.
        self.ignore_reason.set_on_value_changed(self._ignore_reason_changed)
        self.notes = gui.TextEdit()
        self.notes.placeholder_text = "snapshot notes"
        self.complete = gui.Checkbox("ROI exhaustively annotated")
        self.add_fields = [self._number() for _ in range(3)]
        self.pose_step_buttons: list[gui.Button] = []
        self._build_panel()
        self.complete.set_on_checked(self._complete_changed)
        # See the matching comment on `self.ignore_reason` above: this used to
        # be `set_on_text_changed`, firing (and undo-stacking a full document
        # deepcopy) on every keystroke.
        self.notes.set_on_value_changed(self._notes_changed)
        self.window.add_child(self.scene)
        self.window.add_child(self.rgb_view)
        self.window.add_child(self.panel)
        self.window.add_child(self.save_footer)
        self.window.set_on_layout(self._layout)
        self.window.set_on_close(self._on_close)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)
        self._loading_widgets = False
        # See _commit_pending_text_edits: which block the ignore-reason box is
        # showing, and a latch to stop undo/redo re-committing stale text.
        self._ignore_reason_target_id: str | None = None
        self._suppress_text_commit = False
        self._load_snapshot(initial)
        self._warm_up_sam()

    @staticmethod
    def _number() -> gui.NumberEdit:
        field = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        field.decimal_precision = 4
        field.set_limits(-10000, 10000)
        return field

    def _button(self, text: str, callback) -> gui.Button:
        button = gui.Button(text)
        button.set_on_clicked(callback)
        return button

    def _section(
        self,
        title: str,
        *,
        open_by_default: bool = True,
        parent: gui.Widget | None = None,
    ) -> gui.CollapsableVert:
        section = gui.CollapsableVert(title, 4, gui.Margins(4, 0, 0, 4))
        section.set_is_open(open_by_default)
        (self.panel if parent is None else parent).add_child(section)
        return section

    def _build_panel(self) -> None:
        snapshot_section = self._section("Snapshot", open_by_default=False)
        snapshot_section.add_child(self.snapshot_combo)
        nav_row = gui.Horiz(3)
        nav_row.add_child(self.prev_button)
        nav_row.add_child(self.next_button)
        snapshot_section.add_child(nav_row)
        snapshot_section.add_child(self.rgb_toggle)

        preprocessing_section = self._section(WORKFLOW_SECTION_TITLES[0])
        preprocessing_section.add_child(
            gui.Label("Mask the robot's own gripper out of the cloud before fitting.")
        )
        preprocessing_section.add_child(self.self_filter_toggle)
        preprocessing_section.add_child(self.self_filter_reason)

        candidates_section = self._section(WORKFLOW_SECTION_TITLES[1])
        candidates_section.add_child(
            gui.Label("Fits blocks and adds every accepted result as a pending block.")
        )
        candidates_section.add_child(
            self._button("Compute auto-fit suggestions", self._compute_candidates)
        )
        candidates_section.add_child(self.candidate_status)
        candidates_section.add_child(
            gui.Label("Then click any block the fitter missed in the RGB view.")
        )
        candidates_section.add_child(self.sam_status)

        edit_section = self._section(WORKFLOW_SECTION_TITLES[2])
        edit_section.add_child(
            gui.Label("Select a block below or Ctrl-click the cloud to add one.")
        )
        edit_section.add_child(gui.Label("pending=yellow · approved=green · excluded=red"))
        edit_section.add_child(self.target_list)

        # Enabled by the live RGB/SAM point selection, not by a block being
        # selected -- see `_update_create_from_selection_button`. Deliberately
        # not folded into `_update_selection_actions`, which is keyed only on
        # block selection.
        self.create_from_selection_button = self._button(
            "Create block from selection", self._create_block_from_selection
        )
        self._update_create_from_selection_button()
        edit_section.add_child(self.create_from_selection_button)

        edit_section.add_child(gui.Label("Add block at XYZ (metres)"))
        add_row = gui.Horiz(3)
        for field in self.add_fields:
            add_row.add_child(field)
        add_row.add_child(self._button("Add block", self._add_at_fields))
        edit_section.add_child(add_row)

        # Delete is deliberately adjacent to the block list, always visible in
        # the main workflow, and guarded by the existing confirmation dialog.
        self.delete_button = self._button("Delete selected block...", self._confirm_delete_selected)
        edit_section.add_child(self.delete_button)

        edit_section.add_child(
            gui.Label("Position XYZ (m), Euler XYZ (deg; canonical block frame)")
        )
        edit_section.add_child(self.orientation_label)
        # Keep each correction control beside the value it changes.  The old
        # layout put all six fields in one grid and the +/- controls in two
        # separate rows below it, making it difficult to tell which button
        # affected which value.  A compact row per degree of freedom mirrors
        # the pose data and keeps the existing callbacks/button ordering.
        for axis, (label, field) in enumerate(
            zip(("X", "Y", "Z", "Roll", "Pitch", "Yaw"), self.fields, strict=True)
        ):
            row = gui.Horiz(4)
            row.add_child(gui.Label(label))
            # The row label identifies the degree of freedom, so the controls
            # only need compact +/- glyphs and do not repeat the axis name.
            minus = self._button("-", lambda a=axis: self._step(a, -1))
            plus = self._button("+", lambda a=axis: self._step(a, 1))
            self.pose_step_buttons.extend((minus, plus))
            row.add_child(minus)
            row.add_child(field)
            row.add_child(plus)
            edit_section.add_child(row)
        self.rotate_button = self._button("Rotate 90° yaw", self._rotate_selected_90)
        edit_section.add_child(self.rotate_button)
        snap_row = gui.Horiz(3)
        snap_row.add_child(gui.Label("Snap face"))
        snap_row.add_child(self.snap_face)
        self.snap_pick_button = self._button("Pick support plane in cloud", self._toggle_snap_pick)
        snap_row.add_child(self.snap_pick_button)
        edit_section.add_child(snap_row)
        history = gui.Horiz(4)
        history.add_child(self._button("Undo", self._undo))
        history.add_child(self._button("Redo", self._redo))
        history.add_child(self._button("Reset camera", self._reset_camera))
        edit_section.add_child(history)

        details_section = self._section(WORKFLOW_SECTION_TITLES[3])
        details_section.add_child(
            gui.Label("Record visibility, exclusions, and the final review decision.")
        )
        details_section.add_child(gui.Label("Visibility"))
        details_section.add_child(self.visibility)
        details_section.add_child(gui.Label("Exclusion / review reason"))
        details_section.add_child(self.ignore_reason)
        review = gui.Horiz(4)
        self.approve_button = self._button("Approve block", self._approve)
        self.reject_button = self._button("Exclude (reason required)", self._reject)
        review.add_child(self.approve_button)
        review.add_child(self.reject_button)
        details_section.add_child(review)
        details_section.add_child(
            gui.Label("Exclude keeps a detected non-target for audit; Delete removes the block.")
        )

        restore = gui.Horiz(4)
        self.abort_button = self._button("Discard draft / restore prior", self._abort)
        self.reset_button = self._button("Reset pose to saved", self._reset)
        restore.add_child(self.abort_button)
        restore.add_child(self.reset_button)
        details_section.add_child(restore)
        details_section.add_child(
            gui.Label(
                "Discard removes a new draft or restores an imported prior to pending; "
                "Reset changes only its pose."
            )
        )

        save_section = self._section(WORKFLOW_SECTION_TITLES[4], parent=self.save_footer)
        save_section.add_child(self.complete)
        save_section.add_child(gui.Label("Snapshot notes"))
        save_section.add_child(self.notes)
        save_section.add_child(self._button("Validate and save snapshot", self._save))
        save_section.add_child(self.status)

    def _layout(self, context) -> None:
        rect = self.window.content_rect
        panel_width = min(470, int(rect.width * 0.38))
        view_width = rect.width - panel_width
        scene_width = int(view_width * 0.62)
        self.scene.frame = gui.Rect(rect.x, rect.y, scene_width, rect.height)
        self.rgb_view.frame = gui.Rect(
            rect.x + scene_width, rect.y, view_width - scene_width, rect.height
        )
        self.panel.frame = gui.Rect(
            rect.get_right() - panel_width, rect.y, panel_width, max(1, rect.height - 210)
        )
        self.save_footer.frame = gui.Rect(
            rect.get_right() - panel_width, rect.get_bottom() - 210, panel_width, 210
        )

    def _load_snapshot(self, snapshot: Path) -> None:
        manifest = snapshot / "annotations.yaml"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing annotation manifest: {manifest}")
        # Do every load that can raise (FileNotFoundError, malformed tf/cloud,
        # an empty/non-finite capture) into locals first. Committing to
        # `self` only after all of this succeeds means a failure here leaves
        # self.snapshot/self.controller/self.points/the combobox still
        # describing the snapshot that was already loaded, rather than a
        # half-loaded new one escaping this GUI callback in an inconsistent
        # state.
        document = AnnotationDocument.load(manifest)
        tf = load_tf_snapshot(snapshot / "tf.yaml")
        raw_points, _ = load_snapshot(
            snapshot,
            self.config,
            T_world_cloud=tf.T_world_cloud,
            require_registrations=False,
        )
        # ROI is always the full captured cloud, independent of the optional
        # ground/background preprocessing filters (see _apply_point_filters):
        # there is no manual ROI UI, so this unconditionally satisfies
        # AnnotationDocument.validate()'s ROI completeness requirement without
        # the user ever touching it, and without the completeness region
        # shrinking just because a display/candidate filter is toggled on.
        roi_min, roi_max = full_workspace_roi(raw_points)

        # Nothing from here on can fail, so it is now safe to commit.
        self.snapshot = snapshot
        self.controller = AnnotationController(document)
        self.T_world_cloud = tf.T_world_cloud
        self.raw_points = raw_points
        self.controller.document.roi_min, self.controller.document.roi_max = roi_min, roi_max
        self._load_self_filter_boxes(snapshot, tf)
        self.rgb = None
        self.T_world_camera = tf.T_world_camera
        self.camera = None
        self.candidate_masks = {}
        self.candidate_sources_result = None
        # A prior snapshot's SAM/mask pick is a set of world-space points and
        # stays numerically valid across filter changes (see
        # `_apply_point_filters`), but it has no relation to a newly loaded
        # snapshot's cloud, so this is the one place it must be dropped.
        self.selected_rgb_points = np.empty((0, 3))
        self.selected_mask = None
        self._update_create_from_selection_button()
        self.candidate_status.text = (
            "unavailable: annotation_candidates module not found"
            if compute_candidate_sources is None
            else ""
        )
        if self.sam_worker is not None:
            self.sam_worker.invalidate()
        if (snapshot / "rgb.png").is_file():
            try:
                self.rgb = load_rgb(snapshot / "rgb.png")
                self.camera = load_camera_model()
                self.candidate_masks = load_candidate_masks(snapshot, self.rgb.shape[:2])
            except (ValueError, KeyError):
                self.rgb = None
                self.camera = None
        # If the default filters somehow leave nothing (see
        # `_apply_point_filters`'s empty-cloud guard), fall back to this
        # snapshot's own raw cloud rather than whatever the previous
        # snapshot happened to leave in `self.points`.
        self.points = self.raw_points
        self._apply_point_filters()
        if self.rgb is not None:
            self._refresh_rgb()
        else:
            # Without this, a snapshot with no rgb.png left the ImageWidget
            # showing the PREVIOUS snapshot's photo next to the new
            # snapshot's cloud -- nothing else clears it once `self.rgb`
            # itself has already been set to None above.
            self._clear_rgb_view()
        self._changing_snapshot_combo = True
        self.snapshot_combo.selected_index = snapshot_selection_index(self.snapshots, snapshot)
        self._changing_snapshot_combo = False
        self._update_nav_buttons()
        # The text widgets still hold the PREVIOUS snapshot's text while this
        # document is freshly loaded, so a flush here would overwrite the new
        # snapshot's stored notes with stale text. `_go_to_snapshot` has
        # already committed the outgoing snapshot's edits before we got here.
        self._suppress_text_commit = True
        try:
            self._load_document_widgets()
        finally:
            self._suppress_text_commit = False
        self._refresh_scene(reset_camera=True)

    def _on_point_filters_changed(self) -> None:
        try:
            self._apply_point_filters()
        except (ImportError, ValueError) as error:
            self.status.text = f"Preprocessing unchanged: {error}"
            return
        self._refresh_views()

    def _load_self_filter_boxes(self, snapshot: Path, tf) -> None:
        """Configure the PZS100 rail filter from the snapshot's captured TF.

        Captured TF is the only source of rail endpoints. Captures whose TF
        lacks the profile's rail frames simply leave the toggle disabled,
        naming the missing frame so it is obvious why.
        """
        self.pzs100_filter_inputs = None
        self.pzs100_filter_diagnostics = None
        self.self_filter_toggle.checked = False
        self.self_filter_toggle.enabled = False
        try:
            available, native_reason = pzs100_self_filter_available()
            if not available:
                raise ImportError(native_reason)
            profile = load_pzs100_self_filter_config(PZS100_SELF_FILTER_CONFIG)
            if not profile.enabled:
                raise ValueError("profile is disabled")
            parents, rails, rotations, outward_axes = [], [], [], []
            for rail in profile.rails:
                T_world_parent = tf.lookup(tf.reference_frame, rail.parent_frame)
                T_world_rail = tf.lookup(tf.reference_frame, rail.rail_frame)
                parents.append(T_world_parent[:3, 3])
                rails.append(T_world_rail[:3, 3])
                rotations.append(T_world_parent[:3, :3])
                outward_axes.append(rail.outward_axis_parent)
            self.pzs100_filter_inputs = (
                np.asarray(parents, dtype=np.float64),
                np.asarray(rails, dtype=np.float64),
                np.asarray(rotations, dtype=np.float64),
                np.asarray(outward_axes, dtype=np.float64),
                profile.parameters,
            )
        except (ImportError, KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.self_filter_reason.text = f"Disabled: {error}"
            return
        self.self_filter_toggle.checked = True
        self.self_filter_toggle.enabled = True
        self.self_filter_reason.text = (
            "PZS100 native filter ready from captured K9/K11 rails; its filter volume "
            "outlines and rails are shown in 3D"
        )

    def _apply_point_filters(self) -> None:
        """Recompute the working cloud from `raw_points` under the current filter toggles.

        One filtered cloud feeds rendering, Ctrl+click RANSAC snapping, and
        candidate-source computation, so none of them can silently diverge
        from what's displayed. The annotation ROI is deliberately NOT
        recomputed here -- it always covers the full raw capture (see
        `_load_snapshot`), independent of these display/candidate filters.

        The status label is decided by collecting every message this pass
        could show as `(priority, message, is_pzs_message)` candidates, then
        emitting only the highest-priority one at the end -- rather than
        directly overwriting `self.status.text` at each step and separately
        re-deriving "was the filter active last time" by string-matching the
        label's own previous text against a fixed prefix tuple.
        """
        had_pzs_status = self._pzs_status_is_current
        if had_pzs_status:
            self.status.text = ""
        # Never retain a diagnostic overlay from a previous filter pass.  In
        # particular, turning the PZS checkbox off, changing snapshots, or a
        # native filter failure must immediately remove the old volumes.
        self.pzs100_filter_diagnostics = None
        pzs_filter_failed = False
        points = self.raw_points
        status_candidates: list[tuple[int, str, bool]] = []

        if self.self_filter_toggle.checked and self.pzs100_filter_inputs is not None:
            try:
                points, removed, diagnostics = filter_pzs100_gripper(
                    points, *self.pzs100_filter_inputs
                )
                self.pzs100_filter_diagnostics = diagnostics
                # The jaws are masked either way; say so when only the
                # above-gripper volume was skipped, so a rig the volume cannot
                # describe is visible rather than silently absent from the 3D view.
                skipped = diagnostics.get("above_exclusion_error") or ""
                status_candidates.append(
                    (
                        3,
                        f"PZS100 gripper removed: {removed} points"
                        + (f" (above-gripper volume skipped: {skipped})" if skipped else ""),
                        True,
                    )
                )
            except (ImportError, ValueError, RuntimeError) as error:
                # A stale .so or malformed capture must not make the annotator
                # unusable.  Keep the cloud untouched and expose the cause.
                self.self_filter_toggle.checked = False
                self.self_filter_toggle.enabled = False
                self.pzs100_filter_inputs = None
                self.self_filter_reason.text = f"Disabled: PZS100 native filter failed ({error})"
                status_candidates.append((3, f"PZS100 gripper filter failed: {error}", True))
                pzs_filter_failed = True
        if not self.self_filter_toggle.checked and not pzs_filter_failed:
            # Make a state transition visible when toggling an otherwise-ready
            # filter off.  If another preprocessing operation produced status
            # in this same pass (e.g. ground removal), leave that more useful
            # result intact.
            if self.pzs100_filter_inputs is not None and had_pzs_status:
                status_candidates.append((2, "PZS100 gripper filter disabled.", True))
            elif self.pzs100_filter_inputs is None:
                detail = self.self_filter_reason.text.removeprefix("Disabled: ")
                status_candidates.append((2, f"PZS100 gripper filter unavailable: {detail}", True))

        if status_candidates:
            _priority, message, is_pzs = max(status_candidates, key=lambda item: item[0])
            self.status.text = message
            self._pzs_status_is_current = is_pzs
        elif had_pzs_status:
            self.status.text = ""
            self._pzs_status_is_current = False

        if not len(points):
            # "max range (m)" accepts values down to 0.1 and the workspace
            # min/max fields accept min > max, so this combination can
            # legitimately zero out the cloud. Adopting an empty
            # `self.points` would hand a degenerate array to rendering and to
            # Ctrl+click picking (`_show_click_ray` takes `np.ptp` over it,
            # which raises on a zero-size array), so keep the last working
            # cloud and explain why the filters were not applied instead.
            self.status.text = (
                "Filter combination would leave an empty cloud; keeping the previous "
                f"{len(self.points)}-point cloud instead."
            )
            self._pzs_status_is_current = False
            return
        self.points = points
        # The SAM/mask selection is a set of world-space points and stays
        # valid under any of these filters -- it is only invalidated by
        # loading a different snapshot (see `_load_snapshot`), so it is
        # deliberately not cleared here.
        self.candidate_sources_result = None
        if self.camera is not None and self.rgb is not None:
            try:
                self.rgb_colors, _ = colorize_world_points(
                    self.points, self.T_world_camera, self.camera, self.rgb
                )
                self.projection_cache = build_projection_cache(
                    self.points, self.T_world_camera, self.camera
                )
            except (ValueError, KeyError):
                self.rgb_colors = None
                self.projection_cache = None
                self.camera = None
        else:
            self.rgb_colors = None
            self.projection_cache = None

    def _update_nav_buttons(self) -> None:
        self.prev_button.enabled = adjacent_snapshot(self.snapshots, self.snapshot, -1) is not None
        self.next_button.enabled = adjacent_snapshot(self.snapshots, self.snapshot, 1) is not None

    def _navigate(self, step: int) -> None:
        chosen = adjacent_snapshot(self.snapshots, self.snapshot, step)
        if chosen is None:
            return
        self._go_to_snapshot(chosen)

    def _load_document_widgets(self) -> None:
        # Every reload overwrites the text widgets from the document, so
        # anything typed but not yet blurred has to be captured first.
        self._commit_pending_text_edits()
        self._loading_widgets = True
        document = self.controller.document
        self.target_list.set_items(
            [
                f"{item.id}  [{item.review_status}] ({target_provenance_label(item)})"
                for item in document.targets
            ]
        )
        if self.controller.selected is not None:
            self.target_list.selected_index = self.controller.selected
        self.complete.checked = document.annotation_complete
        self.notes.text_value = document.notes
        self._load_target_widgets()
        self._loading_widgets = False

    def _load_target_widgets(self) -> None:
        target = self.controller.target
        self._update_selection_actions(target is not None)
        if target is None:
            self._ignore_reason_target_id = None
            self.orientation_label.text = "No block selected"
            return
        self._loading_widgets = True
        for field, value in zip(self.fields, [*target.position, *target.euler_deg], strict=True):
            field.double_value = float(value)
        self.visibility.selected_index = VISIBILITIES.index(target.visibility)
        self.ignore_reason.text_value = target.ignore_reason or ""
        # Record whose reason the widget now shows, so a later flush cannot
        # attribute this text to a different block (see _commit_pending_text_edits).
        self._ignore_reason_target_id = target.id
        self.orientation_label.text = orientation_summary(target.rotation)
        self._loading_widgets = False

    def _update_selection_actions(self, has_selection: bool) -> None:
        """Keep actions that require a block honest about their availability."""
        for widget in (
            self.delete_button,
            self.rotate_button,
            self.snap_pick_button,
            self.approve_button,
            self.reject_button,
            self.abort_button,
            self.reset_button,
            self.visibility,
            self.ignore_reason,
            *self.fields,
            *self.pose_step_buttons,
        ):
            widget.enabled = has_selection

    def _refresh_views(self) -> None:
        """Refresh the 3D scene and the RGB overlay together.

        Both panes independently redraw the same `self.controller.document.
        targets` (3D wireframes in `_refresh_scene`, projected wireframes in
        `_refresh_rgb`). Calling only one after a pose/selection/review change
        left the other showing a stale block -- e.g. moving a block in 3D
        left its RGB wireframe behind at the old pose. `_refresh_rgb` already
        no-ops when `self.rgb` is None, so this is always safe to call.
        """
        self._refresh_scene()
        self._refresh_rgb()

    def _refresh_scene(self, reset_camera: bool = False) -> None:
        self.scene.scene.clear_geometry()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        if self.rgb_toggle.checked and self.rgb_colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(self.rgb_colors)
        else:
            cloud.paint_uniform_color((0.55, 0.55, 0.55))
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = BASE_POINT_SIZE
        self.scene.scene.add_geometry("cloud", cloud, material)
        self._add_pzs100_filter_geometries()
        if len(self.selected_rgb_points):
            selected = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.selected_rgb_points))
            selected.paint_uniform_color((1.0, 0.1, 0.8))
            selected_material = rendering.MaterialRecord()
            selected_material.shader = "defaultUnlit"
            selected_material.point_size = 6.0
            self.scene.scene.add_geometry("rgb-mask-points", selected, selected_material)
        for index, target in enumerate(self.controller.document.targets):
            box = o3d.geometry.OrientedBoundingBox(
                target.position, target.rotation, target.dimensions
            )
            lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            color = BOX_COLORS[target.review_status]
            if index == self.controller.selected:
                color = (0.1, 0.7, 1.0, 1.0)
            line_material = rendering.MaterialRecord()
            line_material.shader = "unlitLine"
            line_material.line_width = 4.0 if index == self.controller.selected else 2.0
            line_material.base_color = color
            self.scene.scene.add_geometry(f"block-{index}", lines, line_material)
        self._add_candidate_source_geometries()
        if reset_camera:
            bounds = self.scene.scene.bounding_box
            # Match visualize_snapshot.py's initial view exactly: place the
            # viewer at the world origin, keep world +Z upright, and look down
            # world -X. setup_camera establishes suitable projection/clipping
            # planes for the scene; look_at then replaces only the view pose.
            self.scene.setup_camera(60.0, bounds, bounds.get_center())
            self._reset_camera()
        self.scene.force_redraw()

    def _add_pzs100_filter_geometries(self) -> None:
        """Render native PZS filter diagnostics as read-only scene context.

        The native core supplies the exact oriented exclusion boxes used for
        masking, plus the captured rail endpoints, and optionally the
        above-gripper "dead zone" volume (see `_add_above_gripper_geometry`).
        This routine deliberately has no connection to editable targets or
        candidate computation: it is a visual explanation of an
        already-applied preprocessing step only.
        """
        diagnostics = self.pzs100_filter_diagnostics
        if diagnostics is None:
            return
        try:
            centers = np.asarray(diagnostics["centers"], dtype=float)
            rotations = np.asarray(diagnostics["rotations"], dtype=float)
            sizes = np.asarray(diagnostics["sizes"], dtype=float)
            rail_starts = np.asarray(diagnostics["rail_starts"], dtype=float)
            rail_ends = np.asarray(diagnostics["rail_ends"], dtype=float)
            count = len(centers)
            if (
                centers.shape != (count, 3)
                or rotations.shape != (count, 3, 3)
                or sizes.shape != (count, 3)
                or rail_starts.shape != (count, 3)
                or rail_ends.shape != (count, 3)
                or not np.all(np.isfinite(centers))
                or not np.all(np.isfinite(rotations))
                or not np.all(np.isfinite(sizes))
                or not np.all(np.isfinite(rail_starts))
                or not np.all(np.isfinite(rail_ends))
                or np.any(sizes <= 0.0)
            ):
                raise ValueError("invalid native rail-box diagnostics")
        except (KeyError, TypeError, ValueError) as error:
            # Filtering remains valid even if an incompatible native build
            # returns malformed optional diagnostics.  Do not let decoration
            # prevent operators from viewing or annotating the cloud.
            self.pzs100_filter_diagnostics = None
            self.status.text = f"PZS100 filter active; overlay unavailable ({error})"
            return

        outline_material = rendering.MaterialRecord()
        outline_material.shader = "unlitLine"
        outline_material.base_color = PZS_FILTER_OUTLINE_COLOR
        outline_material.line_width = PZS_FILTER_OUTLINE_WIDTH
        rail_material = rendering.MaterialRecord()
        rail_material.shader = "unlitLine"
        rail_material.base_color = PZS_FILTER_RAIL_COLOR
        rail_material.line_width = PZS_FILTER_RAIL_WIDTH

        # A single LineSet keeps both rails visually consistent and makes their
        # captured endpoints obvious.  The filter volumes are outlines only:
        # this makes their active region clear without hiding cloud points or
        # relying on backend-specific transparent materials.
        rail_lines = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack((rail_starts, rail_ends))),
            lines=o3d.utility.Vector2iVector(
                np.asarray([[index, index + count] for index in range(count)], dtype=np.int32)
            ),
        )
        self.scene.scene.add_geometry("pzs-filter-rails", rail_lines, rail_material)
        for index, (center, rotation, size) in enumerate(
            zip(centers, rotations, sizes, strict=True)
        ):
            box = o3d.geometry.OrientedBoundingBox(center, rotation, size)
            outline = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            self.scene.scene.add_geometry(f"pzs-filter-outline-{index}", outline, outline_material)
        self._add_above_gripper_geometry(diagnostics)

    def _add_above_gripper_geometry(self, diagnostics: dict[str, np.ndarray]) -> None:
        """Render the above-gripper "dead zone" exclusion volume, if present.

        ``above_centers``/``above_rotations``/``above_sizes`` are newer
        diagnostics keys (see `annotation_preprocess.filter_pzs100_gripper`);
        an older compiled ``_core`` extension's diagnostics dict simply lacks
        them. `.get()` plus the early return below keeps the rail/outline
        overlay drawn by the caller intact instead of losing the whole
        overlay over this optional addition.
        """
        above_centers = diagnostics.get("above_centers")
        above_rotations = diagnostics.get("above_rotations")
        above_sizes = diagnostics.get("above_sizes")
        if above_centers is None or above_rotations is None or above_sizes is None:
            return
        try:
            above_centers = np.asarray(above_centers, dtype=float)
            above_rotations = np.asarray(above_rotations, dtype=float)
            above_sizes = np.asarray(above_sizes, dtype=float)
            above_count = len(above_centers)
            if (
                above_centers.shape != (above_count, 3)
                or above_rotations.shape != (above_count, 3, 3)
                or above_sizes.shape != (above_count, 3)
                or not np.all(np.isfinite(above_centers))
                or not np.all(np.isfinite(above_rotations))
                or not np.all(np.isfinite(above_sizes))
                or np.any(above_sizes <= 0.0)
            ):
                raise ValueError("invalid native above-gripper diagnostics")
        except (TypeError, ValueError):
            # Malformed optional diagnostics must not break the rail/outline
            # overlay that already rendered successfully in the caller.
            return
        if not above_count:
            return  # Feature off: empty-shaped arrays, nothing to draw.

        above_material = rendering.MaterialRecord()
        above_material.shader = "unlitLine"
        above_material.base_color = PZS_FILTER_ABOVE_GRIPPER_COLOR
        above_material.line_width = PZS_FILTER_ABOVE_GRIPPER_WIDTH
        for index, (center, rotation, size) in enumerate(
            zip(above_centers, above_rotations, above_sizes, strict=True)
        ):
            box = o3d.geometry.OrientedBoundingBox(center, rotation, size)
            outline = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            self.scene.scene.add_geometry(f"pzs-filter-above-{index}", outline, above_material)

    def _reset_camera(self) -> None:
        focus = initial_camera_focus(self.points, INITIAL_EYE, INITIAL_DIRECTION)
        self.scene.look_at(focus, INITIAL_EYE, INITIAL_UP)
        self.scene.force_redraw()

    def _active_candidate_sources(self) -> list[tuple[str, tuple]]:
        """The sources actually used, chosen automatically rather than by checkbox.

        `pcl`, `rgb_edge_prior` and `dino_verifier_reranked` all re-score the
        SAME raw hypotheses, so they are alternatives, not additions: only the
        most refined one available is used. Measured over this repo's 23
        snapshots, the RGB edge reranker strictly dominates raw PCL -- same
        75% recall, 88% precision against 84% -- so there is nothing to choose
        between them and no reason to ask the operator to.

        `dino_proposer` is the exception: it runs its own detection over RGB
        region proposals and can surface blocks the geometry stage never
        proposed, so its candidates are added rather than substituted.
        """
        result = self.candidate_sources_result
        if result is None:
            return []
        active: list[tuple[str, tuple]] = []
        for key in ("dino_verifier_reranked", "rgb_edge_prior", "pcl"):
            candidates = tuple(getattr(result, key, ()))
            if candidates:
                active.append((key, candidates))
                break
        proposer = tuple(getattr(result, "dino_proposer", ()))
        if proposer:
            active.append(("dino_proposer", proposer))
        return active

    def _add_candidate_source_geometries(self) -> None:
        """Draw read-only comparison boxes for the candidates actually in play.

        These are plain LineSet overlays, never EditableTargets: they are not
        registered anywhere `pick_oriented_box` looks, so clicking them cannot
        create or select an editable target. Accepted candidates are promoted
        immediately, so what remains drawn is the rejected near-misses -- the
        useful signal being "the fitter saw something here and did not take it".
        """
        for key, candidates in self._active_candidate_sources():
            color = CANDIDATE_SOURCE_COLORS[key]
            for candidate_index, candidate in enumerate(candidates):
                if candidate.id in self.controller.adopted_candidate_ids:
                    continue
                pose = candidate.pose
                box = o3d.geometry.OrientedBoundingBox(pose.position, pose.rotation, pose.dims)
                lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                material.line_width = 3.0 if candidate.accepted else 1.5
                material.base_color = color
                self.scene.scene.add_geometry(f"candidate-{key}-{candidate_index}", lines, material)

    def _compute_candidates(self) -> None:
        if compute_candidate_sources is None:
            self.candidate_status.text = "unavailable: annotation_candidates module not found"
            return
        self.candidate_status.text = "Computing candidates..."
        try:
            result = compute_candidate_sources(
                self.points,
                self.rgb,
                self.T_world_camera,
                self.camera,
                T_world_cloud=self.T_world_cloud,
                dino_proposer_checkpoint=self.args.dino_proposer_checkpoint,
                dino_verifier_checkpoint=self.args.dino_verifier_checkpoint,
                # `None` when the PZS100 filter is off/unavailable, which
                # `compute_candidate_sources` treats as a no-op -- callers
                # with no gripper geometry are unaffected.
                above_gripper_boxes=self.pzs100_filter_diagnostics,
            )
        except Exception as error:  # noqa: BLE001 - surface any backend failure in the status bar
            self.candidate_sources_result = None
            self.candidate_status.text = f"Candidate computation failed: {error}"
            self._refresh_views()
            return
        self.candidate_sources_result = result
        try:
            promoted = self._promote_accepted_candidates()
        except ValueError as error:
            self.candidate_status.text = f"Candidate promotion failed: {error}"
            self._refresh_views()
            return
        message = f"Computed candidates for {self.snapshot.name}."
        if promoted:
            message += f" Added {len(promoted)} pending block(s)."
        if result.warnings:
            message += " " + " ".join(result.warnings)
        self.candidate_status.text = message
        self._refresh_views()

    def _promote_accepted_candidates(self) -> list[str]:
        """Promote every accepted result from the active sources as one undoable action.

        De-duplicates by centre distance (`CANDIDATE_DEDUP_RADIUS_M`), not by
        `candidate.id` alone: `dino_proposer` candidate ids are namespaced
        `dino_proposer:{proposal_index}:{hypothesis.id}` (see
        `annotation_candidates.py::_dino_proposer_candidates`) while the
        active base source's (pcl/rgb_edge_prior/dino_verifier_reranked) ids
        are the bare hypothesis id, so a proposer candidate describing the
        SAME physical block as an already-accepted base candidate never
        collides by id and both would otherwise get promoted. Measured on 8
        snapshots: 43 pending targets auto-adopted for ~20 physical blocks;
        18 of 23 proposer candidates sat within 0.25 m of an
        already-promoted base candidate, three of them long/short-axis
        swapped twins (rotation deltas 93.5, 94.4, 95.2 degrees) adopted
        right beside the correct block.

        Runs against targets already in `self.controller.document`, not just
        the current batch: `adopted_candidate_ids` only suppresses
        re-adoption by id, so on a recompute a proposer twin of a previously
        adopted target would still slip through a batch-local-only check.
        On a centre-distance tie the base/whole-cloud candidate wins over the
        region-sourced one -- the region crop is the smaller, less-evidenced
        context (see the crop-vs-full `SensorContext` TODO in
        `annotation_candidates.py::_dino_proposer_candidates`).
        """
        pending: list[tuple[bool, object]] = []
        for key, candidates in self._active_candidate_sources():
            is_region_sourced = key == "dino_proposer"
            for candidate in candidates:
                if candidate.accepted and candidate.id not in self.controller.adopted_candidate_ids:
                    pending.append((is_region_sourced, candidate))
        if not pending:
            return []
        # Base/whole-cloud candidates first, so a centre-distance tie against
        # a region-sourced candidate keeps the base one and drops the region
        # duplicate (stable sort preserves each group's original order).
        pending.sort(key=lambda item: item[0])

        kept_positions = [
            np.asarray(target.position, dtype=np.float64)
            for target in self.controller.document.targets
        ]
        unique: dict[str, object] = {}
        for _is_region_sourced, candidate in pending:
            position = np.asarray(candidate.pose.position, dtype=np.float64)
            if any(
                np.linalg.norm(position - other) <= CANDIDATE_DEDUP_RADIUS_M
                for other in kept_positions
            ):
                continue
            unique[candidate.id] = candidate
            kept_positions.append(position)

        specs = [
            (candidate.id, candidate.pose.position, candidate.pose.rotation, candidate.pose.dims)
            for candidate in unique.values()
        ]
        if not specs:
            return []
        target_ids = self.controller.adopt_candidates(specs)
        self._load_document_widgets()
        return target_ids

    def _refresh_rgb(self) -> None:
        if self.rgb is None:
            return
        shown = self.rgb.copy()
        if self.selected_mask is not None:
            overlay = np.zeros_like(shown)
            overlay[..., 0] = 255
            shown[self.selected_mask] = (
                0.55 * shown[self.selected_mask] + 0.45 * overlay[self.selected_mask]
            ).astype(np.uint8)
        if self.camera is not None and self.T_world_camera is not None:
            for index, target in enumerate(self.controller.document.targets):
                color = np.asarray(BOX_COLORS[target.review_status][:3]) * 255.0
                if index == self.controller.selected:
                    color = np.array([26, 179, 255])
                draw_projected_edges(
                    shown,
                    cuboid_corners(target.position, target.rotation, target.dimensions),
                    self.T_world_camera,
                    self.camera,
                    color.astype(np.uint8),
                )
        self.rgb_view.update_image(o3d.geometry.Image(np.ascontiguousarray(shown)))

    def _clear_rgb_view(self) -> None:
        """Blank the RGB pane so a stale photo from a previous snapshot never lingers.

        `gui.ImageWidget` has no "clear" call, only `update_image`; a
        zero-sized image is the documented way to make it show nothing.
        """
        self.rgb_view.update_image(o3d.geometry.Image())

    def _on_rgb_mouse(self, event):
        if event.type != gui.MouseEvent.BUTTON_DOWN or not event.is_button_down(
            gui.MouseButton.LEFT
        ):
            return gui.Widget.EventCallbackResult.IGNORED
        if self.rgb is None or self.projection_cache is None:
            self.status.text = "This snapshot has no usable RGB/calibration data."
            return gui.Widget.EventCallbackResult.HANDLED
        point = map_widget_to_image(
            event.x - self.rgb_view.frame.x,
            event.y - self.rgb_view.frame.y,
            self.rgb_view.frame.width,
            self.rgb_view.frame.height,
            self.rgb.shape[1],
            self.rgb.shape[0],
        )
        if point is None:
            self.status.text = "Click inside the displayed RGB image."
            return gui.Widget.EventCallbackResult.HANDLED
        snapshot = self.snapshot
        if self.sam_backend is None:
            candidate = candidate_mask_at(self.candidate_masks, point)
            if candidate is None:
                self.status.text = (
                    "No stored candidate mask here, and segmentation is off (--no-sam)."
                )
            else:
                self._apply_rgb_mask(candidate[1], point, f"candidate {candidate[0]}")
            return gui.Widget.EventCallbackResult.HANDLED

        assert self.sam_worker is not None

        def schedule(callback) -> None:
            gui.Application.instance.post_to_main_thread(self.window, callback)

        def finish(mask, error, completed_snapshot) -> None:
            if completed_snapshot != self.snapshot:
                return
            if error is not None or mask is None:
                fallback = candidate_mask_at(self.candidate_masks, point)
                if fallback is None:
                    self.status.text = f"SAM failed: {error}"
                else:
                    self._apply_rgb_mask(
                        fallback[1], point, f"candidate {fallback[0]} (SAM failed)"
                    )
            else:
                self._apply_rgb_mask(mask, point, "SAM")

        accepted = self.sam_worker.submit(self.rgb, point, snapshot, schedule, finish)
        self.status.text = (
            "Running SAM segmentation..."
            if accepted
            else "SAM is busy; wait for the current prompt to finish."
        )
        return gui.Widget.EventCallbackResult.HANDLED

    def _warm_up_sam(self) -> None:
        """Load SAM weights in the background so the first real click is instant.

        `TransformersSam2Backend._load` is lazy, which previously meant the
        first RGB click blocked the GUI for however long the model took to
        resolve, download and move to the device -- with no feedback. Doing it
        here on the worker thread keeps the window responsive and turns the
        wait into a visible readiness line instead of an unexplained stall.
        """
        if self.sam_worker is None or self.sam_backend is None:
            return
        probe = np.zeros((64, 64, 3), dtype=np.uint8)

        def schedule(callback) -> None:
            gui.Application.instance.post_to_main_thread(self.window, callback)

        def finish(_mask, error, _context) -> None:
            self.sam_status.text = (
                f"Segmentation ready: {self.args.sam_model} on {self.sam_backend.device}"
                if error is None
                else f"Segmentation unavailable: {error}"
            )

        self.sam_worker.submit(probe, (32, 32), "warm-up", schedule, finish)

    def _apply_rgb_mask(self, mask: np.ndarray, prompt: tuple[int, int], source: str) -> None:
        assert self.projection_cache is not None
        result = select_points_from_mask(
            self.points,
            self.projection_cache,
            mask,
            prompt,
            MaskSelectionParams(),
        )
        self.selected_mask = result.mask
        self.selected_rgb_points = result.points
        self._update_create_from_selection_button()
        self._refresh_views()
        if not result.success:
            self.status.text = f"{source} selection failed: {result.reason}."
            return
        # Seed the plane snapper at the actual prompted surface, not at the
        # centroid/median (which can lie inside the block or between faces).
        selected_uv = self.projection_cache.uv[result.indices]
        pixel_distance = np.linalg.norm(selected_uv - np.asarray(prompt), axis=1)
        near_prompt = np.flatnonzero(pixel_distance <= max(float(pixel_distance.min()), 8.0))
        seed_local = near_prompt[
            int(np.argmin(self.projection_cache.depth[result.indices[near_prompt]]))
        ]
        seed = result.points[seed_local]
        # The RGB mask provides explicit object identity, so a deterministic
        # best candidate is more useful than rejecting translational ambiguity.
        # It remains pending and visibly requires human validation.
        self._block_from_selected_points(result.points, seed, source)

    def _update_create_from_selection_button(self) -> None:
        """Keep the button's label/enabled state honest about the live RGB/SAM selection.

        Unlike `_update_selection_actions`, this is gated on whether a mask
        selection exists, not on whether a block is selected.
        """
        count = len(self.selected_rgb_points)
        self.create_from_selection_button.text = f"Create block from selection ({count} points)"
        self.create_from_selection_button.enabled = count > 0

    def _create_block_from_selection(self) -> None:
        """Turn the current RGB/SAM point selection into a pending block.

        This is the dead-end fix: previously, when the automatic fit
        triggered by clicking in the RGB pane failed, an operator with a
        visibly correct (magenta) point selection had no way to turn it into
        an annotatable block at all. This lets them do so on demand, from
        whatever selection is currently highlighted.
        """
        points = self.selected_rgb_points
        if not len(points):
            self.status.text = "Select RGB/SAM points before creating a block."
            return
        centroid = points.mean(axis=0)
        seed = points[int(np.argmin(np.linalg.norm(points - centroid, axis=1)))]
        self._block_from_selected_points(points, seed, "Selection")

    def _block_from_selected_points(
        self, points: np.ndarray, seed: np.ndarray, source: str
    ) -> None:
        """Create one pending block from a point selection, fitted if it can be.

        Shared by the RGB click and the explicit button so the two cannot
        diverge. They previously did: the click path bailed with an
        instruction to press the button, which put a second action in front of
        exactly the blocks that need annotating most. A gripped block is the
        normal case here -- the gripper occludes its top face, so the native
        fitter cannot find the top/side pair a canonical placement needs, and
        the automatic path failed every time on precisely the blocks a human
        clicked deliberately.

        Clicking a block in the RGB view is an unambiguous statement that
        there is a block there, so it always produces one. An approximate
        placement the operator nudges beats no placement at all, and its
        provenance records which of the two it was.
        """
        snap = snap_block_to_cloud(points, seed, SnapParams(reject_ambiguous=False))
        if snap.success:
            assert snap.position is not None and snap.rotation is not None
            assert snap.dimensions is not None
            self.controller.add_manual_with_pose(
                snap.position, snap.rotation, snap.dimensions, provenance="rgb_mask"
            )
            self.status.text = (
                f"{source}: fitted pending block from {len(points)} selected points "
                f"(score {snap.score:.2f}). Correct and approve, or abort."
            )
        else:
            # No supported top/side placement exists (a partial or occluded
            # view) -- place a canonical box on the selection, oriented by its
            # principal axes, for the operator to correct by hand.
            position, rotation = pca_oriented_block_pose(points)
            self.controller.add_manual_with_pose(
                position, rotation, BLOCK_DIMS, provenance="rgb_mask_pca_fallback"
            )
            self.status.text = (
                f"{source}: no exact fit ({snap.reason}), so an approximate block was placed "
                f"from {len(points)} selected points. Correct its pose, then approve or discard."
            )
        self._load_document_widgets()
        self._refresh_views()

    def _pose_fields_changed(self) -> None:
        if self._loading_widgets or self.controller.target is None:
            return
        values = np.array([field.double_value for field in self.fields])
        self.controller.set_pose(values[:3], values[3:])
        self._refresh_views()

    def _metadata_changed(self, _text: str, index: int) -> None:
        if self._loading_widgets or self.controller.target is None:
            return
        self.controller.set_visibility(VISIBILITIES[index])

    def _ignore_reason_changed(self, text: str) -> None:
        if not self._loading_widgets:
            self.controller.set_ignore_reason(text)

    def _complete_changed(self, checked: bool) -> None:
        # Commit into the document immediately (routed through the controller,
        # so it is undoable) rather than only at save time: any of the many
        # buttons that reload widgets from the document -- Approve, Reject,
        # Abort, Undo, etc. -- would otherwise silently overwrite this
        # checkbox with the last-saved value, discarding what the operator
        # just set.
        if self._loading_widgets:
            return
        self.controller.set_annotation_complete(checked)
        self.status.text = "Unsaved changes... press Ctrl+S or Validate and save snapshot."

    def _notes_changed(self, text: str) -> None:
        # `self.notes`/`self.ignore_reason` are wired to `set_on_value_changed`
        # (see __init__), which fires once when the widget loses focus or the
        # user presses Enter -- NOT on every keystroke -- so a whole burst of
        # typing becomes one undo entry instead of one per character. That
        # alone would still lose text typed right before Save/Prev/Next/Close
        # if the widget never loses focus first, so `_commit_pending_text_edits`
        # additionally reads the live widget value at each of those points.
        if self._loading_widgets:
            return
        self.controller.set_notes(text)
        self.status.text = "Unsaved changes... press Ctrl+S or Validate and save snapshot."

    def _commit_pending_text_edits(self) -> None:
        """Flush notes/ignore-reason text into the document without waiting on focus-loss.

        `set_on_value_changed` only fires once the widget loses focus or the
        user presses Enter, so text typed and then immediately acted on -- via
        a button, a 3D-scene click, or an async SAM result landing mid-typing
        -- needs an explicit read here, otherwise the last burst of typing is
        silently dropped. That is why this is called from the top of
        `_load_document_widgets` and not only from the exit points: any reload
        overwrites the widgets from the document, so every reload is a chance
        to lose text.

        Two ordering hazards make the guards below load-bearing:

        * The ignore-reason box belongs to ONE block. Callers such as
          `add_manual` move the selection before the GUI reloads, so a blind
          flush would write the previous block's reason onto the new one.
          `_ignore_reason_target_id` records whose text the widget is actually
          showing, and the flush is skipped unless it still matches.
        * Undo/redo replace the whole document. Flushing during their reload
          would re-commit the pre-undo widget text on top of the restored
          state, silently defeating the undo. They flush FIRST (so typing
          becomes its own undo entry, reversible with one more press) and then
          suppress the flush while repainting.
        """
        if self._loading_widgets or self._suppress_text_commit:
            return
        target = self.controller.target
        # A None marker means the box has not been repainted for any block yet
        # (startup), where the widget is empty and flushing is a harmless
        # no-op. Only a marker naming a DIFFERENT block means the selection
        # moved out from under the text, which is the case worth skipping.
        if target is not None and self._ignore_reason_target_id in (None, target.id):
            self.controller.set_ignore_reason(self.ignore_reason.text_value)
        self.controller.set_notes(self.notes.text_value)

    def _step(self, axis: int, sign: int) -> None:
        step = 0.01 if axis < 3 else 1.0
        self.fields[axis].double_value += sign * step
        self._pose_fields_changed()
        self._load_target_widgets()

    def _rotate_selected_90(self) -> None:
        """Rotate the selected block exactly +90 degrees about world Z (yaw)."""
        self._commit_pending_text_edits()
        target = self.controller.target
        if target is None:
            self.status.text = "Select a block before rotating."
            return
        # Apply the rotation matrix directly (`set_pose_matrix`) instead of
        # round-tripping through "xyz" Euler angles: a STANDING block's
        # canonical long axis is vertical, exactly the gimbal-lock singularity
        # for that convention, so decomposing to Euler and rebuilding risked
        # drift/misbehavior in precisely the case the tool's docs call out.
        rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix() @ target.rotation
        self.controller.set_pose_matrix(target.position, rotation)
        self._load_document_widgets()
        self._refresh_views()
        self.status.text = "Rotated block +90 degrees around world Z (yaw). Review and approve."

    def _toggle_snap_pick(self) -> None:
        self.snap_pick_mode = not self.snap_pick_mode
        self.snap_pick_button.text = (
            "Cancel support-plane pick" if self.snap_pick_mode else "Pick support plane in cloud"
        )
        self.status.text = (
            "Click a nearby cloud point to snap the selected face."
            if self.snap_pick_mode
            else "Support-plane pick cancelled."
        )

    def _approve(self) -> None:
        if self.controller.target is None:
            return
        self._commit_pending_text_edits()
        self.controller.set_review("approved")
        self._load_document_widgets()
        self._refresh_views()

    def _reject(self) -> None:
        self._commit_pending_text_edits()
        reason = self.ignore_reason.text_value.strip()
        if not reason:
            self.status.text = "Enter an exclusion reason before excluding this block."
            return
        # Persist the reason as the block's ignore_reason too, not only as the
        # rejection_reason. Otherwise the reload below repaints the visible
        # "Exclusion / review reason" box from `target.ignore_reason` (still
        # None) and the text the operator just typed vanishes from the screen
        # the instant they exclude the block.
        self.controller.set_ignore_reason(reason)
        self.controller.set_review("rejected", reason)
        self._load_document_widgets()
        self._refresh_views()

    def _abort(self) -> None:
        self._commit_pending_text_edits()
        self.controller.abort_selected()
        self._load_document_widgets()
        self._refresh_views()

    def _reset(self) -> None:
        self._commit_pending_text_edits()
        self.controller.reset_selected()
        self._load_document_widgets()
        self._refresh_views()

    def _confirm_delete_selected(self) -> None:
        target = self.controller.target
        if target is None:
            self.status.text = "Select a block before deleting."
            return
        target_id, target_status = target.id, target.review_status
        dialog = gui.Dialog("Delete annotation candidate")
        layout = gui.Vert(8, gui.Margins(12, 12, 12, 12))
        layout.add_child(
            gui.Label(
                f"Delete {target_id!r} [{target_status}]?\n"
                "This affects only the in-memory draft until you save the snapshot."
            )
        )
        buttons = gui.Horiz(6)
        buttons.add_child(
            self._button(
                f"Delete {target_id}",
                lambda: self._finish_delete(target_id, target_status),
            )
        )
        buttons.add_child(self._button("Cancel", self.window.close_dialog))
        layout.add_child(buttons)
        dialog.add_child(layout)
        self.window.show_dialog(dialog)

    def _finish_delete(self, target_id: str, target_status: str) -> None:
        deleted = self.controller.delete_selected(
            expected_id=target_id, expected_status=target_status
        )
        self.window.close_dialog()
        if not deleted:
            self.status.text = "Selection changed; nothing was deleted."
            return
        self._load_document_widgets()
        self._refresh_views()
        self.status.text = (
            f"Deleted {target_id!r} [{target_status}] from the draft. "
            "Undo is available; save the snapshot to persist."
        )

    def _snap_selected(self, seed: np.ndarray) -> bool:
        target = self.controller.target
        if target is None:
            self.status.text = "Select a block before snapping."
            return False
        face = SNAP_FACE_CHOICES[self.snap_face.selected_index]
        result = snap_block_face_to_plane(
            self.points, seed, target.position, target.rotation, target.dimensions, face
        )
        if not result.success:
            self.status.text = f"Snap failed: {result.reason}; pose unchanged."
            return False
        assert result.position is not None
        assert result.rotation is not None
        assert result.dimensions is not None
        self.controller.apply_snapped_pose(result.position, result.rotation, result.dimensions)
        self._load_document_widgets()
        self._refresh_views()
        self.status.text = (
            f"Snapped pending proposal: score {result.score:.2f}, "
            f"{result.inlier_count} surface points. Review and approve manually."
        )
        return True

    def _add_at_fields(self) -> None:
        self._commit_pending_text_edits()
        self.controller.add_manual(np.array([field.double_value for field in self.add_fields]))
        self._load_document_widgets()
        self._refresh_views()

    def _undo(self) -> None:
        # Commit typing BEFORE the history move, so it becomes its own
        # reversible entry rather than being clobbered or re-applied.
        self._commit_pending_text_edits()
        if self.controller.undo():
            self._suppress_text_commit = True
            try:
                self._load_document_widgets()
            finally:
                self._suppress_text_commit = False
            self._refresh_views()

    def _redo(self) -> None:
        # Commit typing BEFORE the history move, so it becomes its own
        # reversible entry rather than being clobbered or re-applied.
        self._commit_pending_text_edits()
        if self.controller.redo():
            self._suppress_text_commit = True
            try:
                self._load_document_widgets()
            finally:
                self._suppress_text_commit = False
            self._refresh_views()

    def _on_target_selected(self, _value: str, _double_click: bool) -> None:
        self._commit_pending_text_edits()
        self.controller.selected = self.target_list.selected_index
        self._load_target_widgets()
        self._refresh_views()

    def _ray(self, event) -> tuple[np.ndarray, np.ndarray]:
        x = event.x - self.scene.frame.x
        y = event.y - self.scene.frame.y
        camera = self.scene.scene.camera
        dimensions = (self.scene.frame.width, self.scene.frame.height)
        # Depth=1 may represent infinity in Open3D's projection and produced a
        # zero/NaN direction on real clicks. Use the camera eye plus a stable
        # mid-depth world point instead.
        inverse_view = np.linalg.inv(np.asarray(camera.get_view_matrix(), dtype=float))
        eye = inverse_view[:3, 3]
        point = camera.unproject(x, y, 0.5, *dimensions)
        return camera_ray(eye, point)

    def _show_click_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        candidate: np.ndarray | None,
        accepted: bool,
    ) -> None:
        try:
            origin, unit = camera_ray(origin, np.asarray(origin) + np.asarray(direction))
        except ValueError as error:
            self.status.text = f"Cannot draw click ray: {error}"
            return
        # An empty or degenerate working cloud (see `_apply_point_filters`'s
        # empty-cloud guard) must not reach `np.ptp`, which raises on a
        # zero-size array and escapes this Open3D mouse callback.
        spread = float(np.linalg.norm(np.ptp(self.points, axis=0))) if len(self.points) else 0.0
        extent = max(spread, 20.0) if np.isfinite(spread) else 20.0
        end = origin + unit * extent
        ray = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(np.vstack([origin, end])),
            o3d.utility.Vector2iVector([[0, 1]]),
        )
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 4.0
        material.base_color = (0.1, 0.8, 1.0, 1.0)
        for name in ("click-ray", "click-candidate"):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
        self.scene.scene.add_geometry("click-ray", ray, material)
        if candidate is not None:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
            marker.translate(candidate)
            marker.compute_vertex_normals()
            marker_material = rendering.MaterialRecord()
            marker_material.shader = "defaultLit"
            marker_material.base_color = (0.1, 1.0, 0.2, 1.0) if accepted else (1.0, 0.1, 0.1, 1.0)
            self.scene.scene.add_geometry("click-candidate", marker, marker_material)
        self.scene.force_redraw()

    def _on_mouse(self, event):
        if event.type != gui.MouseEvent.BUTTON_DOWN or not event.is_button_down(
            gui.MouseButton.LEFT
        ):
            return gui.Widget.EventCallbackResult.IGNORED
        now = time.monotonic()
        if now - self._last_button_down_s < 0.25:
            # Open3D reports a double-click as two button-down events. Treat it
            # as one selection/add operation; double-click has no separate action.
            return gui.Widget.EventCallbackResult.HANDLED
        self._last_button_down_s = now
        try:
            origin, direction = self._ray(event)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.status.text = f"Cannot compute click ray: {error}"
            return gui.Widget.EventCallbackResult.HANDLED
        if self.snap_pick_mode:
            candidate = nearest_point_to_ray_with_distance(origin, direction, self.points)
            point = (
                candidate[0] if candidate is not None and candidate[1] <= self.pick_radius else None
            )
            if point is None:
                self.status.text = "No nearby cloud point found for support-plane snap."
            else:
                self._snap_selected(point)
                self.snap_pick_mode = False
                self.snap_pick_button.text = "Pick support plane in cloud"
            return gui.Widget.EventCallbackResult.HANDLED
        if event.is_modifier_down(gui.KeyModifier.CTRL):
            candidate = nearest_point_to_ray_with_distance(origin, direction, self.points)
            point = (
                candidate[0] if candidate is not None and candidate[1] <= self.pick_radius else None
            )
            if point is not None:
                self.controller.add_manual(point)
                self._load_document_widgets()
                self._refresh_views()
                self._show_click_ray(origin, direction, point, True)
                self.status.text = (
                    "Manual block added. Choose a face and pick a support plane to snap it."
                )
            else:
                nearest = None if candidate is None else candidate[0]
                distance = None if candidate is None else candidate[1]
                self._show_click_ray(origin, direction, nearest, False)
                self.status.text = (
                    f"No cloud point within {self.pick_radius:.2f} m of click ray; "
                    + (
                        "no forward point found."
                        if distance is None
                        else f"nearest is {distance:.2f} m away (red marker)."
                    )
                )
            return gui.Widget.EventCallbackResult.HANDLED
        selected = pick_oriented_box(origin, direction, self.controller.document.targets)
        if selected is not None:
            # SceneWidget mouse events are dispatched outside the ImGui frame,
            # so the ignore-reason box has had no chance to blur and fire its
            # value-changed callback. Capture it before the selection moves.
            self._commit_pending_text_edits()
            self.controller.selected = selected
            self.target_list.selected_index = selected
            self._load_target_widgets()
            self._refresh_views()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_key(self, event):
        if event.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.is_repeat:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == ord("S") and event.is_modifier_down(gui.KeyModifier.CTRL):
            self._save()
            return gui.Widget.EventCallbackResult.HANDLED
        # Z/Y are deliberately conventional even if platform modifier delivery differs.
        if event.key == ord("Z"):
            self._undo()
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == ord("Y"):
            self._redo()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _save(self) -> bool:
        # The completeness checkbox commits on every click (`_complete_changed`),
        # but notes/ignore-reason only commit on focus-loss/Enter (see
        # __init__) -- flush them explicitly so a burst of typing right before
        # Save is never silently dropped.
        self._commit_pending_text_edits()
        try:
            backup = save_annotations_atomic(
                self.controller.document, self.snapshot / "annotations.yaml"
            )
        except (ValueError, OSError) as error:
            self.status.text = f"Not saved: {error}"
            return False
        self.controller.mark_saved()
        self.status.text = f"Saved. Backup: {backup.name if backup else 'none'}"
        return True

    def _confirm_dirty(self, action) -> None:
        dialog = gui.Dialog("Unsaved annotations")
        layout = gui.Vert(8, gui.Margins(12, 12, 12, 12))
        layout.add_child(gui.Label("Save changes before continuing?"))
        buttons = gui.Horiz(6)
        buttons.add_child(self._button("Save", lambda: self._finish_dialog(action, True)))
        buttons.add_child(self._button("Discard", lambda: self._finish_dialog(action, False)))
        buttons.add_child(self._button("Cancel", self._cancel_dirty_dialog))
        layout.add_child(buttons)
        dialog.add_child(layout)
        self.window.show_dialog(dialog)

    def _finish_dialog(self, action, save: bool) -> None:
        if save and not self._save():
            return
        if not save:
            self.controller.dirty = False
        self.window.close_dialog()
        action()

    def _cancel_dirty_dialog(self) -> None:
        self._changing_snapshot_combo = True
        self.snapshot_combo.selected_index = snapshot_selection_index(self.snapshots, self.snapshot)
        self._changing_snapshot_combo = False
        self.window.close_dialog()

    def _on_snapshot_changed(self, _name: str, index: int) -> None:
        if self._changing_snapshot_combo:
            return
        self._go_to_snapshot(self.snapshots[index])

    def _go_to_snapshot(self, chosen: Path) -> None:
        """Dirty-check-then-load path shared by the combobox and Prev/Next."""
        if chosen == self.snapshot:
            return
        # Flush a not-yet-blurred notes/ignore-reason edit before the dirty
        # check below, so it is both included in the save-prompt decision and
        # never left behind in the widget when `_load_snapshot` repopulates it
        # from the next snapshot's document.
        self._commit_pending_text_edits()
        if self.sam_worker is not None:
            self.sam_worker.invalidate()

        def action() -> None:
            self._load_snapshot(chosen)

        if self.controller.dirty:
            self._confirm_dirty(action)
        else:
            action()

    def _on_close(self) -> bool:
        # Same reasoning as `_go_to_snapshot`: a not-yet-blurred edit must
        # count towards the dirty check that decides whether to prompt to save.
        self._commit_pending_text_edits()
        if self.controller.dirty:
            if self.sam_worker is not None:
                self.sam_worker.invalidate()
            self._confirm_dirty(self.window.close)
            return False
        if self.sam_worker is not None:
            self.sam_worker.close()
        return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pick-radius",
        type=float,
        default=DEFAULT_PICK_RADIUS_M,
        help="World-space radius around a Ctrl-click ray used to find a cloud point "
        f"(default: {DEFAULT_PICK_RADIUS_M:.2f} m).",
    )
    parser.add_argument(
        "--sam-model",
        default=DEFAULT_SAM_MODEL,
        help="Hugging Face model for RGB point-prompt segmentation "
        f"(default: {DEFAULT_SAM_MODEL}).",
    )
    parser.add_argument(
        "--no-sam",
        dest="sam_model",
        action="store_const",
        const=None,
        help="Disable RGB segmentation and fall back to stored candidate masks. Segmentation "
        "is on by default because auto-fit misses roughly a quarter of blocks and clicking "
        "them in RGB is the fastest way to recover them.",
    )
    parser.add_argument(
        "--sam-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device for SAM (default: auto).",
    )
    parser.add_argument(
        "--sam-offline",
        action="store_true",
        help="Use only locally cached SAM configuration and weights (no network).",
    )
    parser.add_argument(
        "--dino-proposer-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint enabling the DINO candidate proposer source (default: unavailable).",
    )
    parser.add_argument(
        "--dino-verifier-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint enabling the DINO candidate verifier/reranker source "
        "(default: unavailable).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not np.isfinite(args.pick_radius) or args.pick_radius <= 0:
        parser.error("--pick-radius must be a positive finite number")
    args.root = args.root.resolve()
    args.config = args.config.resolve()
    snapshots = discover_snapshot_dirs(args.root)
    if not snapshots:
        raise ValueError(f"No annotatable snapshots under {args.root}")
    initial = (args.snapshot or snapshots[0]).resolve()
    if initial.parent != args.root:
        raise ValueError(f"Snapshot {initial} must be a direct child of {args.root}")
    gui.Application.instance.initialize()
    # Keep the Python owner alive for the entire GUI event loop. Discarding the
    # temporary AnnotatorWindow before run() can destroy Open3D's renderer and
    # GLX pbuffer while Filament still uses it (GLXBadPbuffer on startup/click).
    annotator = AnnotatorWindow(args, initial)
    gui.Application.instance.run()
    del annotator


if __name__ == "__main__":
    main()
