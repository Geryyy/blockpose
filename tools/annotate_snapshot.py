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
from open3d.visualization import gui, rendering
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_snap import SnapParams, snap_block_to_cloud  # noqa: E402
from blockpose.annotation_tool import (  # noqa: E402
    VISIBILITIES,
    AnnotationController,
    AnnotationDocument,
    adjacent_snapshot,
    camera_ray,
    default_max_range_m,
    full_workspace_roi,
    initial_camera_focus,
    nearest_point_to_ray_with_distance,
    next_symmetry_rotation,
    orientation_summary,
    pick_oriented_box,
    remove_background_by_range,
    save_annotations_atomic,
    snapshot_selection_index,
)
from blockpose.detection import DetectionParams, remove_ground  # noqa: E402
from blockpose.io import (  # noqa: E402
    load_camera_model,
    load_rgb,
    load_snapshot,
    load_tf_snapshot,
)
from blockpose.pose import BlockPose  # noqa: E402
from blockpose.projection import (  # noqa: E402
    colorize_world_points,
    project_points,
    world_to_camera,
)
from blockpose.refine import assign_points  # noqa: E402
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
BOX_COLORS = {
    "pending": (1.0, 0.75, 0.0, 1.0),
    "approved": (0.1, 0.9, 0.2, 1.0),
    "rejected": (0.9, 0.15, 0.15, 1.0),
}
# One comparison-overlay color per candidate source, distinct from BOX_COLORS and
# from each other: thin gray for the raw geometric proposer, orange for the
# accepted RGB-edge-prior refinement, purple for the DINO proposer, teal for the
# reranked DINO verifier.
CANDIDATE_SOURCES = (
    ("pcl", "PCL geometry"),
    ("rgb_edge_prior", "+ RGB edge prior"),
    ("dino_proposer", "+ DINO proposer"),
    ("dino_verifier_reranked", "+ DINO verifier (reranked)"),
)
CANDIDATE_SOURCE_COLORS = {
    "pcl": (0.6, 0.6, 0.6, 1.0),
    "rgb_edge_prior": (1.0, 0.55, 0.0, 1.0),
    "dino_proposer": (0.6, 0.2, 0.85, 1.0),
    "dino_verifier_reranked": (0.0, 0.65, 0.65, 1.0),
}
SELECTED_BLOCK_POINTS_COLOR = (0.0, 0.9, 0.9)
SELECTED_BLOCK_POINTS_BAND_M = 0.10
BASE_POINT_SIZE = 3.5
INITIAL_EYE = np.array([0.0, 0.0, 0.0])
INITIAL_UP = np.array([0.0, 0.0, 1.0])
INITIAL_DIRECTION = np.array([-1.0, 0.0, 0.0])
DEFAULT_PICK_RADIUS_M = 0.30
BOX_EDGES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if bin(first ^ second).count("1") == 1
)
BOX_SIGNS = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
)


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
    return np.asarray(position, dtype=float) + (BOX_SIGNS * (np.asarray(dimensions) / 2.0)) @ np.asarray(rotation, dtype=float).T


def draw_projected_edges(
    image: np.ndarray, corners_world: np.ndarray, T_world_camera: np.ndarray, camera, color: np.ndarray
) -> None:
    """Draw in-frame edges of a world-frame cuboid without another GUI dependency."""
    camera_points = world_to_camera(corners_world, T_world_camera)
    uv, in_front = project_points(camera_points, camera)
    height, width = image.shape[:2]
    for first, second in BOX_EDGES:
        if not (in_front[first] and in_front[second]):
            continue
        start, end = uv[first], uv[second]
        if max(start[0], end[0]) < 0 or min(start[0], end[0]) >= width or max(start[1], end[1]) < 0 or min(start[1], end[1]) >= height:
            continue
        start = np.clip(np.rint(start).astype(int), [0, 0], [width - 1, height - 1])
        end = np.clip(np.rint(end).astype(int), [0, 0], [width - 1, height - 1])
        steps = max(abs(end - start).max(), 1)
        for point in np.linspace(start, end, int(steps) + 1).astype(int):
            u, v = point
            image[max(0, v - 1):min(height, v + 2), max(0, u - 1):min(width, u + 2)] = color


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
        self._rotate_cycle_state: tuple[str, np.ndarray, int] | None = None
        self._rotate_cycle_last_applied: np.ndarray | None = None
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

        app = gui.Application.instance
        self.window = app.create_window("Block annotation", 1500, 900)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.rgb_view = gui.ImageWidget()
        self.rgb_view.set_on_mouse(self._on_rgb_mouse)
        self.panel = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self.status = gui.Label("")
        self.snapshot_combo = gui.Combobox()
        for item in self.snapshots:
            self.snapshot_combo.add_item(item.name)
        self.snapshot_combo.set_on_selection_changed(self._on_snapshot_changed)
        self.prev_button = self._button("◀ Prev", lambda: self._navigate(-1))
        self.next_button = self._button("Next ▶", lambda: self._navigate(1))
        self.target_list = gui.ListView()
        self.target_list.set_max_visible_items(8)
        self.target_list.set_on_selection_changed(self._on_target_selected)
        self.rgb_toggle = gui.Checkbox("RGB-colored cloud")
        self.rgb_toggle.checked = True
        self.rgb_toggle.set_on_checked(lambda _checked: self._refresh_scene())
        self.remove_ground_toggle = gui.Checkbox("Remove ground")
        self.remove_ground_toggle.checked = True
        self.remove_ground_toggle.set_on_checked(lambda _checked: self._on_point_filters_changed())
        self.remove_background_toggle = gui.Checkbox("Remove background (range cutoff)")
        self.remove_background_toggle.checked = True
        self.remove_background_toggle.set_on_checked(
            lambda _checked: self._on_point_filters_changed()
        )
        self.max_range_field = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.max_range_field.decimal_precision = 2
        self.max_range_field.set_limits(0.1, 1000.0)
        self.max_range_field.set_on_value_changed(lambda _value: self._on_point_filters_changed())
        self.candidate_checks = {key: gui.Checkbox(label) for key, label in CANDIDATE_SOURCES}
        for checkbox in self.candidate_checks.values():
            checkbox.checked = False
            checkbox.set_on_checked(lambda _checked: self._refresh_scene())
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
        self.ignore_reason.set_on_text_changed(self._ignore_reason_changed)
        self.notes = gui.TextEdit()
        self.notes.placeholder_text = "snapshot notes"
        self.complete = gui.Checkbox("ROI exhaustively annotated")
        self.add_fields = [self._number() for _ in range(3)]
        self._build_panel()
        self.complete.set_on_checked(lambda _checked: self._mark_widget_dirty())
        self.notes.set_on_text_changed(lambda _text: self._mark_widget_dirty())
        self.window.add_child(self.scene)
        self.window.add_child(self.rgb_view)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._layout)
        self.window.set_on_close(self._on_close)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)
        self._loading_widgets = False
        self._load_snapshot(initial)

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

    def _section(self, title: str) -> gui.CollapsableVert:
        section = gui.CollapsableVert(title, 4, gui.Margins(4, 0, 0, 4))
        section.set_is_open(True)
        self.panel.add_child(section)
        return section

    def _build_panel(self) -> None:
        snapshot_section = self._section("Snapshot")
        snapshot_section.add_child(self.snapshot_combo)
        nav_row = gui.Horiz(3)
        nav_row.add_child(self.prev_button)
        nav_row.add_child(self.next_button)
        snapshot_section.add_child(nav_row)
        snapshot_section.add_child(self.rgb_toggle)

        preprocessing_section = self._section("Preprocessing")
        preprocessing_section.add_child(self.remove_ground_toggle)
        preprocessing_section.add_child(self.remove_background_toggle)
        range_row = gui.Horiz(3)
        range_row.add_child(gui.Label("max range (m)"))
        range_row.add_child(self.max_range_field)
        preprocessing_section.add_child(range_row)

        candidates_section = self._section("Candidate sources")
        for key, _label in CANDIDATE_SOURCES:
            candidates_section.add_child(self.candidate_checks[key])
        candidates_section.add_child(self._button("Compute candidates", self._compute_candidates))
        candidates_section.add_child(self.candidate_status)

        blocks_section = self._section("Blocks")
        blocks_section.add_child(
            gui.Label("Blocks: pending=yellow, approved=green, rejected=red")
        )
        blocks_section.add_child(self.target_list)

        pose_section = self._section("Pose")
        pose_section.add_child(
            gui.Label("Position XYZ (m), Euler XYZ (deg; canonical block frame)")
        )
        pose_section.add_child(self.orientation_label)
        grid = gui.VGrid(2, 4)
        for label, field in zip(("X", "Y", "Z", "Roll", "Pitch", "Yaw"), self.fields, strict=True):
            grid.add_child(gui.Label(label))
            grid.add_child(field)
        pose_section.add_child(grid)
        steps = gui.Horiz(3)
        for axis, label in enumerate(("X", "Y", "Z")):
            steps.add_child(self._button(f"{label}-", lambda a=axis: self._step(a, -1)))
            steps.add_child(self._button(f"{label}+", lambda a=axis: self._step(a, 1)))
        pose_section.add_child(steps)
        turns = gui.Horiz(3)
        for axis, label in enumerate(("R", "P", "Y"), start=3):
            turns.add_child(self._button(f"{label}-", lambda a=axis: self._step(a, -1)))
            turns.add_child(self._button(f"{label}+", lambda a=axis: self._step(a, 1)))
        pose_section.add_child(turns)
        pose_section.add_child(
            self._button("Rotate 90° (alternative face labeling)", self._rotate_selected_90)
        )

        review_section = self._section("Review")
        review = gui.Horiz(4)
        review.add_child(self._button("Approve", self._approve))
        review.add_child(self._button("Reject", self._reject))
        review.add_child(self._button("Abort edit", self._abort))
        review.add_child(self._button("Reset", self._reset))
        review_section.add_child(review)
        review_section.add_child(gui.Label("Visibility / ignored target"))
        review_section.add_child(self.visibility)
        review_section.add_child(self.ignore_reason)

        actions_section = self._section("Block actions")
        actions_section.add_child(gui.Label("Add manual draft at XYZ (or Ctrl-click cloud)"))
        add_row = gui.Horiz(3)
        for field in self.add_fields:
            add_row.add_child(field)
        add_row.add_child(self._button("Add", self._add_at_fields))
        actions_section.add_child(add_row)
        actions_section.add_child(self._button("Delete selected…", self._confirm_delete_selected))
        actions_section.add_child(self._button("Snap selected to cloud", self._snap_selected))

        history_section = self._section("History")
        history = gui.Horiz(4)
        history.add_child(self._button("Undo", self._undo))
        history.add_child(self._button("Redo", self._redo))
        history.add_child(self._button("Reset camera", self._reset_camera))
        history_section.add_child(history)

        self.panel.add_child(self.complete)
        self.panel.add_child(self.notes)
        self.panel.add_child(self._button("Validate & Save", self._save))
        self.panel.add_child(self.status)

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
            rect.get_right() - panel_width, rect.y, panel_width, rect.height
        )

    def _load_snapshot(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        manifest = snapshot / "annotations.yaml"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing annotation manifest: {manifest}")
        self.controller = AnnotationController(AnnotationDocument.load(manifest))
        tf = load_tf_snapshot(snapshot / "tf.yaml")
        self.T_world_cloud = tf.T_world_cloud
        self.raw_points, _ = load_snapshot(
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
        self.controller.document.roi_min, self.controller.document.roi_max = full_workspace_roi(
            self.raw_points
        )
        self.max_range_field.double_value = default_max_range_m(
            self.raw_points, self.T_world_cloud[:3, 3]
        )
        self.rgb = None
        self.T_world_camera = tf.T_world_camera
        self.camera = None
        self.candidate_masks = {}
        self.candidate_sources_result = None
        self.candidate_status.text = (
            "unavailable: annotation_candidates module not found"
            if compute_candidate_sources is None
            else ""
        )
        self._rotate_cycle_state = None
        self._rotate_cycle_last_applied = None
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
        self._apply_point_filters()
        if self.rgb is not None:
            self._refresh_rgb()
        self._changing_snapshot_combo = True
        self.snapshot_combo.selected_index = snapshot_selection_index(self.snapshots, snapshot)
        self._changing_snapshot_combo = False
        self._update_nav_buttons()
        self._load_document_widgets()
        self._refresh_scene(reset_camera=True)

    def _on_point_filters_changed(self) -> None:
        self._apply_point_filters()
        self._refresh_scene()

    def _apply_point_filters(self) -> None:
        """Recompute the working cloud from `raw_points` under the current filter toggles.

        One filtered cloud feeds rendering, Ctrl+click RANSAC snapping, and
        candidate-source computation, so none of them can silently diverge
        from what's displayed. The annotation ROI is deliberately NOT
        recomputed here -- it always covers the full raw capture (see
        `_load_snapshot`), independent of these display/candidate filters.
        """
        points = self.raw_points
        if self.remove_ground_toggle.checked and len(points) >= 3:
            points, _ = remove_ground(points, DetectionParams())
        if self.remove_background_toggle.checked and self.T_world_cloud is not None:
            points = remove_background_by_range(
                points, self.T_world_cloud[:3, 3], self.max_range_field.double_value
            )
        self.points = points
        self.selected_rgb_points = np.empty((0, 3))
        self.selected_mask = None
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
        self._loading_widgets = True
        document = self.controller.document
        self.target_list.set_items(
            [f"{item.id}  [{item.review_status}]" for item in document.targets]
        )
        if self.controller.selected is not None:
            self.target_list.selected_index = self.controller.selected
        self.complete.checked = document.annotation_complete
        self.notes.text_value = document.notes
        self._load_target_widgets()
        self._loading_widgets = False

    def _load_target_widgets(self) -> None:
        target = self.controller.target
        if target is None:
            self.orientation_label.text = "No block selected"
            return
        self._loading_widgets = True
        for field, value in zip(self.fields, [*target.position, *target.euler_deg], strict=True):
            field.double_value = float(value)
        self.visibility.selected_index = VISIBILITIES.index(target.visibility)
        self.ignore_reason.text_value = target.ignore_reason or ""
        self.orientation_label.text = orientation_summary(target.rotation)
        self._loading_widgets = False

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
        if len(self.selected_rgb_points):
            selected = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.selected_rgb_points))
            selected.paint_uniform_color((1.0, 0.1, 0.8))
            selected_material = rendering.MaterialRecord()
            selected_material.shader = "defaultUnlit"
            selected_material.point_size = 6.0
            self.scene.scene.add_geometry("rgb-mask-points", selected, selected_material)
        selected_target = self.controller.target
        if selected_target is not None:
            selected_pose = BlockPose(
                position=selected_target.position,
                rotation=selected_target.rotation,
                dims=selected_target.dimensions,
            )
            assigned = assign_points(
                [selected_pose], self.points, None, band=SELECTED_BLOCK_POINTS_BAND_M
            )[0]
            if len(assigned):
                block_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(assigned))
                block_points.paint_uniform_color(SELECTED_BLOCK_POINTS_COLOR)
                block_material = rendering.MaterialRecord()
                block_material.shader = "defaultUnlit"
                block_material.point_size = 5.0
                self.scene.scene.add_geometry("selected-block-points", block_points, block_material)
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

    def _reset_camera(self) -> None:
        focus = initial_camera_focus(self.points, INITIAL_EYE, INITIAL_DIRECTION)
        self.scene.look_at(focus, INITIAL_EYE, INITIAL_UP)
        self.scene.force_redraw()

    def _add_candidate_source_geometries(self) -> None:
        """Draw read-only comparison boxes for each checked candidate source.

        These are plain LineSet overlays, never EditableTargets: they are not
        registered anywhere `pick_oriented_box` looks, so clicking them cannot
        create or select an editable target.
        """
        if self.candidate_sources_result is None:
            return
        for key, _label in CANDIDATE_SOURCES:
            if not self.candidate_checks[key].checked:
                continue
            candidates = getattr(self.candidate_sources_result, key, ())
            color = CANDIDATE_SOURCE_COLORS[key]
            for candidate_index, candidate in enumerate(candidates):
                pose = candidate.pose
                box = o3d.geometry.OrientedBoundingBox(pose.position, pose.rotation, pose.dims)
                lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                material.line_width = 3.0 if candidate.accepted else 1.5
                material.base_color = color
                self.scene.scene.add_geometry(
                    f"candidate-{key}-{candidate_index}", lines, material
                )

    def _compute_candidates(self) -> None:
        if compute_candidate_sources is None:
            self.candidate_status.text = "unavailable: annotation_candidates module not found"
            return
        self.candidate_status.text = "Computing candidates…"
        try:
            result = compute_candidate_sources(
                self.points,
                self.rgb,
                self.T_world_camera,
                self.camera,
                T_world_cloud=self.T_world_cloud,
                dino_proposer_checkpoint=self.args.dino_proposer_checkpoint,
                dino_verifier_checkpoint=self.args.dino_verifier_checkpoint,
            )
        except Exception as error:  # noqa: BLE001 - surface any backend failure in the status bar
            self.candidate_sources_result = None
            self.candidate_status.text = f"Candidate computation failed: {error}"
            self._refresh_scene()
            return
        self.candidate_sources_result = result
        message = f"Computed candidates for {self.snapshot.name}."
        if result.warnings:
            message += " " + " ".join(result.warnings)
        self.candidate_status.text = message
        self._refresh_scene()

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
                self.status.text = "No stored candidate mask at this pixel; enable --sam-model."
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
            "Running SAM segmentation…"
            if accepted
            else "SAM is busy; wait for the current prompt to finish."
        )
        return gui.Widget.EventCallbackResult.HANDLED

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
        self._refresh_rgb()
        self._refresh_scene()
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
        snap = snap_block_to_cloud(result.points, seed, SnapParams(reject_ambiguous=False))
        if not snap.success:
            self.status.text = (
                f"{source}: selected {len(result.points)} points, but fit failed: "
                f"{snap.reason}. No annotation was changed."
            )
            return
        assert snap.position is not None and snap.rotation is not None
        assert snap.dimensions is not None
        self.controller.add_manual(snap.position)
        self.controller.apply_snapped_pose(snap.position, snap.rotation, snap.dimensions)
        self._load_document_widgets()
        self._refresh_scene()
        self.status.text = (
            f"{source}: fitted pending block from {len(result.points)} selected points "
            f"(score {snap.score:.2f}). Correct and approve, or abort."
        )

    def _pose_fields_changed(self) -> None:
        if self._loading_widgets or self.controller.target is None:
            return
        values = np.array([field.double_value for field in self.fields])
        self.controller.set_pose(values[:3], values[3:])
        self._refresh_scene()

    def _metadata_changed(self, _text: str, index: int) -> None:
        if self._loading_widgets or self.controller.target is None:
            return
        self.controller._change(
            lambda: setattr(self.controller.target, "visibility", VISIBILITIES[index])
        )

    def _ignore_reason_changed(self, text: str) -> None:
        if not self._loading_widgets:
            self.controller.set_ignore_reason(text)

    def _mark_widget_dirty(self) -> None:
        if not self._loading_widgets:
            self.controller.dirty = True
            self.status.text = "Unsaved changes — press Ctrl+S or Validate & Save."

    def _step(self, axis: int, sign: int) -> None:
        step = 0.01 if axis < 3 else 1.0
        self.fields[axis].double_value += sign * step
        self._pose_fields_changed()
        self._load_target_widgets()

    def _rotate_selected_90(self) -> None:
        """Cycle the selected target through alternative box-face labelings.

        This is a pose edit like any manual field change: it goes through
        `AnnotationController.set_pose`, so it is undoable and marks the
        document dirty rather than silently rewriting the target in place.
        """
        target = self.controller.target
        if target is None:
            self.status.text = "Select a block before rotating."
            return
        state = self._rotate_cycle_state
        continuing = (
            state is not None
            and state[0] == target.id
            and self._rotate_cycle_last_applied is not None
            and np.allclose(target.rotation, self._rotate_cycle_last_applied, atol=1e-9)
        )
        if continuing:
            _, base, index = state
        else:
            base, index = target.rotation.copy(), 0
        index += 1
        rotation = next_symmetry_rotation(base, index, target.dimensions)
        self._rotate_cycle_state = (target.id, base, index)
        self._rotate_cycle_last_applied = rotation.copy()
        euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        self.controller.set_pose(target.position, euler)
        self._load_document_widgets()
        self._refresh_scene()
        self.status.text = (
            "Rotated to an alternative face labeling of the same box; "
            "review and approve if this matches the physical block."
        )

    def _approve(self) -> None:
        if self.controller.target is None:
            return
        self.controller.set_review("approved")
        self._load_document_widgets()
        self._refresh_scene()

    def _reject(self) -> None:
        reason = self.ignore_reason.text_value.strip()
        if not reason:
            self.status.text = "Enter a reason before rejecting."
            return
        self.controller.set_review("rejected", reason)
        self._load_document_widgets()
        self._refresh_scene()

    def _abort(self) -> None:
        self.controller.abort_selected()
        self._load_document_widgets()
        self._refresh_scene()

    def _reset(self) -> None:
        self.controller.reset_selected()
        self._load_document_widgets()
        self._refresh_scene()

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
                "This affects only the in-memory draft until Validate & Save."
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
        self._refresh_scene()
        self.status.text = (
            f"Deleted {target_id!r} [{target_status}] from the draft. "
            "Undo is available; use Validate & Save to persist."
        )

    def _snap_selected(self, seed: np.ndarray | None = None) -> bool:
        target = self.controller.target
        if target is None:
            self.status.text = "Select a block before snapping."
            return False
        result = snap_block_to_cloud(
            self.points, target.position if seed is None else np.asarray(seed, dtype=float)
        )
        if not result.success:
            self.status.text = f"Snap failed: {result.reason}; pose unchanged."
            return False
        assert result.position is not None
        assert result.rotation is not None
        assert result.dimensions is not None
        self.controller.apply_snapped_pose(result.position, result.rotation, result.dimensions)
        self._load_document_widgets()
        self._refresh_scene()
        self.status.text = (
            f"Snapped pending proposal: score {result.score:.2f}, "
            f"{result.inlier_count} surface points. Review and approve manually."
        )
        return True

    def _add_at_fields(self) -> None:
        self.controller.add_manual(np.array([field.double_value for field in self.add_fields]))
        self._load_document_widgets()
        self._refresh_scene()

    def _undo(self) -> None:
        if self.controller.undo():
            self._load_document_widgets()
            self._refresh_scene()

    def _redo(self) -> None:
        if self.controller.redo():
            self._load_document_widgets()
            self._refresh_scene()

    def _on_target_selected(self, _value: str, _double_click: bool) -> None:
        self.controller.selected = self.target_list.selected_index
        self._load_target_widgets()
        self._refresh_scene()

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
        extent = max(float(np.linalg.norm(np.ptp(self.points, axis=0))), 20.0)
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
        if event.is_modifier_down(gui.KeyModifier.CTRL):
            candidate = nearest_point_to_ray_with_distance(origin, direction, self.points)
            point = (
                candidate[0] if candidate is not None and candidate[1] <= self.pick_radius else None
            )
            if point is not None:
                self.controller.add_manual(point)
                self._load_document_widgets()
                self._refresh_scene()
                snapped = self._snap_selected(point)
                self._show_click_ray(origin, direction, point, True)
                if not snapped:
                    # _snap_selected already explains that the original draft
                    # was retained.  Keep the successful pick visible as well.
                    self.status.text += " Click shown in green."
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
            self.controller.selected = selected
            self.target_list.selected_index = selected
            self._load_target_widgets()
            self._refresh_scene()
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

    def _commit_document_widgets(self) -> None:
        # ROI is set unconditionally to the full workspace in _load_snapshot and
        # is never edited from the UI, so only these two fields need committing.
        document = self.controller.document
        document.annotation_complete = self.complete.checked
        document.notes = self.notes.text_value

    def _save(self) -> bool:
        self._commit_document_widgets()
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
        if self.sam_worker is not None:
            self.sam_worker.invalidate()

        def action() -> None:
            self._load_snapshot(chosen)

        if self.controller.dirty:
            self._confirm_dirty(action)
        else:
            action()

    def _on_close(self) -> bool:
        if self.controller.dirty:
            if self.sam_worker is not None:
                self.sam_worker.invalidate()
            self._confirm_dirty(self.window.close)
            return False
        if self.sam_worker is not None:
            self.sam_worker.close()
        return True


def main() -> None:
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
        nargs="?",
        const="facebook/sam2.1-hiera-small",
        default=None,
        help="Enable RGB point-prompt segmentation (default model when flag has no value: "
        "facebook/sam2.1-hiera-small). Without it, stored candidate masks are used.",
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
        help="Use only locally cached SAM configuration and weights.",
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
