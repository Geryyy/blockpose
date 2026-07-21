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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.annotation_snap import snap_block_to_cloud  # noqa: E402
from blockpose.annotation_tool import (  # noqa: E402
    VISIBILITIES,
    AnnotationController,
    AnnotationDocument,
    camera_ray,
    nearest_point_to_ray_with_distance,
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
from blockpose.projection import colorize_world_points  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "data" / "registration_snapshots"
DEFAULT_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
BOX_COLORS = {
    "pending": (1.0, 0.75, 0.0, 1.0),
    "approved": (0.1, 0.9, 0.2, 1.0),
    "rejected": (0.9, 0.15, 0.15, 1.0),
}
INITIAL_EYE = np.array([0.0, 0.0, 0.0])
INITIAL_LOOK_AT = np.array([-1.0, 0.0, 0.0])
INITIAL_UP = np.array([0.0, 0.0, 1.0])
DEFAULT_PICK_RADIUS_M = 0.30


def discover_snapshot_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Snapshot root does not exist: {root}")
    return sorted(
        item
        for item in root.iterdir()
        if item.is_dir()
        and (item / "cloud.pcd").is_file()
        and (item / "annotations.yaml").is_file()
    )


class AnnotatorWindow:
    def __init__(self, root: Path, initial: Path, config: Path, pick_radius: float):
        self.root = root
        self.config = config
        self.pick_radius = pick_radius
        self.snapshots = discover_snapshot_dirs(root)
        if not self.snapshots:
            raise ValueError(f"No snapshots containing cloud.pcd under {root}")
        self.snapshot = initial
        self._changing_snapshot_combo = False
        self.points = np.empty((0, 3))
        self.rgb_colors: np.ndarray | None = None
        self.controller: AnnotationController
        self._last_button_down_s = -np.inf

        app = gui.Application.instance
        self.window = app.create_window("Block annotation", 1500, 900)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.panel = gui.Vert(6, gui.Margins(8, 8, 8, 8))
        self.status = gui.Label("")
        self.snapshot_combo = gui.Combobox()
        for item in self.snapshots:
            self.snapshot_combo.add_item(item.name)
        self.snapshot_combo.set_on_selection_changed(self._on_snapshot_changed)
        self.target_list = gui.ListView()
        self.target_list.set_max_visible_items(8)
        self.target_list.set_on_selection_changed(self._on_target_selected)
        self.rgb_toggle = gui.Checkbox("RGB-colored cloud")
        self.rgb_toggle.checked = True
        self.rgb_toggle.set_on_checked(lambda _checked: self._refresh_scene())
        self.fields = [self._number() for _ in range(6)]
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
        self.roi_fields = [self._number() for _ in range(6)]
        self.add_fields = [self._number() for _ in range(3)]
        self._build_panel()
        self.complete.set_on_checked(lambda _checked: self._mark_widget_dirty())
        self.notes.set_on_text_changed(lambda _text: self._mark_widget_dirty())
        for field in self.roi_fields:
            field.set_on_value_changed(lambda _value: self._mark_widget_dirty())
        self.window.add_child(self.scene)
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

    def _build_panel(self) -> None:
        self.panel.add_child(gui.Label("Snapshot"))
        self.panel.add_child(self.snapshot_combo)
        self.panel.add_child(self.rgb_toggle)
        self.panel.add_child(gui.Label("Blocks: pending=yellow, approved=green, rejected=red"))
        self.panel.add_child(self.target_list)
        self.panel.add_child(gui.Label("Position XYZ (m), Euler XYZ (deg)"))
        grid = gui.VGrid(2, 4)
        for label, field in zip(("X", "Y", "Z", "Roll", "Pitch", "Yaw"), self.fields, strict=True):
            grid.add_child(gui.Label(label))
            grid.add_child(field)
        self.panel.add_child(grid)
        steps = gui.Horiz(3)
        for axis, label in enumerate(("X", "Y", "Z")):
            steps.add_child(self._button(f"{label}-", lambda a=axis: self._step(a, -1)))
            steps.add_child(self._button(f"{label}+", lambda a=axis: self._step(a, 1)))
        self.panel.add_child(steps)
        turns = gui.Horiz(3)
        for axis, label in enumerate(("R", "P", "Y"), start=3):
            turns.add_child(self._button(f"{label}-", lambda a=axis: self._step(a, -1)))
            turns.add_child(self._button(f"{label}+", lambda a=axis: self._step(a, 1)))
        self.panel.add_child(turns)
        review = gui.Horiz(4)
        review.add_child(self._button("Approve", self._approve))
        review.add_child(self._button("Reject", self._reject))
        review.add_child(self._button("Abort edit", self._abort))
        review.add_child(self._button("Reset", self._reset))
        self.panel.add_child(review)
        self.panel.add_child(self._button("Snap selected to cloud", self._snap_selected))
        self.panel.add_child(gui.Label("Visibility / ignored target"))
        self.panel.add_child(self.visibility)
        self.panel.add_child(self.ignore_reason)
        self.panel.add_child(gui.Label("Add manual draft at XYZ (or Ctrl-click cloud)"))
        add_row = gui.Horiz(3)
        for field in self.add_fields:
            add_row.add_child(field)
        add_row.add_child(self._button("Add", self._add_at_fields))
        self.panel.add_child(add_row)
        history = gui.Horiz(4)
        history.add_child(self._button("Undo", self._undo))
        history.add_child(self._button("Redo", self._redo))
        self.panel.add_child(history)
        self.panel.add_child(self.complete)
        self.panel.add_child(gui.Label("ROI min XYZ / max XYZ"))
        roi_grid = gui.VGrid(3, 3)
        for field in self.roi_fields:
            roi_grid.add_child(field)
        self.panel.add_child(roi_grid)
        self.panel.add_child(self.notes)
        self.panel.add_child(self._button("Validate & Save", self._save))
        self.panel.add_child(self.status)

    def _layout(self, context) -> None:
        rect = self.window.content_rect
        panel_width = min(470, int(rect.width * 0.38))
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
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
        self.points, _ = load_snapshot(snapshot, self.config, T_world_cloud=tf.T_world_cloud)
        self.rgb_colors = None
        if (snapshot / "rgb.png").is_file():
            try:
                self.rgb_colors, _ = colorize_world_points(
                    self.points,
                    tf.T_world_camera,
                    load_camera_model(),
                    load_rgb(snapshot / "rgb.png"),
                )
            except (ValueError, KeyError):
                self.rgb_colors = None
        self._changing_snapshot_combo = True
        self.snapshot_combo.selected_index = snapshot_selection_index(self.snapshots, snapshot)
        self._changing_snapshot_combo = False
        self._load_document_widgets()
        self._refresh_scene(reset_camera=True)

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
        values = (
            [0.0] * 6
            if document.roi_min is None or document.roi_max is None
            else [*document.roi_min, *document.roi_max]
        )
        for field, value in zip(self.roi_fields, values, strict=True):
            field.double_value = float(value)
        self._load_target_widgets()
        self._loading_widgets = False

    def _load_target_widgets(self) -> None:
        target = self.controller.target
        if target is None:
            return
        self._loading_widgets = True
        for field, value in zip(self.fields, [*target.position, *target.euler_deg], strict=True):
            field.double_value = float(value)
        self.visibility.selected_index = VISIBILITIES.index(target.visibility)
        self.ignore_reason.text_value = target.ignore_reason or ""
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
        material.point_size = 2.0
        self.scene.scene.add_geometry("cloud", cloud, material)
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
        if reset_camera:
            bounds = self.scene.scene.bounding_box
            # Match visualize_snapshot.py's initial view exactly: place the
            # viewer at the world origin, keep world +Z upright, and look down
            # world -X. setup_camera establishes suitable projection/clipping
            # planes for the scene; look_at then replaces only the view pose.
            self.scene.setup_camera(60.0, bounds, bounds.get_center())
            self.scene.look_at(INITIAL_LOOK_AT, INITIAL_EYE, INITIAL_UP)
        self.scene.force_redraw()

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
        self.controller.apply_snapped_pose(
            result.position, result.rotation, result.dimensions
        )
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
            marker_material.base_color = (
                (0.1, 1.0, 0.2, 1.0) if accepted else (1.0, 0.1, 0.1, 1.0)
            )
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
                candidate[0]
                if candidate is not None and candidate[1] <= self.pick_radius
                else None
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
        document = self.controller.document
        document.annotation_complete = self.complete.checked
        document.notes = self.notes.text_value
        values = np.array([field.double_value for field in self.roi_fields])
        if document.annotation_complete or document.roi_min is not None or np.any(values != 0):
            document.roi_min, document.roi_max = values[:3], values[3:]
        else:
            document.roi_min = document.roi_max = None

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
        self.snapshot_combo.selected_index = snapshot_selection_index(
            self.snapshots, self.snapshot
        )
        self._changing_snapshot_combo = False
        self.window.close_dialog()

    def _on_snapshot_changed(self, _name: str, index: int) -> None:
        if self._changing_snapshot_combo:
            return
        chosen = self.snapshots[index]
        if chosen == self.snapshot:
            return
        def action() -> None:
            self._load_snapshot(chosen)
        if self.controller.dirty:
            self._confirm_dirty(action)
        else:
            action()

    def _on_close(self) -> bool:
        if self.controller.dirty:
            self._confirm_dirty(self.window.close)
            return False
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
    args = parser.parse_args()
    if not np.isfinite(args.pick_radius) or args.pick_radius <= 0:
        parser.error("--pick-radius must be a positive finite number")
    root = args.root.resolve()
    snapshots = discover_snapshot_dirs(root)
    if not snapshots:
        raise ValueError(f"No annotatable snapshots under {root}")
    initial = (args.snapshot or snapshots[0]).resolve()
    if initial.parent != root:
        raise ValueError(f"Snapshot {initial} must be a direct child of {root}")
    gui.Application.instance.initialize()
    # Keep the Python owner alive for the entire GUI event loop. Discarding the
    # temporary AnnotatorWindow before run() can destroy Open3D's renderer and
    # GLX pbuffer while Filament still uses it (GLXBadPbuffer on startup/click).
    annotator = AnnotatorWindow(root, initial, args.config.resolve(), args.pick_radius)
    gui.Application.instance.run()
    del annotator


if __name__ == "__main__":
    main()
