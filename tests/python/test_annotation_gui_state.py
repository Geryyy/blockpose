"""Headless coverage for annotator state-management fixes.

Uses the same fake/stub style as ``test_annotation_gui_pzs_overlay.py``: call
the real ``AnnotatorWindow`` methods (unbound) against a minimal fake ``self``
so the production logic runs unmodified, without needing a real Open3D window.
Widgets that are cheap and side-effect-free to construct headlessly (Label,
TextEdit, Checkbox, NumberEdit, Combobox, ListView, Button) are real Open3D
GUI objects; the 3D scene is a small recorder fake.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from blockpose.annotation_tool import (  # noqa: E402
    VISIBILITIES,
    AnnotationController,
    AnnotationDocument,
    EditableTarget,
)
from blockpose.pose import BLOCK_DIMS  # noqa: E402
from open3d.visualization import gui  # noqa: E402

import tools.annotate_snapshot as annotator  # noqa: E402

gui.Application.instance.initialize()


class _SceneRecorder:
    """Fakes the subset of `rendering.Open3DScene` that `_refresh_scene` touches."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, object, object]] = []

    def clear_geometry(self) -> None:
        self.entries = []

    def add_geometry(self, name: str, geometry: object, material: object) -> None:
        self.entries.append((name, geometry, material))

    def has_geometry(self, name: str) -> bool:
        return any(entry_name == name for entry_name, _geometry, _material in self.entries)

    def remove_geometry(self, name: str) -> None:
        self.entries = [entry for entry in self.entries if entry[0] != name]


def _document_with_pending_target() -> AnnotationDocument:
    return AnnotationDocument(
        "scene",
        [EditableTarget("prior", np.zeros(3), np.eye(3), provenance="upstream_registration")],
    )


# `_approve`/`_reject`/`_abort`/`_reset`/`_undo`/`_redo` all call back into
# other `self._method()` helpers. A plain SimpleNamespace has no such
# attributes, so bind the real implementations onto each fake instance --
# this still runs the exact production code, just without a real Open3D
# window underneath it.
_BOUND_METHODS = (
    "_load_document_widgets",
    "_load_target_widgets",
    "_update_selection_actions",
    "_update_create_from_selection_button",
    "_refresh_views",
    "_refresh_scene",
    "_refresh_rgb",
    "_add_pzs100_filter_geometries",
    "_add_candidate_source_geometries",
    "_active_candidate_sources",
    "_commit_pending_text_edits",
    "_block_from_selected_points",
)


def _gui_window(document: AnnotationDocument) -> SimpleNamespace:
    """A fake `AnnotatorWindow` carrying real widgets wherever that's cheap.

    Covers exactly the attributes touched by `_notes_changed`/`_complete_changed`,
    `_load_document_widgets`, `_load_target_widgets`, `_update_selection_actions`,
    `_approve`/`_reject`/`_abort`/`_reset`/`_undo`/`_redo`, and `_refresh_views`
    (`_refresh_scene` with `reset_camera=False` plus `_refresh_rgb`, the only
    combination any of those call; `rgb=None` makes `_refresh_rgb` a no-op).
    """
    controller = AnnotationController(document)
    visibility = gui.Combobox()
    for value in VISIBILITIES:
        visibility.add_item(value)
    window = SimpleNamespace(
        controller=controller,
        _loading_widgets=False,
        # Mirrors AnnotatorWindow.__init__: which block the ignore-reason box
        # is showing, and the undo/redo latch (see _commit_pending_text_edits).
        _ignore_reason_target_id=None,
        _suppress_text_commit=False,
        status=gui.Label(""),
        notes=gui.TextEdit(),
        complete=gui.Checkbox("ROI exhaustively annotated"),
        target_list=gui.ListView(),
        fields=[gui.NumberEdit(gui.NumberEdit.DOUBLE) for _ in range(6)],
        visibility=visibility,
        ignore_reason=gui.TextEdit(),
        orientation_label=gui.Label(""),
        delete_button=gui.Button("delete"),
        rotate_button=gui.Button("rotate"),
        snap_pick_button=gui.Button("snap"),
        approve_button=gui.Button("approve"),
        reject_button=gui.Button("reject"),
        abort_button=gui.Button("abort"),
        reset_button=gui.Button("reset"),
        pose_step_buttons=[],
        add_fields=[gui.NumberEdit(gui.NumberEdit.DOUBLE) for _ in range(3)],
        create_from_selection_button=gui.Button("Create block from selection (0 points)"),
        scene=SimpleNamespace(scene=_SceneRecorder(), force_redraw=lambda: None),
        points=np.zeros((1, 3)),
        selected_rgb_points=np.empty((0, 3)),
        rgb_toggle=SimpleNamespace(checked=False),
        rgb_colors=None,
        rgb=None,
        pzs100_filter_diagnostics=None,
        candidate_sources_result=None,
    )
    for name in _BOUND_METHODS:
        setattr(window, name, MethodType(getattr(annotator.AnnotatorWindow, name), window))
    return window


# --- 0.1: notes / annotation_complete must not be discarded by a reload ----


def test_notes_and_complete_survive_an_approve_and_participate_in_undo() -> None:
    window = _gui_window(_document_with_pending_target())

    # Simulate the operator typing a note and ticking the completeness box
    # before pressing "Approve block" -- the exact sequence that used to lose
    # both fields, because they were only written into the document at save
    # time while _load_document_widgets (called by Approve) unconditionally
    # overwrote the widgets FROM the document.
    # Type into the widget WITHOUT firing its value-changed callback: that
    # callback only fires on blur/Enter, so pressing a button straight after
    # typing is precisely the case where the document has not seen the text
    # yet. Calling _notes_changed here instead would assume the very commit
    # under test had already happened, and the test could never fail.
    window.notes.text_value = "left block is partly occluded"
    # The checkbox, by contrast, does fire immediately on click.
    annotator.AnnotatorWindow._complete_changed(window, True)
    annotator.AnnotatorWindow._approve(window)

    assert window.controller.document.notes == "left block is partly occluded"
    assert window.controller.document.annotation_complete is True
    # _load_document_widgets ran as part of _approve; the widgets must reflect
    # what was committed, not silently revert to blank/unchecked.
    assert window.notes.text_value == "left block is partly occluded"
    assert window.complete.checked is True
    assert window.controller.target.review_status == "approved"

    # Both fields are ordinary undoable edits. The pending text is flushed at
    # the START of the action, so the history reads complete -> notes ->
    # approve and undo unwinds the approve first, which is what an operator
    # pressing Undo straight after Approve expects.
    annotator.AnnotatorWindow._undo(window)
    assert window.controller.target.review_status == "pending"
    assert window.controller.document.notes == "left block is partly occluded"
    assert window.controller.document.annotation_complete is True

    annotator.AnnotatorWindow._undo(window)
    assert window.controller.document.notes == ""
    assert window.controller.document.annotation_complete is True

    annotator.AnnotatorWindow._undo(window)
    assert window.controller.document.annotation_complete is False


def test_notes_and_complete_change_callbacks_noop_while_loading_widgets() -> None:
    """Programmatic widget updates (e.g. from `_load_document_widgets`) must not re-enter."""
    window = _gui_window(_document_with_pending_target())
    window._loading_widgets = True

    annotator.AnnotatorWindow._notes_changed(window, "should be ignored")
    annotator.AnnotatorWindow._complete_changed(window, True)

    assert window.controller.document.notes == ""
    assert window.controller.document.annotation_complete is False
    assert not window.controller.dirty


def test_repeated_identical_notes_edits_do_not_spam_the_undo_stack() -> None:
    controller = AnnotationController(_document_with_pending_target())
    controller.set_notes("same text")
    controller.set_notes("same text")
    controller.set_notes("same text")
    assert controller.undo()
    assert controller.document.notes == ""
    assert not controller.undo()


# --- 0.2: filters must not destroy a SAM/mask selection --------------------


def _filter_state(**overrides) -> SimpleNamespace:
    base = dict(
        raw_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        T_world_cloud=np.eye(4),
        self_filter_toggle=SimpleNamespace(checked=False, enabled=False),
        pzs100_filter_inputs=None,
        pzs100_filter_diagnostics=None,
        self_filter_reason=SimpleNamespace(
            text="Disabled: no PZS100 TF rails or explicit exclusions"
        ),
        status=SimpleNamespace(text=""),
        selected_rgb_points=np.array([[9.0, 9.0, 9.0]]),
        selected_mask=np.ones((2, 2), dtype=bool),
        candidate_sources_result=object(),
        camera=None,
        rgb=None,
        points=np.empty((0, 3)),
        _pzs_status_is_current=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_point_filters_preserves_rgb_mask_selection(monkeypatch) -> None:
    window = _filter_state(
        self_filter_toggle=SimpleNamespace(checked=True, enabled=True),
        pzs100_filter_inputs=(object(),),
    )
    kept = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    monkeypatch.setattr(
        annotator, "filter_pzs100_gripper", lambda points, *_a: (kept, 1, {"centers": []})
    )

    annotator.AnnotatorWindow._apply_point_filters(window)

    # Toggling a cloud filter must not destroy the SAM/mask selection: it is
    # expressed in world coordinates and stays valid until a different
    # snapshot is loaded.
    np.testing.assert_array_equal(window.selected_rgb_points, [[9.0, 9.0, 9.0]])
    assert window.selected_mask is not None
    # Sanity: the filter itself did apply.
    np.testing.assert_array_equal(window.points, kept)


# --- 0.3: an empty working cloud must not be adopted or crash --------------


def test_apply_point_filters_keeps_previous_cloud_when_result_is_empty(monkeypatch) -> None:
    previous_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    window = _filter_state(
        raw_points=np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        self_filter_toggle=SimpleNamespace(checked=True, enabled=True),
        pzs100_filter_inputs=(object(),),
        selected_rgb_points=np.empty((0, 3)),
        selected_mask=None,
        candidate_sources_result=None,
        points=previous_points,
    )
    monkeypatch.setattr(
        annotator, "filter_pzs100_gripper", lambda points, *_a: (np.empty((0, 3)), 2, {})
    )

    annotator.AnnotatorWindow._apply_point_filters(window)

    # A gripper close to the sensor can legitimately mask the whole cloud; the
    # previous working cloud must stay in place instead of handing an empty
    # array to rendering/picking.
    np.testing.assert_array_equal(window.points, previous_points)
    assert "empty cloud" in window.status.text


def test_show_click_ray_handles_empty_points_without_raising() -> None:
    window = SimpleNamespace(
        points=np.empty((0, 3)),
        status=SimpleNamespace(text=""),
        scene=SimpleNamespace(scene=_SceneRecorder(), force_redraw=lambda: None),
    )

    # Previously `np.ptp` on a zero-size array raised ValueError here,
    # escaping the Open3D mouse callback entirely.
    annotator.AnnotatorWindow._show_click_ray(
        window, np.zeros(3), np.array([1.0, 0.0, 0.0]), None, False
    )

    assert any(name == "click-ray" for name, _geometry, _material in window.scene.scene.entries)


# --- 0.4: a failed snapshot load must not leave partially-committed state --


def test_load_snapshot_does_not_commit_state_when_loading_fails(tmp_path, monkeypatch) -> None:
    old_snapshot = tmp_path / "old"
    old_snapshot.mkdir()
    new_snapshot = tmp_path / "new"
    new_snapshot.mkdir()
    (new_snapshot / "annotations.yaml").write_text(
        "schema_version: 2\nsnapshot: new\nannotation_complete: false\nblocks: []\n"
    )

    old_controller = AnnotationController(AnnotationDocument("old", []))
    old_raw_points = np.ones((1, 3))
    window = SimpleNamespace(
        snapshot=old_snapshot,
        controller=old_controller,
        raw_points=old_raw_points,
        config=tmp_path,
    )

    def _raise(_path):
        raise FileNotFoundError("no tf.yaml for this test")

    monkeypatch.setattr(annotator, "load_tf_snapshot", _raise)

    with pytest.raises(FileNotFoundError):
        annotator.AnnotatorWindow._load_snapshot(window, new_snapshot)

    # A failure partway through loading (here: the tf.yaml read) must leave
    # self.snapshot/self.controller/self.raw_points describing the snapshot
    # that was already loaded, not a half-loaded new one.
    assert window.snapshot == old_snapshot
    assert window.controller is old_controller
    assert np.array_equal(window.raw_points, old_raw_points)


def test_load_snapshot_rejects_missing_manifest_without_touching_state() -> None:
    old_controller = AnnotationController(AnnotationDocument("old", []))
    window = SimpleNamespace(
        snapshot=Path("old"), controller=old_controller, raw_points=np.ones((1, 3))
    )

    missing = Path("/nonexistent/snapshot/dir/definitely-not-there")
    with pytest.raises(FileNotFoundError):
        annotator.AnnotatorWindow._load_snapshot(window, missing)

    assert window.snapshot == Path("old")
    assert window.controller is old_controller


# --- Create block from RGB/SAM selection ------------------------------------


def _laid_block_surface(center: np.ndarray, seed: int = 8) -> np.ndarray:
    """A synthetic top+side point cloud for a canonical laid block, axis-aligned.

    Dense enough (450 points) to clear `SnapParams`' default min_points/min_inliers
    so `snap_block_to_cloud` actually succeeds via the native fitter.
    """
    rng = np.random.default_rng(seed)
    half = np.asarray(BLOCK_DIMS) / 2.0
    top = rng.uniform(-half, half, size=(300, 3))
    top[:, 2] = half[2]
    side = rng.uniform(-half, half, size=(150, 3))
    side[:, 1] = -half[1]
    points = np.vstack([top, side]) + center
    points += rng.normal(0.0, 0.004, points.shape)
    return points


def test_create_from_selection_button_disabled_and_labeled_for_empty_selection() -> None:
    window = _gui_window(_document_with_pending_target())
    window.selected_rgb_points = np.empty((0, 3))

    annotator.AnnotatorWindow._update_create_from_selection_button(window)

    assert not window.create_from_selection_button.enabled
    assert "(0 points)" in window.create_from_selection_button.text


def test_create_from_selection_button_label_and_enabled_track_live_selection() -> None:
    window = _gui_window(_document_with_pending_target())
    window.selected_rgb_points = np.ones((317, 3))

    annotator.AnnotatorWindow._update_create_from_selection_button(window)

    assert window.create_from_selection_button.enabled
    assert "(317 points)" in window.create_from_selection_button.text


def test_create_block_from_selection_is_noop_with_empty_selection() -> None:
    window = _gui_window(_document_with_pending_target())
    window.selected_rgb_points = np.empty((0, 3))
    before = len(window.controller.document.targets)

    annotator.AnnotatorWindow._create_block_from_selection(window)

    assert len(window.controller.document.targets) == before
    assert "Select RGB/SAM points" in window.status.text


def test_create_block_from_selection_uses_native_fit_when_it_succeeds() -> None:
    window = _gui_window(_document_with_pending_target())
    center = np.array([1.5, -0.4, 0.9])
    window.selected_rgb_points = _laid_block_surface(center)
    before = len(window.controller.document.targets)

    annotator.AnnotatorWindow._create_block_from_selection(window)

    assert len(window.controller.document.targets) == before + 1
    created = window.controller.target
    assert created is not None
    assert created.provenance == "rgb_mask"
    assert created.dimensions == BLOCK_DIMS
    assert np.linalg.norm(created.position - center) < 0.05
    assert "fitted pending block" in window.status.text

    # One Undo must remove the whole created block, not leave an orphan.
    assert window.controller.undo()
    assert len(window.controller.document.targets) == before


def test_create_block_from_selection_falls_back_to_pca_as_one_undo_step() -> None:
    """Too few points for the native fitter must still produce a valid, undoable block.

    This is the production dead-end scenario: a correct-looking RGB/SAM
    selection whose native fit fails must still become a pending block the
    operator can correct by hand, as one undoable action.
    """
    window = _gui_window(_document_with_pending_target())
    rng = np.random.default_rng(4)
    window.selected_rgb_points = rng.uniform(-0.2, 0.2, (10, 3)) + np.array([3.0, 1.0, 0.5])
    before = len(window.controller.document.targets)

    annotator.AnnotatorWindow._create_block_from_selection(window)

    assert len(window.controller.document.targets) == before + 1
    created = window.controller.target
    assert created is not None
    assert created.provenance == "rgb_mask_pca_fallback"
    assert created.dimensions == BLOCK_DIMS
    rotation = created.rotation
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) > 0.0
    assert "no exact fit" in window.status.text

    # One Undo must remove the whole created block, not leave an orphan.
    assert window.controller.undo()
    assert len(window.controller.document.targets) == before


def test_apply_rgb_mask_places_a_block_even_when_the_native_fit_fails(monkeypatch) -> None:
    """Clicking a block in RGB must always produce a block.

    Mask selection succeeds (so `_apply_rgb_mask` reaches the native-fit call)
    but the cluster is too sparse for `snap_block_to_cloud` to fit a canonical
    block. That is the ordinary case for a GRIPPED block, whose top face the
    gripper occludes, so the fitter never finds the top/side pair it needs.
    The click path used to bail here and tell the operator to press a button,
    putting a second action in front of exactly the blocks a human had just
    pointed at deliberately.
    """
    window = _gui_window(_document_with_pending_target())
    window._refresh_rgb = lambda: None
    sparse_points = np.random.default_rng(2).uniform(-0.05, 0.05, (10, 3)) + np.array(
        [2.0, 2.0, 2.0]
    )
    indices = np.arange(10)
    window.projection_cache = SimpleNamespace(
        uv=np.tile(np.array([50.0, 60.0]), (10, 1)), depth=np.linspace(1.0, 2.0, 10)
    )

    def _stub_select(*_args, **_kwargs):
        return SimpleNamespace(
            mask=np.ones((1, 1), dtype=bool),
            points=sparse_points,
            success=True,
            reason="",
            indices=indices,
        )

    monkeypatch.setattr(annotator, "select_points_from_mask", _stub_select)
    before = len(window.controller.document.targets)

    annotator.AnnotatorWindow._apply_rgb_mask(window, np.ones((1, 1), dtype=bool), (50, 60), "SAM")

    # One new block, placed approximately, in a single undoable step.
    assert len(window.controller.document.targets) == before + 1
    created = window.controller.document.targets[-1]
    assert created.provenance == "rgb_mask_pca_fallback", "the placement must stay auditable"
    assert created.review_status == "pending"
    assert "no exact fit" in window.status.text
    np.testing.assert_array_equal(window.selected_rgb_points, sparse_points)

    annotator.AnnotatorWindow._undo(window)
    assert len(window.controller.document.targets) == before, "must be one undo step"


# --- notes/ignore-reason: coalesce keystrokes, never lose typed text -------


def test_notes_and_ignore_reason_commit_on_value_changed_not_every_keystroke() -> None:
    """`set_on_text_changed` fires per keystroke; `set_on_value_changed` fires once
    on focus-loss/Enter. Each commit deep-copies the whole document onto the
    undo stack, so wiring notes/ignore-reason to the former turned a 100
    character note into 100 undo entries.
    """
    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    assert "self.notes.set_on_value_changed(self._notes_changed)" in source
    assert "self.ignore_reason.set_on_value_changed(self._ignore_reason_changed)" in source
    assert "self.notes.set_on_text_changed" not in source
    assert "self.ignore_reason.set_on_text_changed" not in source


def test_notes_changed_callback_commits_a_full_burst_as_one_undo_entry() -> None:
    """The value-changed callback fires once per completed edit, so invoking it
    once with the final text (as Open3D does after a burst of typing) must
    produce exactly one undo entry, however long the burst was.
    """
    window = _gui_window(_document_with_pending_target())
    before = len(window.controller._undo)

    annotator.AnnotatorWindow._notes_changed(window, "a whole sentence typed in one sitting")

    assert len(window.controller._undo) == before + 1
    assert window.controller.document.notes == "a whole sentence typed in one sitting"


def test_ignore_reason_changed_callback_commits_a_full_burst_as_one_undo_entry() -> None:
    window = _gui_window(_document_with_pending_target())
    before = len(window.controller._undo)

    annotator.AnnotatorWindow._ignore_reason_changed(window, "occluded by the gripper jaw")

    assert len(window.controller._undo) == before + 1
    assert window.controller.target.ignore_reason == "occluded by the gripper jaw"


def test_save_commits_a_not_yet_blurred_notes_and_ignore_reason_edit(tmp_path) -> None:
    """Save must never lose text that was typed but never lost focus/Enter.

    `set_on_value_changed` alone would drop this text if the operator clicks
    Save immediately after typing without tabbing away first; `_save` must
    explicitly flush the live widget value before writing the manifest.
    """
    window = _gui_window(_document_with_pending_target())
    window.snapshot = tmp_path
    window.notes.text_value = "typed but not yet blurred"
    window.ignore_reason.text_value = "reason typed but not yet blurred"

    assert annotator.AnnotatorWindow._save(window)

    assert window.controller.document.notes == "typed but not yet blurred"
    assert window.controller.target.ignore_reason == "reason typed but not yet blurred"


def test_go_to_snapshot_commits_a_not_yet_blurred_notes_edit_before_the_dirty_check() -> None:
    window = _gui_window(_document_with_pending_target())
    window.snapshot = Path("old-snapshot")
    window.sam_worker = None
    window.notes.text_value = "typed but not yet blurred"
    # The confirm-save dialog is real Open3D UI, out of scope here; the
    # guarantee under test is that the text already reached the document
    # BEFORE this (stubbed) dirty-check/dialog runs at all.
    window._confirm_dirty = lambda _action: None

    annotator.AnnotatorWindow._go_to_snapshot(window, Path("new-snapshot"))

    assert window.controller.document.notes == "typed but not yet blurred"


def test_on_close_commits_a_not_yet_blurred_notes_edit_before_the_dirty_check() -> None:
    window = _gui_window(_document_with_pending_target())
    window.sam_worker = None
    window.window = SimpleNamespace(close=lambda: None)
    window.notes.text_value = "typed but not yet blurred"
    window._confirm_dirty = lambda _action: None

    annotator.AnnotatorWindow._on_close(window)

    assert window.controller.document.notes == "typed but not yet blurred"


# --- RGB pane must never show a pose/selection the 3D scene has moved on from --


def test_refresh_views_refreshes_both_the_scene_and_the_rgb_pane() -> None:
    window = _gui_window(_document_with_pending_target())
    calls: list[str] = []
    window._refresh_scene = lambda: calls.append("scene")
    window._refresh_rgb = lambda: calls.append("rgb")

    annotator.AnnotatorWindow._refresh_views(window)

    assert calls == ["scene", "rgb"]


def test_refresh_rgb_is_a_noop_when_the_snapshot_has_no_rgb_image() -> None:
    window = _gui_window(_document_with_pending_target())
    assert window.rgb is None

    # Must not raise even though none of the RGB-only attributes below are
    # meaningfully populated.
    annotator.AnnotatorWindow._refresh_rgb(window)


@pytest.mark.parametrize(
    "action,prepare",
    [
        ("_approve", lambda w: None),
        ("_reject", lambda w: setattr(w.ignore_reason, "text_value", "clutter")),
        ("_abort", lambda w: None),
        ("_reset", lambda w: w.controller.set_pose(np.ones(3), np.zeros(3))),
        ("_undo", lambda w: w.controller.set_review("approved")),
        ("_redo", lambda w: (w.controller.set_review("approved"), w.controller.undo())),
        ("_add_at_fields", lambda w: None),
    ],
)
def test_target_mutating_actions_refresh_the_rgb_pane_too(action, prepare) -> None:
    """Each of these used to call only `_refresh_scene`, leaving the RGB pane's
    projected wireframes at the block's PREVIOUS pose/status after the 3D
    scene had already moved on.
    """
    window = _gui_window(_document_with_pending_target())
    prepare(window)
    rgb_calls: list[bool] = []
    window._refresh_rgb = lambda: rgb_calls.append(True)

    getattr(annotator.AnnotatorWindow, action)(window)

    assert rgb_calls, f"{action} must refresh the RGB pane, not just the 3D scene"


def test_pose_fields_changed_refreshes_the_rgb_pane_too() -> None:
    window = _gui_window(_document_with_pending_target())
    for field, value in zip(window.fields, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], strict=True):
        field.double_value = value
    rgb_calls: list[bool] = []
    window._refresh_rgb = lambda: rgb_calls.append(True)

    annotator.AnnotatorWindow._pose_fields_changed(window)

    assert rgb_calls
    assert np.allclose(window.controller.target.position, [1.0, 2.0, 3.0])


def test_on_target_selected_refreshes_the_rgb_pane_too() -> None:
    window = _gui_window(_document_with_pending_target())
    window.controller.add_manual(np.array([2.0, 0.0, 0.0]))
    window.target_list.set_items(["prior", "manual_0"])
    window.target_list.selected_index = 0
    rgb_calls: list[bool] = []
    window._refresh_rgb = lambda: rgb_calls.append(True)

    annotator.AnnotatorWindow._on_target_selected(window, "prior", False)

    assert rgb_calls
    assert window.controller.selected == 0


def test_clear_rgb_view_blanks_the_image_widget_without_raising() -> None:
    window = SimpleNamespace(rgb_view=gui.ImageWidget())
    annotator.AnnotatorWindow._clear_rgb_view(window)


def test_load_snapshot_clears_the_rgb_view_when_the_new_snapshot_has_no_rgb_png(
    tmp_path, monkeypatch
) -> None:
    """A snapshot with no rgb.png must not leave the PREVIOUS snapshot's photo
    on screen next to the new snapshot's cloud -- nothing else clears the
    ImageWidget once `self.rgb` has already been set to None.
    """
    snapshot = tmp_path / "no-rgb-snapshot"
    snapshot.mkdir()
    (snapshot / "annotations.yaml").write_text(
        "schema_version: 2\nsnapshot: no-rgb-snapshot\nannotation_complete: false\nblocks: []\n"
    )

    window = _gui_window(_document_with_pending_target())
    window.snapshots = [snapshot]
    window.config = tmp_path
    window.sam_worker = None
    window.rgb_view = gui.ImageWidget()
    window.candidate_status = gui.Label("")
    window.workspace_fields = [gui.NumberEdit(gui.NumberEdit.DOUBLE) for _ in range(6)]
    window.max_range_field = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    window.snapshot_combo = SimpleNamespace(selected_index=0)
    window._changing_snapshot_combo = False
    window._load_self_filter_boxes = lambda _snapshot, _tf: None
    window._apply_point_filters = lambda: None
    window._update_nav_buttons = lambda: None
    window._refresh_scene = lambda reset_camera=False: None
    cleared: list[bool] = []
    window._clear_rgb_view = lambda: cleared.append(True)

    raw_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    monkeypatch.setattr(
        annotator,
        "load_tf_snapshot",
        lambda _path: SimpleNamespace(T_world_cloud=np.eye(4), T_world_camera=np.eye(4)),
    )
    monkeypatch.setattr(annotator, "load_snapshot", lambda *_a, **_k: (raw_points, None))

    annotator.AnnotatorWindow._load_snapshot(window, snapshot)

    assert cleared
    assert window.rgb is None


# --- Rotate 90: apply the rotation matrix directly, no Euler round-trip ----


def test_rotate_selected_90_refreshes_the_rgb_pane_too() -> None:
    window = _gui_window(_document_with_pending_target())
    rgb_calls: list[bool] = []
    window._refresh_rgb = lambda: rgb_calls.append(True)

    annotator.AnnotatorWindow._rotate_selected_90(window)

    assert rgb_calls


def test_rotate_selected_90_uses_set_pose_matrix_not_a_euler_round_trip() -> None:
    """`_rotate_selected_90` must call `set_pose_matrix`, not `set_pose` (which
    would rebuild the rotation from "xyz" Euler angles -- exactly the
    gimbal-lock singularity for a STANDING block's vertical long axis).
    """
    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    rotate_start = source.index("def _rotate_selected_90")
    rotate_body = source[rotate_start : source.index("\n    def ", rotate_start + 1)]
    assert "self.controller.set_pose_matrix(" in rotate_body
    assert "self.controller.set_pose(" not in rotate_body


def test_repeated_rotate_selected_90_stays_exact_for_a_standing_block() -> None:
    document = AnnotationDocument(
        "scene",
        [
            EditableTarget(
                "standing",
                np.zeros(3),
                # Canonical long axis (local +X) vertical: exactly the "xyz"
                # Euler gimbal-lock case the tool's docs call out.
                np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            )
        ],
    )
    window = _gui_window(document)
    window._refresh_rgb = lambda: None
    starting_rotation = window.controller.target.rotation.copy()

    for _ in range(4):
        annotator.AnnotatorWindow._rotate_selected_90(window)

    # Four +90 degree yaw steps return exactly to the start, with no drift
    # from decomposing to Euler angles and rebuilding.
    np.testing.assert_allclose(window.controller.target.rotation, starting_rotation, atol=1e-12)


# --- visibility must go through the same public-method path as everything else --


def test_metadata_changed_uses_public_set_visibility() -> None:
    window = _gui_window(_document_with_pending_target())
    assert window.controller.target.visibility == "unknown"

    annotator.AnnotatorWindow._metadata_changed(window, "full", VISIBILITIES.index("full"))

    assert window.controller.target.visibility == "full"
    assert window.controller.dirty
    assert window.controller.undo()
    assert window.controller.target.visibility == "unknown"


def test_metadata_changed_noops_while_loading_widgets() -> None:
    window = _gui_window(_document_with_pending_target())
    window._loading_widgets = True

    annotator.AnnotatorWindow._metadata_changed(window, "partial", VISIBILITIES.index("partial"))

    assert window.controller.target.visibility == "unknown"
    assert not window.controller.dirty


def test_metadata_changed_does_not_reach_into_the_private_change_method() -> None:
    source = Path("tools/annotate_snapshot.py").read_text(encoding="utf-8")
    assert "self.controller._change(" not in source
