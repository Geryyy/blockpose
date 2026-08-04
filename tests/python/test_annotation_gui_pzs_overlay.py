"""Headless coverage for the PZS100 filter context drawn by the annotator."""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
import tools.annotate_snapshot as annotator  # noqa: E402


class _SceneRecorder:
    def __init__(self) -> None:
        self.entries: list[tuple[str, object, object]] = []

    def add_geometry(self, name: str, geometry: object, material: object) -> None:
        self.entries.append((name, geometry, material))


class _ClearableCombo:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear_items(self) -> None:
        self.clear_count += 1


def _diagnostics(*, with_above_gripper: bool = False) -> dict[str, np.ndarray]:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    diagnostics: dict[str, np.ndarray] = {
        "centers": np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]]),
        "rotations": np.array([rotation, np.eye(3)]),
        "sizes": np.array([[0.8, 0.4, 1.2], [1.1, 0.5, 0.3]]),
        "rail_starts": np.array([[0.0, 0.0, 0.0], [-3.0, 1.0, 0.5]]),
        "rail_ends": np.array([[1.0, 0.0, 0.0], [-2.0, 1.0, 0.5]]),
    }
    if with_above_gripper:
        # A newer native core's diagnostics dict additionally carries the
        # above-gripper "dead zone" volume(s); older builds simply lack these
        # three keys (see the absent-keys tests below).
        diagnostics["above_centers"] = np.array([[0.5, -0.5, 1.6]])
        diagnostics["above_rotations"] = np.array([np.eye(3)])
        diagnostics["above_sizes"] = np.array([[0.6, 0.9, 0.4]])
    return diagnostics


def _overlay_window(diagnostics: dict[str, np.ndarray] | None) -> tuple[object, _SceneRecorder]:
    recorder = _SceneRecorder()
    window = SimpleNamespace(
        pzs100_filter_diagnostics=diagnostics,
        scene=SimpleNamespace(scene=recorder),
        status=SimpleNamespace(text=""),
    )
    # `_add_pzs100_filter_geometries` delegates to this real method; bind it
    # onto the fake window the same way production `self._method()` calls
    # resolve, so this test drives the actual production code.
    window._add_above_gripper_geometry = MethodType(
        annotator.AnnotatorWindow._add_above_gripper_geometry, window
    )
    return window, recorder


def _assert_same_points(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) < 1e-9


def test_pzs_overlay_draws_native_rails_and_oriented_box_outlines() -> None:
    diagnostics = _diagnostics()
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    assert [name for name, _geometry, _material in recorder.entries] == [
        "pzs-filter-rails",
        "pzs-filter-outline-0",
        "pzs-filter-outline-1",
    ]
    rails, rail_geometry, rail_material = recorder.entries[0]
    assert rails == "pzs-filter-rails"
    np.testing.assert_allclose(
        np.asarray(rail_geometry.points),
        np.vstack((diagnostics["rail_starts"], diagnostics["rail_ends"])),
    )
    np.testing.assert_array_equal(np.asarray(rail_geometry.lines), [[0, 2], [1, 3]])
    assert rail_material.shader == "unlitLine"
    assert rail_material.line_width == annotator.PZS_FILTER_RAIL_WIDTH

    _name, outline, outline_material = recorder.entries[1]
    assert outline_material.shader == "unlitLine"
    assert outline_material.line_width == annotator.PZS_FILTER_OUTLINE_WIDTH
    expected = annotator.cuboid_corners(
        diagnostics["centers"][0], diagnostics["rotations"][0], diagnostics["sizes"][0]
    )
    _assert_same_points(np.asarray(outline.points), expected)


def test_pzs_overlay_malformed_native_diagnostics_fail_safe() -> None:
    diagnostics = _diagnostics()
    diagnostics.pop("rail_ends")
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    assert not recorder.entries
    assert window.pzs100_filter_diagnostics is None
    assert window.status.text.startswith("PZS100 filter active; overlay unavailable")


def test_pzs_overlay_draws_above_gripper_volume_when_present() -> None:
    diagnostics = _diagnostics(with_above_gripper=True)
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    names = [name for name, _geometry, _material in recorder.entries]
    assert names == [
        "pzs-filter-rails",
        "pzs-filter-outline-0",
        "pzs-filter-outline-1",
        "pzs-filter-above-0",
    ]
    _name, above_outline, above_material = recorder.entries[-1]
    assert above_material.shader == "unlitLine"
    assert above_material.line_width == annotator.PZS_FILTER_ABOVE_GRIPPER_WIDTH
    np.testing.assert_allclose(
        np.asarray(above_material.base_color), annotator.PZS_FILTER_ABOVE_GRIPPER_COLOR
    )
    # Distinct from both existing overlay colours, so the dead zone reads as
    # its own thing rather than blending into the rails or the mask outline.
    assert not np.allclose(
        np.asarray(above_material.base_color), annotator.PZS_FILTER_OUTLINE_COLOR
    )
    assert not np.allclose(np.asarray(above_material.base_color), annotator.PZS_FILTER_RAIL_COLOR)
    expected = annotator.cuboid_corners(
        diagnostics["above_centers"][0],
        diagnostics["above_rotations"][0],
        diagnostics["above_sizes"][0],
    )
    _assert_same_points(np.asarray(above_outline.points), expected)


def test_pzs_overlay_above_gripper_keys_absent_still_draws_rails_and_outlines() -> None:
    """An older compiled `_core` .so lacks the above-gripper keys entirely.

    That must degrade to "no dead-zone overlay", not "no overlay at all" --
    the existing rail/outline boxes (from the always-present keys) must still
    render exactly as before.
    """
    diagnostics = _diagnostics(with_above_gripper=False)
    assert "above_centers" not in diagnostics
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    names = [name for name, _geometry, _material in recorder.entries]
    assert names == ["pzs-filter-rails", "pzs-filter-outline-0", "pzs-filter-outline-1"]
    assert not any(name.startswith("pzs-filter-above") for name in names)
    assert window.pzs100_filter_diagnostics is not None
    assert window.status.text == ""


def test_pzs_overlay_above_gripper_empty_when_feature_off_draws_nothing_extra() -> None:
    """Empty-shaped `above_*` arrays (feature off) must draw zero above-gripper boxes."""
    diagnostics = _diagnostics(with_above_gripper=True)
    diagnostics["above_centers"] = np.empty((0, 3))
    diagnostics["above_rotations"] = np.empty((0, 3, 3))
    diagnostics["above_sizes"] = np.empty((0, 3))
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    names = [name for name, _geometry, _material in recorder.entries]
    assert names == ["pzs-filter-rails", "pzs-filter-outline-0", "pzs-filter-outline-1"]


def test_pzs_overlay_malformed_above_gripper_keys_keep_rails_and_outlines() -> None:
    """A malformed optional above-gripper triple must not take down the whole overlay."""
    diagnostics = _diagnostics(with_above_gripper=True)
    diagnostics["above_sizes"] = np.array([[-0.1, 0.9, 0.4]])  # non-positive size: invalid
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    names = [name for name, _geometry, _material in recorder.entries]
    assert names == ["pzs-filter-rails", "pzs-filter-outline-0", "pzs-filter-outline-1"]
    # Unlike a malformed *required* key, the rail/outline overlay this pass
    # already drew successfully must survive: only the optional dead-zone
    # decoration is skipped.
    assert window.pzs100_filter_diagnostics is not None


def _filter_window(*, pzs_inputs: object | None, checked: bool = True) -> object:
    return SimpleNamespace(
        pzs100_filter_diagnostics=_diagnostics(),
        raw_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        self_filter_toggle=SimpleNamespace(checked=checked, enabled=True),
        pzs100_filter_inputs=pzs_inputs,
        candidate_sources_result=object(),
        candidate_combo=_ClearableCombo(),
        adopt_candidate_button=SimpleNamespace(enabled=True),
        camera=None,
        rgb=None,
        rgb_colors=None,
        projection_cache=object(),
        selected_rgb_points=np.ones((1, 3)),
        selected_mask=np.ones((1, 1), dtype=bool),
        status=SimpleNamespace(text=""),
        self_filter_reason=SimpleNamespace(text=""),
        T_world_cloud=None,
        _pzs_status_is_current=False,
    )


def test_pzs_overlay_lifecycle_clears_when_toggled_off_or_native_filter_fails(monkeypatch) -> None:
    inputs = (np.zeros((1, 3)),) * 4 + (object(),)
    window = _filter_window(pzs_inputs=inputs)
    filtered = np.array([[1.0, 0.0, 0.0]])

    monkeypatch.setattr(
        annotator,
        "filter_pzs100_gripper",
        lambda _points, *_inputs: (filtered, 1, _diagnostics()),
    )
    annotator.AnnotatorWindow._apply_point_filters(window)
    np.testing.assert_allclose(window.points, filtered)
    assert window.pzs100_filter_diagnostics is not None

    window.self_filter_toggle.checked = False
    annotator.AnnotatorWindow._apply_point_filters(window)
    assert window.pzs100_filter_diagnostics is None
    assert window.status.text == "PZS100 gripper filter disabled."

    window.self_filter_toggle.checked = True
    window.pzs100_filter_diagnostics = _diagnostics()

    def fail(*_args):
        raise RuntimeError("stale extension")

    monkeypatch.setattr(annotator, "filter_pzs100_gripper", fail)
    annotator.AnnotatorWindow._apply_point_filters(window)
    assert window.pzs100_filter_diagnostics is None
    assert window.pzs100_filter_inputs is None
    assert not window.self_filter_toggle.checked
    assert window.self_filter_reason.text == (
        "Disabled: PZS100 native filter failed (stale extension)"
    )
    assert window.status.text == "PZS100 gripper filter failed: stale extension"


def test_pzs_overlay_lifecycle_reports_missing_rails_instead_of_stale_success() -> None:
    window = _filter_window(pzs_inputs=None, checked=False)
    window.self_filter_toggle.enabled = False
    window.self_filter_reason.text = "Disabled: captured K9/K11 rails unavailable"
    window.status.text = "PZS100 gripper removed: 42 points"

    annotator.AnnotatorWindow._apply_point_filters(window)

    assert window.pzs100_filter_diagnostics is None
    assert window.status.text == (
        "PZS100 gripper filter unavailable: captured K9/K11 rails unavailable"
    )


def _candidates_window(*, pzs_diagnostics: dict[str, np.ndarray] | None) -> object:
    window = SimpleNamespace(
        points=np.zeros((1, 3)),
        rgb=None,
        T_world_camera=None,
        camera=None,
        T_world_cloud=None,
        args=SimpleNamespace(dino_proposer_checkpoint=None, dino_verifier_checkpoint=None),
        pzs100_filter_diagnostics=pzs_diagnostics,
        candidate_status=SimpleNamespace(text=""),
        candidate_sources_result=None,
        snapshot=SimpleNamespace(name="scene-1"),
        controller=SimpleNamespace(adopted_candidate_ids=set()),
        _refresh_views=lambda: None,
    )
    # `_compute_candidates` calls `self._promote_accepted_candidates()`; bind
    # the real implementation the same way production `self._method()` calls
    # resolve (the stub result exposes no candidate sources, so this returns
    # `[]` without touching a document).
    window._active_candidate_sources = MethodType(
        annotator.AnnotatorWindow._active_candidate_sources, window
    )
    window._promote_accepted_candidates = MethodType(
        annotator.AnnotatorWindow._promote_accepted_candidates, window
    )
    return window


def test_compute_candidates_passes_pzs_diagnostics_as_above_gripper_boxes(monkeypatch) -> None:
    """The GUI must wire the live PZS100 diagnostics into candidate rejection.

    `compute_candidate_sources(above_gripper_boxes=...)` rejects hypotheses
    centred in the above-gripper dead zone; the GUI previously never passed
    it, so gripper-region candidates were never actually rejected.
    """
    captured: dict[str, object] = {}

    def _stub_compute(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(warnings=[])

    monkeypatch.setattr(annotator, "compute_candidate_sources", _stub_compute)
    diagnostics = _diagnostics(with_above_gripper=True)
    window = _candidates_window(pzs_diagnostics=diagnostics)

    annotator.AnnotatorWindow._compute_candidates(window)

    assert captured["above_gripper_boxes"] is diagnostics
    assert "Computed candidates for scene-1." in window.candidate_status.text


def test_compute_candidates_passes_none_above_gripper_boxes_when_filter_off(monkeypatch) -> None:
    """`above_gripper_boxes=None` (filter off/unavailable) must remain a no-op, not an error."""
    captured: dict[str, object] = {}

    def _stub_compute(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(warnings=[])

    monkeypatch.setattr(annotator, "compute_candidate_sources", _stub_compute)
    window = _candidates_window(pzs_diagnostics=None)

    annotator.AnnotatorWindow._compute_candidates(window)

    assert captured["above_gripper_boxes"] is None
    assert "Computed candidates for scene-1." in window.candidate_status.text


def test_pzs_overlay_lifecycle_clears_when_loading_a_snapshot_without_native_filter(
    monkeypatch, tmp_path
) -> None:
    window = SimpleNamespace(
        pzs100_filter_diagnostics=_diagnostics(),
        pzs100_filter_inputs=object(),
        self_filter_toggle=SimpleNamespace(checked=True, enabled=True),
        self_filter_reason=SimpleNamespace(text="ready"),
    )
    monkeypatch.setattr(annotator, "pzs100_self_filter_available", lambda: (False, "old core"))

    annotator.AnnotatorWindow._load_self_filter_boxes(window, tmp_path, object())

    assert window.pzs100_filter_diagnostics is None
    assert window.pzs100_filter_inputs is None
    assert not window.self_filter_toggle.checked
    assert not window.self_filter_toggle.enabled
    assert window.self_filter_reason.text == "Disabled: old core"
