"""Headless coverage for the PZS100 filter context drawn by the annotator."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _diagnostics() -> dict[str, np.ndarray]:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return {
        "centers": np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]]),
        "rotations": np.array([rotation, np.eye(3)]),
        "sizes": np.array([[0.8, 0.4, 1.2], [1.1, 0.5, 0.3]]),
        "rail_starts": np.array([[0.0, 0.0, 0.0], [-3.0, 1.0, 0.5]]),
        "rail_ends": np.array([[1.0, 0.0, 0.0], [-2.0, 1.0, 0.5]]),
        "outward_normals": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    }


def _overlay_window(diagnostics: dict[str, np.ndarray] | None) -> tuple[object, _SceneRecorder]:
    recorder = _SceneRecorder()
    window = SimpleNamespace(
        pzs100_filter_diagnostics=diagnostics,
        scene=SimpleNamespace(scene=recorder),
        status=SimpleNamespace(text=""),
    )
    return window, recorder


def _assert_same_points(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) < 1e-9


def test_pzs_overlay_draws_native_rails_and_translucent_oriented_volumes() -> None:
    diagnostics = _diagnostics()
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    assert [name for name, _geometry, _material in recorder.entries] == [
        "pzs-filter-rails",
        "pzs-filter-volume-0",
        "pzs-filter-outline-0",
        "pzs-filter-volume-1",
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

    _name, volume, volume_material = recorder.entries[1]
    assert volume_material.shader == "defaultUnlitTransparency"
    np.testing.assert_allclose(volume_material.base_color, annotator.PZS_FILTER_VOLUME_COLOR)
    assert volume_material.has_alpha
    _name, outline, outline_material = recorder.entries[2]
    assert outline_material.shader == "unlitLine"
    assert outline_material.line_width == annotator.PZS_FILTER_OUTLINE_WIDTH
    expected = annotator.cuboid_corners(
        diagnostics["centers"][0], diagnostics["rotations"][0], diagnostics["sizes"][0]
    )
    _assert_same_points(np.asarray(outline.points), expected)
    _assert_same_points(np.asarray(volume.vertices), expected)


def test_pzs_overlay_malformed_native_diagnostics_fail_safe() -> None:
    diagnostics = _diagnostics()
    diagnostics.pop("rail_ends")
    window, recorder = _overlay_window(diagnostics)

    annotator.AnnotatorWindow._add_pzs100_filter_geometries(window)

    assert not recorder.entries
    assert window.pzs100_filter_diagnostics is None
    assert window.status.text.startswith("PZS100 filter active; overlay unavailable")


def _filter_window(*, pzs_inputs: object | None, checked: bool = True) -> object:
    return SimpleNamespace(
        pzs100_filter_diagnostics=_diagnostics(),
        raw_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        remove_ground_toggle=SimpleNamespace(checked=False),
        workspace_toggle=SimpleNamespace(checked=False),
        remove_background_toggle=SimpleNamespace(checked=False),
        self_filter_toggle=SimpleNamespace(checked=checked, enabled=True),
        pzs100_filter_inputs=pzs_inputs,
        self_filter_boxes=None,
        candidate_sources_result=object(),
        candidate_choices=[("pcl", 0, object())],
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


def test_pzs_overlay_lifecycle_clears_when_loading_a_snapshot_without_native_filter(
    monkeypatch, tmp_path
) -> None:
    window = SimpleNamespace(
        pzs100_filter_diagnostics=_diagnostics(),
        pzs100_filter_inputs=object(),
        self_filter_boxes=object(),
        self_filter_toggle=SimpleNamespace(checked=True, enabled=True),
        self_filter_reason=SimpleNamespace(text="ready"),
    )
    monkeypatch.setattr(annotator, "pzs100_self_filter_available", lambda: (False, "old core"))

    annotator.AnnotatorWindow._load_self_filter_boxes(window, tmp_path, object())

    assert window.pzs100_filter_diagnostics is None
    assert window.pzs100_filter_inputs is None
    assert window.self_filter_boxes is None
    assert not window.self_filter_toggle.checked
    assert not window.self_filter_toggle.enabled
    assert window.self_filter_reason.text == "Disabled: old core"
