from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from blockpose.annotation_tool import (
    AnnotationController,
    AnnotationDocument,
    EditableTarget,
    camera_ray,
    initial_camera_focus,
    nearest_point_to_ray,
    nearest_point_to_ray_with_distance,
    orientation_summary,
    pick_oriented_box,
    save_annotations_atomic,
    snapshot_selection_index,
)
from blockpose.annotations import load_annotations
from scipy.spatial.transform import Rotation


def _document() -> AnnotationDocument:
    return AnnotationDocument(
        "scene",
        [EditableTarget("prior", np.zeros(3), np.eye(3), provenance="upstream_registration")],
    )


def test_schema_v1_priors_load_pending(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "annotations.yaml"
    manifest.write_text(
        """schema_version: 1
snapshot: scene
annotation_complete: false
annotated_roi: null
blocks:
- id: prior
  provenance: upstream_registration
  pose: {position: [0, 0, 0], orientation_xyzw: [0, 0, 0, 1]}
"""
    )
    loaded = load_annotations(manifest)
    assert loaded.targets[0].review_status == "pending"
    assert not loaded.targets[0].is_evaluation_target


def test_document_load_migrates_permuted_dimensions_preserving_world_obb(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    rotation = Rotation.from_euler("xyz", [13, -27, 41], degrees=True).as_matrix()
    manifest = scene / "annotations.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "snapshot": "scene",
                "annotation_complete": False,
                "blocks": [{
                    "id": "upright",
                    "dimensions": [0.6, 0.6, 0.9],
                    "pose": {
                        "position": [1, 2, 3],
                        "orientation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
                    },
                }],
            }
        )
    )
    original_bytes = manifest.read_bytes()
    signs = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    )
    before = sorted(map(tuple, np.round((signs * [0.6, 0.6, 0.9]) @ rotation.T, 12)))
    document = AnnotationDocument.load(manifest)
    target = document.targets[0]
    after = sorted(map(tuple, np.round((signs * target.dimensions) @ target.rotation.T, 12)))
    assert target.dimensions == (0.9, 0.6, 0.6)
    assert before == after
    assert manifest.read_bytes() == original_bytes


def test_document_load_rejects_noncanonical_physical_dimensions(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "annotations.yaml"
    manifest.write_text(
        """schema_version: 2
snapshot: scene
annotation_complete: false
blocks:
- id: wrong
  dimensions: [1.0, 0.6, 0.6]
  pose: {position: [0, 0, 0], orientation_xyzw: [0, 0, 0, 1]}
"""
    )
    with np.testing.assert_raises_regex(ValueError, "not a permutation"):
        AnnotationDocument.load(manifest)


def test_orientation_summary_uses_canonical_long_axis_and_is_symmetry_stable() -> None:
    laying = np.eye(3)
    standing = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    assert orientation_summary(laying).startswith("Laying")
    assert "elevation 0.0" in orientation_summary(laying)
    assert orientation_summary(standing).startswith("Standing")
    assert "elevation 90.0" in orientation_summary(standing)
    flipped = laying @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
    assert orientation_summary(flipped) == orientation_summary(laying)


def test_controller_edit_review_and_undo_redo() -> None:
    controller = AnnotationController(_document())
    controller.set_pose(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    controller.set_review("approved")
    assert controller.target is not None
    assert controller.target.review_status == "approved"
    assert np.allclose(controller.target.position, [1, 2, 3])
    assert controller.undo()
    assert controller.target.review_status == "pending"
    assert controller.redo()
    assert controller.target.review_status == "approved"


def test_delete_selected_middle_is_undoable_redoable_and_serialized() -> None:
    targets = [
        EditableTarget(name, np.array([float(index), 0, 0]), np.eye(3), review_status=status)
        for index, (name, status) in enumerate(
            (("first", "pending"), ("middle", "approved"), ("last", "rejected"))
        )
    ]
    targets[-1].rejection_reason = "clutter"
    controller = AnnotationController(AnnotationDocument("scene", targets))
    controller.selected = 1
    assert controller.delete_selected(expected_id="middle", expected_status="approved")
    assert [item.id for item in controller.document.targets] == ["first", "last"]
    assert controller.target is not None and controller.target.id == "last"
    assert [item["id"] for item in controller.document.as_mapping()["blocks"]] == [
        "first",
        "last",
    ]
    assert controller.undo()
    assert [item.id for item in controller.document.targets] == ["first", "middle", "last"]
    assert controller.target is not None and controller.target.id == "middle"
    assert controller.redo()
    assert [item.id for item in controller.document.targets] == ["first", "last"]


def test_delete_selected_handles_last_only_and_confirmation_mismatch() -> None:
    controller = AnnotationController(
        AnnotationDocument(
            "scene",
            [
                EditableTarget("first", np.zeros(3), np.eye(3)),
                EditableTarget("last", np.ones(3), np.eye(3), review_status="approved"),
            ],
        )
    )
    controller.selected = 1
    assert not controller.delete_selected(expected_id="wrong", expected_status="approved")
    assert not controller.dirty
    assert not controller.delete_selected(expected_id="last", expected_status="pending")
    assert len(controller.document.targets) == 2
    assert controller.delete_selected(expected_id="last", expected_status="approved")
    assert controller.selected == 0
    assert controller.delete_selected(expected_id="first", expected_status="pending")
    assert controller.selected is None and controller.target is None
    assert not controller.delete_selected()
    assert controller.undo()
    assert controller.target is not None and controller.target.id == "first"


def test_delete_selected_accepts_each_review_status() -> None:
    for status in ("pending", "approved", "rejected"):
        target = EditableTarget("candidate", np.zeros(3), np.eye(3), review_status=status)
        if status == "rejected":
            target.rejection_reason = "not a block"
        controller = AnnotationController(AnnotationDocument("scene", [target]))
        assert controller.delete_selected(expected_id="candidate", expected_status=status)
        assert not controller.document.targets


def test_ignore_reason_is_immediately_bound_and_undoable() -> None:
    controller = AnnotationController(_document())
    controller.set_ignore_reason("  too occluded  ")
    assert controller.target is not None
    assert controller.target.ignore_reason == "too occluded"
    assert controller.dirty
    assert controller.undo()
    assert controller.target.ignore_reason is None


def test_abort_retracts_new_manual_draft_but_restores_existing_prior() -> None:
    controller = AnnotationController(_document())
    controller.add_manual(np.ones(3))
    assert len(controller.document.targets) == 2
    controller.abort_selected()
    assert [target.id for target in controller.document.targets] == ["prior"]
    assert controller.undo()
    assert len(controller.document.targets) == 2
    controller.selected = 0
    controller.set_pose(np.ones(3), np.zeros(3))
    controller.abort_selected()
    assert controller.target is not None
    assert np.array_equal(controller.target.position, np.zeros(3))
    assert controller.target.review_status == "pending"


def test_reset_returns_to_original_prior_not_previous_increment() -> None:
    controller = AnnotationController(_document())
    controller.set_review("approved")
    controller.set_pose(np.array([1.0, 0, 0]), np.zeros(3))
    controller.set_pose(np.array([2.0, 0, 0]), np.zeros(3))
    controller.reset_selected()
    assert controller.target is not None
    assert np.array_equal(controller.target.position, np.zeros(3))
    assert controller.target.review_status == "approved"


def test_successful_save_rebaselines_reset_and_abort() -> None:
    controller = AnnotationController(_document())
    controller.set_pose(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    controller.set_review("approved")
    controller.mark_saved()
    assert not controller.undo()
    controller.set_pose(np.array([4.0, 5.0, 6.0]), np.zeros(3))
    controller.reset_selected()
    assert controller.target is not None
    assert np.array_equal(controller.target.position, [1.0, 2.0, 3.0])
    assert controller.target.review_status == "approved"


def test_snapshot_selection_index_restores_loaded_snapshot() -> None:
    snapshots = [Path("first"), Path("second")]
    assert snapshot_selection_index(snapshots, Path("first")) == 0
    try:
        snapshot_selection_index(snapshots, Path("missing"))
    except ValueError as error:
        assert "not in the snapshot list" in str(error)
    else:
        raise AssertionError("missing current snapshot accepted")


def test_ray_box_picking_chooses_nearest() -> None:
    targets = [
        EditableTarget("far", np.array([5.0, 0, 0]), np.eye(3)),
        EditableTarget("near", np.array([2.0, 0, 0]), np.eye(3)),
    ]
    assert pick_oriented_box(np.zeros(3), np.array([1.0, 0, 0]), targets) == 1
    assert pick_oriented_box(np.zeros(3), np.array([0.0, 1, 0]), targets) is None


def test_point_ray_pick_rejects_empty_area_and_points_behind_camera() -> None:
    points = np.array([[2.0, 0.02, 0.0], [1.0, 0.2, 0.0], [-1.0, 0.0, 0.0]])
    picked = nearest_point_to_ray(np.zeros(3), np.array([1.0, 0, 0]), points, 0.08)
    assert picked is not None and np.array_equal(picked, points[0])
    assert nearest_point_to_ray(np.zeros(3), np.array([0, 1.0, 0]), points, 0.08) is None
    candidate = nearest_point_to_ray_with_distance(
        np.zeros(3), np.array([1.0, 0, 0]), points
    )
    assert candidate is not None
    assert np.array_equal(candidate[0], points[0])
    assert np.isclose(candidate[1], 0.02)


def test_camera_ray_normalizes_and_rejects_degenerate_endpoints() -> None:
    origin, direction = camera_ray(np.zeros(3), np.array([2.0, 0.0, 0.0]))
    assert np.array_equal(origin, np.zeros(3))
    assert np.array_equal(direction, np.array([1.0, 0.0, 0.0]))
    for point in (np.zeros(3), np.array([np.nan, 0.0, 0.0])):
        try:
            camera_ray(np.zeros(3), point)
        except ValueError as error:
            assert "degenerate" in str(error)
        else:
            raise AssertionError("invalid camera ray accepted")


def test_initial_camera_focus_uses_robust_forward_depth_and_preserves_direction() -> None:
    points = np.array([[-2.0, 20, 0], [-4.0, -20, 3], [-6.0, 0, 1], [3.0, 0, 0]])
    focus = initial_camera_focus(points, np.zeros(3), np.array([-1.0, 0, 0]))
    assert np.allclose(focus, [-4.0, 0, 0])


def test_initial_camera_focus_fallback_clamp_and_invalid_inputs() -> None:
    assert np.allclose(
        initial_camera_focus(
            np.array([[np.nan, 0, 0], [2.0, 0, 0]]),
            np.zeros(3),
            np.array([-1.0, 0, 0]),
            fallback_depth=80,
            max_depth=20,
        ),
        [-20.0, 0, 0],
    )
    assert np.allclose(
        initial_camera_focus(
            np.array([[-0.1, 0, 0]]), np.zeros(3), np.array([-1.0, 0, 0]), min_depth=1
        ),
        [-1.0, 0, 0],
    )
    with np.testing.assert_raises_regex(ValueError, "direction"):
        initial_camera_focus(np.empty((0, 3)), np.zeros(3), np.zeros(3))


def test_point_ray_and_manual_draft_reject_nonfinite_coordinates() -> None:
    points = np.array([[np.nan, 0, 0], [1.0, 0, 0]])
    assert nearest_point_to_ray(np.full(3, np.nan), np.ones(3), points, 0.1) is None
    assert nearest_point_to_ray(np.zeros(3), np.full(3, np.inf), points, 0.1) is None
    assert nearest_point_to_ray(np.zeros(3), np.array([1.0, 0, 0]), points, 0.1) is not None
    controller = AnnotationController(_document())
    try:
        controller.add_manual(np.array([np.nan, 0, 0]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite manual draft accepted")


def test_complete_document_requires_resolved_targets_and_roi() -> None:
    document = _document()
    document.annotation_complete = True
    assert "requires an ROI" in "; ".join(document.validate())
    document.roi_min = np.array([-1.0, -1.0, -1.0])
    document.roi_max = np.array([1.0, 1.0, 1.0])
    assert "pending prior" in "; ".join(document.validate())


def test_atomic_save_writes_v2_backup_and_round_trips(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "annotations.yaml"
    manifest.write_text("old manifest\n")
    document = _document()
    controller = AnnotationController(document)
    controller.set_review("approved")
    backup = save_annotations_atomic(
        document, manifest, timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    )
    assert backup is not None
    assert backup.read_text() == "old manifest\n"
    raw = yaml.safe_load(manifest.read_text())
    assert raw["schema_version"] == 2
    loaded = load_annotations(manifest)
    assert loaded.targets[0].is_evaluation_target


def test_rejected_target_outside_complete_roi_saves_and_reloads(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "annotations.yaml"
    document = AnnotationDocument(
        "scene",
        [
            EditableTarget(
                "rejected",
                np.array([5.0, 5.0, 5.0]),
                np.eye(3),
                review_status="rejected",
                rejection_reason="not a block",
            )
        ],
        annotation_complete=True,
        roi_min=np.zeros(3),
        roi_max=np.ones(3),
    )
    save_annotations_atomic(document, manifest)
    loaded = load_annotations(manifest)
    assert loaded.annotation_complete
    assert loaded.targets[0].review_status == "rejected"


def test_schema_v2_rejection_requires_reason(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest = scene / "annotations.yaml"
    manifest.write_text(
        """schema_version: 2
snapshot: scene
annotation_complete: false
blocks:
- id: rejected
  review_status: rejected
  pose: {position: [0, 0, 0], orientation_xyzw: [0, 0, 0, 1]}
"""
    )
    try:
        load_annotations(manifest)
    except ValueError as error:
        assert "requires rejection_reason" in str(error)
    else:
        raise AssertionError("invalid rejection loaded")
