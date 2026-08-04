from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import yaml
from blockpose.annotation_tool import (
    AnnotationController,
    AnnotationDocument,
    EditableTarget,
    adjacent_snapshot,
    camera_ray,
    default_max_range_m,
    full_workspace_roi,
    initial_camera_focus,
    orientation_summary,
    pca_oriented_block_pose,
    pick_oriented_box,
    remove_background_by_range,
    save_annotations_atomic,
    snapshot_selection_index,
)
from blockpose.annotations import load_annotations
from blockpose.pose import BLOCK_DIMS
from scipy.spatial.transform import Rotation

UTC = timezone.utc


def _document() -> AnnotationDocument:
    return AnnotationDocument(
        "scene",
        [EditableTarget("prior", np.zeros(3), np.eye(3), provenance="upstream_registration")],
    )


def test_full_workspace_roi_contains_cloud_with_margin() -> None:
    points = np.array([[-2.0, 3.0, 1.0], [4.0, -1.0, 5.0]])
    low, high = full_workspace_roi(points)
    np.testing.assert_allclose(low, [-2.05, -1.05, 0.95])
    np.testing.assert_allclose(high, [4.05, 3.05, 5.05])


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
                "blocks": [
                    {
                        "id": "upright",
                        "dimensions": [0.6, 0.6, 0.9],
                        "pose": {
                            "position": [1, 2, 3],
                            "orientation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
                        },
                    }
                ],
            }
        )
    )
    original_bytes = manifest.read_bytes()
    signs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
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
    # Open3D's bundled font renders U+2014 (em dash) as "?"; ASCII only here.
    assert "—" not in orientation_summary(laying)


def test_set_notes_and_set_annotation_complete_are_undoable_and_debounced() -> None:
    controller = AnnotationController(_document())
    assert not controller.dirty

    # A no-op edit (same value) must not push an undo entry, so the operator
    # typing normally does not spam the undo stack with one entry per
    # keystroke.
    controller.set_notes("")
    assert not controller.dirty
    controller.set_annotation_complete(False)
    assert not controller.dirty

    controller.set_notes("occluded on the left side")
    assert controller.dirty
    assert controller.document.notes == "occluded on the left side"
    controller.set_annotation_complete(True)
    assert controller.document.annotation_complete is True

    # Repeating the same value again must still no-op.
    before = len(controller._undo)
    controller.set_notes("occluded on the left side")
    assert len(controller._undo) == before

    assert controller.undo()
    assert controller.document.annotation_complete is False
    assert controller.document.notes == "occluded on the left side"
    assert controller.undo()
    assert controller.document.notes == ""
    assert not controller.undo()


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


def test_set_visibility_is_undoable_and_noops_on_unchanged_or_invalid() -> None:
    controller = AnnotationController(_document())
    assert controller.target is not None
    assert controller.target.visibility == "unknown"

    controller.set_visibility("partial")
    assert controller.target.visibility == "partial"
    assert controller.dirty

    # No-op on an unchanged value: no new undo entry.
    before = len(controller._undo)
    controller.set_visibility("partial")
    assert len(controller._undo) == before

    # No-op on a value outside VISIBILITIES too, rather than corrupting the
    # document with an unvalidated string.
    controller.set_visibility("not-a-real-visibility")
    assert controller.target.visibility == "partial"
    assert len(controller._undo) == before

    assert controller.undo()
    assert controller.target.visibility == "unknown"


def test_set_pose_matrix_applies_rotation_without_euler_round_trip() -> None:
    controller = AnnotationController(_document())
    position = np.array([1.0, 2.0, 3.0])
    rotation = Rotation.from_euler("xyz", [12.0, -34.0, 56.0], degrees=True).as_matrix()

    controller.set_pose_matrix(position, rotation)

    assert controller.target is not None
    assert np.array_equal(controller.target.position, position)
    np.testing.assert_array_equal(controller.target.rotation, rotation)
    assert controller.dirty
    assert controller.undo()
    assert np.array_equal(controller.target.position, np.zeros(3))


def test_set_pose_matrix_rejects_invalid_transform() -> None:
    controller = AnnotationController(_document())
    with np.testing.assert_raises_regex(ValueError, "proper rotation matrix"):
        controller.set_pose_matrix(np.ones(3), np.zeros((3, 3)))
    assert np.array_equal(controller.target.position, np.zeros(3))


def test_repeated_rotate_90_via_set_pose_matrix_stays_exact_for_a_standing_block() -> None:
    """A STANDING block's canonical long axis is vertical -- exactly the "xyz"
    Euler gimbal-lock singularity. Applying the rotation matrix directly
    (`set_pose_matrix`, as `_rotate_selected_90` now does) instead of
    round-tripping through Euler angles must keep the composed rotation
    numerically exact across repeated 90-degree yaw steps, with no drift.
    """
    controller = AnnotationController(_document())
    standing = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    controller.set_pose_matrix(np.zeros(3), standing)
    quarter_turn = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    rotation = standing
    for _ in range(4):
        rotation = quarter_turn @ rotation
        controller.set_pose_matrix(controller.target.position, rotation)
        np.testing.assert_array_equal(controller.target.rotation, rotation)

    # Four quarter-turns about the same axis return exactly to the start.
    np.testing.assert_allclose(controller.target.rotation, standing, atol=1e-12)


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


def test_add_manual_with_pose_is_one_undo_step_leaving_no_orphan() -> None:
    """The old add_manual + apply_snapped_pose sequence left a two-`_change` trail.

    A single Undo after that sequence only unwound the pose, leaving an
    orphan identity-rotation block at the seed point behind. `add_manual_with_pose`
    must create the block with its final pose in one `_change`, so one Undo
    removes it entirely.
    """
    controller = AnnotationController(_document())
    rotation = Rotation.from_euler("xyz", [0, 0, 40], degrees=True).as_matrix()
    position = np.array([1.0, 2.0, 3.0])
    controller.add_manual_with_pose(position, rotation, BLOCK_DIMS, provenance="rgb_mask")
    assert len(controller.document.targets) == 2
    assert controller.target is not None
    assert np.array_equal(controller.target.position, position)
    assert controller.target.dimensions == BLOCK_DIMS
    assert controller.target.provenance == "rgb_mask"
    assert controller.target.review_status == "pending"

    assert controller.undo()
    # One Undo must remove the created block entirely -- not merely reset its
    # pose to an identity-rotation orphan at the seed point.
    assert [target.id for target in controller.document.targets] == ["prior"]
    assert not controller.undo()


def test_add_manual_with_pose_rejects_invalid_transform() -> None:
    controller = AnnotationController(_document())
    with np.testing.assert_raises_regex(ValueError, "finite rigid transform"):
        controller.add_manual_with_pose(np.ones(3), np.zeros((3, 3)), BLOCK_DIMS)
    assert [target.id for target in controller.document.targets] == ["prior"]


def test_pca_oriented_block_pose_is_a_valid_right_handed_rotation() -> None:
    rng = np.random.default_rng(21)
    # A cluster elongated along one direction, like a real top+side surface
    # selection: PCA's dominant axis should track the elongation.
    local = np.column_stack(
        [
            rng.uniform(-0.45, 0.45, 400),
            rng.uniform(-0.05, 0.05, 400),
            rng.uniform(-0.05, 0.05, 400),
        ]
    )
    true_rotation = Rotation.from_euler("xyz", [5.0, 10.0, 63.0], degrees=True).as_matrix()
    center = np.array([2.0, -1.0, 0.5])
    points = local @ true_rotation.T + center

    position, rotation = pca_oriented_block_pose(points)

    np.testing.assert_allclose(position, center, atol=0.02)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) > 0.0
    # The dominant elongation direction becomes local +X, up to sign.
    assert abs(abs(np.dot(rotation[:, 0], true_rotation[:, 0])) - 1.0) < 1e-3


def test_pca_oriented_block_pose_handles_degenerate_input_without_raising() -> None:
    position, rotation = pca_oriented_block_pose(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) > 0.0

    with np.testing.assert_raises_regex(ValueError, "non-empty finite"):
        pca_oriented_block_pose(np.empty((0, 3)))


def test_pca_fallback_pose_feeds_add_manual_with_pose_as_one_undo_step() -> None:
    """End-to-end: the fallback's output must satisfy the controller's rigid-transform check."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.3, 0.3, (50, 3)) + np.array([5.0, 5.0, 1.0])
    position, rotation = pca_oriented_block_pose(points)

    controller = AnnotationController(_document())
    controller.add_manual_with_pose(
        position, rotation, BLOCK_DIMS, provenance="rgb_mask_pca_fallback"
    )
    assert controller.target is not None
    assert controller.target.dimensions == BLOCK_DIMS
    assert controller.target.provenance == "rgb_mask_pca_fallback"
    assert controller.undo()
    assert [target.id for target in controller.document.targets] == ["prior"]


def test_adopt_candidates_promotes_batch_with_one_undo() -> None:
    controller = AnnotationController(_document())
    rotation = Rotation.from_euler("xyz", [0, 0, 30], degrees=True).as_matrix()
    ids = controller.adopt_candidates(
        [
            ("fit-a", np.ones(3), rotation, BLOCK_DIMS),
            ("fit-b", np.full(3, 2.0), rotation, BLOCK_DIMS),
        ]
    )
    assert ids == ["auto_0", "auto_1"]
    assert controller.adopted_candidate_ids == {"fit-a", "fit-b"}
    assert controller.undo()
    assert [target.id for target in controller.document.targets] == ["prior"]
    assert controller.adopted_candidate_ids == set()


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


def test_adjacent_snapshot_clamps_without_wrapping() -> None:
    snapshots = [Path("a"), Path("b"), Path("c")]
    assert adjacent_snapshot(snapshots, Path("b"), -1) == Path("a")
    assert adjacent_snapshot(snapshots, Path("b"), 1) == Path("c")
    assert adjacent_snapshot(snapshots, Path("a"), -1) is None
    assert adjacent_snapshot(snapshots, Path("c"), 1) is None


def test_remove_background_by_range_keeps_only_points_within_radius() -> None:
    origin = np.array([1.0, 2.0, 0.0])
    points = np.array(
        [
            [1.0, 2.0, 0.0],  # at the origin: distance 0
            [1.0, 2.0, 4.0],  # distance 4, inside
            [1.0, 2.0, 6.0],  # distance 6, right on the boundary
            [1.0, 2.0, 6.5],  # distance 6.5, outside
        ]
    )
    kept = remove_background_by_range(points, origin, max_range_m=6.0)
    assert kept.shape == (3, 3)
    assert np.allclose(kept, points[:3])


def test_remove_background_by_range_handles_empty_input() -> None:
    kept = remove_background_by_range(np.empty((0, 3)), np.zeros(3), max_range_m=5.0)
    assert kept.shape == (0, 3)


def test_remove_background_by_range_rejects_non_positive_max_range() -> None:
    points = np.zeros((1, 3))
    try:
        remove_background_by_range(points, np.zeros(3), max_range_m=0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-positive max_range_m accepted")


def test_default_max_range_m_is_a_mild_trim_not_the_full_extent() -> None:
    origin = np.zeros(3)
    # 100 points spread from 1m to 10m, plus one far outlier at 50m.
    near = np.column_stack([np.linspace(1.0, 10.0, 100), np.zeros(100), np.zeros(100)])
    outlier = np.array([[50.0, 0.0, 0.0]])
    points = np.vstack([near, outlier])
    default = default_max_range_m(points, origin)
    # A 95th-percentile trim should sit well below the outlier but still
    # cover nearly all of the near-range cloud.
    assert 9.0 < default < 50.0


def test_ray_box_picking_chooses_nearest() -> None:
    targets = [
        EditableTarget("far", np.array([5.0, 0, 0]), np.eye(3)),
        EditableTarget("near", np.array([2.0, 0, 0]), np.eye(3)),
    ]
    assert pick_oriented_box(np.zeros(3), np.array([1.0, 0, 0]), targets) == 1
    assert pick_oriented_box(np.zeros(3), np.array([0.0, 1, 0]), targets) is None


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


def test_abort_selected_restores_an_approved_prior_in_one_undo_step() -> None:
    """One button press must be one undo step.

    Restoring the baseline pose and returning the block to pending used to be
    two separate changes, so Undo left the operator in a state they never
    asked for -- pose restored but still approved -- and a second press was
    needed to actually reverse the button.
    """
    document = AnnotationDocument(
        "snap",
        [
            EditableTarget(
                "prior", np.zeros(3), np.eye(3), provenance="registration", review_status="pending"
            )
        ],
    )
    controller = AnnotationController(document)
    controller.set_pose(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    controller.set_review("approved")
    before = len(controller._undo)

    controller.abort_selected()

    assert len(controller._undo) - before == 1
    assert controller.target.review_status == "pending"
    assert np.allclose(controller.target.position, np.zeros(3))

    controller.undo()
    assert controller.target.review_status == "approved"
    assert np.allclose(controller.target.position, [1.0, 2.0, 3.0])


def test_pca_oriented_block_pose_is_fast_and_bounded_on_a_large_selection() -> None:
    """The fallback runs on real mask selections; it must not allocate an (N, N) matrix."""
    points = np.random.default_rng(0).normal(size=(60000, 3))
    start = perf_counter()
    centroid, rotation = pca_oriented_block_pose(points)
    assert perf_counter() - start < 2.0
    assert centroid.shape == (3,)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_adopt_candidates_rejects_a_duplicate_identity_on_recompute() -> None:
    """Candidate ids are stable across recomputes, so re-adopting one is an error."""
    controller = AnnotationController(AnnotationDocument("snap", []))
    controller.adopt_candidates([("pcl:hyp-7", np.ones(3), np.eye(3), BLOCK_DIMS)])
    assert controller.adopted_candidate_ids == {"pcl:hyp-7"}
    with pytest.raises(ValueError):
        controller.adopt_candidates([("pcl:hyp-7", np.full(3, 2.0), np.eye(3), BLOCK_DIMS)])
    # Rejected within a single batch too, not just against the document.
    with pytest.raises(ValueError):
        controller.adopt_candidates(
            [("dup", np.ones(3), np.eye(3), BLOCK_DIMS)] * 2,
        )


def test_adopt_candidates_validates_the_whole_batch_before_touching_the_document() -> None:
    """A malformed proposal must not leave a partial adoption behind."""
    controller = AnnotationController(AnnotationDocument("snap", []))
    with pytest.raises(ValueError):
        controller.adopt_candidates(
            [
                ("good", np.ones(3), np.eye(3), BLOCK_DIMS),
                ("bad", np.ones(3), np.zeros((3, 3)), BLOCK_DIMS),
            ]
        )
    assert controller.document.targets == []
    assert controller.adopted_candidate_ids == set()


def test_add_manual_rejects_nonfinite_coordinates() -> None:
    controller = AnnotationController(AnnotationDocument("snap", []))
    with pytest.raises(ValueError):
        controller.add_manual(np.array([np.nan, 0.0, 0.0]))
    assert controller.document.targets == []
