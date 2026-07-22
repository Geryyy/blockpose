from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from blockpose.annotation_roi import (
    FULL_CLOUD_ROI_NOTE,
    cuboid_corners,
    full_cloud_world_roi,
    prepare_roi_completion,
)
from blockpose.pose import BlockPose

TOOLS_DIR = Path(__file__).parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import complete_snapshot_rois  # noqa: E402


def test_full_cloud_roi_transforms_finite_points_and_pads() -> None:
    points = np.array([[0, 0, 0], [1, 2, 3], [np.nan, 4, 5]], dtype=float)
    transform = np.eye(4)
    transform[:3, 3] = [10, -2, 0.5]
    low, high = full_cloud_world_roi(points, transform)
    np.testing.assert_allclose(low, [9.99, -2.01, 0.49])
    np.testing.assert_allclose(high, [11.01, 0.01, 3.51])


def test_full_cloud_roi_single_point_is_non_degenerate() -> None:
    low, high = full_cloud_world_roi(np.array([[1, 2, 3]]), np.eye(4))
    np.testing.assert_allclose(high - low, [0.02, 0.02, 0.02])


@pytest.mark.parametrize(
    "points, message",
    [(np.empty((0, 3)), "no finite"), (np.full((2, 3), np.nan), "no finite")],
)
def test_full_cloud_roi_rejects_cloud_without_finite_points(points, message) -> None:
    with pytest.raises(ValueError, match=message):
        full_cloud_world_roi(points, np.eye(4))


def test_full_cloud_roi_rejects_invalid_transform() -> None:
    transform = np.eye(4)
    transform[0, 0] = 2
    with pytest.raises(ValueError, match="not rigid"):
        full_cloud_world_roi(np.zeros((1, 3)), transform)


def test_cuboid_corners_uses_pose_dimensions_and_rotation() -> None:
    pose = BlockPose(np.array([1, 2, 3]), np.eye(3), dims=(0.9, 0.6, 0.6))
    corners = cuboid_corners(pose)
    np.testing.assert_allclose(corners.min(axis=0), [0.55, 1.7, 2.7])
    np.testing.assert_allclose(corners.max(axis=0), [1.45, 2.3, 3.3])


def test_prepare_completion_preserves_manifest_without_writing(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "scene"
    snapshot.mkdir()
    manifest = snapshot / "annotations.yaml"
    original = (
        "schema_version: 2\nsnapshot: scene\nannotation_complete: false\n"
        "annotated_roi: null\nnotes: reviewed\nblocks: []\n"
        "unmatched_detection_reviews: {det_1: clutter}\n"
    )
    manifest.write_text(original)
    monkeypatch.setattr(
        "blockpose.annotation_roi.load_tf_snapshot",
        lambda _path: type("TF", (), {"T_world_cloud": np.eye(4)})(),
    )
    monkeypatch.setattr(
        "blockpose.annotation_roi.load_pcd", lambda _path: np.array([[0.0, 1.0, 2.0]])
    )

    completion = prepare_roi_completion(snapshot)

    assert manifest.read_text() == original
    assert completion.document["unmatched_detection_reviews"] == {"det_1": "clutter"}
    assert completion.document["annotation_complete"] is True
    assert FULL_CLOUD_ROI_NOTE in completion.document["notes"]


def test_prepare_completion_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        prepare_roi_completion(tmp_path)


def test_complete_snapshots_dry_run_honors_exclusions_and_does_not_write(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "active").mkdir()
    (tmp_path / "excluded").mkdir()
    (tmp_path / "snapshot_exclusions.yaml").write_text(
        "schema_version: 1\nexcluded_snapshots:\n  - {snapshot: excluded, reason: duplicate}\n"
    )
    seen = []

    def prepare(snapshot):
        seen.append(snapshot.name)
        return type("Completion", (), {"snapshot": snapshot, "document": {}})()

    monkeypatch.setattr(complete_snapshot_rois, "prepare_roi_completion", prepare)
    monkeypatch.setattr(
        complete_snapshot_rois,
        "_write_atomic",
        lambda *_args: pytest.fail("dry-run attempted to overwrite a manifest"),
    )
    complete_snapshot_rois.complete_snapshots(tmp_path)
    assert seen == ["active"]
