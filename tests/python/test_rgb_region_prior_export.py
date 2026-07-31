"""Union-silhouette target construction for the ``dino_proposer`` exporter.

Offline unit tests: no GPU, no network, synthetic snapshots only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from blockpose.io import CameraModel, load_camera_model  # noqa: E402
from blockpose.pose import BlockPose  # noqa: E402

from experiments.rgb_region_prior import export  # noqa: E402

_WIDTH, _HEIGHT = 64, 64


def _camera() -> CameraModel:
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    return CameraModel(K=K, width=_WIDTH, height=_HEIGHT)


def _identity_extrinsics() -> np.ndarray:
    return np.eye(4)


def _pose(position: tuple[float, float, float], dims=(0.2, 0.2, 0.2)) -> BlockPose:
    return BlockPose(np.array(position, dtype=np.float64), np.eye(3), dims=dims)


# ---------------------------------------------------------------------------
# block_silhouette_mask / union_mask
# ---------------------------------------------------------------------------


def test_block_silhouette_mask_is_nonempty_for_a_block_in_front_of_camera() -> None:
    camera = _camera()
    mask = export.block_silhouette_mask(_pose((0.0, 0.0, 2.0)), _identity_extrinsics(), camera)
    assert mask is not None
    assert mask.shape == (_HEIGHT, _WIDTH)
    assert mask.dtype == bool
    assert np.any(mask)
    # projects near the principal point (cx=cy=32)
    ys, xs = np.nonzero(mask)
    assert abs(float(xs.mean()) - 32.0) < 10.0
    assert abs(float(ys.mean()) - 32.0) < 10.0


def test_block_silhouette_mask_none_when_block_behind_camera() -> None:
    camera = _camera()
    mask = export.block_silhouette_mask(_pose((0.0, 0.0, -2.0)), _identity_extrinsics(), camera)
    assert mask is None


def test_union_mask_combines_two_disjoint_blocks() -> None:
    camera = _camera()
    left = _pose((-0.6, 0.0, 2.0))
    right = _pose((0.6, 0.0, 2.0))
    from blockpose.annotations import AnnotationTarget

    targets = [
        AnnotationTarget(id="left", pose=left, provenance="manual", review_status="approved"),
        AnnotationTarget(id="right", pose=right, provenance="manual", review_status="approved"),
    ]
    mask, contributing, dropped = export.union_mask(targets, _identity_extrinsics(), camera)
    assert set(contributing) == {"left", "right"}
    assert dropped == []
    left_only, _, _ = export.union_mask(targets[:1], _identity_extrinsics(), camera)
    right_only, _, _ = export.union_mask(targets[1:], _identity_extrinsics(), camera)
    assert np.array_equal(mask, left_only | right_only)
    assert np.any(left_only & mask)
    assert not np.array_equal(left_only, mask)  # union is strictly bigger


def test_union_mask_drops_ungated_block_but_keeps_others() -> None:
    camera = _camera()
    from blockpose.annotations import AnnotationTarget

    targets = [
        AnnotationTarget(
            id="visible", pose=_pose((0.0, 0.0, 2.0)), provenance="manual", review_status="approved"
        ),
        AnnotationTarget(
            id="behind", pose=_pose((0.0, 0.0, -2.0)), provenance="manual", review_status="approved"
        ),
    ]
    mask, contributing, dropped = export.union_mask(targets, _identity_extrinsics(), camera)
    assert contributing == ["visible"]
    assert dropped == ["behind"]
    assert np.any(mask)


def test_union_mask_empty_for_no_targets() -> None:
    camera = _camera()
    mask, contributing, dropped = export.union_mask([], _identity_extrinsics(), camera)
    assert not np.any(mask)
    assert contributing == []
    assert dropped == []


# ---------------------------------------------------------------------------
# approved_targets: review_status + occlusion filtering
# ---------------------------------------------------------------------------


def test_approved_targets_excludes_pending_rejected_and_occluded() -> None:
    from blockpose.annotations import AnnotationTarget, SnapshotAnnotations

    pose = _pose((0.0, 0.0, 2.0))
    targets = (
        AnnotationTarget(
            id="ok", pose=pose, provenance="manual", review_status="approved", visibility="full"
        ),
        AnnotationTarget(id="pending", pose=pose, provenance="manual", review_status="pending"),
        AnnotationTarget(
            id="rejected", pose=pose, provenance="manual", review_status="rejected",
            rejection_reason="bad",
        ),
        AnnotationTarget(
            id="heavy_occlusion", pose=pose, provenance="manual", review_status="approved",
            visibility="heavy_occlusion",
        ),
        AnnotationTarget(
            id="unknown_vis", pose=pose, provenance="manual", review_status="approved",
            visibility="unknown",
        ),
        AnnotationTarget(
            id="partial", pose=pose, provenance="manual", review_status="approved",
            visibility="partial",
        ),
    )
    annotations = SnapshotAnnotations(2, "snap", True, None, None, targets)
    kept = {t.id for t in export.approved_targets(annotations)}
    assert kept == {"ok", "partial"}


# ---------------------------------------------------------------------------
# _health_report
# ---------------------------------------------------------------------------


def test_health_report_flags_zero_block_snapshot_and_thin_sessions() -> None:
    records = [
        {"sample_id": "s1", "split_group": "g1", "block_count": 2},
        {"sample_id": "s2", "split_group": "g2", "block_count": 0},
    ]
    per_snapshot = [
        {"snapshot": "s1", "status": "exported"},
        {"snapshot": "s2", "status": "exported"},
        {
            "snapshot": "s3",
            "status": "gate_failed",
            "gate": "sync",
            "reason": "sync_delta_exceeds_tolerance",
        },
    ]
    health = export._health_report(records, per_snapshot)
    assert health["zero_block_snapshots"] == ["s2"]
    assert health["total_blocks"] == 2
    assert health["gate_failures"] == {"sync:sync_delta_exceeds_tolerance": 1}
    assert any("only 2 sessions" in w for w in health["warnings"])
    assert any("g2" in w and "zero approved blocks" in w for w in health["warnings"])


# ---------------------------------------------------------------------------
# export_snapshot end-to-end on a fully synthetic snapshot directory
# ---------------------------------------------------------------------------


def _write_calibration(path: Path) -> None:
    K = [100.0, 0.0, 32.0, 0.0, 100.0, 32.0, 0.0, 0.0, 1.0]
    calib = {"image_width": _WIDTH, "image_height": _HEIGHT, "K": K, "D": []}
    path.write_text(yaml.safe_dump(calib))


def _write_tf(path: Path) -> None:
    identity = np.eye(4).tolist()
    path.write_text(
        yaml.safe_dump(
            {
                "reference_frame": "world",
                "image_frame": "camera",
                "cloud_frame": "cloud",
                "transforms": [
                    {"name": "T_world_camera", "available": True, "matrix": identity},
                    {"name": "T_world_cloud", "available": True, "matrix": identity},
                ],
            }
        )
    )


def _write_metadata(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "image": {"stamp": {"sec": 100, "nanosec": 0}},
                "cloud": {"stamp": {"sec": 100, "nanosec": 20_000_000}},
            }
        )
    )


def _write_annotations(path: Path, snapshot_name: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "snapshot": snapshot_name,
                "annotation_complete": True,
                "annotated_roi": {"min": [-5.0, -5.0, -5.0], "max": [5.0, 5.0, 5.0]},
                "blocks": [
                    {
                        "id": "block_0",
                        "review_status": "approved",
                        "provenance": "manual",
                        "visibility": "full",
                        "dimensions": [0.2, 0.2, 0.2],
                        "pose": {
                            "position": [0.0, 0.0, 2.0],
                            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            }
        )
    )


def _build_snapshot(root: Path, name: str) -> Path:
    snapshot = root / name
    snapshot.mkdir(parents=True)
    Image.fromarray(np.full((_HEIGHT, _WIDTH, 3), 128, dtype=np.uint8)).save(snapshot / "rgb.png")
    _write_tf(snapshot / "tf.yaml")
    _write_metadata(snapshot / "metadata.yaml")
    _write_annotations(snapshot / "annotations.yaml", name)
    return snapshot


def test_export_snapshot_writes_record_and_mask(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path / "snapshots", "snap_a")
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()

    records: list[dict] = []
    status = export.export_snapshot(
        snapshot,
        export_root=export_root,
        camera=load_camera_model(calibration_path),
        calibration_checksum="deadbeef",
        split_group="snap_a",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "exported"
    assert status["block_count"] == 1
    assert len(records) == 1
    record = records[0]
    assert record["sample_id"] == "snap_a"
    assert record["block_count"] == 1
    assert record["block_ids"] == ["block_0"]

    mask_path = export_root / record["mask"]
    image_path = export_root / record["image"]
    assert mask_path.is_file()
    assert image_path.is_file()
    mask = np.asarray(Image.open(mask_path)) > 0
    assert mask.shape == (_HEIGHT, _WIDTH)
    assert np.any(mask)


def test_export_snapshot_gate_failed_on_sync_drops_sample(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path / "snapshots", "snap_b")
    # blow the sync tolerance
    (snapshot / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "image": {"stamp": {"sec": 100, "nanosec": 0}},
                "cloud": {"stamp": {"sec": 105, "nanosec": 0}},
            }
        )
    )
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()

    records: list[dict] = []
    status = export.export_snapshot(
        snapshot,
        export_root=export_root,
        camera=load_camera_model(calibration_path),
        calibration_checksum="deadbeef",
        split_group="snap_b",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "gate_failed"
    assert status["gate"] == "sync"
    assert records == []


def test_export_snapshot_zero_blocks_still_exports(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path / "snapshots", "snap_c")
    annotations = yaml.safe_load((snapshot / "annotations.yaml").read_text())
    annotations["blocks"][0]["review_status"] = "pending"
    (snapshot / "annotations.yaml").write_text(yaml.safe_dump(annotations))
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()

    records: list[dict] = []
    status = export.export_snapshot(
        snapshot,
        export_root=export_root,
        camera=load_camera_model(calibration_path),
        calibration_checksum="deadbeef",
        split_group="snap_c",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "exported"
    assert status["block_count"] == 0
    assert len(records) == 1
    assert records[0]["block_count"] == 0


def test_export_snapshot_skips_incomplete_annotations(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path / "snapshots", "snap_d")
    annotations = yaml.safe_load((snapshot / "annotations.yaml").read_text())
    annotations["annotation_complete"] = False
    (snapshot / "annotations.yaml").write_text(yaml.safe_dump(annotations))
    calibration_path = tmp_path / "calib.yaml"
    _write_calibration(calibration_path)
    export_root = tmp_path / "out"
    export_root.mkdir()

    records: list[dict] = []
    status = export.export_snapshot(
        snapshot,
        export_root=export_root,
        camera=load_camera_model(calibration_path),
        calibration_checksum="deadbeef",
        split_group="snap_d",
        sync_tolerance_s=0.10,
        records=records,
    )
    assert status["status"] == "skipped_annotation_incomplete"
    assert records == []
