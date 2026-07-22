from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from blockpose import annotation_batch
from blockpose.annotation_batch import _publish_no_replace, discover_snapshots, initialize_snapshot
from blockpose.annotation_snap import SnapResult
from blockpose.annotations import load_annotations


def _snapshot(tmp_path: Path, name: str = "sample") -> Path:
    snapshot = tmp_path / name
    snapshot.mkdir()
    for filename in annotation_batch.REQUIRED_FILES:
        (snapshot / filename).write_bytes(b"placeholder")
    (snapshot / "metadata.yaml").write_text("candidate_masks: []\n")
    (snapshot / "registrations.yaml").write_text("records: []\n")
    return snapshot


def _patch_valid_loaders(monkeypatch, *, masks=None) -> None:
    monkeypatch.setattr(annotation_batch, "load_rgb", lambda _path: np.zeros((4, 5, 3), np.uint8))
    monkeypatch.setattr(annotation_batch, "load_pcd", lambda _path: np.zeros((50, 3)))
    monkeypatch.setattr(
        annotation_batch,
        "load_tf_snapshot",
        lambda _path: SimpleNamespace(T_world_cloud=np.eye(4), T_world_camera=np.eye(4)),
    )
    monkeypatch.setattr(
        annotation_batch,
        "load_camera_model",
        lambda _path: SimpleNamespace(height=4, width=5),
    )
    monkeypatch.setattr(
        annotation_batch,
        "build_projection_cache",
        lambda *_args: SimpleNamespace(
            uv=np.zeros((50, 2)), depth=np.ones(50), valid=np.ones(50, bool)
        ),
    )
    monkeypatch.setattr(annotation_batch, "load_candidate_masks", lambda *_args: masks or {})


def test_discovery_includes_malformed_directories_for_audit(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "file").write_text("not a snapshot")
    assert [path.name for path in discover_snapshots(tmp_path)] == ["a", "b"]


def test_discovery_excludes_manifest_entries_by_default(tmp_path: Path) -> None:
    (tmp_path / "keep").mkdir()
    (tmp_path / "drop").mkdir()
    (tmp_path / "snapshot_exclusions.yaml").write_text(
        "schema_version: 1\nexcluded_snapshots:\n"
        "  - {snapshot: drop, reason: duplicate, retained_snapshot: keep}\n"
    )
    assert [path.name for path in discover_snapshots(tmp_path)] == ["keep"]
    assert [path.name for path in discover_snapshots(tmp_path, include_excluded=True)] == [
        "drop",
        "keep",
    ]


def test_atomic_publication_never_replaces_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "annotations.yaml"
    assert _publish_no_replace("first", destination)
    assert not _publish_no_replace("second", destination)
    assert destination.read_text() == "first"


def test_initializer_never_overwrites_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    _patch_valid_loaders(monkeypatch)
    first = initialize_snapshot(snapshot)
    assert first.status == "created"
    original = (snapshot / "annotations.yaml").read_bytes()
    second = initialize_snapshot(snapshot)
    assert second.status == "skipped_existing"
    assert (snapshot / "annotations.yaml").read_bytes() == original
    assert not list(snapshot.glob("annotations.yaml.backup-*"))


def test_registration_import_is_pending_and_ids_are_unique(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    records = {
        "records": [
            {
                "registration_ok": True,
                "assigned_id": "same",
                "confidence": 0.8,
                "pose": {"position": [1, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
            },
            {
                "registration_ok": True,
                "assigned_id": "same",
                "pose": {"position": [2, 2, 3], "orientation_xyzw": [0, 0, 0, 1]},
            },
        ]
    }
    (snapshot / "registrations.yaml").write_text(yaml.safe_dump(records))
    _patch_valid_loaders(monkeypatch)
    result = initialize_snapshot(snapshot)
    loaded = load_annotations(snapshot / "annotations.yaml")
    assert result.registration_targets == 2
    assert [target.id for target in loaded.targets] == ["same", "same_2"]
    assert all(target.review_status == "pending" for target in loaded.targets)
    assert all(target.provenance == "upstream_registration" for target in loaded.targets)
    assert not loaded.annotation_complete and loaded.roi_min is None


def test_bad_cloud_or_tf_is_integrity_failure_without_manifest(tmp_path: Path, monkeypatch) -> None:
    for bad in ("cloud", "tf"):
        snapshot = _snapshot(tmp_path, bad)
        _patch_valid_loaders(monkeypatch)
        if bad == "cloud":
            monkeypatch.setattr(
                annotation_batch,
                "load_pcd",
                lambda _path: (_ for _ in ()).throw(ValueError("bad PCD")),
            )
        else:
            transform = np.eye(4)
            transform[0, 0] = np.nan
            monkeypatch.setattr(
                annotation_batch,
                "load_tf_snapshot",
                lambda _path, value=transform: SimpleNamespace(
                    T_world_cloud=value, T_world_camera=np.eye(4)
                ),
            )
        result = initialize_snapshot(snapshot)
        assert result.status == "failed_integrity"
        assert not (snapshot / "annotations.yaml").exists()


def test_malformed_mask_is_reported_without_target(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / "metadata.yaml").write_text(
        "candidate_masks:\n  - {detection_id: 7, file: masks/bad.png}\n"
    )
    _patch_valid_loaders(monkeypatch)
    result = initialize_snapshot(snapshot)
    assert result.status == "created" and result.rgb_targets == 0
    assert any(failure.source == "candidate mask 7" for failure in result.failures)
    assert load_annotations(snapshot / "annotations.yaml").targets == ()


def test_rgb_fits_are_pending_and_deduplicated(tmp_path: Path, monkeypatch) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / "metadata.yaml").write_text(
        "candidate_masks:\n"
        "  - {detection_id: 1, file: masks/one.png}\n"
        "  - {detection_id: 2, file: masks/two.png}\n"
    )
    mask = np.ones((4, 5), bool)
    _patch_valid_loaders(monkeypatch, masks={1: mask, 2: mask})
    selection = SimpleNamespace(
        success=True,
        indices=np.array([0]),
        points=np.zeros((50, 3)),
        reason="ok",
    )
    monkeypatch.setattr(
        annotation_batch, "select_points_from_mask", lambda *_args, **_kw: selection
    )
    monkeypatch.setattr(
        annotation_batch,
        "snap_block_to_cloud",
        lambda *_args, **_kw: SnapResult(
            True, np.array([1.0, 2.0, 3.0]), np.eye(3), (0.6, 0.6, 0.9), 0.7, 50, "ok"
        ),
    )
    result = initialize_snapshot(snapshot)
    loaded = load_annotations(snapshot / "annotations.yaml")
    assert result.rgb_targets == 1
    assert len(loaded.targets) == 1
    assert loaded.targets[0].id == "rgb_candidate_1"
    assert loaded.targets[0].provenance == "rgb_candidate_fit"
    assert loaded.targets[0].review_status == "pending"
    assert loaded.targets[0].pose.dims == (0.9, 0.6, 0.6)
    assert any("deduplicated" in failure.reason for failure in result.failures)
