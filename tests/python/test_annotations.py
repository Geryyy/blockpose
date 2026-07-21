from pathlib import Path

import pytest
from blockpose.annotations import load_annotations

REPO_ROOT = Path(__file__).parents[2]


def test_all_real_snapshots_have_valid_review_manifests() -> None:
    paths = sorted((REPO_ROOT / "data" / "registration_snapshots").glob("*/annotations.yaml"))
    assert len(paths) == 4
    annotations = [load_annotations(path) for path in paths]
    # These development snapshots remain partial until a human explicitly
    # defines and exhaustively reviews an ROI.
    assert all(not item.annotation_complete for item in annotations)
    assert all(not item.has_complete_roi for item in annotations)
    targets = [target for item in annotations for target in item.targets]
    assert targets
    assert all(target.review_status in {"pending", "approved", "rejected"} for target in targets)
    assert {target.provenance for target in targets} <= {
        "manual",
        "upstream_registration",
    }
    assert {target.provenance for target in targets} == {
        "manual",
        "upstream_registration",
    }
    assert all(
        len({target.id for target in item.targets}) == len(item.targets) for item in annotations
    )


def test_complete_manifest_requires_roi(tmp_path: Path) -> None:
    manifest = tmp_path / "annotations.yaml"
    manifest.write_text(
        f"schema_version: 1\nsnapshot: {tmp_path.name}\nannotation_complete: true\n"
    )
    with pytest.raises(ValueError, match="require an annotated_roi"):
        load_annotations(manifest)


def test_complete_target_must_be_inside_inclusive_roi_and_ids_unique(tmp_path: Path) -> None:
    manifest = tmp_path / "annotations.yaml"
    prefix = (
        f"schema_version: 1\nsnapshot: {tmp_path.name}\nannotation_complete: true\n"
        "annotated_roi: {min: [0, 0, 0], max: [1, 1, 1]}\nblocks:\n"
    )
    block = (
        "  - id: edge\n    pose:\n      position: [1, 1, 1]\n"
        "      orientation_xyzw: [0, 0, 0, 1]\n"
    )
    manifest.write_text(prefix + block)
    assert load_annotations(manifest).targets[0].id == "edge"

    manifest.write_text(prefix + block.replace("[1, 1, 1]", "[1.01, 1, 1]"))
    with pytest.raises(ValueError, match="outside annotated_roi"):
        load_annotations(manifest)

    manifest.write_text(prefix + block + block)
    with pytest.raises(ValueError, match="duplicate block id"):
        load_annotations(manifest)


def test_snapshot_name_must_match_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "annotations.yaml"
    manifest.write_text("schema_version: 1\nsnapshot: another\nannotation_complete: false\n")
    with pytest.raises(ValueError, match="does not match directory"):
        load_annotations(manifest)
