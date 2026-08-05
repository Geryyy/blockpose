from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_snapshot_manifest import build_manifest, has_backup, render, scene_of  # noqa: E402

ROI = "annotated_roi: {min: [-1, -1, -1], max: [1, 1, 1]}\n"


def _block(
    block_id: str,
    *,
    review_status: str = "approved",
    visibility: str = "partial",
    provenance: str = "auto_fit",
    ignore_reason: str | None = None,
    rejection_reason: str | None = None,
) -> str:
    lines = [
        f"  - id: {block_id}\n",
        f"    review_status: {review_status}\n",
        f"    provenance: {provenance}\n",
        f"    visibility: {visibility}\n",
        "    pose:\n",
        "      position: [0, 0, 0]\n",
        "      orientation_xyzw: [0, 0, 0, 1]\n",
    ]
    if ignore_reason is not None:
        lines.append(f"    ignore_reason: {ignore_reason}\n")
    if rejection_reason is not None:
        lines.append(f"    rejection_reason: {rejection_reason}\n")
    return "".join(lines)


def _write_snapshot(
    root: Path,
    name: str,
    *,
    complete: bool,
    blocks: list[str] = (),
    with_backup: bool = True,
) -> Path:
    snapshot = root / name
    snapshot.mkdir(parents=True)
    body = (
        f"schema_version: 2\nsnapshot: {name}\nannotation_complete: {str(complete).lower()}\n"
        + (ROI if complete or blocks else "")
        + "blocks:\n"
        + "".join(blocks)
    )
    (snapshot / "annotations.yaml").write_text(body)
    if with_backup and complete and blocks:
        (snapshot / "annotations.yaml.backup-20260101T000000.000000Z").write_text(body)
    return snapshot


def test_scene_of_splits_on_double_underscore() -> None:
    assert scene_of("2026_07_07_wall__1783429196_422551296") == "2026_07_07_wall"
    assert scene_of("no_double_underscore") == "no_double_underscore"


def test_complete_with_approved_block_qualifies(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path, "sceneA__001", complete=True, blocks=[_block("b0", review_status="approved")]
    )
    manifest = build_manifest(tmp_path)
    assert manifest["totals"]["annotated_snapshots"] == 1
    assert "sceneA" in manifest["scenes"]
    assert manifest["scenes"]["sceneA"]["annotated"][0]["name"] == "sceneA__001"
    assert manifest["scenes"]["sceneA"]["annotated"][0]["blocks"] == 1


def test_incomplete_with_approved_block_does_not_qualify(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneB__001",
        complete=False,
        blocks=[_block("b0", review_status="approved")],
        with_backup=False,
    )
    manifest = build_manifest(tmp_path, warn=lambda _msg: None)
    assert manifest["totals"]["annotated_snapshots"] == 0
    assert "sceneB" in manifest["scenes_without_annotation"]
    assert manifest["scenes_without_annotation"]["sceneB"]["snapshots_in_scene"] == 1
    assert "sceneB" not in manifest["scenes"]


def test_complete_with_only_pending_block_does_not_qualify(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneC__001",
        complete=True,
        blocks=[_block("b0", review_status="pending")],
        with_backup=False,
    )
    manifest = build_manifest(tmp_path, warn=lambda _msg: None)
    assert manifest["totals"]["annotated_snapshots"] == 0
    assert "sceneC" in manifest["scenes_without_annotation"]


def test_approved_but_ignored_block_does_not_count_as_evaluation_target(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneD__001",
        complete=True,
        blocks=[_block("b0", review_status="approved", ignore_reason="duplicate")],
        with_backup=False,
    )
    manifest = build_manifest(tmp_path, warn=lambda _msg: None)
    assert manifest["totals"]["annotated_snapshots"] == 0
    assert "sceneD" in manifest["scenes_without_annotation"]


def test_disagreeing_signals_warn_but_do_not_raise(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneM__001",
        complete=True,
        blocks=[_block("b0", review_status="approved")],
        with_backup=False,
    )
    messages: list[str] = []
    manifest = build_manifest(tmp_path, warn=messages.append)
    assert manifest["totals"]["annotated_snapshots"] == 1
    assert len(messages) == 1
    assert "sceneM__001" in messages[0]


def test_scene_grouping_collects_multiple_snapshots_under_one_scene(tmp_path: Path) -> None:
    for stamp in ("001", "002", "003"):
        _write_snapshot(
            tmp_path,
            f"sceneE__{stamp}",
            complete=True,
            blocks=[_block("b0", review_status="approved")],
        )
    # an unannotated near-duplicate frame in the same scene
    (tmp_path / "sceneE__004").mkdir()
    (tmp_path / "sceneE__004" / "annotations.yaml").write_text(
        "schema_version: 2\nsnapshot: sceneE__004\nannotation_complete: false\nblocks:\n"
    )

    manifest = build_manifest(tmp_path)
    assert manifest["scenes"]["sceneE"]["snapshots_in_scene"] == 4
    assert len(manifest["scenes"]["sceneE"]["annotated"]) == 3
    names = [entry["name"] for entry in manifest["scenes"]["sceneE"]["annotated"]]
    assert names == sorted(names)


def test_approximate_pose_provenance_is_counted(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneF__001",
        complete=True,
        blocks=[
            _block("b0", review_status="approved", provenance="auto_fit"),
            _block("b1", review_status="approved", provenance="rgb_mask_pca_fallback"),
        ],
    )
    manifest = build_manifest(tmp_path)
    entry = manifest["scenes"]["sceneF"]["annotated"][0]
    assert entry["blocks"] == 2
    assert entry["approximate_poses"] == 1
    assert manifest["totals"]["approximate_poses"] == 1


def test_counts_exclude_non_approved_blocks_of_a_qualifying_snapshot(tmp_path: Path) -> None:
    """blocks/approximate_poses/visibility count evaluation targets only.

    No snapshot in data/seibersdorf_dataset mixes approved with
    pending/rejected/ignored blocks, so the live acceptance test cannot see
    this: replacing the per-snapshot `is_evaluation_target` filter with
    `list(annotations.targets)` left every other test in this file green.
    An operator annotating a fresh bag will absolutely have rejected and
    still-pending blocks in an otherwise complete snapshot, and counting
    them would inflate `approved_blocks` -- the manifest's headline number.
    """
    _write_snapshot(
        tmp_path,
        "sceneN__001",
        complete=True,
        blocks=[
            _block("b0", review_status="approved", visibility="full"),
            _block("b1", review_status="pending", visibility="partial"),
            _block(
                "b2",
                review_status="rejected",
                rejection_reason="not a block",
                visibility="partial",
            ),
            _block(
                "b3",
                review_status="approved",
                ignore_reason="out_of_roi",
                visibility="partial",
                provenance="rgb_mask_pca_fallback",
            ),
        ],
    )
    manifest = build_manifest(tmp_path)
    entry = manifest["scenes"]["sceneN"]["annotated"][0]
    assert entry["blocks"] == 1
    assert entry["approximate_poses"] == 0
    assert entry["visibility"] == {"full": 1}
    assert manifest["totals"]["approved_blocks"] == 1
    assert manifest["totals"]["approximate_poses"] == 0


def test_visibility_is_tallied_per_snapshot(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneG__001",
        complete=True,
        blocks=[
            _block("b0", review_status="approved", visibility="full"),
            _block("b1", review_status="approved", visibility="partial"),
            _block("b2", review_status="approved", visibility="partial"),
        ],
    )
    manifest = build_manifest(tmp_path)
    entry = manifest["scenes"]["sceneG"]["annotated"][0]
    assert entry["visibility"] == {"full": 1, "partial": 2}


def test_scenes_without_annotation_omits_covered_by_scene_fields(tmp_path: Path) -> None:
    snapshot = tmp_path / "sceneH__001"
    snapshot.mkdir()
    (snapshot / "annotations.yaml").write_text(
        "schema_version: 2\nsnapshot: sceneH__001\nannotation_complete: false\nblocks:\n"
    )
    manifest = build_manifest(tmp_path)
    entry = manifest["scenes_without_annotation"]["sceneH"]
    assert entry == {"snapshots_in_scene": 1}
    assert "covered_by_scene" not in entry
    assert "rgb_similarity_to_covering_scene" not in entry


def test_directory_without_annotations_yaml_counts_toward_scene_total_only(tmp_path: Path) -> None:
    (tmp_path / "sceneI__001").mkdir()
    manifest = build_manifest(tmp_path)
    assert manifest["totals"]["snapshots_in_dataset"] == 1
    assert manifest["totals"]["annotated_snapshots"] == 0
    assert manifest["scenes_without_annotation"]["sceneI"]["snapshots_in_scene"] == 1


def test_missing_backup_still_qualifies_via_completeness_signal(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path,
        "sceneJ__001",
        complete=True,
        blocks=[_block("b0", review_status="approved")],
        with_backup=False,
    )
    manifest = build_manifest(tmp_path, warn=lambda _msg: None)
    assert manifest["totals"]["annotated_snapshots"] == 1


def test_has_backup_matches_annotate_snapshot_naming(tmp_path: Path) -> None:
    snapshot = tmp_path / "sceneK__001"
    snapshot.mkdir()
    assert has_backup(snapshot) is False
    (snapshot / "annotations.yaml.backup-20260101T000000.000000Z").write_text("x")
    assert has_backup(snapshot) is True


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        build_manifest(tmp_path / "does-not-exist")


def test_render_is_valid_yaml_and_carries_a_do_not_hand_edit_header(tmp_path: Path) -> None:
    _write_snapshot(
        tmp_path, "sceneL__001", complete=True, blocks=[_block("b0", review_status="approved")]
    )
    manifest = build_manifest(tmp_path)
    rendered = render(manifest)
    assert "do not hand-edit" in rendered
    round_tripped = yaml.safe_load(rendered)
    assert round_tripped["totals"] == manifest["totals"]


def test_reproduces_committed_seibersdorf_manifest_selection() -> None:
    """The acceptance test: regenerate against the real dataset and compare.

    Verified separately (see tools/build_snapshot_manifest.py's module
    docstring) that `scenes`, `totals`, and `scenes_without_annotation`
    (minus the two hand-computed RGB-similarity fields this generator
    deliberately omits) match config/annotated_snapshots_seibersdorf.yaml
    exactly. Kept as a regression test here so a future change to the
    dataset or the generator that breaks reproduction is caught by the
    Python test suite, not just discovered by hand.
    """
    dataset_root = REPO_ROOT / "data" / "seibersdorf_dataset"
    if not dataset_root.is_dir():
        pytest.skip("data/seibersdorf_dataset not present in this checkout")

    committed = yaml.safe_load(
        (REPO_ROOT / "config" / "annotated_snapshots_seibersdorf.yaml").read_text()
    )
    # No dataset_label: the default invocation must reproduce the committed
    # `dataset:` value, not just the selection.
    generated = build_manifest(dataset_root)

    assert generated["dataset"] == committed["dataset"]
    assert generated["totals"] == committed["totals"]
    assert generated["scenes"] == committed["scenes"]
    assert set(generated["scenes_without_annotation"]) == set(
        committed["scenes_without_annotation"]
    )
    for scene, entry in generated["scenes_without_annotation"].items():
        assert (
            entry["snapshots_in_scene"]
            == committed["scenes_without_annotation"][scene]["snapshots_in_scene"]
        )


def test_scene_labels_match_rgb_ablation_splits() -> None:
    """The `grouping:` field claims these groups match the ablation folds.

    The claim is the whole reason this manifest composes with
    config/rgb_ablation_splits_seibersdorf.yaml's leave-one-scenario-out
    folds instead of defining a second, competing grouping, so it is checked
    rather than asserted in a comment.
    """
    dataset_root = REPO_ROOT / "data" / "seibersdorf_dataset"
    if not dataset_root.is_dir():
        pytest.skip("data/seibersdorf_dataset not present in this checkout")

    splits = yaml.safe_load(
        (REPO_ROOT / "config" / "rgb_ablation_splits_seibersdorf.yaml").read_text()
    )
    manifest = build_manifest(dataset_root)
    derived = set(manifest["scenes"]) | set(manifest["scenes_without_annotation"])
    assert derived == set(splits["groups"])
