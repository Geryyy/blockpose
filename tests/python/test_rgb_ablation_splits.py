"""Frozen diagnostic split manifest checks for RGB proposal replay."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from ablate_rgb_proposals import load_split_manifest  # noqa: E402


def test_diagnostic_manifest_has_explicit_dev_final_semantics() -> None:
    manifest = load_split_manifest(REPO_ROOT / "config" / "rgb_ablation_splits.yaml")
    assert not manifest["selection_eligible"]
    assert not manifest["final_enabled"]
    assert "not_validated_run_ids" in manifest["grouping_evidence"]
    assert {group["split"] for group in manifest["groups"].values()} == {"development", "final"}
    assert all(
        "fold" in group for group in manifest["groups"].values() if group["split"] == "development"
    )


def test_seibersdorf_manifest_is_all_development_leave_one_scenario_out() -> None:
    manifest = load_split_manifest(REPO_ROOT / "config" / "rgb_ablation_splits_seibersdorf.yaml")
    assert not manifest["selection_eligible"]
    assert not manifest["final_enabled"]
    assert "not_validated_run_ids" in manifest["grouping_evidence"]
    assert {group["split"] for group in manifest["groups"].values()} == {"development"}
    folds = sorted(group["fold"] for group in manifest["groups"].values())
    assert folds == list(range(len(manifest["groups"])))


def test_manifest_rejects_duplicate_snapshot_across_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "selection_eligible: false\n"
        "final_enabled: false\n"
        "grouping_evidence: provisional\n"
        "groups:\n"
        "  a: {split: development, fold: 0, snapshots: [same]}\n"
        "  b: {split: final, snapshots: [same]}\n"
    )
    with pytest.raises(ValueError, match="unique direct-child"):
        load_split_manifest(manifest)


@pytest.mark.parametrize(
    ("key", "value"),
    (("selection_eligible", '"false"'), ("final_enabled", "0")),
)
def test_manifest_requires_real_yaml_boole(tmp_path: Path, key: str, value: str) -> None:
    manifest = tmp_path / "bad_bool.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        f"selection_eligible: {value if key == 'selection_eligible' else 'false'}\n"
        f"final_enabled: {value if key == 'final_enabled' else 'false'}\n"
        "grouping_evidence: provisional\n"
        "groups:\n"
        "  a: {split: development, fold: 0, snapshots: [one]}\n"
    )
    with pytest.raises(ValueError, match=f"{key} must be a YAML boolean"):
        load_split_manifest(manifest)


def test_timestamp_grouping_cannot_be_enabled_as_final_by_flag_flip(tmp_path: Path) -> None:
    source = (REPO_ROOT / "config" / "rgb_ablation_splits.yaml").read_text()
    manifest = tmp_path / "unsafe_final.yaml"
    manifest.write_text(
        source.replace("selection_eligible: false", "selection_eligible: true").replace(
            "final_enabled: false", "final_enabled: true"
        )
    )
    with pytest.raises(ValueError, match="validated_recording_run_ids"):
        load_split_manifest(manifest)


def test_final_manifest_requires_explicit_unique_run_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_runs.yaml"
    manifest.write_text(
        "schema_version: 1\n"
        "selection_eligible: true\n"
        "final_enabled: true\n"
        "grouping_evidence: validated_recording_run_ids\n"
        "groups:\n"
        "  a: {split: development, fold: 0, snapshots: [one]}\n"
        "  b: {split: final, snapshots: [two]}\n"
    )
    with pytest.raises(ValueError, match="unique explicit run_id"):
        load_split_manifest(manifest)
