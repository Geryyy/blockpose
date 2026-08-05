"""Promotion-gate split manifest generation, and fold_safety consumability.

`config/rgb_ablation_splits_seibersdorf.yaml` groups by directory-name prefix
and is explicitly `selection_eligible: false` because that grouping is not a
verified recording session. `tools/build_promotion_splits.py` builds the same
manifest shape from an EXPLICIT operator-supplied session mapping instead, so
these tests cover: reserved sessions actually land in `final` and disappear
from development folds, the leak guard fires when a session is asked to be
both, an empty reserve is refused, a final set with zero approved blocks is
refused, and the output is consumable by `experiments/fold_safety.py`'s
functions without modification (Track A3 requirement 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT))

from build_promotion_splits import build_promotion_splits, load_session_map  # noqa: E402

from experiments.fold_safety import (  # noqa: E402
    development_fold_of_snapshot,
    fold_of_group,
    select_development_groups,
)

# Two snapshots each in three sessions; session_c is the one we reserve.
_SESSION_MAP = {
    "s1a": "session_a",
    "s1b": "session_a",
    "s2a": "session_b",
    "s2b": "session_b",
    "s3a": "session_c",
    "s3b": "session_c",
}
_APPROVED_BLOCKS = {"s1a": 2, "s1b": 1, "s2a": 3, "s2b": 0, "s3a": 1, "s3b": 2}


def test_reserved_session_lands_in_final_and_excluded_from_development() -> None:
    manifest = build_promotion_splits(_SESSION_MAP, ["session_c"], _APPROVED_BLOCKS)

    assert manifest["selection_eligible"] is True
    assert manifest["final_enabled"] is True
    assert manifest["grouping_evidence"] == "operator_supplied_session_mapping"
    assert manifest["groups"]["session_c"]["split"] == "final"
    assert "fold" not in manifest["groups"]["session_c"]
    assert sorted(manifest["groups"]["session_c"]["snapshots"]) == ["s3a", "s3b"]
    assert {
        group["split"] for name, group in manifest["groups"].items() if name != "session_c"
    } == {"development"}

    snapshot_fold = development_fold_of_snapshot(manifest)
    assert "s3a" not in snapshot_fold
    assert "s3b" not in snapshot_fold
    assert set(snapshot_fold) == {"s1a", "s1b", "s2a", "s2b"}


def test_development_folds_are_one_per_session_leave_one_out() -> None:
    manifest = build_promotion_splits(_SESSION_MAP, ["session_c"], _APPROVED_BLOCKS)
    dev_groups = {
        name: group for name, group in manifest["groups"].items() if group["split"] == "development"
    }
    assert {name: group["fold"] for name, group in dev_groups.items()} == {
        "session_a": 0,
        "session_b": 1,
    }
    assert dev_groups["session_a"]["snapshots"] == ["s1a", "s1b"]
    assert dev_groups["session_b"]["snapshots"] == ["s2a", "s2b"]


def test_explicit_leak_is_refused() -> None:
    with pytest.raises(ValueError, match="both development and final"):
        build_promotion_splits(
            _SESSION_MAP,
            ["session_c"],
            _APPROVED_BLOCKS,
            development_sessions=["session_a", "session_c"],
        )


def test_empty_reserved_set_is_refused() -> None:
    with pytest.raises(ValueError, match="reserve_sessions is empty"):
        build_promotion_splits(_SESSION_MAP, [], _APPROVED_BLOCKS)


def test_unknown_reserved_session_is_refused() -> None:
    with pytest.raises(ValueError, match="not present in the session mapping"):
        build_promotion_splits(_SESSION_MAP, ["session_nonexistent"], _APPROVED_BLOCKS)


def test_reserved_set_with_zero_approved_blocks_is_refused() -> None:
    # session_b's snapshots have 3 and 0 approved blocks; hold out only the
    # empty one by carving development explicitly around it.
    blocks = dict(_APPROVED_BLOCKS)
    blocks["s3a"] = 0
    blocks["s3b"] = 0
    with pytest.raises(ValueError, match="zero approved blocks"):
        build_promotion_splits(_SESSION_MAP, ["session_c"], blocks)


def test_reserving_every_session_leaves_no_development_and_is_refused() -> None:
    with pytest.raises(ValueError, match="no development sessions remain"):
        build_promotion_splits(
            _SESSION_MAP, ["session_a", "session_b", "session_c"], _APPROVED_BLOCKS
        )


def test_load_session_map_rejects_non_string_or_empty_session_id(tmp_path: Path) -> None:
    bad = tmp_path / "sessions.yaml"
    bad.write_text("snap_a: 3\n")
    with pytest.raises(ValueError, match="non-empty string"):
        load_session_map(bad)


def test_load_session_map_reads_the_real_seibersdorf_mapping() -> None:
    mapping = load_session_map(REPO_ROOT / "config" / "seibersdorf_sessions.yaml")
    assert mapping["2026_07_07_grip_at_bottom__1783428244_423788800"] == "2026_07_07_grip_at_bottom"
    assert len(mapping) > 20


# --- fold_safety consumability (Track A3 requirement 4) --------------------


class _FakeSample:
    def __init__(self, split_group: str, snapshot: str) -> None:
        self.split_group = split_group
        self.raw = {"source": {"snapshot": snapshot}}


class _FakeDataset:
    def __init__(self, root: Path, samples: tuple) -> None:
        self.root = root
        self.samples = samples

    def groups(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for sample in self.samples:
            out.setdefault(sample.split_group, []).append(sample)
        return out


def test_fold_safety_consumes_generated_manifest_end_to_end() -> None:
    manifest = build_promotion_splits(_SESSION_MAP, ["session_c"], _APPROVED_BLOCKS)
    snapshot_fold = development_fold_of_snapshot(manifest)

    # fold_of_group resolves a whole split_group of samples via their own
    # recorded source.snapshot, never by string-matching split_group itself.
    session_a_samples = [_FakeSample("session_a", "s1a"), _FakeSample("session_a", "s1b")]
    assert fold_of_group(session_a_samples, snapshot_fold) == 0

    # A final-only session resolves to no development fold at all.
    session_c_samples = [_FakeSample("session_c", "s3a"), _FakeSample("session_c", "s3b")]
    assert fold_of_group(session_c_samples, snapshot_fold) is None

    # select_development_groups drops every sample belonging to the reserved
    # (final) session and keeps development samples grouped by fold.
    all_samples = (
        *session_a_samples,
        _FakeSample("session_b", "s2a"),
        _FakeSample("session_b", "s2b"),
        *session_c_samples,
    )
    dataset = _FakeDataset(root=Path("/nonexistent"), samples=all_samples)
    dev_dataset, group_fold = select_development_groups(dataset, snapshot_fold)

    assert group_fold == {"session_a": 0, "session_b": 1}
    kept_snapshots = {sample.raw["source"]["snapshot"] for sample in dev_dataset.samples}
    assert kept_snapshots == {"s1a", "s1b", "s2a", "s2b"}
    assert isinstance(dev_dataset, _FakeDataset)
