"""Session/group assignment for offline RGB export splits."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.rgb_candidate_prior.sessions import (  # noqa: E402
    assign_scenario_groups,
    assign_sessions,
)


def test_assign_sessions_clusters_by_timestamp_gap() -> None:
    names = ["100_a", "110_b", "500_c", "510_d"]
    mapping = assign_sessions(names, gap_seconds=60.0)
    assert mapping["100_a"] == mapping["110_b"]
    assert mapping["500_c"] == mapping["510_d"]
    assert mapping["100_a"] != mapping["500_c"]


def test_assign_sessions_groups_unparseable_names_together() -> None:
    names = ["100_a", "no_timestamp_here", "also_bad"]
    mapping = assign_sessions(names, gap_seconds=60.0)
    assert mapping["no_timestamp_here"] == "session_unknown"
    assert mapping["also_bad"] == "session_unknown"


def test_assign_sessions_overrides_win() -> None:
    names = ["100_a", "110_b"]
    mapping = assign_sessions(names, gap_seconds=60.0, overrides={"100_a": "manual_session"})
    assert mapping["100_a"] == "manual_session"
    assert mapping["110_b"] != "manual_session"


def test_assign_scenario_groups_uses_label_before_timestamp() -> None:
    names = [
        "grip_at_bottom__1783428244_423788800",
        "grip_at_bottom__1783428245_523785984",
        "wall__1783429000_318426368",
    ]
    mapping = assign_scenario_groups(names)
    assert mapping["grip_at_bottom__1783428244_423788800"] == "grip_at_bottom"
    assert mapping["grip_at_bottom__1783428245_523785984"] == "grip_at_bottom"
    assert mapping["wall__1783429000_318426368"] == "wall"


def test_assign_scenario_groups_flags_unmatched_names() -> None:
    mapping = assign_scenario_groups(["not_the_right_shape"])
    assert mapping["not_the_right_shape"] == "group_unknown"


def test_assign_scenario_groups_overrides_win() -> None:
    names = ["grip_at_bottom__1783428244_423788800"]
    mapping = assign_scenario_groups(
        names, overrides={"grip_at_bottom__1783428244_423788800": "manual_group"}
    )
    assert mapping["grip_at_bottom__1783428244_423788800"] == "manual_group"
