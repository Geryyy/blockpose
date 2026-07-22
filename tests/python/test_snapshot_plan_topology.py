from __future__ import annotations

from pathlib import Path

import pytest
from blockpose.snapshot_plan_topology import load_snapshot_plan_topology


def test_snapshot_topology_loads_before_gt_and_covers_safe_priors() -> None:
    topology = load_snapshot_plan_topology()
    assert sum(len(value) for value in topology.priors_by_snapshot.values()) == 15
    assert topology.provenance["ground_truth_loaded"] is False
    assert topology.provenance["prior_source"] == "cross_snapshot_loso"


def test_snapshot_topology_rejects_wrong_search_window(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_text("schema_version: 1\nsearch_window_m: 0.11\ncourses: []\n")
    with pytest.raises(ValueError, match="search_window_m"):
        load_snapshot_plan_topology(path)


def test_snapshot_topology_rejects_unknown_course_block(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_text(
        """schema_version: 1
search_window_m: 0.10
courses:
  - snapshot_id: 1783427923_718112512_seq5
    course_id: bad
    block_ids: [does_not_exist]
"""
    )
    with pytest.raises(ValueError, match="unknown topology block"):
        load_snapshot_plan_topology(path)
