from __future__ import annotations

from pathlib import Path

import numpy as np
from blockpose.detection import DetectionParams, detect_blocks_detailed
from blockpose.snapshot_plan_fixtures import load_snapshot_plan_fixtures
from blockpose.snapshot_scenarios import (
    acceptance_scenarios,
    audit_manifest,
    load_snapshot_scenarios,
    taxonomy_acceptance_summary,
)

ROOT = Path(__file__).parents[2]
SNAPSHOTS = ROOT / "data/registration_snapshots"


def test_taxonomy_exactly_covers_frozen_approved_targets() -> None:
    report = audit_manifest(SNAPSHOTS)
    assert report["approved_targets"] == report["configured_targets"] == 61
    assert report["missing"] == report["extra"] == []
    assert report["approximate"] == [
        "1783428141_224458752_seq3/manual_1",
        "1783428170_524189696_seq9/manual_2",
    ]
    assert report["classification_status_counts"] == {"provisional": 57, "reviewed": 4}
    reviewed = [
        value
        for value in load_snapshot_scenarios().values()
        if value.classification_status == "reviewed"
    ]
    assert len(reviewed) == 4
    assert all(value.scenario == "gripped" for value in reviewed)
    assert all(value.query == "estimate_gripped" for value in reviewed)
    eligible = acceptance_scenarios(load_snapshot_scenarios())
    assert set(eligible) == {value.target_id for value in reviewed}
    assert not any(value.classification_status == "provisional" for value in eligible.values())
    assert taxonomy_acceptance_summary(load_snapshot_scenarios()) == {
        "eligible_targets": 4,
        "excluded_provisional_targets": 57,
        "queries": ["estimate_gripped"],
        "status": "unavailable_until_query_specific_evaluation",
    }


def test_every_verify_target_has_leakage_safe_prior_or_exclusion() -> None:
    scenarios = load_snapshot_scenarios()
    fixtures = load_snapshot_plan_fixtures()
    verify = {key for key, value in scenarios.items() if value.query == "verify_placed"}
    assert set(fixtures) == verify
    assert sum(value.prior is not None for value in fixtures.values()) == 15
    assert sum(value.prior is None for value in fixtures.values()) == 6
    for key, fixture in fixtures.items():
        assert key.split("/", 1)[0] not in fixture.source_snapshot_ids
        assert fixture.search_window_m == 0.10


def test_cross_snapshot_slot_ids_are_stable_for_same_physical_pose() -> None:
    fixtures = load_snapshot_plan_fixtures()
    first = fixtures["1783428498_520086272_seq3/rgb_candidate_1"]
    second = fixtures["1783428548_719285504_seq11/rgb_candidate_1"]
    assert first.stable_slot_id == second.stable_slot_id


def test_cross_snapshot_fixtures_do_not_require_unused_plan_asset(tmp_path: Path) -> None:
    fixtures = load_snapshot_plan_fixtures(plan_style_path=tmp_path / "does-not-exist.yaml")
    assert fixtures
    assert all(value.plan_style is None for value in fixtures.values())


def test_trace_is_opt_in_and_does_not_change_detection() -> None:
    rng = np.random.default_rng(2)
    points = rng.normal(size=(250, 3))
    params = DetectionParams(ground_ransac_iters=10, ransac_num_iterations=10)
    plain = detect_blocks_detailed(points, params)
    traced = detect_blocks_detailed(points, params, collect_trace=True)
    assert plain.trace is None
    assert [pose.position.tolist() for pose in plain.poses] == [
        pose.position.tolist() for pose in traced.poses
    ]
    assert plain.stage_counts == traced.stage_counts
    assert plain.diagnostics == traced.diagnostics
    assert traced.trace is not None


def test_trace_exists_on_every_valid_early_exit() -> None:
    for points in (np.empty((0, 3)), np.zeros((2, 3)), np.zeros((3, 3))):
        result = detect_blocks_detailed(points, collect_trace=True)
        assert result.trace is not None
        assert result.trace.raw_hypotheses == ()


def test_trace_ids_and_outputs_are_point_permutation_stable() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(250, 3))
    params = DetectionParams(ground_ransac_iters=10, ransac_num_iterations=10)
    first = detect_blocks_detailed(points, params, collect_trace=True)
    second = detect_blocks_detailed(
        points[rng.permutation(len(points))], params, collect_trace=True
    )
    assert first.trace is not None and second.trace is not None
    assert [item.id for item in first.trace.raw_hypotheses] == [
        item.id for item in second.trace.raw_hypotheses
    ]
    assert [pose.position.tolist() for pose in first.poses] == [
        pose.position.tolist() for pose in second.poses
    ]
    raw_ids = {item.id for item in first.trace.raw_hypotheses}
    threshold_ids = {item.id for item in first.trace.threshold_hypotheses}
    final_ids = {item.id for item in first.trace.final_hypotheses}
    assert final_ids <= threshold_ids <= raw_ids
    assert all(
        item.fate
        in {
            "final",
            "score_rejected",
            "pre_refinement_selection_rejected",
            "post_refinement_bounds_rejected",
            "post_refinement_nms_rejected",
        }
        for item in first.trace.raw_hypotheses
    )
