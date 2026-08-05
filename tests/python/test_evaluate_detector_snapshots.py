"""Fast, GPU-free tests for `tools/evaluate_detector_snapshots.py`'s pure logic.

Covers overall/per-scene/per-visibility aggregation and manifest-optional
directory discovery with synthetic inputs -- no GPU, no C++ detector, no
dataset. Mirrors `tests/python/test_dino_proposer_pooled.py`'s style and its
namespace-package import trick (`tools/` has no `__init__.py`).
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.annotation_tool import VISIBILITIES, EditableTarget
from blockpose.pose import BLOCK_DIMS, BlockPose

from tools.evaluate_detector_snapshots import (
    VISIBILITY_CLASSES,
    SkippedSnapshot,
    SnapshotResult,
    _rate,
    aggregate_overall,
    aggregate_per_scene,
    aggregate_per_visibility,
    approved_targets,
    discover_snapshots,
)


def _target(
    id_: str,
    visibility: str = "unknown",
    review_status: str = "approved",
    ignore_reason: str | None = None,
) -> EditableTarget:
    return EditableTarget(
        id=id_,
        position=np.zeros(3),
        rotation=np.eye(3),
        dimensions=BLOCK_DIMS,
        visibility=visibility,
        review_status=review_status,
        ignore_reason=ignore_reason,
        rejection_reason="n/a" if review_status == "rejected" else None,
    )


def _pose(position=(0.0, 0.0, 0.0)) -> BlockPose:
    return BlockPose(np.asarray(position, dtype=np.float64), np.eye(3), dims=BLOCK_DIMS)


def _result(
    scene: str,
    name: str,
    n_gt: int,
    matched_gt: set[int],
    strict_gt: set[int],
    n_det: int,
    matched_det: set[int],
    visibilities: list[str] | None = None,
) -> SnapshotResult:
    visibilities = visibilities or ["unknown"] * n_gt
    gt = [_target(f"{name}:{i}", visibility=visibilities[i]) for i in range(n_gt)]
    detections = [_pose((float(i), 0.0, 0.0)) for i in range(n_det)]
    return SnapshotResult(
        scene=scene,
        name=name,
        gt=gt,
        detections=detections,
        matched_gt=matched_gt,
        strict_gt=strict_gt,
        matched_det=matched_det,
    )


# --------------------------------------------------------------------------
# approved_targets: GT definition mirrors AnnotationTarget.is_evaluation_target
# --------------------------------------------------------------------------


class _FakeDoc:
    def __init__(self, targets: list[EditableTarget]) -> None:
        self.targets = targets


def test_approved_targets_excludes_pending_and_rejected() -> None:
    doc = _FakeDoc(
        [
            _target("a", review_status="approved"),
            _target("b", review_status="pending"),
            _target("c", review_status="rejected"),
        ]
    )
    assert [t.id for t in approved_targets(doc)] == ["a"]


def test_approved_targets_excludes_ignored_even_if_approved() -> None:
    doc = _FakeDoc(
        [
            _target("a", review_status="approved"),
            _target("b", review_status="approved", ignore_reason="clutter"),
        ]
    )
    assert [t.id for t in approved_targets(doc)] == ["a"]


# --------------------------------------------------------------------------
# aggregate_overall / aggregate_per_scene
# --------------------------------------------------------------------------


def test_aggregate_overall_sums_across_snapshots() -> None:
    results = [
        _result(
            "scene_a", "s1", n_gt=3, matched_gt={0, 1}, strict_gt={0}, n_det=4, matched_det={0, 1}
        ),
        _result("scene_a", "s2", n_gt=2, matched_gt={0}, strict_gt={0}, n_det=1, matched_det={0}),
    ]
    stats = aggregate_overall(results)
    assert stats.gt_total == 5
    assert stats.det_total == 5
    assert stats.matched_centre == 3
    assert stats.matched_det == 3
    assert stats.matched_strict == 2


def test_aggregate_overall_empty_is_all_zero() -> None:
    stats = aggregate_overall([])
    assert (
        stats.gt_total,
        stats.det_total,
        stats.matched_centre,
        stats.matched_det,
        stats.matched_strict,
    ) == (0, 0, 0, 0, 0)


def test_aggregate_overall_precision_denominator_counts_unmatched_detections() -> None:
    """Precision must be true-positives over ALL detections, never derived from
    recall. This snapshot has 5 detections of which 1 matches: precision is
    1/5, not 1/2 (recall's denominator) and not 1/1 (matched-only). Note the
    numerators are necessarily equal -- `match_detections` is a 1-to-1
    Hungarian assignment, so `|matched_det| == |matched_gt|` always holds on
    real data and no test over valid inputs can separate them; the denominator
    is where a recall-derived precision would give itself away.
    """
    results = [
        _result("scene_a", "s1", n_gt=2, matched_gt={0}, strict_gt={0}, n_det=5, matched_det={0}),
    ]
    stats = aggregate_overall(results)
    assert (stats.gt_total, stats.det_total) == (2, 5)
    assert _rate(stats.matched_det, stats.det_total) == "1/5 (20.0%)"
    assert _rate(stats.matched_centre, stats.gt_total) == "1/2 (50.0%)"


def test_aggregate_per_scene_groups_by_scene_not_snapshot() -> None:
    results = [
        _result("scene_a", "s1", n_gt=2, matched_gt={0}, strict_gt=set(), n_det=1, matched_det={0}),
        _result(
            "scene_a", "s2", n_gt=1, matched_gt=set(), strict_gt=set(), n_det=0, matched_det=set()
        ),
        _result(
            "scene_b",
            "s3",
            n_gt=3,
            matched_gt={0, 1, 2},
            strict_gt={0, 1, 2},
            n_det=3,
            matched_det={0, 1, 2},
        ),
    ]
    per_scene = aggregate_per_scene(results)
    assert set(per_scene) == {"scene_a", "scene_b"}
    assert per_scene["scene_a"].gt_total == 3
    assert per_scene["scene_a"].matched_centre == 1
    assert per_scene["scene_b"].matched_centre == 3
    assert per_scene["scene_b"].matched_strict == 3


# --------------------------------------------------------------------------
# aggregate_per_visibility
# --------------------------------------------------------------------------


def test_aggregate_per_visibility_buckets_by_gt_label() -> None:
    results = [
        _result(
            "scene_a",
            "s1",
            n_gt=3,
            matched_gt={0, 2},
            strict_gt=set(),
            n_det=2,
            matched_det={0, 1},
            visibilities=["full", "partial", "heavy_occlusion"],
        ),
    ]
    per_visibility = aggregate_per_visibility(results)
    assert per_visibility["full"] == (1, 1)  # matched
    assert per_visibility["partial"] == (0, 1)  # missed
    assert per_visibility["heavy_occlusion"] == (1, 1)  # matched
    assert per_visibility["unknown"] == (0, 0)


def test_visibility_classes_cover_the_annotation_tool_vocabulary() -> None:
    """The report's buckets must be the annotation tool's own vocabulary
    (`annotation_tool.VISIBILITIES`), only reordered worst-to-best for
    reading. If a class is ever added there, every block carrying it would
    otherwise be silently re-bucketed into "unknown" and the occlusion
    breakdown -- the whole point of that table -- would quietly go wrong.
    """
    assert set(VISIBILITY_CLASSES) == set(VISIBILITIES)


def test_aggregate_per_visibility_unrecognised_label_falls_back_to_unknown() -> None:
    results = [
        _result(
            "scene_a",
            "s1",
            n_gt=1,
            matched_gt=set(),
            strict_gt=set(),
            n_det=0,
            matched_det=set(),
            visibilities=["not_a_real_class"],
        ),
    ]
    per_visibility = aggregate_per_visibility(results)
    assert per_visibility["unknown"] == (0, 1)
    assert "not_a_real_class" not in per_visibility


def test_aggregate_per_visibility_precision_has_no_class_but_recall_totals_match_gt() -> None:
    """Every GT block ends up in exactly one visibility bucket -- the sum of
    bucketed GT totals equals the overall GT total, so the breakdown never
    silently drops a block."""
    results = [
        _result(
            "scene_a",
            "s1",
            n_gt=4,
            matched_gt={0, 1},
            strict_gt=set(),
            n_det=2,
            matched_det={0, 1},
            visibilities=["full", "partial", "heavy_occlusion", "unknown"],
        ),
    ]
    per_visibility = aggregate_per_visibility(results)
    assert sum(total for _, total in per_visibility.values()) == aggregate_overall(results).gt_total


# --------------------------------------------------------------------------
# _rate
# --------------------------------------------------------------------------


def test_rate_formats_fraction_and_percentage() -> None:
    assert _rate(46, 62) == "46/62 (74.2%)"


def test_rate_zero_denominator_is_not_a_crash() -> None:
    assert _rate(0, 0) == "n/a"


# --------------------------------------------------------------------------
# discover_snapshots: manifest-optional directory selection
# --------------------------------------------------------------------------


def _write_annotations(
    path: Path,
    snapshot_name: str,
    *,
    complete: bool = True,
    blocks: list[dict] | None = None,
) -> None:
    data = {
        "schema_version": 2,
        "snapshot": snapshot_name,
        "annotation_complete": complete,
        "notes": "",
        "blocks": blocks if blocks is not None else [],
    }
    if complete:
        data["annotated_roi"] = {"min": [-10.0, -10.0, -1.0], "max": [10.0, 10.0, 5.0]}
    path.write_text(yaml.safe_dump(data))


def _approved_block(id_: str) -> dict:
    return {
        "id": id_,
        "review_status": "approved",
        "provenance": "manual",
        "visibility": "full",
        "dimensions": list(BLOCK_DIMS),
        "confidence": 0.0,
        "pose": {
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }


def test_discover_snapshots_includes_complete_with_approved_block(tmp_path: Path) -> None:
    d = tmp_path / "scene__001"
    d.mkdir()
    _write_annotations(
        d / "annotations.yaml", d.name, complete=True, blocks=[_approved_block("b0")]
    )

    included, skipped = discover_snapshots(tmp_path)

    assert included == [("scene", "scene__001")]
    assert skipped == []


def test_discover_snapshots_skips_missing_annotations(tmp_path: Path) -> None:
    d = tmp_path / "scene__001"
    d.mkdir()  # no annotations.yaml at all

    included, skipped = discover_snapshots(tmp_path)

    assert included == []
    assert len(skipped) == 1
    assert skipped[0] == SkippedSnapshot("scene__001", "no annotations.yaml")


def test_discover_snapshots_skips_incomplete(tmp_path: Path) -> None:
    d = tmp_path / "scene__001"
    d.mkdir()
    _write_annotations(d / "annotations.yaml", d.name, complete=False, blocks=[])

    included, skipped = discover_snapshots(tmp_path)

    assert included == []
    assert skipped[0].name == "scene__001"
    assert "annotation_complete" in skipped[0].reason


def test_discover_snapshots_skips_zero_approved_blocks(tmp_path: Path) -> None:
    d = tmp_path / "scene__001"
    d.mkdir()
    pending_block = _approved_block("b0")
    pending_block["review_status"] = "pending"
    _write_annotations(d / "annotations.yaml", d.name, complete=True, blocks=[pending_block])

    included, skipped = discover_snapshots(tmp_path)

    assert included == []
    assert skipped[0].reason == "zero approved targets"


def test_discover_snapshots_ignores_non_directory_entries(tmp_path: Path) -> None:
    (tmp_path / "stray_file.txt").write_text("not a snapshot")

    included, skipped = discover_snapshots(tmp_path)

    assert included == []
    assert skipped == []


def test_discover_snapshots_scene_is_directory_prefix_before_double_underscore(
    tmp_path: Path,
) -> None:
    d = tmp_path / "my_scene_label__1783428244_423788800"
    d.mkdir()
    _write_annotations(
        d / "annotations.yaml", d.name, complete=True, blocks=[_approved_block("b0")]
    )

    included, _ = discover_snapshots(tmp_path)

    assert included == [("my_scene_label", d.name)]


def test_discover_snapshots_mixed_directory_reports_correct_denominator(tmp_path: Path) -> None:
    good = tmp_path / "scene_a__001"
    good.mkdir()
    _write_annotations(
        good / "annotations.yaml", good.name, complete=True, blocks=[_approved_block("b0")]
    )

    incomplete = tmp_path / "scene_b__002"
    incomplete.mkdir()
    _write_annotations(incomplete / "annotations.yaml", incomplete.name, complete=False, blocks=[])

    no_annotations = tmp_path / "scene_c__003"
    no_annotations.mkdir()

    included, skipped = discover_snapshots(tmp_path)

    assert len(included) == 1
    assert len(skipped) == 2
    assert included[0][1] == good.name
