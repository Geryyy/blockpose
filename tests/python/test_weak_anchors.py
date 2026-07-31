"""Multi-frame agreement matching and health-report accounting for weak_anchors.py.

Synthetic hypothesis/camera data only -- no real point clouds, no GPU, no torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.detection import TraceHypothesis
from blockpose.io import CameraModel
from blockpose.pose import BLOCK_DIMS, BlockPose

from experiments.rgb_candidate_prior.weak_anchors import (
    _health_report,
    anchors_for_group,
    find_agreements,
)


def _pose(x: float, y: float, z: float, *, yaw_deg: float = 0.0) -> BlockPose:
    theta = np.radians(yaw_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    return BlockPose(position=np.array([x, y, z]), rotation=rotation, dims=BLOCK_DIMS)


def _hyp(hyp_id: str, pose: BlockPose, score: float) -> TraceHypothesis:
    return TraceHypothesis(id=hyp_id, pose=pose, score=score, stage="raw", fate="retained")


def _camera() -> CameraModel:
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    return CameraModel(K=K, width=640, height=480, D=np.zeros(0))


_IDENTITY = np.eye(4)


def _capture_outcome(hypotheses: list[TraceHypothesis], *, name: str) -> dict:
    return {
        "capture": name,
        "status": "ok",
        "T_world_camera": _IDENTITY,
        "hypotheses": tuple(hypotheses),
        "image_stamp_ns": 0,
        "cloud_stamp_ns": 0,
        "sync_delta_s": 0.0,
        "rgb_checksum": "deadbeef",
        "tf_checksum": "deadbeef",
    }


# ---------------------------------------------------------------------------
# find_agreements
# ---------------------------------------------------------------------------


def test_find_agreements_matches_within_tolerance() -> None:
    anchor = _hyp("a", _pose(0.0, 0.0, 5.0), 0.6)
    close = _hyp("b", _pose(0.05, 0.0, 5.0), 0.6)
    others = [("capture_2", close)]
    agreements = find_agreements(anchor, others, pos_tol_m=0.25, rot_tol_deg=30.0)
    assert len(agreements) == 1
    assert agreements[0].capture == "capture_2"
    assert agreements[0].hypothesis_id == "b"
    assert agreements[0].translation_error_m == pytest.approx(0.05)
    assert agreements[0].rotation_error_deg == pytest.approx(0.0)


def test_find_agreements_rejects_beyond_position_tolerance() -> None:
    anchor = _hyp("a", _pose(0.0, 0.0, 5.0), 0.6)
    far = _hyp("b", _pose(1.0, 0.0, 5.0), 0.6)
    agreements = find_agreements(anchor, [("capture_2", far)], pos_tol_m=0.25, rot_tol_deg=30.0)
    assert agreements == []


def test_find_agreements_rejects_beyond_rotation_tolerance() -> None:
    anchor = _hyp("a", _pose(0.0, 0.0, 5.0, yaw_deg=0.0), 0.6)
    rotated = _hyp("b", _pose(0.0, 0.0, 5.0, yaw_deg=80.0), 0.6)
    agreements = find_agreements(anchor, [("capture_2", rotated)], pos_tol_m=0.25, rot_tol_deg=30.0)
    assert agreements == []


def test_find_agreements_ignores_same_capture_by_caller_contract() -> None:
    # find_agreements itself has no capture-identity notion; the caller
    # (anchors_for_group) is responsible for excluding a hypothesis's own
    # capture from its "others" list. This test documents that division.
    anchor = _hyp("a", _pose(0.0, 0.0, 5.0), 0.6)
    same_capture_dup = _hyp("a", _pose(0.0, 0.0, 5.0), 0.6)
    agreements = find_agreements(
        anchor, [("capture_1", same_capture_dup)], pos_tol_m=0.25, rot_tol_deg=30.0
    )
    assert len(agreements) == 1  # would match; anchors_for_group excludes this case


# ---------------------------------------------------------------------------
# anchors_for_group
# ---------------------------------------------------------------------------


def test_anchors_for_group_keeps_multiframe_agreeing_strong_hypotheses() -> None:
    pose_a = _pose(0.0, 0.0, 5.0)
    pose_b = _pose(0.03, 0.0, 5.0)  # within tolerance of pose_a
    outcomes = {
        "capture_1": _capture_outcome([_hyp("h0", pose_a, 0.6)], name="capture_1"),
        "capture_2": _capture_outcome([_hyp("h0", pose_b, 0.6)], name="capture_2"),
    }
    records, counts = anchors_for_group(
        "group_x",
        outcomes,
        camera=_camera(),
        calibration_checksum="cksum",
        min_score=0.4,
        agreement_pos_tol_m=0.25,
        agreement_rot_tol_deg=30.0,
        crop_margin_frac=0.25,
    )
    assert counts["anchors_kept"] == 2  # one anchor per source capture
    assert counts["hypotheses_strong"] == 2
    assert counts["no_agreement"] == 0
    captures = {record["source_capture"] for record in records}
    assert captures == {"capture_1", "capture_2"}
    for record in records:
        assert record["provenance"] == "weak_pcl_multiframe_anchor"
        assert record["multiframe_agreement"]["agreement_count"] == 1
        assert record["pcl_evidence"]["score"] == 0.6


def test_anchors_for_group_drops_single_frame_only_hypothesis() -> None:
    pose_lonely = _pose(2.0, 2.0, 5.0)  # far from anything in the other capture
    outcomes = {
        "capture_1": _capture_outcome([_hyp("h0", pose_lonely, 0.6)], name="capture_1"),
        "capture_2": _capture_outcome([_hyp("h0", _pose(0.0, 0.0, 5.0), 0.6)], name="capture_2"),
    }
    records, counts = anchors_for_group(
        "group_x",
        outcomes,
        camera=_camera(),
        calibration_checksum="cksum",
        min_score=0.4,
        agreement_pos_tol_m=0.25,
        agreement_rot_tol_deg=30.0,
        crop_margin_frac=0.25,
    )
    assert records == []
    assert counts["no_agreement"] == 2
    assert counts["anchors_kept"] == 0


def test_anchors_for_group_ignores_hypotheses_below_score_threshold() -> None:
    pose_a = _pose(0.0, 0.0, 5.0)
    pose_b = _pose(0.03, 0.0, 5.0)
    outcomes = {
        "capture_1": _capture_outcome([_hyp("h0", pose_a, 0.2)], name="capture_1"),  # weak
        "capture_2": _capture_outcome([_hyp("h0", pose_b, 0.6)], name="capture_2"),  # strong
    }
    records, counts = anchors_for_group(
        "group_x",
        outcomes,
        camera=_camera(),
        calibration_checksum="cksum",
        min_score=0.4,
        agreement_pos_tol_m=0.25,
        agreement_rot_tol_deg=30.0,
        crop_margin_frac=0.25,
    )
    # The weak hypothesis is never promoted to "strong", so it can't supply
    # agreement evidence and the lone strong one has no strong partner.
    assert records == []
    assert counts["hypotheses_strong"] == 1
    assert counts["hypotheses_considered"] == 2


def test_anchors_for_group_drops_anchors_that_fail_projection() -> None:
    # Behind the camera (Z < 0 in the optical frame == world frame here since
    # T_world_camera is identity): agreement holds but projection must fail.
    pose_a = _pose(0.0, 0.0, -5.0)
    pose_b = _pose(0.03, 0.0, -5.0)
    outcomes = {
        "capture_1": _capture_outcome([_hyp("h0", pose_a, 0.6)], name="capture_1"),
        "capture_2": _capture_outcome([_hyp("h0", pose_b, 0.6)], name="capture_2"),
    }
    records, counts = anchors_for_group(
        "group_x",
        outcomes,
        camera=_camera(),
        calibration_checksum="cksum",
        min_score=0.4,
        agreement_pos_tol_m=0.25,
        agreement_rot_tol_deg=30.0,
        crop_margin_frac=0.25,
    )
    assert records == []
    assert counts["projection_dropped"] == 2
    assert counts["no_agreement"] == 0


# ---------------------------------------------------------------------------
# _health_report
# ---------------------------------------------------------------------------


def test_health_report_flags_single_capture_and_zero_anchor_groups() -> None:
    per_group = [
        {
            "group": "lonely_scenario",
            "captures": [{"capture": "c1", "status": "ok"}],
            "counts": {
                "hypotheses_considered": 1,
                "hypotheses_strong": 1,
                "no_agreement": 1,
                "projection_dropped": 0,
                "anchors_kept": 0,
            },
        },
        {
            "group": "empty_scenario",
            "captures": [{"capture": "c2", "status": "ok"}, {"capture": "c3", "status": "ok"}],
            "counts": {
                "hypotheses_considered": 2,
                "hypotheses_strong": 0,
                "no_agreement": 0,
                "projection_dropped": 0,
                "anchors_kept": 0,
            },
        },
        {
            "group": "productive_scenario",
            "captures": [
                {"capture": "c4", "status": "ok"},
                {"capture": "c5", "status": "ok"},
                {
                    "capture": "c6",
                    "status": "gate_failed",
                    "gate": "sync",
                    "reason": "sync_delta_exceeds_tolerance",
                    "details": {},
                },
            ],
            "counts": {
                "hypotheses_considered": 4,
                "hypotheses_strong": 3,
                "no_agreement": 1,
                "projection_dropped": 0,
                "anchors_kept": 2,
            },
        },
    ]
    health = _health_report(per_group)

    assert health["totals"]["anchors_kept"] == 2
    assert health["totals"]["hypotheses_considered"] == 7
    assert health["gate_failures"] == {"sync:sync_delta_exceeds_tolerance": 1}
    assert health["capture_status"] == {"ok": 5, "gate_failed": 1}
    assert health["errors"] == []

    warnings_text = " ".join(health["warnings"])
    assert "lonely_scenario" in warnings_text
    assert "empty_scenario" in warnings_text
    assert "productive_scenario" not in warnings_text


def test_health_report_reports_errors_separately_from_gate_failures() -> None:
    per_group = [
        {
            "group": "broken_scenario",
            "captures": [{"capture": "*", "status": "error", "error": "RuntimeError('boom')"}],
            "counts": {
                "hypotheses_considered": 0,
                "hypotheses_strong": 0,
                "no_agreement": 0,
                "projection_dropped": 0,
                "anchors_kept": 0,
            },
        }
    ]
    health = _health_report(per_group)
    assert health["errors"] == [
        {"group": "broken_scenario", "capture": "*", "error": "RuntimeError('boom')"}
    ]
    assert health["capture_status"] == {"error": 1}
