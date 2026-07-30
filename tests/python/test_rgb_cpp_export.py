"""Focused contracts for production-trace RGB candidate exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from blockpose.annotations import AnnotationTarget
from blockpose.io import CameraModel
from blockpose.pose import BlockPose
from PIL import Image

from experiments.rgb_candidate_prior import gates, render
from experiments.rgb_candidate_prior.candidates import CandidateSetParams
from experiments.rgb_candidate_prior.cpp_export import (
    CppCandidate,
    _record_target,
    load_detector_trace,
)


def _pose(x: float) -> BlockPose:
    return BlockPose(
        position=np.array([x, 0.0, 5.0]),
        rotation=np.eye(3),
        dims=(0.9, 0.6, 0.6),
    )


def test_candidates_for_one_target_share_the_exact_rgb_crop(tmp_path: Path) -> None:
    """Candidate-specific conditioning must not be accompanied by RGB crop leakage."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    rgb = np.indices((100, 200)).sum(axis=0).astype(np.uint8)
    rgb = np.repeat(rgb[..., None], 3, axis=2)
    render.save_png(rgb, snapshot / "rgb.png")
    camera = CameraModel(np.array([[80, 0, 100], [0, 80, 50], [0, 0, 1]]), 200, 100)
    target = AnnotationTarget("block", _pose(0.0), "manual", visibility="full")
    candidates = [
        CppCandidate("left", _pose(-0.2), "c++_geometry", {}),
        CppCandidate("right", _pose(0.2), "c++_geometry", {}),
    ]
    records: list[dict] = []
    written, dropped = _record_target(
        snapshot=snapshot,
        target=target,
        candidates=candidates,
        export_root=tmp_path / "out",
        rgb=rgb,
        camera=camera,
        T_world_camera=np.eye(4),
        image_stamp_ns=1,
        cloud_stamp_ns=1,
        sync_delta_s=0.0,
        calibration_checksum="calibration",
        trace_checksum="trace",
        tf_checksum="tf",
        split_group="recording_a",
        crop_margin_frac=0.25,
        out_size=64,
        params=CandidateSetParams(),
        records=records,
    )

    assert (written, dropped) == (2, 0)
    assert len({tuple(record["crop_bbox_xyxy"]) for record in records}) == 1
    rendered = [np.asarray(Image.open(tmp_path / "out" / record["rgb"])) for record in records]
    np.testing.assert_array_equal(rendered[0], rendered[1])
    assert not np.array_equal(
        np.asarray(Image.open(tmp_path / "out" / records[0]["conditioning"])),
        np.asarray(Image.open(tmp_path / "out" / records[1]["conditioning"])),
    )


def test_cpp_trace_requires_post_refinement_candidate_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"schema_version": 2, "snapshots": [{"snapshot": "snap", "raw_lineage": []}]})
    )
    with pytest.raises(ValueError, match="candidate_trace"):
        load_detector_trace(trace)


def test_cpp_trace_rejects_schema1_capture_shape(tmp_path: Path) -> None:
    """Feeding the detector_candidates.yaml capture shape must name the mismatch, not KeyError."""
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id_namespace": "request_local",
                "reference_frame": "world",
                "candidates": [
                    {
                        "candidate_id": "detector_candidate_000",
                        "pose": {
                            "position": [-11.5555, 2.63818, 0.693449],
                            "orientation_xyzw": [-0.00766093, 0.0287269, 0.685699, 0.727278],
                        },
                        "dimensions": [0.9, 0.6, 0.6],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="schema_version 1 'detector_candidates.yaml'"):
        load_detector_trace(trace)


def test_cpp_trace_loads_runner_pose_shape(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "snapshots": [
                    {
                        "snapshot": "snap",
                        "candidate_trace_stage": "post_refinement_pre_nms",
                        "candidate_trace": [
                            {
                                "id": "component-0/seed",
                                "source": "fk_seed",
                                "stage": "post_refinement_pre_nms",
                                "pose": {
                                    "position": [1.0, 2.0, 3.0],
                                    "rotation": np.eye(3).tolist(),
                                    "dims": [0.9, 0.6, 0.6],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )
    loaded, header = load_detector_trace(trace)
    assert header["schema_version"] == 2
    assert loaded["snap"][0].candidate_id == "component-0/seed"
    assert loaded["snap"][0].origin == "fk_seed"
    np.testing.assert_allclose(loaded["snap"][0].pose.position, [1.0, 2.0, 3.0])


def test_capture_metadata_uses_rgb_stamp_section(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        "rgb:\n  stamp: {sec: 4, nanosec: 5}\ncloud:\n  stamp: {sec: 6, nanosec: 7}\n"
    )
    assert gates.read_stamps(metadata) == (4_000_000_005, 6_000_000_007)
