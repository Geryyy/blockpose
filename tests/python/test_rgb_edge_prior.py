"""Tests for classical candidate-conditioned RGB edge evidence."""

from __future__ import annotations

import numpy as np
from blockpose.io import CameraModel
from blockpose.pose import BlockPose
from blockpose.rgb_edge_prior import EdgePriorParams, build_edge_map, score_projected_cuboid
from blockpose.viz import _EDGES, box_corners


def _camera() -> CameraModel:
    intrinsics = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
    return CameraModel(intrinsics, 160, 120)


def _pose(yaw_deg: float = 0.0) -> BlockPose:
    yaw = np.radians(yaw_deg)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    return BlockPose(np.array([0.0, 0.0, 5.0]), rotation, dims=(1.4, 0.6, 0.6))


def _wireframe_image(pose: BlockPose, camera: CameraModel) -> np.ndarray:
    image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    corners = box_corners(pose, pose.dims)
    uv = corners[:, :2] / corners[:, 2:3]
    uv = uv * np.array([camera.fx, camera.fy]) + np.array([camera.cx, camera.cy])
    for start, end in _EDGES:
        count = max(2, int(np.linalg.norm(uv[end] - uv[start])) + 1)
        line = np.rint(
            uv[start] + np.linspace(0.0, 1.0, count)[:, None] * (uv[end] - uv[start])
        ).astype(int)
        valid = (
            (line[:, 0] >= 0)
            & (line[:, 0] < camera.width)
            & (line[:, 1] >= 0)
            & (line[:, 1] < camera.height)
        )
        image[line[valid, 1], line[valid, 0]] = 255
    return image


def test_correct_projection_beats_yaw_sibling() -> None:
    camera, true_pose = _camera(), _pose()
    edge_map = build_edge_map(
        _wireframe_image(true_pose, camera), EdgePriorParams(edge_percentile=85.0)
    )
    correct = score_projected_cuboid(edge_map, true_pose, np.eye(4), camera)
    wrong = score_projected_cuboid(edge_map, _pose(90.0), np.eye(4), camera)
    assert correct.evidence and wrong.evidence
    assert correct.score > wrong.score + 0.08


def test_excluded_mask_and_missing_input_are_neutral() -> None:
    camera, pose = _camera(), _pose()
    edge_map = build_edge_map(_wireframe_image(pose, camera))
    excluded = np.ones((camera.height, camera.width), dtype=bool)
    excluded_score = score_projected_cuboid(
        edge_map, pose, np.eye(4), camera, excluded_mask=excluded
    )
    assert not excluded_score.evidence
    assert not score_projected_cuboid(None, pose, np.eye(4), camera).evidence


def test_behind_camera_and_repeat_score_are_safe_and_deterministic() -> None:
    camera, pose = _camera(), _pose()
    edge_map = build_edge_map(_wireframe_image(pose, camera))
    first = score_projected_cuboid(edge_map, pose, np.eye(4), camera)
    second = score_projected_cuboid(edge_map, pose, np.eye(4), camera)
    assert first.score == second.score
    behind = BlockPose(np.array([0.0, 0.0, -2.0]), np.eye(3), dims=pose.dims)
    assert not score_projected_cuboid(edge_map, behind, np.eye(4), camera).evidence


def test_downsampled_edge_map_keeps_source_coordinates() -> None:
    camera, pose = _camera(), _pose()
    edge_map = build_edge_map(
        _wireframe_image(pose, camera),
        EdgePriorParams(edge_percentile=85.0, max_image_dimension_px=60),
    )
    assert edge_map is not None
    assert edge_map.source_shape == (camera.height, camera.width)
    assert max(edge_map.distance_px.shape) == 60
    assert score_projected_cuboid(edge_map, pose, np.eye(4), camera).evidence
    source_mask = np.ones((camera.height, camera.width), dtype=bool)
    excluded = score_projected_cuboid(edge_map, pose, np.eye(4), camera, excluded_mask=source_mask)
    assert not excluded.evidence
    assert excluded.reason == "insufficient_visible_samples"
