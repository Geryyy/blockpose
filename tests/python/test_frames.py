import time

import numpy as np
import pytest
from blockpose.frames import FrameBurst, PointFrame, PoseTrajectory, TimedPose


def _translation(x: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def test_deskew_interpolates_each_point_pose() -> None:
    trajectory = PoseTrajectory(
        (TimedPose(10.0, _translation(0)), TimedPose(11.0, _translation(1)))
    )
    frame = PointFrame(
        np.zeros((3, 3)), timestamp=10.0, relative_time=np.array([0.0, 0.5, 1.0]),
        sensor_trajectory=trajectory,
    )
    np.testing.assert_allclose(frame.points_in_world()[:, 0], [0.0, 0.5, 1.0])


def test_burst_fusion_preserves_provenance_and_unavailable_evidence() -> None:
    frames = FrameBurst(
        (
            PointFrame(np.ones((2, 3)), timestamp=2.0, frame_id="world"),
            PointFrame(np.zeros((1, 3)), timestamp=1.0, frame_id="world"),
        )
    )
    fused = frames.fuse()
    assert fused.points_world.shape == (3, 3)
    assert fused.source_frame_index.tolist() == [1, 0, 0]
    assert fused.source_point_index.tolist() == [0, 0, 1]
    assert "intensity" in fused.unavailable_evidence


def test_non_world_frame_requires_trajectory() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        PointFrame(np.zeros((1, 3))).points_in_world()


def test_fusion_retains_only_complete_optional_channels() -> None:
    burst = FrameBurst(
        (
            PointFrame(np.zeros((1, 3)), frame_id="world", intensity=np.array([2.0])),
            PointFrame(np.zeros((1, 3)), frame_id="world", intensity=np.array([3.0])),
        )
    )
    np.testing.assert_array_equal(burst.fuse().intensity, [2.0, 3.0])
    partial = FrameBurst((burst.frames[0], PointFrame(np.zeros((1, 3)), frame_id="world")))
    assert partial.fuse().intensity is None


def test_deskew_preserves_per_point_origins_and_no_return_ray_origins() -> None:
    trajectory = PoseTrajectory(
        (TimedPose(10.0, _translation(0.0)), TimedPose(11.0, _translation(1.0)))
    )
    frame = PointFrame(
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        timestamp=10.0,
        relative_time=np.array([0.0, 1.0]),
        no_return_rays=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        no_return_relative_time=np.array([0.0, 1.0]),
        sensor_trajectory=trajectory,
    )
    np.testing.assert_allclose(frame.point_origins_in_world(), [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    origins, directions = frame.no_return_rays_in_world()  # type: ignore[misc]
    np.testing.assert_allclose(origins, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(directions, frame.no_return_rays)


def test_moving_trajectory_withholds_untimed_no_return_rays() -> None:
    trajectory = PoseTrajectory(
        (TimedPose(10.0, _translation(0.0)), TimedPose(11.0, _translation(1.0)))
    )
    frame = PointFrame(
        np.array([[1.0, 0.0, 0.0]]),
        timestamp=10.0,
        sensor_trajectory=trajectory,
        no_return_rays=np.array([[1.0, 0.0, 0.0]]),
    )
    assert frame.no_return_rays_in_world() is None
    assert "no_return_relative_time" in frame.unavailable_evidence
    assert "no_return_ray_origins" in frame.unavailable_evidence


def test_static_trajectory_can_transform_untimed_no_return_rays() -> None:
    trajectory = PoseTrajectory(
        (TimedPose(10.0, _translation(2.0)), TimedPose(11.0, _translation(2.0)))
    )
    frame = PointFrame(
        np.empty((0, 3)),
        timestamp=10.0,
        sensor_trajectory=trajectory,
        no_return_rays=np.array([[1.0, 0.0, 0.0]]),
    )
    origins, directions = frame.no_return_rays_in_world()  # type: ignore[misc]
    np.testing.assert_allclose(origins, [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(directions, [[1.0, 0.0, 0.0]])


def test_burst_fusion_preserves_ray_provenance_in_timestamp_order() -> None:
    first = PointFrame(
        np.array([[1.0, 0.0, 0.0]]),
        timestamp=0.0,
        sensor_trajectory=PoseTrajectory((TimedPose(0.0, _translation(0.0)),)),
        no_return_rays=np.array([[0.0, 1.0, 0.0]]),
    )
    second = PointFrame(
        np.array([[1.0, 0.0, 0.0]]),
        timestamp=1.0,
        sensor_trajectory=PoseTrajectory((TimedPose(1.0, _translation(2.0)),)),
        no_return_rays=np.array([[0.0, -1.0, 0.0]]),
    )
    fused = FrameBurst((second, first)).fuse()
    np.testing.assert_allclose(fused.points_world, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(fused.point_origins_world, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(fused.no_return_origins_world, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        fused.no_return_directions_world, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )


def test_burst_fusion_has_a_cooperative_preprocess_deadline() -> None:
    burst = FrameBurst((PointFrame(np.zeros((1, 3)), frame_id="world"),))
    with pytest.raises(TimeoutError, match="deadline"):
        burst.fuse(deadline_s=time.perf_counter() - 1.0)


def test_burst_keeps_transformable_no_return_rays_from_partial_exports() -> None:
    trajectory = PoseTrajectory((TimedPose(0.0, np.eye(4)),))
    fused = FrameBurst(
        (
            PointFrame(
                np.array([[1.0, 0.0, 0.0]]),
                sensor_trajectory=trajectory,
                no_return_rays=np.array([[0.0, 1.0, 0.0]]),
            ),
            PointFrame(np.array([[1.0, 0.0, 0.0]]), sensor_trajectory=trajectory),
        )
    ).fuse()
    np.testing.assert_allclose(fused.no_return_directions_world, [[0.0, 1.0, 0.0]])
    assert "no_return_rays" in fused.unavailable_evidence
