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
