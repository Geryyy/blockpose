"""Rich point frames, pose trajectories, deskewing, and burst fusion."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _transform_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", transforms[:, :3, :3], points) + transforms[:, :3, 3]


@dataclass(frozen=True)
class TimedPose:
    time: float
    transform: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("pose time must be finite")
        transform = np.asarray(self.transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform must be a finite 4x4 matrix")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError("transform rotation must be a proper orthonormal matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("transform must have homogeneous bottom row [0, 0, 0, 1]")
        object.__setattr__(self, "transform", transform)


@dataclass(frozen=True)
class PoseTrajectory:
    poses: tuple[TimedPose, ...]

    def __post_init__(self) -> None:
        if not self.poses:
            raise ValueError("trajectory must contain at least one pose")
        times = np.array([p.time for p in self.poses])
        if np.any(np.diff(times) <= 0):
            raise ValueError("trajectory times must be strictly increasing")

    def sample(self, times: np.ndarray) -> np.ndarray:
        query = np.asarray(times, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(query)):
            raise ValueError("sample times must be finite")
        source_times = np.array([p.time for p in self.poses])
        if np.any(query < source_times[0]) or np.any(query > source_times[-1]):
            raise ValueError("pose trajectory does not cover all requested times")
        matrices = np.stack([p.transform for p in self.poses])
        if len(self.poses) == 1:
            return np.repeat(matrices, len(query), axis=0)
        rotation_samples = Slerp(source_times, Rotation.from_matrix(matrices[:, :3, :3]))
        rotations = rotation_samples(query).as_matrix()
        translations = np.column_stack(
            [np.interp(query, source_times, matrices[:, axis, 3]) for axis in range(3)]
        )
        output = np.repeat(np.eye(4)[None], len(query), axis=0)
        output[:, :3, :3] = rotations
        output[:, :3, 3] = translations
        return output

    @property
    def is_static(self) -> bool:
        """Whether every sampled sensor pose is the same rigid transform."""
        reference = self.poses[0].transform
        return all(np.allclose(pose.transform, reference, atol=1e-9) for pose in self.poses[1:])


@dataclass(frozen=True)
class PointFrame:
    points: np.ndarray
    timestamp: float = 0.0
    frame_id: str = "sensor"
    intensity: np.ndarray | None = None
    relative_time: np.ndarray | None = None
    beam_id: np.ndarray | None = None
    no_return_rays: np.ndarray | None = None
    no_return_relative_time: np.ndarray | None = None
    sensor_trajectory: PoseTrajectory | None = None
    gripper_trajectory: PoseTrajectory | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must have shape (N, 3) and be finite")
        object.__setattr__(self, "points", points)
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        for name in ("intensity", "relative_time", "beam_id"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value).reshape(-1)
                if len(array) != len(points):
                    raise ValueError(f"{name} must have one value per point")
                if name != "beam_id" and not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be finite")
                if name == "beam_id" and not np.issubdtype(array.dtype, np.integer):
                    raise ValueError("beam_id must contain integers")
                object.__setattr__(self, name, array)
        if self.no_return_rays is not None:
            rays = np.asarray(self.no_return_rays, dtype=np.float64)
            if rays.ndim != 2 or rays.shape[1] != 3 or not np.all(np.isfinite(rays)):
                raise ValueError("no_return_rays must have shape (M, 3) and be finite")
            norms = np.linalg.norm(rays, axis=1)
            if np.any(norms <= 1e-12):
                raise ValueError("no_return_rays must not contain zero vectors")
            object.__setattr__(self, "no_return_rays", rays / norms[:, None])
            if self.no_return_relative_time is not None:
                ray_time = np.asarray(self.no_return_relative_time, dtype=np.float64).reshape(-1)
                if len(ray_time) != len(rays) or not np.all(np.isfinite(ray_time)):
                    raise ValueError(
                        "no_return_relative_time must be finite and match no_return_rays"
                    )
                object.__setattr__(self, "no_return_relative_time", ray_time)
        elif self.no_return_relative_time is not None:
            raise ValueError("no_return_relative_time requires no_return_rays")

    @property
    def unavailable_evidence(self) -> tuple[str, ...]:
        names = ("intensity", "relative_time", "beam_id", "no_return_rays", "gripper_trajectory")
        unavailable = [name for name in names if getattr(self, name) is None]
        if self.sensor_trajectory is None:
            unavailable.extend(("per_point_sensor_origins", "no_return_ray_origins"))
        elif (
            self.no_return_rays is not None
            and self.no_return_relative_time is None
            and not self.sensor_trajectory.is_static
        ):
            unavailable.extend(("no_return_relative_time", "no_return_ray_origins"))
        return tuple(unavailable)

    def points_in_world(self) -> np.ndarray:
        if self.sensor_trajectory is None:
            if self.frame_id != "world":
                raise ValueError("sensor trajectory is required for a non-world frame")
            return self.points.copy()
        relative = (
            self.relative_time if self.relative_time is not None else np.zeros(len(self.points))
        )
        transforms = self.sensor_trajectory.sample(self.timestamp + relative)
        return _transform_points(self.points, transforms)

    def point_origins_in_world(self) -> np.ndarray | None:
        """Per-return sensor centres used to deskew the corresponding point.

        A world-frame XYZ cloud without a trajectory deliberately returns
        ``None``: its points are usable geometry, but it cannot safely invent
        ray origins for free-space reasoning.
        """
        if self.sensor_trajectory is None:
            return None
        relative = (
            self.relative_time if self.relative_time is not None else np.zeros(len(self.points))
        )
        return self.sensor_trajectory.sample(self.timestamp + relative)[:, :3, 3]

    def no_return_rays_in_world(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(origins, directions)`` for no-return rays, if transformable."""
        if self.no_return_rays is None or self.sensor_trajectory is None:
            return None
        if self.no_return_relative_time is None and not self.sensor_trajectory.is_static:
            # A moving scan cannot safely attach all misses to the frame start.
            # Withhold the evidence until acquisition exports each miss time.
            return None
        relative = (
            self.no_return_relative_time
            if self.no_return_relative_time is not None
            else np.zeros(len(self.no_return_rays))
        )
        transforms = self.sensor_trajectory.sample(self.timestamp + relative)
        directions = np.einsum("nij,nj->ni", transforms[:, :3, :3], self.no_return_rays)
        return transforms[:, :3, 3], directions


@dataclass(frozen=True)
class FusedFrame:
    points_world: np.ndarray
    source_frame_index: np.ndarray
    source_point_index: np.ndarray
    query_timestamp: float
    unavailable_evidence: tuple[str, ...]
    intensity: np.ndarray | None = None
    relative_time: np.ndarray | None = None
    beam_id: np.ndarray | None = None
    point_origins_world: np.ndarray | None = None
    no_return_origins_world: np.ndarray | None = None
    no_return_directions_world: np.ndarray | None = None


@dataclass(frozen=True)
class FrameBurst:
    frames: tuple[PointFrame, ...]

    def fuse(
        self,
        query_timestamp: float | None = None,
        max_frames: int | None = None,
        *,
        deadline_s: float | None = None,
    ) -> FusedFrame:
        if not self.frames:
            raise ValueError("frame burst is empty")
        indexed_frames = sorted(enumerate(self.frames), key=lambda item: item[1].timestamp)
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
            indexed_frames = indexed_frames[-max_frames:]
        frames = [frame for _, frame in indexed_frames]
        selected_timestamp = (
            query_timestamp if query_timestamp is not None else frames[-1].timestamp
        )
        if not np.isfinite(selected_timestamp):
            raise ValueError("query_timestamp must be finite")
        if deadline_s is not None and time.perf_counter() >= deadline_s:
            raise TimeoutError("burst fusion deadline exceeded")
        points = []
        point_origins = []
        no_return = []
        for frame in frames:
            points.append(frame.points_in_world())
            point_origins.append(frame.point_origins_in_world())
            no_return.append(frame.no_return_rays_in_world())
            if deadline_s is not None and time.perf_counter() >= deadline_s:
                raise TimeoutError("burst fusion deadline exceeded")
        unavailable = sorted({name for frame in frames for name in frame.unavailable_evidence})
        if any(value is None for value in point_origins):
            unavailable.append("per_point_sensor_origins")
        if any(value is None for value in no_return):
            unavailable.append("no_return_ray_origins")
        all_origins = (
            None if any(value is None for value in point_origins) else np.concatenate(point_origins)
        )
        # No-return rays are independent observations. Keep the rays provided
        # by any transformable scan even if another burst member did not export
        # them; ``unavailable_evidence`` still records that the burst is partial.
        usable_no_return = [value for value in no_return if value is not None]
        all_no_return_origins = (
            np.concatenate([value[0] for value in usable_no_return])
            if usable_no_return
            else None
        )
        all_no_return_directions = (
            np.concatenate([value[1] for value in usable_no_return])
            if usable_no_return
            else None
        )
        return FusedFrame(
            np.concatenate(points),
            np.concatenate(
                [
                    np.full(len(p), original_index, dtype=np.int32)
                    for (original_index, _), p in zip(indexed_frames, points, strict=True)
                ]
            ),
            np.concatenate([np.arange(len(p), dtype=np.int32) for p in points]),
            float(selected_timestamp),
            tuple(sorted(set(unavailable))),
            _fuse_optional(frames, "intensity"),
            _fuse_optional(frames, "relative_time"),
            _fuse_optional(frames, "beam_id"),
            all_origins,
            all_no_return_origins,
            all_no_return_directions,
        )


def _fuse_optional(frames: list[PointFrame], name: str) -> np.ndarray | None:
    """Fuse a channel only when it is present for every retained frame."""
    values = [getattr(frame, name) for frame in frames]
    if any(value is None for value in values):
        return None
    return np.concatenate(values)  # type: ignore[arg-type]
