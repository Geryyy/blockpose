"""GUI-independent state and persistence for snapshot annotation.

The controller deliberately never guesses whether a prior is correct.  Schema-v1
registrations load as pending and only an explicit ``approve`` makes them usable by
the evaluator.
"""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

UTC = timezone.utc

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from blockpose.annotations import AnnotationTarget, load_annotations
from blockpose.pose import BLOCK_DIMS, frame_alignments, rotmat_to_quat_xyzw, symmetry_equivalents

VISIBILITIES = ("unknown", "full", "partial", "heavy_occlusion")


def full_workspace_roi(points: np.ndarray, margin_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return a complete-annotation ROI covering every captured cloud point."""
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud) or not np.all(np.isfinite(cloud)):
        raise ValueError("points must be a non-empty finite (N, 3) cloud")
    if not np.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("ROI margin must be finite and non-negative")
    margin = np.full(3, float(margin_m))
    return cloud.min(axis=0) - margin, cloud.max(axis=0) + margin


def remove_background_by_range(
    points: np.ndarray, sensor_origin: np.ndarray, max_range_m: float
) -> np.ndarray:
    """Keep only points within `max_range_m` of `sensor_origin`.

    Declutters far-range background (walls, containers) that sits well beyond
    the workspace, without fitting any planes -- cheap and sufficient since
    blocks are always well within the reachable workspace radius.
    """
    if not np.isfinite(max_range_m) or max_range_m <= 0.0:
        raise ValueError("max_range_m must be finite and positive")
    origin = np.asarray(sensor_origin, dtype=float).reshape(3)
    cloud = np.asarray(points, dtype=float)
    distances = np.linalg.norm(cloud - origin, axis=1)
    return cloud[distances <= max_range_m]


def default_max_range_m(points: np.ndarray, sensor_origin: np.ndarray) -> float:
    """A mild starting range cutoff: the 95th percentile of point-to-sensor distance.

    Not a fixed workspace constant -- no such config exists -- so this starts
    as a light trim of the very-farthest returns that the annotator can tune
    down further while watching the cloud shrink live.
    """
    origin = np.asarray(sensor_origin, dtype=float).reshape(3)
    cloud = np.asarray(points, dtype=float)
    distances = np.linalg.norm(cloud - origin, axis=1)
    return float(np.percentile(distances, 95.0))


def canonicalize_block_frame(
    rotation: np.ndarray, dimensions: tuple[float, float, float]
) -> np.ndarray:
    """Re-express a physical 0.9 x 0.6 x 0.6 box in the canonical local frame."""
    dims = tuple(float(value) for value in dimensions)
    alignments = frame_alignments(dims, BLOCK_DIMS)
    if not alignments:
        raise ValueError(
            f"block dimensions {dims!r} are not a permutation of canonical {BLOCK_DIMS!r}"
        )
    # frame_alignments is generated in a stable signed-permutation order.  The
    # first representative is deterministic; all alternatives are exact box
    # symmetries and therefore describe the same world-space OBB.
    return np.asarray(rotation, dtype=float) @ alignments[0]


def next_symmetry_rotation(
    base_rotation: np.ndarray,
    cycle_index: int,
    dimensions: tuple[float, float, float] = BLOCK_DIMS,
) -> np.ndarray:
    """Cycle through the alternative ways to label a symmetric box's faces.

    `base_rotation` is the pose the cycle started from (typically the rotation
    at the moment a target was selected, before any Rotate-90 presses).
    `symmetry_equivalents` already enumerates every rotation describing the
    same physical placement -- the 90 degree relabelings about the long axis
    and the 180 degree end-for-end flips -- so this only indexes into it
    rather than constructing any new rotation matrix.
    """
    alternatives = symmetry_equivalents(base_rotation, dimensions)
    return alternatives[cycle_index % len(alternatives)]


def orientation_summary(rotation: np.ndarray) -> str:
    """Human-readable, symmetry-stable direction of canonical local +X."""
    axis = np.asarray(rotation, dtype=float)[:, 0]
    # The long axis is unoriented. Pick its upper hemisphere, then a stable
    # horizontal half-plane at zero elevation, so symmetric frames agree.
    if axis[2] < -1e-12 or (
        abs(axis[2]) <= 1e-12
        and (axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0))
    ):
        axis = -axis
    elevation = float(np.degrees(np.arctan2(axis[2], np.hypot(axis[0], axis[1]))))
    azimuth = float(np.degrees(np.arctan2(axis[1], axis[0]))) % 180.0
    orientation_class = "Standing" if elevation >= 45.0 else "Laying"
    return f"{orientation_class} — long-axis azimuth {azimuth:.1f}°, elevation {elevation:.1f}°"


@dataclass
class EditableTarget:
    id: str
    position: np.ndarray
    rotation: np.ndarray
    dimensions: tuple[float, float, float] = BLOCK_DIMS
    provenance: str = "manual"
    visibility: str = "unknown"
    ignore_reason: str | None = None
    review_status: str = "pending"
    rejection_reason: str | None = None
    confidence: float = 0.0

    @classmethod
    def from_annotation(cls, target: AnnotationTarget) -> EditableTarget:
        return cls(
            target.id,
            target.pose.position.copy(),
            target.pose.rotation.copy(),
            target.pose.dims,
            target.provenance,
            target.visibility,
            target.ignore_reason,
            target.review_status,
            target.rejection_reason,
            target.pose.confidence,
        )

    @property
    def euler_deg(self) -> np.ndarray:
        return Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)


@dataclass
class AnnotationDocument:
    snapshot: str
    targets: list[EditableTarget] = field(default_factory=list)
    annotation_complete: bool = False
    roi_min: np.ndarray | None = None
    roi_max: np.ndarray | None = None
    notes: str = ""
    unmatched_detection_reviews: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> AnnotationDocument:
        loaded = load_annotations(path)
        targets = []
        for target in loaded.targets:
            editable = EditableTarget.from_annotation(target)
            editable.rotation = canonicalize_block_frame(editable.rotation, editable.dimensions)
            editable.dimensions = BLOCK_DIMS
            targets.append(editable)
        return cls(
            loaded.snapshot,
            targets,
            loaded.annotation_complete,
            None if loaded.roi_min is None else loaded.roi_min.copy(),
            None if loaded.roi_max is None else loaded.roi_max.copy(),
            loaded.notes,
            dict(loaded.unmatched_detection_reviews),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        ids = [target.id for target in self.targets]
        if len(ids) != len(set(ids)):
            errors.append("Block IDs must be unique")
        if self.annotation_complete:
            if self.roi_min is None or self.roi_max is None:
                errors.append("A complete annotation requires an ROI")
            elif np.any(self.roi_min >= self.roi_max):
                errors.append("ROI minimum must be strictly below ROI maximum")
            if any(target.review_status == "pending" for target in self.targets):
                errors.append("Resolve every pending prior before marking the ROI complete")
        for target in self.targets:
            if not np.allclose(target.dimensions, BLOCK_DIMS, atol=1e-9, rtol=0.0):
                errors.append(f"{target.id}: dimensions must use canonical {BLOCK_DIMS}")
            if target.visibility not in VISIBILITIES:
                errors.append(f"{target.id}: invalid visibility")
            if target.review_status == "rejected" and not target.rejection_reason:
                errors.append(f"{target.id}: rejected blocks require a reason")
            if self.annotation_complete and self.roi_min is not None and self.roi_max is not None:
                if target.review_status == "approved" and (
                    np.any(target.position < self.roi_min) or np.any(target.position > self.roi_max)
                ):
                    errors.append(f"{target.id}: approved center lies outside ROI")
        return errors

    def as_mapping(self) -> dict:
        errors = [
            target.id
            for target in self.targets
            if not np.allclose(target.dimensions, BLOCK_DIMS, atol=1e-9, rtol=0.0)
        ]
        if errors:
            raise ValueError(f"non-canonical block dimensions for: {', '.join(errors)}")
        blocks = []
        for target in self.targets:
            item = {
                "id": target.id,
                "review_status": target.review_status,
                "provenance": target.provenance,
                "visibility": target.visibility,
                "dimensions": [float(value) for value in target.dimensions],
                "confidence": float(target.confidence),
                "pose": {
                    "position": [float(value) for value in target.position],
                    "orientation_xyzw": [
                        float(value) for value in rotmat_to_quat_xyzw(target.rotation)
                    ],
                },
            }
            if target.ignore_reason:
                item["ignore_reason"] = target.ignore_reason
            if target.rejection_reason:
                item["rejection_reason"] = target.rejection_reason
            blocks.append(item)
        return {
            "schema_version": 2,
            "snapshot": self.snapshot,
            "annotation_complete": self.annotation_complete,
            "annotated_roi": (
                None
                if self.roi_min is None or self.roi_max is None
                else {
                    "min": [float(value) for value in self.roi_min],
                    "max": [float(value) for value in self.roi_max],
                }
            ),
            "notes": self.notes,
            "blocks": blocks,
            "unmatched_detection_reviews": dict(sorted(self.unmatched_detection_reviews.items())),
        }


class AnnotationController:
    """Undoable annotation operations independent of Open3D."""

    def __init__(self, document: AnnotationDocument):
        self.document = document
        self.selected = 0 if document.targets else None
        self.dirty = False
        self._baselines = {target.id: copy.deepcopy(target) for target in document.targets}
        self._initial_ids = set(self._baselines)
        self._undo: list[tuple[AnnotationDocument, int | None]] = []
        self._redo: list[tuple[AnnotationDocument, int | None]] = []

    @property
    def target(self) -> EditableTarget | None:
        if self.selected is None or self.selected >= len(self.document.targets):
            return None
        return self.document.targets[self.selected]

    def _change(self, operation: Callable[[], None]) -> None:
        self._undo.append((copy.deepcopy(self.document), self.selected))
        self._redo.clear()
        operation()
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append((copy.deepcopy(self.document), self.selected))
        self.document, self.selected = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append((copy.deepcopy(self.document), self.selected))
        self.document, self.selected = self._redo.pop()
        self.dirty = True
        return True

    def mark_saved(self) -> None:
        """Make the successfully persisted document the new reset/abort checkpoint."""
        self._baselines = {
            target.id: copy.deepcopy(target) for target in self.document.targets
        }
        self._initial_ids = set(self._baselines)
        self._undo.clear()
        self._redo.clear()
        self.dirty = False

    def set_pose(self, position: np.ndarray, euler_deg: np.ndarray) -> None:
        if self.target is None:
            return
        position = np.asarray(position, dtype=float)
        angles = np.asarray(euler_deg, dtype=float)
        finite = np.all(np.isfinite([*position, *angles]))
        if position.shape != (3,) or angles.shape != (3,) or not finite:
            raise ValueError("pose fields must contain finite XYZ and roll/pitch/yaw values")
        self._change(
            lambda: (
                setattr(self.target, "position", position.copy()),
                setattr(
                    self.target,
                    "rotation",
                    Rotation.from_euler("xyz", angles, degrees=True).as_matrix(),
                ),
            )
        )

    def apply_snapped_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        dimensions: tuple[float, float, float],
    ) -> None:
        """Apply a fitter proposal as one undoable edit without approving it."""
        if self.target is None:
            return
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        dims = tuple(float(value) for value in dimensions)
        valid = (
            position.shape == (3,)
            and rotation.shape == (3, 3)
            and len(dims) == 3
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(dims))
            and np.all(np.asarray(dims) > 0)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.linalg.det(rotation) > 0.0
        )
        if not valid:
            raise ValueError("snapped pose must contain a finite rigid transform and dimensions")
        rotation = canonicalize_block_frame(rotation, dims)

        def apply() -> None:
            assert self.target is not None
            self.target.position = position.copy()
            self.target.rotation = rotation.copy()
            self.target.dimensions = BLOCK_DIMS
            self.target.review_status = "pending"
            self.target.rejection_reason = None

        self._change(apply)

    def set_review(self, status: str, reason: str | None = None) -> None:
        if self.target is None or status not in {"pending", "approved", "rejected"}:
            return
        if status == "rejected" and not reason:
            raise ValueError("rejection requires a reason")
        self._change(
            lambda: (
                setattr(self.target, "review_status", status),
                setattr(self.target, "rejection_reason", reason if status == "rejected" else None),
            )
        )

    def set_ignore_reason(self, reason: str | None) -> None:
        if self.target is None:
            return
        normalized = reason.strip() if reason else None
        normalized = normalized or None
        if self.target.ignore_reason == normalized:
            return
        self._change(lambda: setattr(self.target, "ignore_reason", normalized))

    def abort_selected(self) -> None:
        """Retract a manual draft, or restore an imported prior to pending."""
        target = self.target
        if target is None:
            return
        if target.id not in self._initial_ids and target.review_status != "approved":
            index = self.selected
            assert index is not None
            self._change(lambda: self.document.targets.pop(index))
            self.selected = (
                min(index, len(self.document.targets) - 1) if self.document.targets else None
            )
            return
        self.reset_selected()
        self.set_review("pending")

    def delete_selected(
        self,
        *,
        expected_id: str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        """Delete the selected target as one undoable, in-memory operation.

        The optional expectations let a confirmation dialog prove that its
        answer still refers to the item that was displayed to the user.
        """
        target = self.target
        if target is None:
            return False
        if expected_id is not None and target.id != expected_id:
            return False
        if expected_status is not None and target.review_status != expected_status:
            return False
        index = self.selected
        assert index is not None
        self._change(lambda: self.document.targets.pop(index))
        self.selected = (
            min(index, len(self.document.targets) - 1) if self.document.targets else None
        )
        return True

    def reset_selected(self) -> None:
        if self.target is None:
            return
        match = self._baselines.get(self.target.id)
        if match is not None:
            position = match.position.copy()
            rotation = match.rotation.copy()
            dimensions = match.dimensions

            def restore_pose() -> None:
                assert self.target is not None
                self.target.position = position
                self.target.rotation = rotation
                self.target.dimensions = dimensions

            self._change(restore_pose)

    def add_manual(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("manual block position must contain three finite numbers")
        used = {target.id for target in self.document.targets}
        index = 0
        while f"manual_{index}" in used:
            index += 1
        target = EditableTarget(
            f"manual_{index}", position.copy(), np.eye(3), provenance="manual"
        )
        self._change(lambda: self.document.targets.append(target))
        self._baselines[target.id] = copy.deepcopy(target)
        self.selected = len(self.document.targets) - 1


def pick_oriented_box(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    targets: list[EditableTarget],
) -> int | None:
    """Return the nearest ray/OBB intersection using a slab test."""
    direction = np.asarray(ray_direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    direction /= norm
    best: tuple[float, int] | None = None
    for index, target in enumerate(targets):
        origin = target.rotation.T @ (np.asarray(ray_origin) - target.position)
        local_direction = target.rotation.T @ direction
        lower = -np.asarray(target.dimensions) / 2
        upper = -lower
        near, far = -np.inf, np.inf
        for axis in range(3):
            if abs(local_direction[axis]) < 1e-12:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    far = -np.inf
                    break
                continue
            first = (lower[axis] - origin[axis]) / local_direction[axis]
            second = (upper[axis] - origin[axis]) / local_direction[axis]
            near, far = max(near, min(first, second)), min(far, max(first, second))
        distance = max(near, 0.0)
        if near <= far and far >= 0 and (best is None or distance < best[0]):
            best = (distance, index)
    return None if best is None else best[1]


def nearest_point_to_ray(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    points: np.ndarray,
    max_distance: float,
) -> np.ndarray | None:
    """Pick a forward cloud point only when it lies close enough to the view ray."""
    candidate = nearest_point_to_ray_with_distance(ray_origin, ray_direction, points)
    if candidate is None or not np.isfinite(max_distance) or max_distance < 0:
        return None
    point, distance = candidate
    return point if distance <= max_distance else None


def nearest_point_to_ray_with_distance(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Return the nearest finite forward point and its perpendicular ray distance."""
    points = np.asarray(points, dtype=float)
    origin = np.asarray(ray_origin, dtype=float)
    direction = np.asarray(ray_direction, dtype=float)
    norm = np.linalg.norm(direction)
    valid_input = (
        points.ndim == 2
        and points.shape[1:] == (3,)
        and origin.shape == (3,)
        and direction.shape == (3,)
        and np.all(np.isfinite(origin))
        and np.all(np.isfinite(direction))
        and np.isfinite(norm)
        and norm >= 1e-12
    )
    if not valid_input:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return None
    unit = direction / norm
    along = (points - origin) @ unit
    valid = along > 0
    if not np.any(valid):
        return None
    candidates = points[valid]
    projected = origin + along[valid, None] * unit
    distances = np.linalg.norm(candidates - projected, axis=1)
    nearest = int(np.argmin(distances))
    return candidates[nearest].copy(), float(distances[nearest])


def camera_ray(eye: np.ndarray, unprojected_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite, non-degenerate world-space ray from a camera eye and point."""
    origin = np.asarray(eye, dtype=float).reshape(-1)
    point = np.asarray(unprojected_point, dtype=float).reshape(-1)
    if origin.shape != (3,) or point.shape != (3,):
        raise ValueError("camera eye and unprojected point must each contain three values")
    direction = point - origin
    norm = np.linalg.norm(direction)
    if (
        not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(point))
        or not np.isfinite(norm)
        or norm < 1e-9
    ):
        raise ValueError("camera produced a non-finite or degenerate click ray")
    return origin, direction / norm


def initial_camera_focus(
    points: np.ndarray,
    eye: np.ndarray,
    direction: np.ndarray,
    *,
    fallback_depth: float = 5.0,
    min_depth: float = 1.0,
    max_depth: float = 50.0,
) -> np.ndarray:
    """Choose a robust look-at point while preserving the requested view direction.

    A realistic focus depth makes Open3D's mouse pan scale useful. Invalid or
    behind-camera clouds use the finite fallback; extreme depths are clamped.
    """
    cloud = np.asarray(points, dtype=float)
    origin = np.asarray(eye, dtype=float)
    vector = np.asarray(direction, dtype=float)
    if origin.shape != (3,) or vector.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("camera eye and direction must contain three finite values")
    norm = float(np.linalg.norm(vector))
    limits = np.asarray([fallback_depth, min_depth, max_depth], dtype=float)
    if not np.isfinite(norm) or norm < 1e-12 or not np.all(np.isfinite(limits)):
        raise ValueError("camera direction and depth limits must be finite")
    if min_depth <= 0 or max_depth < min_depth:
        raise ValueError("camera depth limits must satisfy 0 < min_depth <= max_depth")
    unit = vector / norm
    depth = fallback_depth
    if cloud.ndim == 2 and cloud.shape[1:] == (3,):
        finite = cloud[np.all(np.isfinite(cloud), axis=1)]
        if len(finite):
            forward = (finite - origin) @ unit
            forward = forward[forward > 0]
            if len(forward):
                depth = float(np.median(forward))
    depth = float(np.clip(depth, min_depth, max_depth))
    return origin + unit * depth


def snapshot_selection_index(snapshots: list[Path], current: Path) -> int:
    """Index used to restore a snapshot selector after canceling navigation."""
    try:
        return snapshots.index(current)
    except ValueError as error:
        raise ValueError(f"current snapshot {current} is not in the snapshot list") from error


def adjacent_snapshot(snapshots: list[Path], current: Path, step: int) -> Path | None:
    """Return the snapshot `step` positions from `current`, or None past either end.

    Never wraps: stepping before the first or past the last snapshot returns
    None so Prev/Next buttons can clamp and disable at the ends.
    """
    index = snapshot_selection_index(snapshots, current) + step
    return snapshots[index] if 0 <= index < len(snapshots) else None


def save_annotations_atomic(
    document: AnnotationDocument,
    path: str | Path,
    *,
    timestamp: datetime | None = None,
) -> Path | None:
    """Validate, back up an existing manifest, then atomically write schema v2."""
    errors = document.validate()
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = (timestamp or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = path.with_name(f"{path.name}.backup-{stamp}")
        # Exclusive creation prevents silently overwriting even at identical timestamps.
        with path.open("rb") as source, backup.open("xb") as destination:
            destination.write(source.read())
    rendered = yaml.safe_dump(document.as_mapping(), sort_keys=False, allow_unicode=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        # Do not unlink a failed temporary file: repository policy forbids deletion.
        raise
    return backup
