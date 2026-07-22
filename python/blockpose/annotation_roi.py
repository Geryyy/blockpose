"""Derive exhaustive annotation ROIs from complete snapshot point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from blockpose.annotations import load_annotations
from blockpose.io import apply_transform, load_pcd, load_tf_snapshot
from blockpose.pose import BlockPose

FULL_CLOUD_ROI_NOTE = (
    "User confirmed every visible block in the entire snapshot is annotated. "
    "The annotated ROI is the full finite point-cloud AABB in world coordinates "
    "with 0.01 m padding."
)


@dataclass(frozen=True)
class RoiCompletion:
    snapshot: Path
    roi_min: np.ndarray
    roi_max: np.ndarray
    document: dict[str, Any]


def validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Return a validated cloud-to-world rigid transform."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_world_cloud must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_world_cloud has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3
    ):
        raise ValueError("T_world_cloud rotation is not rigid")
    return transform


def full_cloud_world_roi(
    points: np.ndarray, transform: np.ndarray, *, padding_m: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a padded world AABB from every finite cloud point."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("cloud points must have shape (N, 3)")
    if not np.isfinite(padding_m) or padding_m <= 0:
        raise ValueError("padding_m must be positive and finite")
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) == 0:
        raise ValueError("point cloud contains no finite points")
    world = apply_transform(finite, validate_rigid_transform(transform))
    return world.min(axis=0) - padding_m, world.max(axis=0) + padding_m


def cuboid_corners(pose: BlockPose) -> np.ndarray:
    """Return the eight world-space corners of a block pose."""
    signs = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    local = signs * (np.asarray(pose.dims, dtype=np.float64) / 2.0)
    return (pose.rotation @ local.T).T + pose.position


def prepare_roi_completion(snapshot: str | Path, *, padding_m: float = 0.01) -> RoiCompletion:
    """Prepare and fully validate one manifest update without writing it."""
    snapshot = Path(snapshot)
    manifest = snapshot / "annotations.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"annotation manifest not found: {manifest}")
    annotations = load_annotations(manifest)
    unresolved = [target.id for target in annotations.targets if target.review_status == "pending"]
    if unresolved:
        raise ValueError(f"snapshot has pending annotation targets: {', '.join(unresolved)}")

    transform = load_tf_snapshot(snapshot / "tf.yaml").T_world_cloud
    points = load_pcd(snapshot / "cloud.pcd")
    roi_min, roi_max = full_cloud_world_roi(points, transform, padding_m=padding_m)

    finite_world = apply_transform(
        np.asarray(points)[np.all(np.isfinite(points), axis=1)], validate_rigid_transform(transform)
    )
    tolerance = 1e-9
    if np.any(finite_world < roi_min - tolerance) or np.any(finite_world > roi_max + tolerance):
        raise AssertionError("derived ROI does not enclose every finite cloud point")
    for target in annotations.targets:
        if target.review_status != "approved":
            continue
        corners = cuboid_corners(target.pose)
        if np.any(corners < roi_min - tolerance) or np.any(corners > roi_max + tolerance):
            raise ValueError(f"approved block {target.id!r} is not fully enclosed by the cloud ROI")

    with manifest.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"annotation manifest is not a mapping: {manifest}")
    document["annotation_complete"] = True
    document["annotated_roi"] = {
        "min": [float(value) for value in roi_min],
        "max": [float(value) for value in roi_max],
    }
    notes = str(document.get("notes", "")).strip()
    if FULL_CLOUD_ROI_NOTE not in notes:
        document["notes"] = f"{notes}\n{FULL_CLOUD_ROI_NOTE}".strip()
    return RoiCompletion(snapshot, roi_min, roi_max, document)
