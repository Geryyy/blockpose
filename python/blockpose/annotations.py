"""Versioned, completeness-aware annotations for real registration snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from blockpose.pose import BLOCK_DIMS, BlockPose, quat_xyzw_to_rotmat

ANNOTATION_SCHEMA_VERSION = 2
SUPPORTED_ANNOTATION_SCHEMA_VERSIONS = frozenset({1, 2})
REVIEW_STATUSES = frozenset({"pending", "approved", "rejected"})


@dataclass(frozen=True)
class AnnotationTarget:
    id: str
    pose: BlockPose
    provenance: str
    visibility: str = "unknown"
    ignore_reason: str | None = None
    # Programmatically constructed targets historically represented accepted GT.
    # The loader explicitly assigns old/on-disk priors to pending instead.
    review_status: str = "approved"
    rejection_reason: str | None = None

    @property
    def is_evaluation_target(self) -> bool:
        """Only a human-approved, non-ignored block is ground truth."""
        return self.review_status == "approved" and self.ignore_reason is None


@dataclass(frozen=True)
class SnapshotAnnotations:
    schema_version: int
    snapshot: str
    annotation_complete: bool
    roi_min: np.ndarray | None
    roi_max: np.ndarray | None
    targets: tuple[AnnotationTarget, ...] = ()
    unmatched_detection_reviews: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def has_complete_roi(self) -> bool:
        return self.annotation_complete and self.roi_min is not None and self.roi_max is not None


def _vec3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def load_annotations(path: str | Path) -> SnapshotAnnotations:
    """Load and validate an annotation manifest without assuming completeness."""
    path = Path(path)
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    version = int(data.get("schema_version", 0))
    if version not in SUPPORTED_ANNOTATION_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported annotation schema_version {version}")
    complete = bool(data.get("annotation_complete", False))
    snapshot = str(data.get("snapshot", path.parent.name))
    if snapshot != path.parent.name:
        raise ValueError(
            f"annotation snapshot {snapshot!r} does not match directory {path.parent.name!r}"
        )
    roi = data.get("annotated_roi")
    roi_min = roi_max = None
    if roi is not None:
        roi_min = _vec3(roi["min"], "annotated_roi.min")
        roi_max = _vec3(roi["max"], "annotated_roi.max")
        if np.any(roi_min >= roi_max):
            raise ValueError("annotated_roi min must be strictly less than max")
    if complete and roi is None:
        raise ValueError("complete annotations require an annotated_roi")

    blocks = data.get("blocks") or []
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list or null")
    targets = []
    target_ids: set[str] = set()
    for item in blocks:
        target_id = str(item.get("id", "")).strip()
        if not target_id:
            raise ValueError("block id must be non-empty")
        if target_id in target_ids:
            raise ValueError(f"duplicate block id {target_id!r}")
        target_ids.add(target_id)
        pose_data = item["pose"]
        legacy_default = "approved" if version == 1 and complete else "pending"
        review_status = str(item.get("review_status", legacy_default))
        dims = tuple(float(v) for v in item.get("dimensions", BLOCK_DIMS))
        if len(dims) != 3 or any(not np.isfinite(v) or v <= 0 for v in dims):
            raise ValueError("block dimensions must contain three positive finite numbers")
        position = _vec3(pose_data["position"], "pose.position")
        if complete and review_status == "approved" and (
            np.any(position < roi_min) or np.any(position > roi_max)  # type: ignore[operator]
        ):
            raise ValueError(f"complete annotation target {target_id!r} lies outside annotated_roi")
        confidence = float(item.get("confidence", 0.0))
        if not np.isfinite(confidence):
            raise ValueError("block confidence must be finite")
        targets.append(
            AnnotationTarget(
                id=target_id,
                pose=BlockPose(
                    position,
                    quat_xyzw_to_rotmat(_vec4(pose_data["orientation_xyzw"])),
                    confidence=confidence,
                    source=str(item.get("provenance", "unknown")),
                    dims=dims,
                ),
                provenance=str(item.get("provenance", "unknown")),
                visibility=str(item.get("visibility", "unknown")),
                ignore_reason=item.get("ignore_reason"),
                # Schema 1 contains unreviewed upstream priors, not human GT.
                review_status=review_status,
                rejection_reason=item.get("rejection_reason"),
            )
        )
        if targets[-1].review_status not in REVIEW_STATUSES:
            raise ValueError(
                f"invalid review_status {targets[-1].review_status!r} for block {target_id!r}"
            )
        if targets[-1].review_status == "rejected" and not targets[-1].rejection_reason:
            raise ValueError(f"rejected block {target_id!r} requires rejection_reason")
    reviews = {str(k): str(v) for k, v in data.get("unmatched_detection_reviews", {}).items()}
    invalid_reviews = set(reviews.values()) - {"block", "clutter", "uncertain"}
    if invalid_reviews:
        raise ValueError(f"invalid unmatched detection review values: {sorted(invalid_reviews)}")
    return SnapshotAnnotations(
        version,
        snapshot,
        complete,
        roi_min,
        roi_max,
        tuple(targets),
        reviews,
        str(data.get("notes", "")),
    )


def _vec4(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)) or np.linalg.norm(result) < 1e-12:
        raise ValueError("orientation_xyzw must contain a non-zero finite quaternion")
    return result / np.linalg.norm(result)
