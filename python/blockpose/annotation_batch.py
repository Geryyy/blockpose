"""Safe, deterministic initialization of provisional snapshot annotations."""

from __future__ import annotations

import ctypes
import errno
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage

from blockpose.annotation_snap import SnapParams, snap_block_to_cloud
from blockpose.annotation_tool import AnnotationDocument, EditableTarget
from blockpose.io import (
    apply_transform,
    load_camera_model,
    load_pcd,
    load_rgb,
    load_tf_snapshot,
)
from blockpose.pose import BLOCK_DIMS, quat_xyzw_to_rotmat
from blockpose.rgb_annotation import (
    MaskSelectionParams,
    build_projection_cache,
    load_candidate_masks,
    select_points_from_mask,
)
from blockpose.snapshot_exclusions import load_snapshot_exclusions

REQUIRED_FILES = ("cloud.pcd", "tf.yaml", "rgb.png", "metadata.yaml", "registrations.yaml")


@dataclass(frozen=True)
class CandidateFailure:
    source: str
    reason: str


@dataclass
class InitializationResult:
    snapshot: str
    status: str
    manifest: Path | None = None
    registration_targets: int = 0
    rgb_targets: int = 0
    failures: list[CandidateFailure] = field(default_factory=list)


def discover_snapshots(root: str | Path, *, include_excluded: bool = False) -> list[Path]:
    """Return every snapshot directory, including malformed ones for audit."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"snapshot root does not exist: {root}")
    excluded = {} if include_excluded else load_snapshot_exclusions(root)
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name not in excluded)


def _publish_no_replace(rendered: str, destination: Path) -> bool:
    """Atomically publish text iff destination does not exist (Linux renameat2)."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(temporary_path),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            return False
        raise OSError(error, os.strerror(error), destination)
    except BaseException:
        # Failed temporary publications remain for inspection: repository policy forbids deletion.
        raise


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected a mapping")
    return value


def _validate_transform(transform: np.ndarray, label: str) -> None:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3
    ):
        raise ValueError(f"{label} rotation is not rigid")


def _registration_targets(
    data: dict[str, Any],
) -> tuple[list[EditableTarget], list[CandidateFailure]]:
    records = data.get("records", [])
    if not isinstance(records, list):
        raise ValueError("invalid registrations.yaml: records must be a list")
    targets: list[EditableTarget] = []
    failures: list[CandidateFailure] = []
    used: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not record.get("registration_ok", False):
            continue
        source = f"registration record {index}"
        try:
            pose = record["pose"]
            position = np.asarray(pose["position"], dtype=float)
            quaternion = np.asarray(pose["orientation_xyzw"], dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("position must contain three finite values")
            rotation = quat_xyzw_to_rotmat(quaternion)
            confidence = float(record.get("confidence", 0.0))
            if not np.isfinite(confidence):
                raise ValueError("confidence must be finite")
            base = str(record.get("assigned_id", "")).strip()
            if not base:
                base = f"registration_{record.get('detection_id', index + 1)}"
            target_id = base
            suffix = 2
            while target_id in used:
                target_id = f"{base}_{suffix}"
                suffix += 1
            used.add(target_id)
            targets.append(
                EditableTarget(
                    target_id,
                    position,
                    rotation,
                    dimensions=BLOCK_DIMS,
                    provenance="upstream_registration",
                    review_status="pending",
                    confidence=confidence,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            failures.append(CandidateFailure(source, str(error)))
    return targets, failures


def _mask_prompt(mask: np.ndarray) -> tuple[int, int]:
    """Choose the most interior mask pixel, with stable row/column tie-breaking."""
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("mask is empty or not two-dimensional")
    distance = ndimage.distance_transform_edt(mask)
    row, column = np.argwhere(distance == distance.max())[0]
    return int(column), int(row)


def _near_existing(position: np.ndarray, targets: list[EditableTarget], threshold_m: float) -> bool:
    return any(np.linalg.norm(position - target.position) <= threshold_m for target in targets)


def _declared_mask_ids(metadata: dict[str, Any]) -> list[int]:
    entries = metadata.get("candidate_masks", [])
    if not isinstance(entries, list):
        raise ValueError("invalid metadata.yaml: candidate_masks must be a list")
    result: list[int] = []
    for entry in entries:
        if isinstance(entry, dict):
            try:
                result.append(int(entry["detection_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(set(result))


def initialize_snapshot(
    snapshot: str | Path,
    *,
    camera_calibration: str | Path | None = None,
    dedup_distance_m: float = 0.40,
) -> InitializationResult:
    """Create one incomplete schema-v2 manifest, without replacing any file."""
    snapshot = Path(snapshot)
    result = InitializationResult(snapshot.name, "failed_integrity")
    manifest = snapshot / "annotations.yaml"
    if manifest.exists():
        result.status = "skipped_existing"
        result.manifest = manifest
        return result
    if not np.isfinite(dedup_distance_m) or dedup_distance_m <= 0:
        result.failures.append(
            CandidateFailure("snapshot", "dedup_distance_m must be positive and finite")
        )
        return result
    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    empty = [
        name
        for name in REQUIRED_FILES
        if (snapshot / name).is_file() and not (snapshot / name).stat().st_size
    ]
    if missing or empty:
        detail = [*(f"missing {name}" for name in missing), *(f"empty {name}" for name in empty)]
        result.failures.append(CandidateFailure("snapshot", ", ".join(detail)))
        return result
    try:
        metadata = _load_mapping(snapshot / "metadata.yaml", "metadata.yaml")
        registrations = _load_mapping(snapshot / "registrations.yaml", "registrations.yaml")
        rgb = load_rgb(snapshot / "rgb.png")
        points_sensor = load_pcd(snapshot / "cloud.pcd")
        if not np.all(np.isfinite(points_sensor)):
            raise ValueError("cloud contains non-finite points")
        tf = load_tf_snapshot(snapshot / "tf.yaml")
        T_world_cloud = tf.T_world_cloud
        T_world_camera = tf.T_world_camera
        _validate_transform(T_world_cloud, "T_world_cloud")
        _validate_transform(T_world_camera, "T_world_camera")
        camera = load_camera_model(camera_calibration)
        if rgb.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"RGB shape {rgb.shape[:2]} does not match calibration "
                f"{(camera.height, camera.width)}"
            )
        points_world = apply_transform(points_sensor, T_world_cloud)
        projection = build_projection_cache(points_world, T_world_camera, camera)
        targets, registration_failures = _registration_targets(registrations)
        result.failures.extend(registration_failures)
        result.registration_targets = len(targets)
        declared_ids = _declared_mask_ids(metadata)
        masks = load_candidate_masks(snapshot, rgb.shape[:2])
        for detection_id in declared_ids:
            source = f"candidate mask {detection_id}"
            mask = masks.get(detection_id)
            if mask is None:
                result.failures.append(
                    CandidateFailure(source, "missing, unreadable, or wrong shape")
                )
                continue
            try:
                prompt = _mask_prompt(mask)
                selection = select_points_from_mask(
                    points_world,
                    projection,
                    mask,
                    prompt,
                    MaskSelectionParams(),
                )
                if not selection.success:
                    raise ValueError(selection.reason)
                selected_uv = projection.uv[selection.indices]
                pixel_distance = np.linalg.norm(selected_uv - np.asarray(prompt), axis=1)
                near = np.flatnonzero(pixel_distance <= max(float(pixel_distance.min()), 8.0))
                local_index = near[int(np.argmin(projection.depth[selection.indices[near]]))]
                seed = selection.points[local_index]
                fit = snap_block_to_cloud(
                    selection.points,
                    seed,
                    SnapParams(reject_ambiguous=False),
                )
                if not fit.success or fit.position is None or fit.rotation is None:
                    raise ValueError(fit.reason)
                if fit.dimensions is None:
                    raise ValueError("fit returned no dimensions")
                if _near_existing(fit.position, targets, dedup_distance_m):
                    result.failures.append(
                        CandidateFailure(source, "deduplicated against prior fit")
                    )
                    continue
                base = f"rgb_candidate_{detection_id}"
                target_id = base
                suffix = 2
                used = {target.id for target in targets}
                while target_id in used:
                    target_id = f"{base}_{suffix}"
                    suffix += 1
                targets.append(
                    EditableTarget(
                        target_id,
                        fit.position,
                        fit.rotation,
                        dimensions=BLOCK_DIMS,
                        provenance="rgb_candidate_fit",
                        review_status="pending",
                        confidence=float(fit.score),
                    )
                )
                result.rgb_targets += 1
            except (IndexError, TypeError, ValueError) as error:
                result.failures.append(CandidateFailure(source, str(error)))
        document = AnnotationDocument(
            snapshot.name,
            targets=targets,
            annotation_complete=False,
            roi_min=None,
            roi_max=None,
            notes="Provisional batch initialization; every pending target requires manual review.",
        )
        errors = document.validate()
        if errors:
            raise ValueError("; ".join(errors))
        rendered = yaml.safe_dump(document.as_mapping(), sort_keys=False, allow_unicode=True)
        if not _publish_no_replace(rendered, manifest):
            result.status = "skipped_existing"
            result.manifest = manifest
            return result
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        result.failures.append(CandidateFailure("snapshot", str(error)))
        return result
    result.status = "created"
    result.manifest = manifest
    return result


def initialize_root(
    root: str | Path,
    *,
    camera_calibration: str | Path | None = None,
    dedup_distance_m: float = 0.40,
) -> list[InitializationResult]:
    root = Path(root)
    exclusions = load_snapshot_exclusions(root)
    results = []
    for snapshot in discover_snapshots(root, include_excluded=True):
        if snapshot.name in exclusions:
            exclusion = exclusions[snapshot.name]
            results.append(
                InitializationResult(
                    snapshot.name,
                    "skipped_excluded",
                    failures=[CandidateFailure("exclusion", exclusion.reason)],
                )
            )
            continue
        results.append(
            initialize_snapshot(
                snapshot,
                camera_calibration=camera_calibration,
                dedup_distance_m=dedup_distance_m,
            )
        )
    return results
