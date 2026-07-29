"""Non-negotiable input gates for the RGB candidate-prior exporter.

Every sample must clear these before it is written.  A failed gate returns a
``GateResult`` with ``passed=False`` and a recorded reason so the exporter can
drop the sample instead of guessing one (see the design doc's "input gates"
section).  Nothing here imports a model runtime.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from blockpose.io import CameraModel
from blockpose.projection import project_points, world_to_camera

# Distortion vectors are OpenCV plumb-bob (5) or rational-polynomial (8); empty
# means an ideal pinhole.  Any other length means the calibration is malformed.
_VALID_DISTORTION_LENGTHS = (0, 5, 8)


@dataclass(frozen=True)
class GateResult:
    """Outcome of one input gate; ``details`` is exported for audit."""

    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    """Content checksum, stored in every record for image/calibration/TF identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_ns(stamp: dict[str, Any]) -> int:
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])


def read_stamps(metadata_path: str | Path) -> tuple[int, int]:
    """Return ``(image_stamp_ns, cloud_stamp_ns)`` from a snapshot ``metadata.yaml``."""
    with Path(metadata_path).open() as handle:
        data = yaml.safe_load(handle)
    # Registration snapshots predate direct scene-discovery captures and call
    # the image section `image`; captures intentionally call it `rgb`. Both
    # name the same sensor timestamp, so accept either schema without guessing.
    image = data.get("image", data.get("rgb"))
    if not isinstance(image, dict) or not isinstance(data.get("cloud"), dict):
        raise ValueError("metadata needs image/rgb and cloud stamp sections")
    return _stamp_ns(image["stamp"]), _stamp_ns(data["cloud"]["stamp"])


def sync_gate(image_stamp_ns: int, cloud_stamp_ns: int, *, tolerance_s: float) -> GateResult:
    """RGB and cloud/candidate stamps must be within a recorded sync tolerance."""
    delta_s = abs(image_stamp_ns - cloud_stamp_ns) / 1e9
    passed = delta_s <= tolerance_s
    return GateResult(
        passed=passed,
        reason="ok" if passed else "sync_delta_exceeds_tolerance",
        details={"sync_delta_s": delta_s, "tolerance_s": tolerance_s},
    )


def calibration_gate(camera: CameraModel, rgb_shape: tuple[int, ...]) -> GateResult:
    """Intrinsics, distortion convention, and image resolution must be valid and consistent."""
    if len(camera.D) not in _VALID_DISTORTION_LENGTHS:
        return GateResult(False, "invalid_distortion_length", {"distortion_len": len(camera.D)})
    if not (np.all(np.isfinite(camera.K)) and np.all(np.isfinite(camera.D))):
        return GateResult(False, "non_finite_calibration", {})
    if not (camera.fx > 0.0 and camera.fy > 0.0):
        return GateResult(False, "non_positive_focal_length", {"fx": camera.fx, "fy": camera.fy})
    if rgb_shape[:2] != (camera.height, camera.width):
        return GateResult(
            False,
            "resolution_mismatch",
            {"rgb_shape": list(rgb_shape[:2]), "camera_hw": [camera.height, camera.width]},
        )
    return GateResult(True, "ok", {"image_hw": [camera.height, camera.width]})


def tf_gate(T_world_camera: np.ndarray | None) -> GateResult:
    """A valid, finite ``world -> camera`` transform must be present at the candidate stamp."""
    if T_world_camera is None:
        return GateResult(False, "missing_world_camera_tf", {})
    T = np.asarray(T_world_camera, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return GateResult(False, "invalid_world_camera_tf", {"shape": list(T.shape)})
    return GateResult(True, "ok", {})


def projection_gate(
    corners_world: np.ndarray, T_world_camera: np.ndarray, camera: CameraModel
) -> GateResult:
    """Every projected corner must be finite, in front of the camera, and in image bounds."""
    try:
        corners_cam = world_to_camera(np.asarray(corners_world, dtype=np.float64), T_world_camera)
        uv, in_front = project_points(corners_cam, camera)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return GateResult(False, "projection_failed", {})
    if not np.all(np.isfinite(uv)):
        return GateResult(False, "non_finite_projection", {})
    if not np.all(in_front):
        return GateResult(False, "corner_behind_camera", {"in_front": int(np.sum(in_front))})
    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < camera.height)
    )
    if not np.any(inside):
        return GateResult(False, "projection_out_of_bounds", {"corners_in_bounds": 0})
    return GateResult(
        True,
        "ok",
        {"corners_in_bounds": int(np.sum(inside)), "corners_total": int(len(inside))},
    )
