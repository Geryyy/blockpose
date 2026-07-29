"""Fast classical RGB evidence for already-generated cuboid candidates.

This module deliberately does not detect blocks or infer a metric pose.  It
scores whether a projected cuboid wireframe agrees with image edges, so callers
may use it only to re-rank otherwise geometrically valid hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from blockpose.io import CameraModel
from blockpose.pose import BlockPose
from blockpose.projection import project_points, world_to_camera
from blockpose.viz import _EDGES, box_corners


@dataclass(frozen=True)
class EdgePriorParams:
    """Conservative image-processing and sampling parameters."""

    edge_percentile: float = 92.0
    sample_spacing_px: float = 4.0
    distance_scale_px: float = 3.0
    min_samples: int = 12
    orientation_weight: float = 0.25
    max_image_dimension_px: int = 768
    min_support_fraction: float = 0.15


DEFAULT_PARAMS = EdgePriorParams()


@dataclass(frozen=True)
class EdgeMap:
    """Per-image data shared by every candidate score."""

    distance_px: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    edge_mask: np.ndarray
    source_shape: tuple[int, int]


@dataclass(frozen=True)
class EdgeScore:
    """Bounded candidate evidence and enough diagnostics for an overlay."""

    score: float = 0.0
    evidence: bool = False
    sampled_pixels: np.ndarray | None = None
    sampled_distances_px: np.ndarray | None = None
    used_samples: int = 0
    support_fraction: float = 0.0
    reason: str = "neutral"


def _neutral(reason: str) -> EdgeScore:
    return EdgeScore(
        reason=reason,
        sampled_pixels=np.empty((0, 2), dtype=np.int64),
        sampled_distances_px=np.empty(0, dtype=np.float64),
    )


def build_edge_map(
    rgb: np.ndarray | None, params: EdgePriorParams = DEFAULT_PARAMS
) -> EdgeMap | None:
    """Build a deterministic grayscale edge-distance field, or ``None`` when unusable."""
    if rgb is None:
        return None
    image = np.asarray(rgb)
    if image.ndim not in (2, 3) or image.shape[0] < 2 or image.shape[1] < 2:
        return None
    if image.ndim == 3 and image.shape[2] < 3:
        return None
    if not np.all(np.isfinite(image)):
        return None
    if not (
        0.0 < params.edge_percentile < 100.0
        and params.sample_spacing_px > 0.0
        and params.distance_scale_px > 0.0
        and params.min_samples > 0
        and 0.0 <= params.orientation_weight <= 1.0
        and params.max_image_dimension_px >= 2
        and 0.0 <= params.min_support_fraction <= 1.0
    ):
        return None
    gray = (
        image.astype(np.float64)
        if image.ndim == 2
        else (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2])
    )
    source_shape = tuple(gray.shape)
    largest_dimension = max(source_shape)
    if largest_dimension > params.max_image_dimension_px:
        scale = params.max_image_dimension_px / largest_dimension
        gray = ndimage.zoom(gray, scale, order=1, prefilter=False)
    grad_y = ndimage.sobel(gray, axis=0, mode="reflect")
    grad_x = ndimage.sobel(gray, axis=1, mode="reflect")
    magnitude = np.hypot(grad_x, grad_y)
    nonzero_magnitude = magnitude[magnitude > np.finfo(float).eps]
    if not len(nonzero_magnitude):
        return None
    threshold = float(np.percentile(nonzero_magnitude, params.edge_percentile))
    if not np.isfinite(threshold) or threshold <= np.finfo(float).eps:
        return None
    edge_mask = magnitude >= threshold
    if not np.any(edge_mask):
        return None
    return EdgeMap(
        distance_px=ndimage.distance_transform_edt(~edge_mask),
        grad_x=grad_x,
        grad_y=grad_y,
        edge_mask=edge_mask,
        source_shape=source_shape,
    )


def _sample_projected_edges(
    pose: BlockPose, T_world_camera: np.ndarray, camera: CameraModel, params: EdgePriorParams
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return image samples and local projected edge normals, excluding behind-camera edges."""
    try:
        transform = np.asarray(T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return None
        corners_world = box_corners(pose, pose.dims)
        corners_camera = world_to_camera(corners_world, transform)
        uv, in_front = project_points(corners_camera, camera)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(uv)):
        return None
    pixels: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    corner_signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    camera_origin = transform[:3, 3]
    for start, end in _EDGES:
        if not (in_front[start] and in_front[end]):
            continue
        # A convex cuboid edge is visible when one of its two adjacent faces
        # faces the camera.  Rear-only edges are not expected to match RGB and
        # would otherwise turn valid self-occlusion into negative evidence.
        unchanged_axes = np.flatnonzero(corner_signs[start] == corner_signs[end])
        midpoint = (corners_world[start] + corners_world[end]) / 2.0
        face_visible = any(
            np.dot(pose.rotation[:, axis] * corner_signs[start, axis], camera_origin - midpoint)
            > 0.0
            for axis in unchanged_axes
        )
        if not face_visible:
            continue
        delta = uv[end] - uv[start]
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length < 1e-6:
            continue
        count = max(2, int(np.ceil(length / params.sample_spacing_px)) + 1)
        samples = uv[start] + np.linspace(0.0, 1.0, count)[:, None] * delta
        tangent = delta / length
        pixels.append(samples)
        normals.append(np.repeat(np.array([[-tangent[1], tangent[0]]]), count, axis=0))
    if not pixels:
        return None
    return np.vstack(pixels), np.vstack(normals)


def score_projected_cuboid(
    edge_map: EdgeMap | None,
    pose: BlockPose,
    T_world_camera: np.ndarray | None,
    camera: CameraModel | None,
    *,
    excluded_mask: np.ndarray | None = None,
    params: EdgePriorParams = DEFAULT_PARAMS,
) -> EdgeScore:
    """Score one candidate; unavailable/occluded evidence is safely neutral.

    ``excluded_mask`` is intended for the projected gripper or other known
    occluder.  It skips pixels rather than counting absence of an image edge as
    evidence against the cuboid.
    """
    if edge_map is None or T_world_camera is None or camera is None:
        return _neutral("missing_rgb_or_calibration")
    shape = edge_map.distance_px.shape
    if (
        edge_map.grad_x.shape != shape
        or edge_map.grad_y.shape != shape
        or edge_map.edge_mask.shape != shape
        or edge_map.source_shape != (camera.height, camera.width)
    ):
        return _neutral("incompatible_edge_map")
    if excluded_mask is not None:
        excluded = np.asarray(excluded_mask, dtype=bool)
        if excluded.shape != (camera.height, camera.width):
            return _neutral("invalid_excluded_mask")
    else:
        excluded = None
    sampled = _sample_projected_edges(pose, T_world_camera, camera, params)
    if sampled is None:
        return _neutral("not_projectable")
    uv, normals = sampled
    pixels = np.rint(uv).astype(np.int64)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )
    if excluded is not None:
        valid &= ~excluded[
            np.clip(pixels[:, 1], 0, camera.height - 1), np.clip(pixels[:, 0], 0, camera.width - 1)
        ]
    pixels, normals = pixels[valid], normals[valid]
    if len(pixels) < params.min_samples:
        return _neutral("insufficient_visible_samples")
    scale_x = shape[1] / camera.width
    scale_y = shape[0] / camera.height
    map_pixels = np.rint(pixels * np.array([scale_x, scale_y])).astype(np.int64)
    map_pixels[:, 0] = np.clip(map_pixels[:, 0], 0, shape[1] - 1)
    map_pixels[:, 1] = np.clip(map_pixels[:, 1], 0, shape[0] - 1)
    distances = edge_map.distance_px[map_pixels[:, 1], map_pixels[:, 0]] / scale_x
    gx = edge_map.grad_x[map_pixels[:, 1], map_pixels[:, 0]]
    gy = edge_map.grad_y[map_pixels[:, 1], map_pixels[:, 0]]
    gradient_norm = np.hypot(gx, gy)
    normal_alignment = np.zeros(len(pixels), dtype=np.float64)
    nonzero = gradient_norm > np.finfo(float).eps
    normal_alignment[nonzero] = np.abs(
        (gx[nonzero] * normals[nonzero, 0] + gy[nonzero] * normals[nonzero, 1])
        / gradient_norm[nonzero]
    )
    proximity = np.exp(-distances / params.distance_scale_px)
    support_fraction = float(np.mean(distances <= params.distance_scale_px))
    if support_fraction < params.min_support_fraction:
        return _neutral("insufficient_edge_support")
    per_sample = proximity * (
        (1.0 - params.orientation_weight) + params.orientation_weight * normal_alignment
    )
    return EdgeScore(
        score=float(np.mean(per_sample)),
        evidence=True,
        sampled_pixels=pixels,
        sampled_distances_px=distances,
        used_samples=len(pixels),
        support_fraction=support_fraction,
        reason="ok",
    )
