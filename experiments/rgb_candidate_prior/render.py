"""Conditioning-image and audit-overlay rendering for exported candidates.

The conditioning image tells the model *which* box a crop is being asked about:
without it an embedding cannot say whether *this* cuboid matches the pixels.  It
is rendered in full-image pixel coordinates and then cropped so its geometry
stays consistent with the RGB crop.

The visible-edge rule mirrors ``blockpose.rgb_edge_prior._sample_projected_edges``
(a convex cuboid edge is drawn only when one of its two adjacent faces points at
the camera) so rear-only self-occluded edges are not presented as evidence.
"""

from __future__ import annotations

import numpy as np
from blockpose.io import CameraModel
from blockpose.pose import BlockPose
from blockpose.projection import project_points, world_to_camera
from blockpose.viz import _EDGES, box_corners
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

# Corner sign layout matching ``box_corners``/``_EDGES`` ([-1,1]^3, x outer).
_CORNER_SIGNS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def project_corners(
    pose: BlockPose, T_world_camera: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(corners_world (8,3), uv (8,2), in_front (8,))`` for the posed cuboid."""
    corners_world = box_corners(pose, pose.dims)
    corners_cam = world_to_camera(corners_world, T_world_camera)
    uv, in_front = project_points(corners_cam, camera)
    return corners_world, uv, in_front


def visible_edges(
    pose: BlockPose, corners_world: np.ndarray, in_front: np.ndarray, T_world_camera: np.ndarray
) -> list[tuple[int, int]]:
    """Indices of cuboid edges whose two endpoints are in front and on a camera-facing face."""
    camera_origin = np.asarray(T_world_camera, dtype=np.float64)[:3, 3]
    edges: list[tuple[int, int]] = []
    for start, end in _EDGES:
        if not (in_front[start] and in_front[end]):
            continue
        unchanged_axes = np.flatnonzero(_CORNER_SIGNS[start] == _CORNER_SIGNS[end])
        midpoint = (corners_world[start] + corners_world[end]) / 2.0
        if any(
            np.dot(pose.rotation[:, axis] * _CORNER_SIGNS[start, axis], camera_origin - midpoint)
            > 0.0
            for axis in unchanged_axes
        ):
            edges.append((start, end))
    return edges


def crop_bbox(
    uv: np.ndarray, in_front: np.ndarray, camera: CameraModel, *, margin_frac: float
) -> tuple[int, int, int, int] | None:
    """Axis-aligned crop around the in-front projection, padded by ``margin_frac``.

    Returns ``(x0, y0, x1, y1)`` clamped to image bounds, or ``None`` when no
    corner is in front of the camera.
    """
    points = uv[in_front]
    if len(points) == 0 or not np.all(np.isfinite(points)):
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    pad_x = margin_frac * max(x1 - x0, 1.0)
    pad_y = margin_frac * max(y1 - y0, 1.0)
    x0 = int(np.clip(np.floor(x0 - pad_x), 0, camera.width - 1))
    y0 = int(np.clip(np.floor(y0 - pad_y), 0, camera.height - 1))
    x1 = int(np.clip(np.ceil(x1 + pad_x), x0 + 1, camera.width))
    y1 = int(np.clip(np.ceil(y1 + pad_y), y0 + 1, camera.height))
    return x0, y0, x1, y1


def _hull_polygon(points: np.ndarray) -> list[tuple[float, float]] | None:
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:  # noqa: BLE001 - degenerate/collinear projections are expected
        return None
    return [(float(points[v, 0]), float(points[v, 1])) for v in hull.vertices]


def render_conditioning(
    pose: BlockPose,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    bbox: tuple[int, int, int, int],
    *,
    out_size: int,
) -> np.ndarray:
    """Render an ``(out_size, out_size, 3)`` uint8 conditioning image for one candidate.

    Channels: R = visible-edge wireframe, G = filled silhouette (convex hull of
    the in-front corners), B = 0 (reserved for a future depth/occlusion channel).
    Rendered inside ``bbox`` then resized so it aligns with the RGB crop.
    """
    corners_world, uv, in_front = project_corners(pose, T_world_camera, camera)
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    local = uv - np.array([x0, y0], dtype=np.float64)

    edges = Image.new("L", (width, height), 0)
    silhouette = Image.new("L", (width, height), 0)
    edge_draw = ImageDraw.Draw(edges)
    silhouette_draw = ImageDraw.Draw(silhouette)

    polygon = _hull_polygon(local[in_front])
    if polygon is not None:
        silhouette_draw.polygon(polygon, fill=255)
    for start, end in visible_edges(pose, corners_world, in_front, T_world_camera):
        edge_draw.line([tuple(local[start]), tuple(local[end])], fill=255, width=2)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 0] = np.asarray(edges)
    canvas[..., 1] = np.asarray(silhouette)
    resized = Image.fromarray(canvas).resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(resized)


def render_crop(rgb: np.ndarray, bbox: tuple[int, int, int, int], *, out_size: int) -> np.ndarray:
    """Fixed-size RGB crop for ``bbox`` (``(out_size, out_size, 3)`` uint8)."""
    x0, y0, x1, y1 = bbox
    crop = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[y0:y1, x0:x1])
    return np.asarray(crop.resize((out_size, out_size), Image.BILINEAR))


def render_overlay(
    rgb: np.ndarray,
    pose: BlockPose,
    T_world_camera: np.ndarray,
    camera: CameraModel,
    bbox: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Full-resolution RGB with the candidate wireframe + crop box, for manual audit."""
    corners_world, uv, in_front = project_corners(pose, T_world_camera, camera)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
    for start, end in visible_edges(pose, corners_world, in_front, T_world_camera):
        draw.line([tuple(uv[start]), tuple(uv[end])], fill=color, width=2)
    return np.asarray(image)


def save_png(image: np.ndarray, path) -> None:
    """Write an (H, W, 3) uint8 array as a PNG, creating parent directories."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(out)
