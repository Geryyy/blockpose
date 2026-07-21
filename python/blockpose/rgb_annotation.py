"""RGB-prompted point selection for block annotation.

This module deliberately has no Open3D GUI dependency.  Model loading is lazy,
so projection and mask-selection tests never download weights or import torch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Callable, Hashable, Protocol

import numpy as np
import yaml
from matplotlib import image as mpl_image
from scipy import ndimage
from scipy.spatial import cKDTree

from blockpose.io import CameraModel
from blockpose.projection import project_points, world_to_camera


class SamBackend(Protocol):
    """Minimal promptable segmenter interface used by the annotator."""

    def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
        """Return a boolean mask with the same height and width as ``rgb``."""


def segment_with_backend(
    backend: SamBackend, rgb: np.ndarray, point_xy: tuple[int, int]
) -> np.ndarray:
    """Run a backend and enforce its boundary contract."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    u, v = point_xy
    if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
        raise ValueError("prompt lies outside the RGB image")
    mask = np.asarray(backend.segment(image, point_xy), dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"segmenter mask shape {mask.shape} does not match RGB {image.shape[:2]}")
    return mask


class AsyncSegmentationWorker:
    """Single-flight segmentation with stale/closed callback suppression."""

    def __init__(self, backend: SamBackend):
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-annotator")
        self._lock = RLock()
        self._busy = False
        self._closed = False
        self._generation = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def invalidate(self) -> None:
        """Invalidate results, for example when switching snapshots."""
        with self._lock:
            self._generation += 1

    def submit(
        self,
        rgb: np.ndarray,
        point_xy: tuple[int, int],
        context: Hashable,
        schedule: Callable[[Callable[[], None]], None],
        complete: Callable[[np.ndarray | None, str | None, Hashable], None],
    ) -> bool:
        """Submit unless busy/closed; completion is scheduled only while current."""
        with self._lock:
            if self._closed or self._busy:
                return False
            self._busy = True
            generation = self._generation
        image = np.asarray(rgb).copy()

        def work() -> None:
            try:
                mask = segment_with_backend(self.backend, image, point_xy)
                error = None
            except Exception as caught:  # backends are an optional annotation aid
                mask, error = None, str(caught)
            with self._lock:
                if self._closed or generation != self._generation:
                    self._busy = False
                    return

                def deliver() -> None:
                    with self._lock:
                        if self._closed or generation != self._generation:
                            self._busy = False
                            return
                        self._busy = False
                    complete(mask, error, context)

                # Keep the close/check/schedule operation atomic: after close()
                # returns, no worker can enqueue another GUI callback.
                schedule(deliver)

        try:
            self._executor.submit(work)
        except RuntimeError:
            # close() may win the race after the initial state check.
            with self._lock:
                self._busy = False
            return False
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class ProjectionCache:
    uv: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    pixel_u: np.ndarray
    pixel_v: np.ndarray
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class MaskSelectionParams:
    dilation_px: int = 4
    depth_tolerance_m: float = 0.45
    spatial_radius_m: float = 1.25
    connectivity_radius_m: float = 0.16
    min_points: int = 20


@dataclass(frozen=True)
class MaskSelectionResult:
    success: bool
    indices: np.ndarray
    points: np.ndarray
    mask: np.ndarray
    reason: str
    projected_count: int = 0
    depth_filtered_count: int = 0


def map_widget_to_image(
    x: float,
    y: float,
    widget_width: float,
    widget_height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    """Map a click through an aspect-preserving, centered ImageWidget viewport."""
    values = (x, y, widget_width, widget_height, image_width, image_height)
    if (
        not np.all(np.isfinite(values))
        or min(widget_width, widget_height, image_width, image_height) <= 0
    ):
        return None
    scale = min(widget_width / image_width, widget_height / image_height)
    shown_w, shown_h = image_width * scale, image_height * scale
    left, top = (widget_width - shown_w) / 2.0, (widget_height - shown_h) / 2.0
    if x < left or y < top or x >= left + shown_w or y >= top + shown_h:
        return None
    u = min(image_width - 1, int((x - left) / scale))
    v = min(image_height - 1, int((y - top) / scale))
    return u, v


def build_projection_cache(
    points_world: np.ndarray,
    T_world_camera: np.ndarray,
    camera: CameraModel,
) -> ProjectionCache:
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world must be a finite (N, 3) array")
    camera_points = world_to_camera(points, np.asarray(T_world_camera, dtype=float))
    uv, in_front = project_points(camera_points, camera)
    pixel_u = np.rint(uv[:, 0]).astype(np.int64)
    pixel_v = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        in_front
        & np.all(np.isfinite(uv), axis=1)
        & (pixel_u >= 0)
        & (pixel_u < camera.width)
        & (pixel_v >= 0)
        & (pixel_v < camera.height)
    )
    return ProjectionCache(
        uv,
        camera_points[:, 2],
        valid,
        pixel_u,
        pixel_v,
        (camera.height, camera.width),
    )


def load_candidate_masks(snapshot: Path, image_shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """Safely load declared masks; malformed and escaping paths are ignored."""
    metadata_path = snapshot / "metadata.yaml"
    if not metadata_path.is_file():
        return {}
    try:
        metadata = yaml.safe_load(metadata_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(metadata, dict):
        return {}
    result: dict[int, np.ndarray] = {}
    root = snapshot.resolve()
    for entry in metadata.get("candidate_masks", []):
        try:
            candidate = (snapshot / str(entry["file"])).resolve()
            candidate.relative_to(root)
            detection_id = int(entry["detection_id"])
            raw = np.asarray(mpl_image.imread(candidate))
            if raw.ndim == 3:
                raw = np.any(raw[..., :3] > 0, axis=2)
            mask = np.asarray(raw > 0, dtype=bool)
            if mask.shape == image_shape:
                result[detection_id] = mask
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return result


def candidate_mask_at(
    masks: dict[int, np.ndarray], point_xy: tuple[int, int]
) -> tuple[int, np.ndarray] | None:
    u, v = point_xy
    hits = [
        (key, mask)
        for key, mask in masks.items()
        if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]
    ]
    if not hits:
        return None
    return min(hits, key=lambda item: (int(np.count_nonzero(item[1])), item[0]))


def select_points_from_mask(
    points_world: np.ndarray,
    cache: ProjectionCache,
    mask: np.ndarray,
    prompt_xy: tuple[int, int],
    params: MaskSelectionParams | None = None,
) -> MaskSelectionResult:
    """Project a mask to 3D, then reject depth bleed and disconnected clutter."""
    if params is None:
        params = MaskSelectionParams()
    points = np.asarray(points_world, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        return MaskSelectionResult(
            False, np.empty(0, int), np.empty((0, 3)), mask, "mask must be 2D"
        )
    if mask.shape != cache.image_shape:
        return MaskSelectionResult(
            False,
            np.empty(0, int),
            np.empty((0, 3)),
            mask,
            f"mask shape {mask.shape} does not match image {cache.image_shape}",
        )
    if params.dilation_px < 0 or params.min_points < 1:
        return MaskSelectionResult(
            False, np.empty(0, int), np.empty((0, 3)), mask, "invalid selection parameters"
        )
    expanded = (
        ndimage.binary_dilation(mask, iterations=params.dilation_px)
        if params.dilation_px
        else mask.copy()
    )
    inside = cache.valid.copy()
    valid_idx = np.flatnonzero(inside)
    inside[valid_idx] = expanded[cache.pixel_v[valid_idx], cache.pixel_u[valid_idx]]
    candidates = np.flatnonzero(inside)
    if len(candidates) < params.min_points:
        return MaskSelectionResult(
            False,
            candidates,
            points[candidates],
            expanded,
            "too few projected mask points",
            len(candidates),
        )

    u0, v0 = prompt_xy
    pixel_d2 = (cache.uv[candidates, 0] - u0) ** 2 + (cache.uv[candidates, 1] - v0) ** 2
    prompt_near = candidates[pixel_d2 <= 8.0**2]
    if len(prompt_near):
        seed = prompt_near[int(np.argmin(cache.depth[prompt_near]))]
    else:
        seed = candidates[int(np.argmin(pixel_d2))]
    seed_depth = cache.depth[seed]
    depth_keep = np.abs(cache.depth[candidates] - seed_depth) <= params.depth_tolerance_m
    depth_idx = candidates[depth_keep]
    if len(depth_idx) < params.min_points:
        return MaskSelectionResult(
            False,
            depth_idx,
            points[depth_idx],
            expanded,
            "too few depth-coherent points",
            len(candidates),
            len(depth_idx),
        )

    seed_local = int(np.argmin(np.linalg.norm(points[depth_idx] - points[seed], axis=1)))
    near = (
        np.linalg.norm(points[depth_idx] - points[depth_idx[seed_local]], axis=1)
        <= params.spatial_radius_m
    )
    local_idx = depth_idx[near]
    if len(local_idx) >= params.min_points and params.connectivity_radius_m > 0:
        tree = cKDTree(points[local_idx])
        start = int(np.argmin(np.linalg.norm(points[local_idx] - points[seed], axis=1)))
        visited = {start}
        frontier = [start]
        while frontier:
            for neighbor in tree.query_ball_point(
                points[local_idx[frontier.pop()]], params.connectivity_radius_m
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        local_idx = local_idx[np.fromiter(sorted(visited), dtype=int)]
    if len(local_idx) < params.min_points:
        return MaskSelectionResult(
            False,
            local_idx,
            points[local_idx],
            expanded,
            "too few spatially connected points",
            len(candidates),
            len(depth_idx),
        )
    return MaskSelectionResult(
        True, local_idx, points[local_idx], expanded, "ok", len(candidates), len(depth_idx)
    )


class TransformersSam2Backend:
    """Lazy optional Hugging Face SAM2 backend; no weights load until first prompt."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "auto",
        *,
        local_files_only: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._inference_lock = Lock()

    def _load(self) -> None:
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as error:
            raise RuntimeError("SAM requires the optional 'sam' dependencies") from error
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self._processor = Sam2Processor.from_pretrained(
            self.model_id, local_files_only=self.local_files_only
        )
        self._model = (
            Sam2Model.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            .to(self.device)
            .eval()
        )
        self._torch = torch

    def segment(self, rgb: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
        # Loading/inference on one shared model is serialized. Rapid GUI clicks
        # still use stale-token suppression, without racing model initialization.
        with self._inference_lock:
            if self._model is None:
                self._load()
            assert self._processor is not None and self._model is not None
            inputs = self._processor(
                images=rgb,
                input_points=[[[list(point_xy)]]],
                input_labels=[[[1]]],
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            masks = self._processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
            scores = outputs.iou_scores.reshape(-1)
            return np.asarray(
                masks.reshape(-1, *masks.shape[-2:])[int(scores.argmax())] > 0,
                dtype=bool,
            )
