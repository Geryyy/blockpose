"""Full-image, class-agnostic 2-D region proposals from a frozen DINOv2 backbone.

This implements the `dino_proposer` half of the RGB priors described in
``docs/dino-rgb-prior-ros-integration.md``: it recovers a missed or merged
block by predicting a *high-recall* 2-D region over the whole image. The
detector, not this module, selects already PZS100-filtered LiDAR points
projected into that region and runs its ordinary local cuboid fit. This
module never emits a metric pose, a mask decoder's instance mask, or a
learned uncertainty output -- the doc is explicit that the head stays a
compact per-patch MLP/shallow-conv, not a Mask2Former-style architecture.

Only the small dense head is trainable; the DINOv2 backbone is always frozen.
Torch/transformers are imported lazily inside methods, matching
``rgb_verifier.DinoVerifierHead``/``rgb_annotation.TransformersSam2Backend``,
so importing ``blockpose`` (or this module) never requires torch.

Unlike ``DinoVerifierHead``, which runs on small square candidate crops, this
head runs on the *whole* RGB frame (e.g. 1536x2048 for this dataset) resized
so its long side is ``config.max_image_side`` -- a very different input
scale, so the patch-grid extraction here is aspect-preserving and generally
rectangular, not the square-crop reshape ``DinoVerifierHead._patch_grid``
relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from blockpose._dino_backbone import load_frozen_backbone
from blockpose._head_checkpoint import (
    config_from_payload,
    read_checkpoint_config,
    write_checkpoint_config,
)

if TYPE_CHECKING:
    import torch

    from blockpose.rgb_proposals import ImageProposal

_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DinoRegionProposerConfig:
    """Frozen head configuration; safe to serialize as-is (see `save_checkpoint`)."""

    model_id: str = "facebook/dinov2-small"
    hidden_dim: int = 128
    # Resize the long side of the input image to this before running the
    # backbone. DINOv2's patch size is 14, so this also sets the patch-grid
    # resolution (e.g. 896 -> a ~64x48 grid for this dataset's 2048x1536
    # frames). Larger keeps more spatial detail for small/merged blocks but
    # backbone compute and memory grow ~quadratically with patch count; 896
    # is a reasonable first-pass tradeoff, not a tuned value -- revisit once
    # bag-level promotion-gate metrics (see the ROS integration doc) exist.
    max_image_side: int = 896
    # Probability threshold for extracting proposal instances from the dense map.
    foreground_threshold: float = 0.5
    # Drop connected components smaller than this after thresholding; filters
    # spurious single-patch-scale noise, not intended to suppress real small blocks.
    min_component_area_px: int = 200

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if self.max_image_side < 1:
            raise ValueError("max_image_side must be a positive integer")
        if not 0.0 <= self.foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold must lie in [0, 1]")
        if self.min_component_area_px < 0:
            raise ValueError("min_component_area_px must be non-negative")


def _patch_aligned_size(
    height: int, width: int, max_image_side: int, patch_size: int
) -> tuple[int, int]:
    """Aspect-preserving resize target with the long side ~= `max_image_side`.

    Both dimensions are rounded to the nearest (non-zero) multiple of
    `patch_size` so the backbone's patch grid tiles the resized image exactly.
    """
    scale = max_image_side / max(height, width)
    resized_h = max(patch_size, patch_size * round(height * scale / patch_size))
    resized_w = max(patch_size, patch_size * round(width * scale / patch_size))
    return resized_h, resized_w


class DinoRegionProposerHead:
    """Frozen DINOv2 patch-token backbone + a small trainable dense head.

    Only the dense head (`.dense_head`) is trainable; the backbone is always
    frozen (`requires_grad_(False)`, `.eval()`). Nothing is loaded until the
    first call that needs it (`patch_features`/`predict_proposal_map`, or an
    explicit `_load()`), so constructing a head never requires torch/
    transformers or a network fetch.

    The dense head is a 2-layer conv (3x3 -> ReLU -> 1x1), deliberately not a
    transformer decoder or Mask2Former-style query architecture, per the ROS
    integration doc's explicit "keep the head compact" decision. It maps
    per-patch DINOv2 features to a single foreground logit per patch.
    """

    def __init__(self, config: DinoRegionProposerConfig, *, device: str = "auto"):
        self.config = config
        self.device = device
        self._processor = None
        self._backbone = None
        self._head = None
        self._torch = None
        self._patch_size: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._backbone is not None

    @property
    def dense_head(self) -> "torch.nn.Module":
        """The trainable per-patch dense head; loads the backbone+head if needed."""
        self._load()
        assert self._head is not None
        return self._head

    def _load(self) -> None:
        if self._backbone is not None:
            return
        torch, self._processor, backbone, self.device = load_frozen_backbone(
            self.config.model_id, self.device, head_name="dino_proposer"
        )
        self._backbone = backbone
        self._patch_size = int(backbone.config.patch_size)
        from torch import nn

        hidden_size = backbone.config.hidden_size
        head = nn.Sequential(
            nn.Conv2d(hidden_size, self.config.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.config.hidden_dim, 1, kernel_size=1),  # foreground logit
        ).to(self.device)
        head.eval()
        self._head = head
        self._torch = torch

    def _patch_grid(self, rgb: np.ndarray) -> np.ndarray:
        """Run the frozen backbone; return the `(grid_h, grid_w, D)` patch-token map.

        Resizes `rgb` to `config.max_image_side` on the long side (aspect-
        preserving, patch-grid aligned) instead of the processor's default
        square resize+center-crop, since the proposer must see the whole
        frame rather than a candidate crop.
        """
        self._load()
        assert self._processor is not None
        assert self._backbone is not None and self._torch is not None
        assert self._patch_size is not None
        from PIL import Image

        image = np.asarray(rgb, dtype=np.uint8)
        height, width = image.shape[:2]
        resized_h, resized_w = _patch_aligned_size(
            height, width, self.config.max_image_side, self._patch_size
        )
        pil_image = Image.fromarray(image).resize((resized_w, resized_h), Image.BILINEAR)
        inputs = self._processor(
            images=pil_image, return_tensors="pt", do_resize=False, do_center_crop=False
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._backbone(**inputs)
        tokens = output.last_hidden_state[0]  # (1 + n_patches [+ registers], D)
        patch_tokens = tokens[1:].detach().cpu().numpy()
        grid_h, grid_w = resized_h // self._patch_size, resized_w // self._patch_size
        if grid_h * grid_w != patch_tokens.shape[0]:
            raise RuntimeError(
                f"unexpected patch token count {patch_tokens.shape[0]} for grid "
                f"{grid_h}x{grid_w} ({self.config.model_id})"
            )
        return patch_tokens.reshape(grid_h, grid_w, -1)

    def patch_features(self, rgb: np.ndarray) -> np.ndarray:
        """Frozen `(grid_h, grid_w, D)` DINOv2 patch-token features for one full image.

        Pure backbone forward pass (no grad, frozen weights) -- a training
        loop combines this with `.dense_head` for its own forward/backward,
        exactly as `DinoVerifierHead.embed_regions`/`.mlp` are combined in
        `rgb_verifier.py`.
        """
        return self._patch_grid(rgb)

    def predict_proposal_map(self, rgb: np.ndarray) -> np.ndarray:
        """Full inference path: dense foreground probability at `rgb`'s own resolution.

        Resizes per `config.max_image_side`, runs the frozen backbone plus
        the trainable dense head to get a `(grid_h, grid_w)` probability map,
        then bilinearly upsamples back to `rgb`'s original `(H, W)` -- directly
        usable as an `ImageProposal.mask`-shaped array after thresholding
        (see `extract_region_proposals`).
        """
        self._load()
        assert self._head is not None and self._torch is not None
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        height, width = image.shape[:2]
        patch_tokens = self._patch_grid(image)
        tensor = self._torch.from_numpy(patch_tokens).float().to(self.device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, D, grid_h, grid_w)
        with self._torch.inference_mode():
            logits = self._head(tensor)
            probabilities = self._torch.sigmoid(logits)
            upsampled = self._torch.nn.functional.interpolate(
                probabilities, size=(height, width), mode="bilinear", align_corners=False
            )
        return upsampled[0, 0].detach().cpu().numpy().astype(np.float64)


def _component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def extract_region_proposals(
    probability_map: np.ndarray,
    config: DinoRegionProposerConfig,
    *,
    provider: str = "dino_region_proposer",
    model_id: str | None = None,
    timestamp_ns: int | None = None,
) -> tuple["ImageProposal", ...]:
    """Threshold + connected-component extraction of high-recall regions.

    One `ImageProposal` per connected component of
    `probability_map >= config.foreground_threshold` (via `scipy.ndimage.label`,
    the same primitive used by `rgb_proposals.associate_mask_to_lidar`),
    dropping components smaller than `config.min_component_area_px`. Each
    proposal's `score` is the mean predicted probability within that
    component -- this is the `confidence` in the ROS integration doc's
    `regions[{bbox_px, confidence, uncertainty}]` contract; the proposer has
    no separate learned uncertainty output today, so none is invented here.
    """
    from blockpose.rgb_proposals import ImageProposal

    probabilities = np.asarray(probability_map, dtype=np.float64)
    if probabilities.ndim != 2 or not probabilities.size:
        raise ValueError("probability_map must be a non-empty 2-D array")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probability_map must be finite")
    foreground = probabilities >= config.foreground_threshold
    labeled, num_components = ndimage.label(foreground)
    proposals: list[ImageProposal] = []
    detection_id = 0
    for component_id in range(1, num_components + 1):
        component_mask = labeled == component_id
        if int(component_mask.sum()) < config.min_component_area_px:
            continue
        proposals.append(
            ImageProposal(
                mask=component_mask,
                bbox_xywh=_component_bbox(component_mask),
                score=float(probabilities[component_mask].mean()),
                provider=provider,
                model_id=model_id,
                detection_id=detection_id,
                timestamp_ns=timestamp_ns,
                provenance={"foreground_threshold": str(config.foreground_threshold)},
            )
        )
        detection_id += 1
    return tuple(proposals)


def save_checkpoint(
    path: Path, head: DinoRegionProposerHead, config: DinoRegionProposerConfig
) -> None:
    """Save the trainable dense head and config.

    On-disk format (see `blockpose._head_checkpoint`, shared with
    `rgb_verifier.py` -- callers must go through
    `save_checkpoint`/`load_dino_region_proposer`, never read these files
    directly): a directory with `dense_head.pt` (trainable dense-head state
    dict only -- the frozen backbone is never serialized) and `config.json`.
    """
    import torch

    head._load()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(head.dense_head.state_dict(), path / "dense_head.pt")
    write_checkpoint_config(path, config, _CHECKPOINT_SCHEMA_VERSION)


def load_dino_region_proposer(
    path: Path,
) -> tuple[DinoRegionProposerHead, DinoRegionProposerConfig]:
    """Load a checkpoint written by `save_checkpoint`.

    Raises `FileNotFoundError`/`ValueError` for a missing or unrecognized
    checkpoint rather than silently returning an untrained head.
    """
    import torch

    path = Path(path)
    payload = read_checkpoint_config(
        path,
        weights_filename="dense_head.pt",
        schema_version=_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_kind="dino_region_proposer",
    )
    config = config_from_payload(DinoRegionProposerConfig, payload)
    head = DinoRegionProposerHead(config)
    head._load()
    state_dict = torch.load(path / "dense_head.pt", map_location=head.device)
    head.dense_head.load_state_dict(state_dict)
    head.dense_head.eval()
    return head, config
