"""Candidate-conditioned RGB evidence from a frozen DINOv2 backbone.

This mirrors ``rgb_edge_prior.py``'s contract with a learned feature source
instead of a classical edge map: it scores whether a fitted 3-D candidate's
projected image evidence (an RGB crop plus a rendered conditioning image, see
``experiments.rgb_candidate_prior.render.render_conditioning``) agrees with
that candidate, so callers may use it only to re-rank otherwise
geometrically valid hypotheses within a geometry score band -- never to
override geometry.  It never detects a block or infers a metric pose, and it
must degrade to a safe abstention (no evidence, no penalty) whenever the
model, a crop, or a conditioning image is unavailable or malformed.

Only the small MLP head is trainable; the DINOv2 backbone is always frozen.
Torch/transformers are imported lazily inside methods/functions, matching
``rgb_annotation.TransformersSam2Backend``, so importing ``blockpose`` (or
this module) never requires torch.
"""

from __future__ import annotations

import math
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

# Fixed pooling order; also the on-disk/feature-vector concatenation order.
_REGION_NAMES: tuple[str, ...] = ("interior", "edge_band", "exterior", "gripper_occlusion")
_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DinoVerifierConfig:
    """Frozen head configuration; safe to serialize as-is (see `save_checkpoint`)."""

    model_id: str = "facebook/dinov2-small"
    edge_band_px: int = 6
    hidden_dim: int = 128
    max_score_contribution: float = 0.20
    # Trained via the heteroscedastic term in `ranking_margin_loss`
    # (experiments/rgb_candidate_prior/model.py) -- exp(log_uncertainty) above
    # this threshold abstains. Untrained if a checkpoint's training script
    # never passed `log_uncertainties` into that loss.
    uncertainty_abstain_threshold: float = 1.5


def region_masks(conditioning: np.ndarray, edge_band_px: int) -> dict[str, np.ndarray]:
    """Derive interior/edge_band/exterior/gripper_occlusion boolean masks.

    Only the conditioning image's R (visible-edge wireframe) and G (filled
    silhouette) channels are used, per ``render_conditioning``'s contract:

    - interior = G > 0
    - edge_band = dilate(R > 0, edge_band_px)
    - exterior = ~interior & ~edge_band
    - gripper_occlusion: ``render_conditioning`` never populates the B
      channel today -- its own docstring calls B "reserved for a future
      depth/occlusion channel" -- so this region is always empty and pools
      to a zero vector until that channel is extended.  This is a known,
      deliberate gap, not a heuristic guess.
    """
    image = np.asarray(conditioning)
    if image.ndim != 3 or image.shape[2] < 2:
        raise ValueError("conditioning must be an (H, W, >=2) array with R=edges, G=silhouette")
    if edge_band_px < 0:
        raise ValueError("edge_band_px must be non-negative")
    edges = image[..., 0] > 0
    interior = image[..., 1] > 0
    edge_band = (
        ndimage.binary_dilation(edges, iterations=edge_band_px) if edge_band_px else edges.copy()
    )
    exterior = ~interior & ~edge_band
    gripper_occlusion = np.zeros_like(interior, dtype=bool)
    return {
        "interior": interior,
        "edge_band": edge_band,
        "exterior": exterior,
        "gripper_occlusion": gripper_occlusion,
    }


def _grid_weights(mask: np.ndarray, grid_size: int) -> np.ndarray:
    """Fractional region occupancy resampled to the patch-token grid resolution."""
    if mask.shape[0] == grid_size and mask.shape[1] == grid_size:
        return mask.astype(np.float64)
    zoom = (grid_size / mask.shape[0], grid_size / mask.shape[1])
    return np.clip(ndimage.zoom(mask.astype(np.float64), zoom, order=1), 0.0, 1.0)


def pool_patch_tokens(patch_tokens: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Weighted-average ``(grid, grid, D)`` patch tokens under each region mask.

    Aligns each boolean region mask to the patch-token grid by fractional
    occupancy (so pixel and patch grids need not match resolution), then
    concatenates the four region vectors in `_REGION_NAMES` order. A region
    with zero total weight (e.g. `gripper_occlusion` today) pools to a zero
    vector rather than raising or renormalizing.
    """
    if patch_tokens.ndim != 3 or patch_tokens.shape[0] != patch_tokens.shape[1]:
        raise ValueError("patch_tokens must be a square (grid, grid, D) array")
    grid_size = patch_tokens.shape[0]
    pooled = []
    for name in _REGION_NAMES:
        weights = _grid_weights(masks[name], grid_size)
        total = float(weights.sum())
        if total <= 1e-8:
            pooled.append(np.zeros(patch_tokens.shape[-1], dtype=np.float64))
        else:
            pooled.append(np.tensordot(weights, patch_tokens, axes=([0, 1], [0, 1])) / total)
    return np.concatenate(pooled).astype(np.float64)


class DinoVerifierHead:
    """Frozen DINOv2 patch-token backbone + a small trainable scoring MLP.

    Only the MLP head (`.mlp`) is trainable; the backbone is always frozen
    (`requires_grad_(False)`, `.eval()`).  Nothing is loaded until the first
    call that needs it (`embed_regions`/`forward`, or an explicit `_load()`),
    so constructing a head never requires torch/transformers or a network
    fetch.
    """

    def __init__(self, config: DinoVerifierConfig, *, device: str = "auto"):
        self.config = config
        self.device = device
        self._processor = None
        self._backbone = None
        self._head = None
        self._torch = None

    @property
    def is_loaded(self) -> bool:
        return self._backbone is not None

    @property
    def mlp(self) -> "torch.nn.Module":
        """The trainable scoring/uncertainty MLP; loads the backbone+head if needed."""
        self._load()
        assert self._head is not None
        return self._head

    def _load(self) -> None:
        if self._backbone is not None:
            return
        torch, self._processor, backbone, self.device = load_frozen_backbone(
            self.config.model_id, self.device, head_name="dino_verifier"
        )
        self._backbone = backbone
        from torch import nn

        hidden_size = backbone.config.hidden_size
        head = nn.Sequential(
            nn.Linear(hidden_size * len(_REGION_NAMES), self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 2),  # [score_logit, log_uncertainty]
        ).to(self.device)
        head.eval()
        self._head = head
        self._torch = torch

    def _patch_grid(self, rgb_crop: np.ndarray) -> np.ndarray:
        """Run the frozen backbone; return the (grid, grid, D) patch-token map."""
        self._load()
        assert self._processor is not None
        assert self._backbone is not None and self._torch is not None
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb_crop, dtype=np.uint8))
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._backbone(**inputs)
        tokens = output.last_hidden_state[0]  # (1 + n_patches [+ registers], D)
        patch_tokens = tokens[1:].detach().cpu().numpy()
        n_patches = patch_tokens.shape[0]
        grid = int(round(n_patches**0.5))
        if grid * grid != n_patches:
            raise RuntimeError(
                f"unexpected non-square patch token count {n_patches} for {self.config.model_id}"
            )
        return patch_tokens.reshape(grid, grid, -1)

    def embed_regions(self, rgb_crop: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
        """Pooled `(4 * hidden_size,)` region feature vector for one candidate crop pair."""
        patch_grid = self._patch_grid(rgb_crop)
        masks = region_masks(conditioning, self.config.edge_band_px)
        return pool_patch_tokens(patch_grid, masks)

    def forward(self, rgb_crop: np.ndarray, conditioning: np.ndarray) -> tuple[float, float]:
        """Return `(score in [0, 1], log_uncertainty)` for one candidate crop pair."""
        self._load()
        assert self._head is not None and self._torch is not None
        features = self.embed_regions(rgb_crop, conditioning)
        tensor = self._torch.from_numpy(features).float().to(self.device)
        with self._torch.inference_mode():
            raw = self._head(tensor)
        score = float(self._torch.sigmoid(raw[0]).item())
        log_uncertainty = float(raw[1].item())
        return score, log_uncertainty


def save_checkpoint(
    path: Path, head: DinoVerifierHead, config: DinoVerifierConfig, calibration_temperature: float
) -> None:
    """Save the trainable MLP head, config, and calibration temperature.

    On-disk format (see `blockpose._head_checkpoint`, shared with
    `rgb_region_proposer.py` -- callers must go through
    `save_checkpoint`/`load_dino_verifier`, never read these files directly):
    a directory with `mlp_head.pt` (trainable MLP state dict only -- the
    frozen backbone is never serialized) and `config.json`.
    """
    import torch

    head._load()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(head.mlp.state_dict(), path / "mlp_head.pt")
    write_checkpoint_config(
        path,
        config,
        _CHECKPOINT_SCHEMA_VERSION,
        calibration_temperature=float(calibration_temperature),
    )


def load_dino_verifier(path: Path) -> tuple[DinoVerifierHead, DinoVerifierConfig, float]:
    """Load a checkpoint written by `save_checkpoint`.

    Raises `FileNotFoundError`/`ValueError` for a missing or unrecognized
    checkpoint rather than silently returning an untrained head.
    """
    import torch

    path = Path(path)
    payload = read_checkpoint_config(
        path,
        weights_filename="mlp_head.pt",
        schema_version=_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_kind="dino_verifier",
    )
    calibration_temperature = float(payload["calibration_temperature"])
    config = config_from_payload(DinoVerifierConfig, payload)
    head = DinoVerifierHead(config)
    head._load()
    state_dict = torch.load(path / "mlp_head.pt", map_location=head.device)
    head.mlp.load_state_dict(state_dict)
    head.mlp.eval()
    return head, config, calibration_temperature


def _calibrate_score(raw_score: float, temperature: float) -> float:
    """Temperature-scaled probability calibration (Guo et al. 2017).

    Deliberately duplicated -- not imported -- from
    ``experiments.rgb_candidate_prior.model.apply_calibration``: that module
    is training-only and imports torch at module scope, while this module
    must stay importable without torch.  Keep the two formulas identical.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    probability = min(max(raw_score, 1e-6), 1.0 - 1e-6)
    logit = math.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


@dataclass(frozen=True)
class DinoVerifierScore:
    """Bounded candidate evidence; safely neutral (abstained) when unusable."""

    contribution: float = 0.0
    raw_score: float = 0.5
    calibrated_score: float = 0.5
    uncertainty: float = float("inf")
    abstained: bool = True
    reason: str = "neutral"


def _abstain(reason: str, **overrides: float) -> DinoVerifierScore:
    return DinoVerifierScore(reason=reason, **overrides)


def score_candidate_with_dino(
    head: DinoVerifierHead | None,
    config: DinoVerifierConfig,
    rgb_crop: np.ndarray | None,
    conditioning: np.ndarray | None,
    *,
    calibration_temperature: float = 1.0,
) -> DinoVerifierScore:
    """Bounded, abstain-capable DINO rerank contribution for one fitted candidate.

    Mirrors `rgb_edge_prior.score_projected_cuboid`'s contract: an
    unavailable model, missing/malformed crop or conditioning image, or high
    uncertainty is safely neutral -- never treated as negative evidence. The
    returned `contribution` is clamped to
    `[-config.max_score_contribution, +config.max_score_contribution]` and is
    derived from the *calibrated* score (`calibration_temperature`-scaled per
    `_calibrate_score`, matching the training side's `apply_calibration`),
    not the head's raw `[0, 1]` output.

    This function only *proposes* a bounded contribution/abstention per
    candidate; it must never by itself let RGB override geometry across a
    score band.  That enforcement is the caller's job, exactly as
    `ablate_rgb_edge_prior.py`'s `_rank` restricts the classical edge prior
    to reordering *within* one geometry score band (see
    `tools/ablate_rgb_verifier.py`'s `_rank`).
    """
    if head is None or rgb_crop is None or conditioning is None:
        return _abstain("missing_dino_or_inputs")
    if config.max_score_contribution <= 0.0 or calibration_temperature <= 0.0:
        return _abstain("disabled")
    rgb = np.asarray(rgb_crop)
    cond = np.asarray(conditioning)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or cond.ndim != 3 or cond.shape[2] < 2:
        return _abstain("incompatible_crop_or_conditioning")
    if rgb.shape[:2] != cond.shape[:2]:
        return _abstain("incompatible_crop_or_conditioning")
    if not np.all(np.isfinite(rgb)) or not np.all(np.isfinite(cond)):
        return _abstain("non_finite_input")
    try:
        raw_score, log_uncertainty = head.forward(rgb, cond)
    except RuntimeError:
        # Missing torch/transformers, or a malformed patch grid: RGB absence
        # is not negative evidence, so fall back to geometry-only safely.
        return _abstain("dino_unavailable")
    uncertainty = float(np.exp(log_uncertainty))
    if not np.isfinite(raw_score) or not np.isfinite(uncertainty):
        return _abstain("non_finite_output")
    calibrated_score = _calibrate_score(raw_score, calibration_temperature)
    if uncertainty > config.uncertainty_abstain_threshold:
        return _abstain(
            "uncertain",
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            uncertainty=uncertainty,
        )
    signed = np.clip((calibrated_score - 0.5) * 2.0, -1.0, 1.0)
    contribution = float(signed * config.max_score_contribution)
    return DinoVerifierScore(
        contribution=contribution,
        raw_score=raw_score,
        calibrated_score=calibrated_score,
        uncertainty=uncertainty,
        abstained=False,
        reason="ok",
    )
