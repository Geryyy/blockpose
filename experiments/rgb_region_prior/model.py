"""Training-only pieces for the ``dino_proposer`` head: patch-grid target + loss.

Architecture lives in ``blockpose.rgb_region_proposer`` (frozen DINOv2
backbone + trainable dense per-patch head); this module never redefines it.
It only supplies what ``train.py``/``report.py`` need on top: turning a
full-resolution binary union mask into a soft patch-grid training target, and
a class-imbalance-aware loss for it.

Torch is imported at module scope: like
``experiments.rgb_candidate_prior.model``, this is a GPU-training-only
module, never imported by the runtime/inference path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def resample_mask_to_patch_grid(mask: np.ndarray, grid_h: int, grid_w: int) -> torch.Tensor:
    """Fractional per-patch foreground occupancy in ``[0, 1]``, shape ``(grid_h, grid_w)``.

    Uses true area-weighted average pooling (``adaptive_avg_pool2d``), not a
    nearest/bilinear resize: a patch straddling a block silhouette's edge
    gets an honest soft target (e.g. 0.3 if 30% of its pixels are
    foreground) instead of a lossy 0/1 snap from the nearest pixel. This is
    deliberately different from ``blockpose.rgb_verifier``'s ``_grid_weights``
    (bilinear ``ndimage.zoom``), which resamples an already-continuous
    pooling *weight*, not a binary ground-truth mask that needs an honest
    occupancy fraction as a training target.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if grid_h < 1 or grid_w < 1:
        raise ValueError("grid_h and grid_w must be positive")
    tensor = torch.from_numpy(array.astype(np.float32))[None, None]
    pooled = F.adaptive_avg_pool2d(tensor, (grid_h, grid_w))
    return pooled[0, 0]


def foreground_pos_weight(targets: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Positive-class weight derived from this batch's own foreground mass fraction.

    ``negative_mass / positive_mass`` under the soft occupancy targets (not a
    fixed hyperparameter), so a batch of mostly-empty snapshots gets a larger
    weight than a batch full of close-up blocks. Falls back to a neutral
    weight of ``1.0`` when the batch has (numerically) no foreground mass at
    all, since the ratio would otherwise blow up for no benefit -- a target of
    exactly zero everywhere already contributes nothing to the positive term
    regardless of the weight multiplying it. Also capped at ``targets.numel()``
    (the weight can never need to exceed "as if there were only one positive
    patch"), so a real but tiny sliver of foreground -- e.g. a block clipped
    at the frame edge -- can't produce an unboundedly large loss spike.
    """
    positive_mass = targets.sum()
    negative_mass = targets.numel() - positive_mass
    if float(positive_mass) <= eps:
        return torch.ones((), dtype=targets.dtype, device=targets.device)
    return (negative_mass / positive_mass).clamp(min=eps, max=float(targets.numel()))


def weighted_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Foreground-weighted BCE for the dense per-patch ``dino_proposer`` head.

    Block regions cover a small fraction of a 1536x2048 image, so plain BCE
    would be dominated by background patches and could collapse toward
    predicting all-background (poor recall). ``pos_weight`` comes from
    ``foreground_pos_weight`` (this batch's own statistics), not a fixed
    hyperparameter.

    Focal loss was considered -- it is the more common choice for this shape
    of dense imbalance -- but was not used here: it adds an extra gamma
    hyperparameter to tune, and its main benefit (down-weighting a flood of
    *easy* negatives) is largely covered here by a batch-derived
    ``pos_weight`` without a second knob. Revisit if held-out recall stalls.

    ``targets`` are soft occupancy fractions in ``[0, 1]``
    (``resample_mask_to_patch_grid``'s output), not hard 0/1 labels;
    ``pos_weight`` only multiplies the positive term, which is proportional
    to the target itself, so an all-zero-target batch stays numerically safe
    regardless of how large ``pos_weight`` would otherwise be.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    pos_weight = foreground_pos_weight(targets, eps=eps)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
