"""Shared frozen-DINOv2 backbone loading for the RGB-prior heads.

Both ``DinoVerifierHead`` and ``DinoRegionProposerHead`` are a frozen DINOv2
backbone plus their own small trainable head; only the loading of that
backbone (and its lazy-import/error-message/device-resolution/freeze
sequence) is identical between them.
"""

from __future__ import annotations

from typing import Any


def load_frozen_backbone(
    model_id: str, device: str, *, head_name: str
) -> tuple[Any, Any, Any, str]:
    """Import torch/transformers, load and freeze a DINOv2 backbone+processor.

    Returns ``(torch, processor, backbone, resolved_device)``. Raises
    ``RuntimeError`` (naming ``head_name``) with a `uv sync --extra sam` hint
    if the optional dependencies aren't installed, rather than a raw
    ``ImportError``.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        missing = error.name or "an optional dependency"
        raise RuntimeError(
            f"{head_name} is missing {missing!r}. Install the optional dependencies with "
            "`uv sync --extra sam`, then retry."
        ) from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    backbone = AutoModel.from_pretrained(model_id).to(device).eval()
    backbone.requires_grad_(False)
    return torch, processor, backbone, device
