"""Contracts for the optional native classical RGB edge scorer."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _core_with_vision():
    try:
        core = importlib.import_module("blockpose._core")
    except ImportError as exc:
        pytest.skip(f"native blockpose extension is not built: {exc}")
    if not getattr(core, "vision_enabled", False):
        pytest.skip("native extension was built without optional OpenCV vision support")
    return core


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[:, 16:] = 255
    camera = np.array([[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]])
    return image, camera, np.zeros(5), np.eye(4)


def test_native_rgb_edge_prior_scores_and_validates_inputs() -> None:
    core = _core_with_vision()
    image, camera, distortion, transform = _inputs()
    prior = core.RgbEdgePrior(image, camera, distortion, transform)

    score = prior.score(np.array([0.0, 0.0, 3.0]), np.eye(3), np.array([1.0, 1.0, 1.0]))

    assert prior.ready()
    assert set(score) == {"score", "available"}
    assert isinstance(score["score"], float)
    assert isinstance(score["available"], bool)
    with pytest.raises(ValueError, match="camera_matrix"):
        core.RgbEdgePrior(image, np.eye(2), distortion, transform)
    with pytest.raises(ValueError, match="excluded_mask"):
        core.RgbEdgePrior(image, camera, distortion, transform, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="proper rotation"):
        prior.score(np.array([0.0, 0.0, 3.0]), np.zeros((3, 3)), np.ones(3))
