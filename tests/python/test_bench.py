"""tools/bench.py associates like the world model's selectRefineMatch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("blockpose._core")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from bench import select_refine_match  # noqa: E402


def _pose(x: float, yaw_deg: float, confidence: float = 1.0) -> dict:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return {"position": (x, 0.0, 0.3), "rotation": ((c, -s, 0), (s, c, 0), (0, 0, 1)),
            "confidence": confidence}


def test_nearest_inside_the_window_with_yaw_modulo_pi() -> None:
    prior = np.array([0.0, 0.0, 0.3])
    poses = [_pose(0.30, 0.0), _pose(0.20, 179.0), _pose(0.05, 90.0), _pose(0.01, 0.0, 0.1)]
    assert select_refine_match(poses, prior, 0.0) == 1  # yaw 179 = 0 mod pi; 90 off, 0.1 too weak
    assert select_refine_match([_pose(0.36, 0.0)], prior, 0.0) == -1
