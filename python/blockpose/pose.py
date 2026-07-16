"""Pose representation and SO(3) utilities for blockpose."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BlockPose:
    position: np.ndarray  # (3,) float64, world frame [x, y, z]
    rotation: np.ndarray  # (3,3) float64 rotation matrix R_world_block
    confidence: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q).as_matrix()


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    return Rotation.from_matrix(R).as_quat()


def symmetry_equivalents(R: np.ndarray) -> list[np.ndarray]:
    """Return all 4 rotation matrices equivalent under 4-fold symmetry about block Z axis.

    The concrete block (0.6x0.6x0.9 m) has a square cross-section, so rotations of
    0, 90, 180, 270 degrees about the block's local Z axis are physically identical.
    """
    R = np.asarray(R, dtype=np.float64)
    equivalents = []
    for k in range(4):
        angle = k * np.pi / 2.0
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        equivalents.append(R @ Rz)
    return equivalents
