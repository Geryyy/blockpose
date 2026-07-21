"""Pose representation and SO(3) utilities for blockpose."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

# Canonical block extents (x, y, z) in the block's own frame. Lives here
# rather than in synth so that pose utilities can default to it without
# importing the generator. This is the block laid on its long face -- the
# resting orientation used by every course in config/wall_plans.yaml and the
# one actually observed in data/registration_snapshots/. The two orientations
# have DIFFERENT symmetry groups (see _cuboid_symmetry_group), so pass `dims`
# explicitly whenever a block is not in this default orientation.
BLOCK_DIMS = (0.9, 0.6, 0.6)
# The same physical block standing on end, as in a staging pallet/stack.
# Never assumed -- pass it explicitly wherever a scene models that orientation.
BLOCK_DIMS_UPRIGHT = (0.6, 0.6, 0.9)


@dataclass
class BlockPose:
    position: np.ndarray  # (3,) float64, world frame [x, y, z]
    rotation: np.ndarray  # (3,3) float64 rotation matrix R_world_block
    confidence: float = 0.0
    source: str = ""
    # Extents in the block's own frame. Carried on the pose because a pose alone
    # does not say which way a non-cube block is lying, and the same physical
    # block appears as (0.9, 0.6, 0.6) laid down or (0.6, 0.6, 0.9) standing.
    # Downstream code needs this to pick the right symmetry group and to fit the
    # right model, so it must travel with the pose rather than be assumed.
    dims: tuple[float, float, float] = BLOCK_DIMS

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.dims = tuple(float(value) for value in self.dims)


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q).as_matrix()


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    return Rotation.from_matrix(R).as_quat()


@lru_cache(maxsize=32)
def _signed_permutations() -> tuple[np.ndarray, ...]:
    """The 24 proper rotations of a cube: signed permutations with det +1."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for axis, source in enumerate(permutation):
                matrix[axis, source] = signs[axis]
            if abs(np.linalg.det(matrix) - 1.0) < 1e-9:
                matrices.append(matrix)
    return tuple(matrices)


@lru_cache(maxsize=32)
def frame_alignments(
    dims_from: tuple[float, float, float], dims_to: tuple[float, float, float]
) -> tuple[np.ndarray, ...]:
    """Rotations re-expressing a box described by `dims_from` in `dims_to`'s frame.

    The same physical block can be described with its extents in different axis
    orders -- a laid block is (0.9, 0.6, 0.6) or (0.6, 0.9, 0.6) yawed 90 degrees
    -- so two poses are only comparable once both use the same ordering.
    Comparing rotations directly across orderings reports a spurious 90 degrees.

    Returns every ``S`` with ``rotation_from @ S`` expressed in the `dims_to`
    convention. When the extents match this is exactly the symmetry group, so
    `symmetry_equivalents` is the special case `dims_from == dims_to`.
    """
    source = np.asarray(dims_from, dtype=np.float64)
    target = np.asarray(dims_to, dtype=np.float64)
    return tuple(
        matrix.T
        for matrix in _signed_permutations()
        if np.allclose(np.abs(matrix) @ source, target, atol=1e-9)
    )


@lru_cache(maxsize=8)
def _cuboid_symmetry_group(dims: tuple[float, float, float]) -> tuple[np.ndarray, ...]:
    """Proper rotations mapping a cuboid with these extents onto itself.

    Derived from the extents rather than assumed, because the correct group
    depends on which sides are equal:

    * all three distinct -> D2, order 4 (identity + 180 deg about each axis)
    * exactly two equal  -> D4, order 8 (a square prism)
    * all three equal    -> O,  order 24 (a cube)

    Assuming a fixed 4-fold group about local Z was wrong in both directions for
    a block laid on its side (0.9 x 0.6 x 0.6): it treated a 90 deg yaw -- which
    does *not* map that box onto itself -- as a free symmetry and scored it as
    zero error, while penalising an end-for-end flip that genuinely is a symmetry
    with a full 180 deg.
    """
    extents = np.asarray(dims, dtype=np.float64)
    group = []
    # The 24 proper rotations of a cube are the signed permutation matrices with
    # determinant +1. Keep those that also preserve the extents.
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for axis, source in enumerate(permutation):
                matrix[axis, source] = signs[axis]
            if abs(np.linalg.det(matrix) - 1.0) > 1e-9:
                continue
            # matrix maps local coordinates; the box is invariant when the
            # permuted extents match the originals.
            if np.allclose(extents[list(permutation)], extents):
                group.append(matrix)
    return tuple(group)


def symmetry_equivalents(
    R: np.ndarray, dims: tuple[float, float, float] = BLOCK_DIMS
) -> list[np.ndarray]:
    """Rotations of `R` that describe the same physical block placement.

    `dims` are the block's extents in its own frame and determine the symmetry
    group; see `_cuboid_symmetry_group`. For the default laid 0.9x0.6x0.6
    block this returns 8 rotations (4 about the long axis, each with an
    end-for-end flip), not the 4 an earlier version assumed.
    """
    R = np.asarray(R, dtype=np.float64)
    return [R @ symmetry for symmetry in _cuboid_symmetry_group(tuple(float(d) for d in dims))]
