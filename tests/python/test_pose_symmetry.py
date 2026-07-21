"""Tests for the cuboid symmetry group used by rotation error.

The group depends on which extents are equal. Getting it wrong is not a cosmetic
issue: using the upright block's group on a block laid on its side reports a
90 degree yaw error as zero.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from blockpose.evaluate import rotation_error_deg
from blockpose.pose import BLOCK_DIMS, BlockPose, _cuboid_symmetry_group, symmetry_equivalents

UPRIGHT = (0.6, 0.6, 0.9)
LAID = (0.9, 0.6, 0.6)
CUBE = (0.6, 0.6, 0.6)
DISTINCT = (0.6, 0.7, 0.9)


def _rotation(axis: int, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    matrices = {
        0: [[1, 0, 0], [0, cos, -sin], [0, sin, cos]],
        1: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]],
        2: [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]],
    }
    return np.array(matrices[axis], dtype=np.float64)


@pytest.mark.parametrize(
    ("dims", "order"),
    [(DISTINCT, 4), (UPRIGHT, 8), (LAID, 8), (CUBE, 24)],
)
def test_group_order_follows_the_extents(dims, order) -> None:
    assert len(_cuboid_symmetry_group(dims)) == order


@pytest.mark.parametrize("dims", [DISTINCT, UPRIGHT, LAID, CUBE])
def test_every_element_is_a_proper_rotation_preserving_the_box(dims) -> None:
    for matrix in _cuboid_symmetry_group(dims):
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-9)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        # A symmetry permutes the axes without changing the extents.
        np.testing.assert_allclose(np.abs(matrix) @ np.asarray(dims), dims, atol=1e-9)


@pytest.mark.parametrize("dims", [DISTINCT, UPRIGHT, LAID, CUBE])
def test_group_is_closed_under_composition(dims) -> None:
    group = _cuboid_symmetry_group(dims)
    for first, second in itertools.product(group, repeat=2):
        product = first @ second
        assert any(np.allclose(product, member, atol=1e-9) for member in group)


def test_upright_block_yaw_quarter_turn_is_free() -> None:
    identity = BlockPose(np.zeros(3), np.eye(3))
    turned = BlockPose(np.zeros(3), _rotation(2, 90.0))
    assert rotation_error_deg(turned, identity, UPRIGHT) == pytest.approx(0.0, abs=1e-9)


def test_laid_block_yaw_quarter_turn_is_a_real_error() -> None:
    """The regression that motivated deriving the group from the extents.

    A 0.9 x 0.6 x 0.6 block turned 90 degrees about world Z does not map onto
    itself. The old fixed 4-fold-about-Z group scored this as zero.
    """
    identity = BlockPose(np.zeros(3), np.eye(3))
    turned = BlockPose(np.zeros(3), _rotation(2, 90.0))
    assert rotation_error_deg(turned, identity, LAID) == pytest.approx(90.0, abs=1e-6)


def test_laid_block_roll_quarter_turn_about_its_long_axis_is_free() -> None:
    """The laid block's square cross-section is about local X, not local Z."""
    identity = BlockPose(np.zeros(3), np.eye(3))
    rolled = BlockPose(np.zeros(3), _rotation(0, 90.0))
    assert rotation_error_deg(rolled, identity, LAID) == pytest.approx(0.0, abs=1e-9)


def test_end_for_end_flip_is_free_for_a_square_prism() -> None:
    """A flipped cuboid occupies the same volume, so it is not a 180 degree error."""
    identity = BlockPose(np.zeros(3), np.eye(3))
    flipped = BlockPose(np.zeros(3), _rotation(0, 180.0))
    assert rotation_error_deg(flipped, identity, UPRIGHT) == pytest.approx(0.0, abs=1e-9)


def test_small_errors_survive_the_larger_group() -> None:
    """Widening the group must not swallow genuine small misalignments."""
    identity = BlockPose(np.zeros(3), np.eye(3))
    tilted = BlockPose(np.zeros(3), _rotation(1, 3.0))
    assert rotation_error_deg(tilted, identity, UPRIGHT) == pytest.approx(3.0, abs=1e-6)


def test_default_dims_are_the_laid_block() -> None:
    assert BLOCK_DIMS == LAID
    assert len(symmetry_equivalents(np.eye(3))) == len(_cuboid_symmetry_group(LAID))


def test_equivalents_are_left_multiplied_onto_the_pose() -> None:
    rotation = _rotation(2, 37.0)
    equivalents = symmetry_equivalents(rotation, UPRIGHT)
    assert any(np.allclose(candidate, rotation) for candidate in equivalents)
    for candidate in equivalents:
        np.testing.assert_allclose(np.linalg.det(candidate), 1.0, atol=1e-9)
