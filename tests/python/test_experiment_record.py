"""Provenance stamping and strict detection scoring.

Both exist because an audit found recorded results that could not be
reproduced: one report's numbers were measured at a detector default that had
since changed, with nothing in the artifact saying so, and every headline was
scored on centre distance alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from blockpose.core_types import DetectionParameters
from blockpose.experiment_record import (
    experiment_stamp,
    oriented_box_iou,
    strict_match,
)
from blockpose.pose import BLOCK_DIMS, BlockPose
from scipy.spatial.transform import Rotation


def _pose(position, rotation=None, dims=BLOCK_DIMS) -> BlockPose:
    return BlockPose(
        position=np.asarray(position, dtype=float),
        rotation=np.eye(3) if rotation is None else rotation,
        dims=dims,
    )


def test_stamp_records_the_commit_and_expands_dataclass_parameters() -> None:
    """A number is only meaningful next to the parameters that produced it."""
    stamp = experiment_stamp(detector=DetectionParameters(), match_dist=0.30)
    assert stamp["git_commit"]
    assert stamp["parameters"]["match_dist"] == 0.30
    # Expanded, not stringified: a reader must be able to diff a stale result
    # against current defaults rather than guess which were in force.
    assert stamp["parameters"]["detector"]["cluster_max_extent_xy"] == pytest.approx(2.25)


def test_iou_is_one_for_identical_boxes_and_zero_for_disjoint_ones() -> None:
    assert oriented_box_iou(
        np.zeros(3), np.eye(3), BLOCK_DIMS, np.zeros(3), np.eye(3), BLOCK_DIMS
    ) == pytest.approx(1.0, abs=0.02)
    assert oriented_box_iou(
        np.zeros(3), np.eye(3), BLOCK_DIMS, np.array([5.0, 0, 0]), np.eye(3), BLOCK_DIMS
    ) == pytest.approx(0.0, abs=1e-9)


def test_strict_scoring_rejects_a_near_centre_match_with_wrong_orientation() -> None:
    """The exact failure centre-only scoring hides.

    Measured over the annotated set, two "detections" sat within 0.30 m of a GT
    block while being ~87 degrees out. Centre-only counted them; they are not
    the same block in any useful sense.
    """
    truth = _pose([0.0, 0.0, 0.0])
    # 87 degrees about Y: not a symmetry of a 0.9 x 0.6 x 0.6 block, so it
    # cannot be explained away as a face relabelling.
    wrong = _pose([0.25, 0.0, 0.0], Rotation.from_euler("y", 87.0, degrees=True).as_matrix())
    result = strict_match(wrong, truth)
    assert result["centre_only_match"] is True
    assert result["strict_match"] is False
    assert result["rotation_deg"] > 30.0


def test_strict_scoring_accepts_a_genuine_fit() -> None:
    """Real recoveries measured at 0.006-0.035 m and 1.2-2.9 degrees must pass."""
    truth = _pose([1.0, 2.0, 0.3])
    good = _pose([1.02, 2.01, 0.3], Rotation.from_euler("z", 2.0, degrees=True).as_matrix())
    result = strict_match(good, truth)
    assert result["strict_match"] is True
    assert result["iou"] > 0.8


def test_strict_scoring_is_symmetry_aware() -> None:
    """Relabelling a symmetric block's faces is not an orientation error.

    The 0.6 x 0.6 square cross-section means a 90 degree roll about the long
    axis describes the same physical placement.
    """
    truth = _pose([0.0, 0.0, 0.0])
    relabelled = _pose([0.0, 0.0, 0.0], Rotation.from_euler("x", 90.0, degrees=True).as_matrix())
    result = strict_match(relabelled, truth)
    assert result["rotation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert result["strict_match"] is True


def test_iou_is_deterministic_for_a_fixed_seed() -> None:
    """A recorded result must stay reproducible despite Monte-Carlo sampling."""
    args = (np.zeros(3), np.eye(3), BLOCK_DIMS, np.array([0.1, 0.0, 0.0]), np.eye(3), BLOCK_DIMS)
    assert oriented_box_iou(*args) == oriented_box_iou(*args)
