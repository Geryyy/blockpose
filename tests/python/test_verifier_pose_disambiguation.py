"""Fast, GPU-free tests for the within-block pose-disambiguation tool's pure logic.

Covers hypothesis-set generation and symmetry de-duplication (`is_symmetry_duplicate`,
`generate_pose_hypotheses`) -- no model download, no GPU, no dataset. `tools/` has
no `__init__.py`; it is importable as an implicit namespace package once the repo
root is on `sys.path` (mirrors `tests/python/test_dino_proposer_pooled.py`).
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blockpose.pose import BLOCK_DIMS, BLOCK_DIMS_UPRIGHT, BlockPose
from scipy.spatial.transform import Rotation

from tools.eval_verifier_pose_disambiguation import (
    _rotation_about_axis,
    body_up_axis,
    generate_pose_hypotheses,
    is_symmetry_duplicate,
)


def _pose(position=(0.0, 0.0, 0.0), rotation=None, dims=BLOCK_DIMS) -> BlockPose:
    return BlockPose(
        np.asarray(position, dtype=np.float64),
        np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64),
        dims=dims,
    )


# --------------------------------------------------------------------------
# is_symmetry_duplicate
# --------------------------------------------------------------------------


def test_identity_is_its_own_symmetry_duplicate() -> None:
    assert is_symmetry_duplicate(np.eye(3), BLOCK_DIMS, [np.eye(3)])


def test_180_about_vertical_is_always_a_symmetry_of_a_rectangular_box() -> None:
    """180 deg about a resting block's own up axis flips it end-for-end onto
    itself for ANY rectangular cuboid, not only ones with two equal extents --
    the Klein-four subgroup every box's symmetry group contains."""
    r180 = _rotation_about_axis([0.0, 0.0, 1.0], 180.0)
    assert is_symmetry_duplicate(r180, BLOCK_DIMS, [np.eye(3)])
    # Also true for a box with three DISTINCT extents (no two equal).
    assert is_symmetry_duplicate(r180, (0.9, 0.7, 0.5), [np.eye(3)])


def test_90_about_vertical_is_not_a_symmetry_of_the_laid_block() -> None:
    """0.9x0.6x0.6 laid flat: a 90 deg yaw swaps the long (0.9) axis onto a
    short (0.6) one, which is not a shape-preserving symmetry."""
    r90 = _rotation_about_axis([0.0, 0.0, 1.0], 90.0)
    assert not is_symmetry_duplicate(r90, BLOCK_DIMS, [np.eye(3)])


def test_270_is_a_duplicate_of_a_kept_90_hypothesis() -> None:
    """270 deg = 180 deg (always a symmetry) composed with 90 deg about the
    same axis, so once 90 deg is kept, 270 deg must be recognised as the same
    physical placement, not a third distinct hypothesis."""
    r90 = _rotation_about_axis([0.0, 0.0, 1.0], 90.0)
    r270 = _rotation_about_axis([0.0, 0.0, 1.0], 270.0)
    assert is_symmetry_duplicate(r270, BLOCK_DIMS, [np.eye(3), r90])


def test_symmetry_duplicate_respects_tolerance() -> None:
    r180 = _rotation_about_axis([0.0, 0.0, 1.0], 180.0)
    slightly_off = _rotation_about_axis([0.0, 0.0, 1.0], 178.0) @ np.eye(3)
    assert is_symmetry_duplicate(r180, BLOCK_DIMS, [np.eye(3)], atol_deg=1.0)
    assert not is_symmetry_duplicate(slightly_off, BLOCK_DIMS, [np.eye(3)], atol_deg=1.0)
    assert is_symmetry_duplicate(slightly_off, BLOCK_DIMS, [np.eye(3)], atol_deg=3.0)


# --------------------------------------------------------------------------
# generate_pose_hypotheses
# --------------------------------------------------------------------------


def test_generate_pose_hypotheses_always_includes_base_first() -> None:
    hyps = generate_pose_hypotheses(_pose())
    assert hyps[0].id == "base"
    assert hyps[0].yaw_deg == 0.0
    assert np.allclose(hyps[0].pose.rotation, np.eye(3))


def test_generate_pose_hypotheses_laid_block_yields_base_and_yaw90_only() -> None:
    """For the canonical 0.9x0.6x0.6 laid block, 180 dupes base and 270 dupes
    yaw90 (see the two symmetry tests above), so exactly 2 hypotheses survive
    out of the 3 requested yaw angles."""
    hyps = generate_pose_hypotheses(_pose(), yaw_angles_deg=(90.0, 180.0, 270.0))
    ids = [h.id for h in hyps]
    assert ids == ["base", "yaw90"]


def test_generate_pose_hypotheses_position_unchanged_across_hypotheses() -> None:
    """Rotating about an axis through the candidate's own centre must not move it."""
    position = np.array([1.5, -2.0, 0.3])
    hyps = generate_pose_hypotheses(_pose(position=position))
    for h in hyps:
        assert np.allclose(h.pose.position, position)


def test_generate_pose_hypotheses_dims_preserved() -> None:
    hyps = generate_pose_hypotheses(_pose(dims=BLOCK_DIMS))
    for h in hyps:
        assert h.pose.dims == BLOCK_DIMS


def test_generate_pose_hypotheses_works_from_an_arbitrary_starting_yaw() -> None:
    """The dedup logic must hold regardless of the candidate's own starting
    orientation, not only when it happens to be axis-aligned."""
    rotation = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
    hyps = generate_pose_hypotheses(_pose(rotation=rotation), yaw_angles_deg=(90.0, 180.0, 270.0))
    assert [h.id for h in hyps] == ["base", "yaw90"]


def test_generate_pose_hypotheses_cube_drops_every_yaw_as_symmetric() -> None:
    """A cube (all extents equal) has full O-group symmetry: every yaw about
    its own vertical axis maps it onto itself, so nothing beyond `base`
    should ever be a distinguishable hypothesis."""
    hyps = generate_pose_hypotheses(
        _pose(dims=(0.6, 0.6, 0.6)), yaw_angles_deg=(90.0, 180.0, 270.0)
    )
    assert [h.id for h in hyps] == ["base"]


def test_generate_pose_hypotheses_upright_block_drops_every_yaw_as_symmetric() -> None:
    """0.6x0.6x0.9 upright, resting on its square (0.6x0.6) end: rotating
    about the vertical (local z, the 0.9 axis) swaps the two HORIZONTAL
    extents (local x, y) -- which are equal here (0.6, 0.6) -- so every yaw
    about that axis is a symmetry, unlike the laid orientation where the
    horizontal extents (0.9, 0.6) differ. This is the mirror case of
    `test_generate_pose_hypotheses_laid_block_yields_base_and_yaw90_only`:
    which extents are equal, not just how many are, decides whether yaw is
    distinguishable."""
    hyps = generate_pose_hypotheses(
        _pose(dims=BLOCK_DIMS_UPRIGHT), yaw_angles_deg=(90.0, 180.0, 270.0)
    )
    assert [h.id for h in hyps] == ["base"]


def test_generate_pose_hypotheses_empty_yaw_list_yields_only_base() -> None:
    hyps = generate_pose_hypotheses(_pose(), yaw_angles_deg=())
    assert [h.id for h in hyps] == ["base"]


# --------------------------------------------------------------------------
# Tilted candidates -- the case that actually occurs in the data
# --------------------------------------------------------------------------


def test_body_up_axis_picks_the_body_axis_nearest_world_up() -> None:
    """A laid block tilted off level still yaws about its OWN vertical, not
    world +Z. Picking the wrong axis is not a rounding error: it makes the
    180 deg symmetry inexact and lets a duplicate pose through."""
    tilt = Rotation.from_euler("x", 8.0, degrees=True).as_matrix()
    assert np.allclose(body_up_axis(tilt), tilt[:, 2])
    # A block resting on its side has its long axis vertical; `body_up_axis`
    # must follow the geometry, not assume column 2.
    on_side = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()
    assert np.allclose(np.abs(body_up_axis(on_side)), np.abs(on_side[:, 0]))
    assert body_up_axis(on_side)[2] > 0.0  # always signed upwards


def test_dedup_survives_tilt_that_defeats_a_world_z_yaw() -> None:
    """Regression test for the defect that world-Z yawing hid: the accepted
    candidates on this dataset have a median tilt of 1.70 deg from vertical
    (only 5% under 0.5 deg), and 180 deg about world +Z misses the block's
    true symmetry by roughly twice that -- so a 1 deg dedup tolerance kept
    both duplicates and turned the intended base-vs-yaw90 argmax into a
    best-of-two-per-side one. Yawing about the body axis makes the symmetry
    exact at any tilt."""
    for tilt_deg in (0.5, 1.7, 5.0, 20.0, 35.0):
        rotation = Rotation.from_euler("xz", [tilt_deg, 17.0], degrees=True).as_matrix()
        hyps = generate_pose_hypotheses(
            _pose(rotation=rotation), yaw_angles_deg=(90.0, 180.0, 270.0)
        )
        assert [h.id for h in hyps] == ["base", "yaw90"], f"tilt={tilt_deg}"


def test_tilted_yaw90_hypothesis_keeps_the_block_on_its_own_up_axis() -> None:
    """Yawing about the body axis must not re-tilt the block: the hypothesis
    has to differ from the candidate in yaw ONLY, otherwise it is a tilt
    negative (`tilt_negative_deg`) rather than the yaw negative the verifier
    was trained to discriminate."""
    rotation = Rotation.from_euler("xz", [8.0, 17.0], degrees=True).as_matrix()
    hyps = generate_pose_hypotheses(_pose(rotation=rotation), yaw_angles_deg=(90.0,))
    yaw90 = next(h for h in hyps if h.id == "yaw90")
    assert np.allclose(body_up_axis(yaw90.pose.rotation), body_up_axis(rotation), atol=1e-9)
