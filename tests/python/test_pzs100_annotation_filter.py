"""PZS100 snapshot-adapter coverage for the annotation preprocessing path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from blockpose.annotation_preprocess import (
    Pzs100SelfFilterParams,
    filter_pzs100_gripper,
    load_pzs100_self_filter_config,
    pzs100_self_filter_available,
)
from blockpose.io import TFSnapshot

REPO_ROOT = Path(__file__).parents[2]


def test_tf_lookup_accepts_slash_and_underscore_frame_spellings() -> None:
    transform = np.eye(4)
    tf = TFSnapshot(
        reference_frame="world",
        image_frame="camera",
        cloud_frame="cloud",
        transforms={"T_world_elastic_K9": transform},
        transform_frames={"T_world_elastic_K9": ("world", "elastic/K9")},
    )

    assert tf.lookup("world", "elastic/K9") is transform
    assert tf.lookup("world", "elastic_K9") is transform


def test_native_pzs100_filter_uses_profile_and_removes_outboard_points() -> None:
    profile = load_pzs100_self_filter_config(REPO_ROOT / "config/pzs100_gripper_filter.yaml")
    available, reason = pzs100_self_filter_available()
    assert available, reason
    params = Pzs100SelfFilterParams(
        outboard_extent_m=0.30,
        cross_rail_extent_m=profile.parameters.cross_rail_extent_m,
        rail_end_margin_m=0.0,
    )
    points = np.array(
        [
            [0.5, 0.0, 0.20],  # inside: parent -> rail and outward +Z
            [0.5, 0.0, -0.10],  # behind the outward face
            [1.6, 0.0, 0.20],  # beyond the rail endpoint
        ]
    )
    kept, removed, boxes = filter_pzs100_gripper(
        points,
        parent_positions=np.array([[0.0, 0.0, 0.0]]),
        rail_positions=np.array([[1.0, 0.0, 0.0]]),
        parent_rotations=np.array([np.eye(3)]),
        outward_axes_parent=np.array([[0.0, 0.0, 1.0]]),
        params=params,
    )

    assert removed == 1
    np.testing.assert_allclose(kept, points[1:])
    assert boxes["centers"].shape == (1, 3)
    assert boxes["rotations"].shape == (1, 3, 3)
    assert boxes["sizes"].shape == (1, 3)
    assert boxes["rail_starts"].shape == (1, 3)
    assert boxes["rail_ends"].shape == (1, 3)
    np.testing.assert_allclose(boxes["rail_starts"], [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(boxes["rail_ends"], [[1.0, 0.0, 0.0]])


# Two level, parallel rails 2 m apart (parent -> rail along +X, outward along
# +/-Y), matching a symmetric PZS100 rig with the boom directly above the gap
# between them -- shared by the above-gripper exclusion tests below.
_LEVEL_PARENT_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
_LEVEL_RAIL_POSITIONS = np.array([[1.0, 0.0, 0.0], [1.0, -2.0, 0.0]])
_LEVEL_PARENT_ROTATIONS = np.array([np.eye(3), np.eye(3)])
_LEVEL_OUTWARD_AXES = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])


def test_above_gripper_exclusion_removes_boom_but_never_the_payload_below() -> None:
    """Safety-critical: the payload block hangs BELOW the jaws and must never
    be removed by the above-gripper exclusion volume, however far below it
    lies.
    """
    available, reason = pzs100_self_filter_available()
    assert available, reason
    params = Pzs100SelfFilterParams(
        outboard_extent_m=0.3,
        cross_rail_extent_m=0.4,
        rail_end_margin_m=0.0,
        above_exclusion_height_m=1.0,
        above_exclusion_margin_m=0.1,
    )
    points = np.array(
        [
            [0.5, -1.0, 0.5],  # above the gap between the jaws -> removed
            [0.5, -1.0, -0.5],  # BELOW the jaws: the payload -- must survive
            [0.5, -1.0, -50.0],  # far below -- must also survive
        ]
    )
    kept, removed, boxes = filter_pzs100_gripper(
        points,
        parent_positions=_LEVEL_PARENT_POSITIONS,
        rail_positions=_LEVEL_RAIL_POSITIONS,
        parent_rotations=_LEVEL_PARENT_ROTATIONS,
        outward_axes_parent=_LEVEL_OUTWARD_AXES,
        params=params,
    )

    assert removed == 1
    np.testing.assert_allclose(kept, points[1:])
    assert boxes["above_centers"].shape == (1, 3)
    assert boxes["above_rotations"].shape == (1, 3, 3)
    assert boxes["above_sizes"].shape == (1, 3)
    # Top of each jaw box is at z = cross_rail_extent_m / 2 = 0.2; the volume
    # extrudes above_exclusion_height_m (1.0) upward from there -> centred at
    # z = 0.7.  Width spans the 2 m rail-to-rail gap plus the 0.1 m margin.
    np.testing.assert_allclose(boxes["above_centers"], [[0.5, -1.0, 0.7]])
    np.testing.assert_allclose(boxes["above_sizes"], [[1.0, 2.1, 1.0]])


def test_above_gripper_exclusion_is_inert_at_default_zero_height() -> None:
    """Default parameters (height 0.0) must reproduce the pre-existing
    rail-only filtering behaviour exactly -- this is what makes the feature
    backward compatible.
    """
    params = Pzs100SelfFilterParams(
        outboard_extent_m=0.3, cross_rail_extent_m=0.4, rail_end_margin_m=0.0
    )
    assert params.above_exclusion_height_m == 0.0
    assert params.above_exclusion_margin_m == 0.0
    # Far above the jaws -- would be caught by the above-gripper volume if it
    # were active, so this proves the feature is truly off, not just narrow.
    points = np.array([[0.5, -1.0, 500.0]])

    kept, removed, boxes = filter_pzs100_gripper(
        points,
        parent_positions=_LEVEL_PARENT_POSITIONS,
        rail_positions=_LEVEL_RAIL_POSITIONS,
        parent_rotations=_LEVEL_PARENT_ROTATIONS,
        outward_axes_parent=_LEVEL_OUTWARD_AXES,
        params=params,
    )

    assert removed == 0
    np.testing.assert_allclose(kept, points)
    assert boxes["above_centers"].shape == (0, 3)
    assert boxes["above_rotations"].shape == (0, 3, 3)
    assert boxes["above_sizes"].shape == (0, 3)


def test_above_gripper_exclusion_resolves_up_on_a_tilted_gripper() -> None:
    """A tilted gripper must exclude only structure genuinely above it (in
    the rig's own tilted 'up'), and must still never reach the payload
    hanging below the jaws -- verified against the rebuilt native extension,
    not a Python reimplementation of the geometry.
    """
    angle = np.deg2rad(20.0)
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    parent_positions = _LEVEL_PARENT_POSITIONS @ tilt.T
    rail_positions = _LEVEL_RAIL_POSITIONS @ tilt.T
    parent_rotations = np.array([tilt, tilt])

    params = Pzs100SelfFilterParams(
        outboard_extent_m=0.3,
        cross_rail_extent_m=0.4,
        rail_end_margin_m=0.0,
        above_exclusion_height_m=1.0,
        above_exclusion_margin_m=0.1,
    )
    payload_world = tilt @ np.array([0.5, -1.0, -0.5])
    above_world = tilt @ np.array([0.5, -1.0, 0.5])
    points = np.array([above_world, payload_world])

    kept, removed, boxes = filter_pzs100_gripper(
        points,
        parent_positions=parent_positions,
        rail_positions=rail_positions,
        parent_rotations=parent_rotations,
        outward_axes_parent=_LEVEL_OUTWARD_AXES,
        params=params,
    )

    assert removed == 1
    np.testing.assert_allclose(kept, [payload_world])
    expected_up = tilt @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(boxes["above_rotations"][0][:, 2], expected_up, atol=1e-9)


def test_above_gripper_exclusion_rejects_degenerate_geometry() -> None:
    """A cross-rail axis that is exactly horizontal cannot resolve which way
    is 'up'; the native core must raise rather than silently building a
    wrong-direction exclusion volume that could reach the payload.
    """
    params = Pzs100SelfFilterParams(above_exclusion_height_m=1.0)
    with pytest.raises(ValueError):
        filter_pzs100_gripper(
            np.zeros((1, 3)),
            parent_positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            rail_positions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            parent_rotations=np.array([np.eye(3), np.eye(3)]),
            outward_axes_parent=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            params=params,
        )


def test_config_above_gripper_keys_default_for_legacy_schema_version_1_profiles(
    tmp_path: Path,
) -> None:
    """A pre-existing schema_version: 1 profile that predates this feature
    must keep loading unchanged, with the new keys defaulted to 0.0 (off).
    """
    legacy_payload = {
        "schema_version": 1,
        "model": "pzs100",
        "enabled": True,
        "outboard_extent_m": 0.3,
        "cross_rail_extent_m": 0.9,
        "rail_end_margin_m": 0.0,
        "rails": [
            {
                "name": "left",
                "parent_frame": "elastic/K9",
                "rail_frame": "elastic/K10_left_rail",
                "outward_axis_parent": [0.0, 0.0, 1.0],
            },
            {
                "name": "right",
                "parent_frame": "elastic/K11",
                "rail_frame": "elastic/K12_right_rail",
                "outward_axis_parent": [0.0, 0.0, 1.0],
            },
        ],
    }
    path = tmp_path / "legacy_profile.yaml"
    path.write_text(yaml.safe_dump(legacy_payload), encoding="utf-8")

    config = load_pzs100_self_filter_config(path)

    assert config.parameters.above_exclusion_height_m == 0.0
    assert config.parameters.above_exclusion_margin_m == 0.0


def test_checked_in_config_defines_positive_above_gripper_exclusion() -> None:
    """The checked-in production profile should actually configure the new
    exclusion (not leave it at the off-by-default 0.0), since real snapshots
    do have boom/arm returns above the jaws.
    """
    profile = load_pzs100_self_filter_config(REPO_ROOT / "config/pzs100_gripper_filter.yaml")

    assert profile.parameters.above_exclusion_height_m > 0.0
    assert profile.parameters.above_exclusion_margin_m >= 0.0


def _tilted_rig(degrees: float, axis: str = "x") -> tuple[np.ndarray, ...]:
    """Rotate the whole level two-rail rig rigidly about a world axis."""
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = (
        np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
        if axis == "x"
        else np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    )
    return (
        _LEVEL_PARENT_POSITIONS @ rotation.T,
        _LEVEL_RAIL_POSITIONS @ rotation.T,
        np.array([rotation @ item for item in _LEVEL_PARENT_ROTATIONS]),
        _LEVEL_OUTWARD_AXES,
        rotation,
    )


@pytest.mark.parametrize("degrees", [95.0, 135.0, 179.0, 181.0, 270.0])
def test_above_gripper_exclusion_refuses_an_inverted_rig_instead_of_eating_the_payload(
    degrees: float,
) -> None:
    """A rig past the upright limit must disable the filter, never invert it.

    Resolving "up" from the sign of a world +Z dot product is ambiguous: the
    two rails' cross-axes are antiparallel by construction, so an inverted rig
    is indistinguishable from an upright one and the volume would extrude
    straight down through the gripped block. Refusing is the only safe answer
    the available inputs support.
    """
    parents, rails, rotations, outward, rig = _tilted_rig(degrees)
    parameters = Pzs100SelfFilterParams(
        outboard_extent_m=0.3,
        cross_rail_extent_m=0.9,
        rail_end_margin_m=0.0,
        above_exclusion_height_m=1.2,
        above_exclusion_margin_m=0.15,
    )
    payload = (np.array([[0.5, -1.0, -0.8]]) @ rig.T).copy()
    with pytest.raises((ValueError, RuntimeError)):
        filter_pzs100_gripper(payload, parents, rails, rotations, outward, parameters)


@pytest.mark.parametrize("degrees", [0.0, 30.0, 55.0])
def test_above_gripper_exclusion_still_works_within_the_upright_limit(degrees: float) -> None:
    parents, rails, rotations, outward, rig = _tilted_rig(degrees)
    parameters = Pzs100SelfFilterParams(
        outboard_extent_m=0.3,
        cross_rail_extent_m=0.9,
        rail_end_margin_m=0.0,
        above_exclusion_height_m=1.2,
        above_exclusion_margin_m=0.15,
    )
    payload = np.array([[0.5, -1.0, -0.8]]) @ rig.T
    boom = np.array([[0.5, -1.0, 1.2]]) @ rig.T
    kept, removed, _ = filter_pzs100_gripper(
        np.ascontiguousarray(np.vstack([payload, boom])),
        parents,
        rails,
        rotations,
        outward,
        parameters,
    )
    kept = np.asarray(kept)
    assert removed == 1
    assert any(np.allclose(point, payload[0]) for point in kept), "payload must survive"
    assert not any(np.allclose(point, boom[0]) for point in kept), "boom must be removed"
