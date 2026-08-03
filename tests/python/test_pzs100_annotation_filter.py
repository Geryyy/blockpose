"""PZS100 snapshot-adapter coverage for the annotation preprocessing path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
