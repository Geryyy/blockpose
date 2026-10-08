"""The reference transforms must hold for every `initial_pose` they claim.

The `world_*` entries of `config/transforms_pzs100_initial_pose_1.yaml` are
forward kinematics of `urdf/pzs100_initial_pose_1.urdf` evaluated at
`reference.dynamic_joint_assumption` -- and at nothing else, because every other
joint between `world` and the sensors is fixed. Issue 235 moved the Gazebo
launches' default from pose 1 (outside crane_model's control-safe box) to pose
2; the two initialization files agree, so one transform set covers both. This is
what notices when a future default does not.

Two deliberate limits. The structural test sees joint *types*, so it cannot see
`truck_pitch_clearance`, which is baked into the `truck_pitch` origin -- that one
is caught by the value comparison instead, which is why it has to be in
`dynamic_joint_assumption`. And the initialization yamls read here are
blockpose's vendored copies under `config/initialization/`, not the
`epsilon_crane_description` originals the xacro loads; they are byte-identical
today and this repo is deliberately standalone (no colcon, no ROS), so drift
between the two copies is not detected here. `concrete_block_behavior_tree`'s
`test_start_pose_is_control_safe` goes through the installed share and does see
the originals.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
TRANSFORM_CONFIG = REPO_ROOT / "config" / "transforms_pzs100_initial_pose_1.yaml"
URDF = REPO_ROOT / "urdf" / "pzs100_initial_pose_1.urdf"
INITIALIZATION_DIR = REPO_ROOT / "config" / "initialization"

SENSOR_LINKS = ("sensor_konsole", "seyond", "blackfly_rotated")


def _reference() -> dict:
    return yaml.safe_load(TRANSFORM_CONFIG.read_text(encoding="utf-8"))["reference"]


def test_only_the_assumed_joints_move_between_world_and_the_sensors() -> None:
    """Any other movable joint on the chain would make the set pose-dependent."""
    root = ET.parse(URDF).getroot()
    parent_joint = {
        joint.find("child").get("link"): (
            joint.find("parent").get("link"),
            joint.get("name"),
            joint.get("type"),
        )
        for joint in root.findall("joint")
    }

    movable = set()
    for link in SENSOR_LINKS:
        child = link
        while child in parent_joint:
            parent, name, joint_type = parent_joint[child]
            if joint_type != "fixed":
                movable.add(name)
            child = parent
        assert child == "world", f"{link} does not hang off `world` but off {child}"

    assumed = _reference()["dynamic_joint_assumption"]
    # The config keys the assumption by the initialization yamls' names, not the
    # URDF joint names: theta1_0 is theta1_slewing_joint's initial value, and
    # truck_pitch_clearance is truck_pitch's origin rather than a joint at all.
    assert movable == {"truck_pitch", "truck_roll", "theta1_slewing_joint"}
    assert set(assumed) == {
        "truck_pitch",
        "truck_roll",
        "truck_pitch_clearance",
        "theta1_0",
    }


def test_every_claimed_initial_pose_matches_the_assumption() -> None:
    reference = _reference()
    assumed = reference["dynamic_joint_assumption"]
    applies_to = reference["applies_to"]

    assert reference["default_initial_pose_id"] in applies_to
    assert applies_to[reference["default_initial_pose_id"]] == reference["initial_pose_file"]

    for pose_id, filename in applies_to.items():
        start = yaml.safe_load((INITIALIZATION_DIR / filename).read_text(encoding="utf-8"))
        for key, value in assumed.items():
            assert start[key] == value, (
                f"initial_pose:={pose_id} ({filename}) has {key} = {start[key]}, not "
                f"{value}, so the world_* transforms in {TRANSFORM_CONFIG.name} do not "
                "apply to it -- regenerate them for that pose."
            )
