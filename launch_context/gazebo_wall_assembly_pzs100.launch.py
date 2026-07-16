"""PZS100 Gazebo simulation with CBS wall assembly behavior tree.

Launches the full PZS100 crane simulation (without the epsilon_crane BT)
and adds the CBS wall assembly pipeline: wall_plan_server, world_model,
and a bt_action_server with CBS + epsilon_crane plugins.
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PathSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Plugin config ────────────────────────────────────────────────────
    # epsilon_crane default.yaml provides base settings (loop duration, lifecycle
    # manager, logging). bt_server_override.yaml overrides plugin_lib_names to
    # include both epsilon_crane and CBS plugins.
    base_bt_config = (
        PathSubstitution(FindPackageShare("epsilon_crane_behavior_tree"))
        / "config"
        / "default.yaml"
    )
    override_bt_config = (
        PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
        / "config"
        / "bt_server_override.yaml"
    )
    seed_file = LaunchConfiguration("seed_file")

    # Keyboard TUI
    if os.path.exists("/usr/bin/xterm"):
        spawn_terminal_prefix = "xterm -e "
    elif os.path.exists("/usr/bin/gnome-terminal"):
        spawn_terminal_prefix = "gnome-terminal -- "
    else:
        spawn_terminal_prefix = ""

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="True"),
            DeclareLaunchArgument(
                "gazebo_world_file",
                default_value="epsilon_crane.world",
                description=(
                    "World file from testsite_description/worlds. Use empty.world "
                    "for a clean sim without the Seibersdorf container."
                ),
            ),
            DeclareLaunchArgument(
                "initial_pose",
                default_value="1",
                description="Initial crane pose (see epsilon_crane_description)",
            ),
            DeclareLaunchArgument(
                "controller",
                default_value="pid",
                description="Low-level A2B controller: 'pid' or 'mpc'",
            ),
            DeclareLaunchArgument(
                "place_approach_angle_deg",
                default_value="4.0",
                description="Pre-place lateral approach angle in degrees.",
            ),
            DeclareLaunchArgument(
                "lift_height",
                default_value="1.0",
                description="Vertical post-grasp/post-place lift height for CalcGripMovement.",
            ),
            DeclareLaunchArgument(
                "seed_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("concrete_block_world_model"),
                        "config",
                        "world_model_seed_pick_place.yaml",
                    ]
                ),
                description="World-model/Gazebo seed YAML with world_model.initial_blocks.",
            ),
            # Point the RViz BT panel to concrete_block_behavior_tree
            SetEnvironmentVariable(
                name="BEHAVIOR_TREE_PANEL_BT_PACKAGE",
                value="concrete_block_behavior_tree",
            ),
            SetEnvironmentVariable(
                name="BEHAVIOR_TREE_PANEL_BT_CATALOG",
                value=PathJoinSubstitution(
                    [
                        FindPackageShare("concrete_block_behavior_tree"),
                        "config",
                        "bt_panel_catalog.yaml",
                    ]
                ),
            ),
            # ── PZS100 crane simulation (without epsilon_crane BT) ───────
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
                / "launch"
                / "pzs100_bringup.launch.py",
                launch_arguments={
                    "start_bt_action_server": "False",
                    "start_grip_traj_server": "False",
                    "controller": LaunchConfiguration("controller"),
                    "initial_pose": LaunchConfiguration("initial_pose"),
                    "seed_file": seed_file,
                    "gui": LaunchConfiguration("gui"),
                    "gazebo_world_file": LaunchConfiguration("gazebo_world_file"),
                }.items(),
            ),
            # World model is launched by gazebo_model_bt_pzs100.launch.py.
            # Virtual TCP TF is published by grip_traj_server from tcp_z_offset param.
            # ── Simple grip trajectory server ────────────────────────────
            Node(
                package="concrete_block_motion_planning",
                executable="grip_traj_server_simple.py",
                name="grip_traj_server",
                output="screen",
                parameters=[
                    PathSubstitution(FindPackageShare("concrete_block_motion_planning"))
                    / "config"
                    / "grip_traj_simple.yaml",
                    {
                        "lift_height": ParameterValue(
                            LaunchConfiguration("lift_height"),
                            value_type=float,
                        ),
                    },
                    {"use_sim_time": True},
                ],
            ),
            # ── Wall plan server (assembly planning layer) ───────────────
            Node(
                package="concrete_block_assembly_planning",
                executable="wall_plan_server",
                name="concrete_block_wall_plan_server",
                output="screen",
                parameters=[
                    PathSubstitution(FindPackageShare("concrete_block_assembly_planning"))
                    / "config"
                    / "wall_plan_server.yaml",
                    {
                        "use_sim_time": True,
                        "world_model_service": "/world_model_node/get_coarse_blocks",
                        "world_model_timeout_s": 2.0,
                        "output_frame": "world",
                        "place_approach_angle_deg": ParameterValue(
                            LaunchConfiguration("place_approach_angle_deg"),
                            value_type=float,
                        ),
                        "wall_plans_file": PathSubstitution(
                            FindPackageShare("concrete_block_assembly_planning"))
                        / "config" / "wall_plans.yaml",
                    },
                ],
            ),
            # ── BT action server ─────────────────────────────────────────
            # Loads base config from epsilon_crane, then overrides plugins
            # to include CBS plugins. Same lifecycle pattern as timber framework.
            Node(
                package="lsrl_behavior_tree",
                executable="bt_action_server",
                output="both",
                parameters=[
                    base_bt_config,
                    override_bt_config,
                    {"use_sim_time": True},
                    {
                        "behaviortree": PathSubstitution(
                            FindPackageShare("lsrl_behavior_tree")
                        )
                        / "behavior_trees"
                        / "empty.xml"
                    },
                ],
            ),
            # ── Lifecycle manager ────────────────────────────────────────
            # Reads lifecycle_manager params (node_names, autostart, …) from
            # bt_server_override.yaml (CBS-owned, since upstream removed them
            # from epsilon_crane_behavior_tree/default.yaml).
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                output="screen",
                parameters=[
                    base_bt_config,
                    override_bt_config,
                    {"use_sim_time": True},
                ],
            ),
            # ── Gazebo block spawner ─────────────────────────────────────
            TimerAction(
                period=8.0,
                actions=[
                    Node(
                        package="concrete_block_behavior_tree",
                        executable="gazebo_block_spawner.py",
                        name="gazebo_block_spawner",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": True,
                                "seed_config_file": seed_file,
                                "gazebo_world_frame": "world",
                                "use_precomputed_gazebo_pose": False,
                                # Seed is in `world` (same frame as wall_spec / plan).
                                "seed_frame_id": "world",
                                # Pose of the ROS `world` frame in the Gazebo world
                                # frame. gazebo_base.launch.py spawns the crane at
                                # y=-6, yaw=pi, so the two worlds differ by yaw 180 +
                                # a -6 m y shift. (Reproduces the K0 calibration:
                                # K0-in-gazebo = world->gazebo applied to T_world_K0.)
                                "gazebo_seed_frame_xyz": [0.0, -6.0, 0.0],
                                "gazebo_seed_frame_rpy_deg": [0.0, 0.0, 180.0],
                                "spawn_height_offset": 0.15,
                                "sync_world_model_from_gazebo": False,
                                "settle_time_sec": 3.0,
                                "service_wait_timeout_sec": 60.0,
                                "gazebo_get_entity_state_service": "/gazebo/get_entity_state",
                            },
                        ],
                    ),
                ],
            ),
            # ── Keyboard TUI ────────────────────────────────────────────
            Node(
                package="timber_crane_tui",
                executable="remote_ctrl",
                output="screen",
                parameters=[{"sample_time_ms": 100}],
                prefix=spawn_terminal_prefix,
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("gui"),
                            "'.lower() in ('true', '1', 'yes')",
                        ]
                    )
                ),
            ),
        ]
    )
