"""PZS100 Gazebo simulation with the CBS single-block pick-and-place BT."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PathSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
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

    if os.path.exists("/usr/bin/xterm"):
      spawn_terminal_prefix = "xterm -e "
    elif os.path.exists("/usr/bin/gnome-terminal"):
      spawn_terminal_prefix = "gnome-terminal -- "
    else:
      spawn_terminal_prefix = ""

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="True"),
            DeclareLaunchArgument("gazebo_world_file", default_value="epsilon_crane.world"),
            DeclareLaunchArgument("initial_pose", default_value="1"),
            DeclareLaunchArgument("controller", default_value="pid"),
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
            # PZS100 crane bringup -- common PZS arguments are encoded in
            # pzs100_bringup.launch.py (this same package). Only override the
            # things that differ from its defaults.
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
                / "launch"
                / "pzs100_bringup.launch.py",
                launch_arguments={
                    "planner": "ilqr",
                    "controller": LaunchConfiguration("controller"),
                    "start_bt_action_server": "False",
                    "start_grip_traj_server": "False",
                    "initial_pose": LaunchConfiguration("initial_pose"),
                    "gui": LaunchConfiguration("gui"),
                    "gazebo_world_file": LaunchConfiguration("gazebo_world_file"),
                    "seed_file": seed_file,
                }.items(),
            ),
            Node(
                package="concrete_block_motion_planning",
                executable="grip_traj_server_simple.py",
                name="grip_traj_server",
                output="screen",
                parameters=[
                    PathSubstitution(FindPackageShare("concrete_block_motion_planning"))
                    / "config"
                    / "grip_traj_simple.yaml",
                    {"use_sim_time": True},
                ],
            ),
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
                        "wall_plans_file": PathSubstitution(
                            FindPackageShare("concrete_block_assembly_planning"))
                        / "config" / "wall_plans.yaml",
                    },
                ],
            ),
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
                            FindPackageShare("concrete_block_behavior_tree")
                        )
                        / "behavior_trees"
                        / "basic_pick_and_place.xml"
                    },
                ],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                output="screen",
                parameters=[base_bt_config, {"use_sim_time": True}],
            ),
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
                                # a -6 m y shift.
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
