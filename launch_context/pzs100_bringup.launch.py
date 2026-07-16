from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PathSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    planner = LaunchConfiguration("planner")
    controller = LaunchConfiguration("controller")
    seed_file = LaunchConfiguration("seed_file")
    grasp_detector_config = (
        PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
        / "config"
        / "gripper_grasp_detector_sim.yaml"
    )

    mp_launch_file = PythonExpression(
        ["'mp.launch.py' if '", planner, "' == 'ilqr' else 'mp_esdf.launch.py'"]
    )
    # PZS100 PID and MPC controller configs both live in CBS.
    ctrl_package = "concrete_block_behavior_tree"
    ctrl_config = PythonExpression(
        [
            "'ros2_control/crane_controller_hydraulic_a2b_jtc_pid_pzs100.ros2_control.yaml' if '",
            controller,
            "' == 'pid' else 'ros2_control/crane_controller_hydraulic_a2b_jtc_mpc_pzs100.ros2_control.yaml'",
        ]
    )

    ld = LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner",
                default_value="ilqr",
                description="Planner: 'ilqr' or 'vpsto'",
            ),
            DeclareLaunchArgument(
                "controller",
                default_value="pid",
                description="Controller: 'pid' or 'mpc'",
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
            DeclareLaunchArgument(
                "gui",
                default_value="True",
                description="Flag to launch gazebo+rviz2 GUI.",
            ),
            DeclareLaunchArgument(
                "gazebo_world_file",
                default_value="epsilon_crane.world",
                description=(
                    "World file from testsite_description/worlds. Use empty.world "
                    "for a clean sim without the Seibersdorf container."
                ),
            ),
            DeclareLaunchArgument(
                "start_grasp_detector",
                default_value="True",
                description="Start the q9 gripper grasp detector for PZS100 Gazebo runs.",
            ),
            # Spawn the PZS100 joint-state adapter that publishes
            # /joint_states_rviz with the per-rail-corrected EPSCOPE q9.
            Node(
                package="concrete_block_behavior_tree",
                executable="pzs100_rviz_joint_state_adapter.py",
                name="pzs100_rviz_joint_state_adapter",
                parameters=[{"use_sim_time": True}],
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="concrete_block_behavior_tree",
                executable="gripper_grasp_detector",
                name="gripper_grasp_detector",
                output="screen",
                parameters=[grasp_detector_config, {"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("start_grasp_detector")),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_sim"))
                / "launch"
                / "gazebo_model_bt.launch.py",
                launch_arguments={
                    "tool": "pzs100_description",
                    "controller_common_config_package": "concrete_block_behavior_tree",
                    "controller_common_config": "crane_controller_hydraulic_common_pzs100.ros2_control.yaml",
                    "controller_a2b_config_package": ctrl_package,
                    "controller_a2b_config": ctrl_config,
                    "mp_launch_file": mp_launch_file,
                    "mp_param_file": "mp_parameter_pzs100.yaml",
                    "mp_param_path": PathSubstitution(
                        FindPackageShare("concrete_block_behavior_tree")
                    )
                    / "config/motion_planning/mp_parameter_pzs100.yaml",
                    "grip_traj_param_file": "grip_traj_parameter_pzs100.yaml",
                    "grip_traj_param_path": PathSubstitution(
                        FindPackageShare("concrete_block_behavior_tree")
                    )
                    / "config/motion_planning/grip_traj_parameter_pzs100.yaml",
                    "estimators_param_file": "estimators_pzs100.yaml",
                    "estimators_param_path": PathSubstitution(
                        FindPackageShare("concrete_block_behavior_tree")
                    )
                    / "config/estimators_pzs100.yaml",
                    "enable_gripper_tip_publisher": "False",
                    "enable_pump_flow_estimator": "False",
                    "enable_log_state_estimator": "False",
                    "rviz_config": PathSubstitution(
                        FindPackageShare("concrete_block_behavior_tree")
                    )
                    / "rviz"
                    / "cbs.rviz",
                    "joint_states_topic": "joint_states_rviz",
                    "gui": LaunchConfiguration("gui"),
                    "gazebo_world_file": LaunchConfiguration("gazebo_world_file"),
                }.items(),
            ),
        ]
    )

    # World model is seeded from the same YAML the Gazebo block spawner uses, so
    # markers / GetCoarseBlocks and the spawned Gazebo blocks share one source.
    ld.add_action(
        Node(
            package="concrete_block_world_model",
            executable="world_model_node",
            name="world_model_node",
            parameters=[
                PathSubstitution(FindPackageShare("concrete_block_world_model"))
                / "config"
                / "world_model.yaml",
                seed_file,
                {
                    "use_sim_time": True,
                    # Whole CBS stack (spec, plan, world model, viz) stays in `world`;
                    # the single world -> K0_mounting_base conversion happens in the
                    # GetNextAssemblyTask BT plugin.
                    "world_frame": "world",
                    # TCP frame for FK tracking of grasped (TASK_MOVE) blocks.
                    # Must match the BT CaptureBlockGraspOffset gripper_frame and
                    # exist in the sim TF tree.
                    "refine_grasped.tcp_frame": "K8_tool_center_point",
                    "pipeline_mode": "idle",
                    "perception_mode": "IDLE",
                },
            ],
            remappings=[
                ("block_world_model", "/cbp/block_world_model"),
                ("block_world_model_markers", "/cbp/block_world_model_markers"),
                ("block_goal_markers", "/cbp/block_goal_markers"),
            ],
            output="screen",
        )
    )

    return ld
