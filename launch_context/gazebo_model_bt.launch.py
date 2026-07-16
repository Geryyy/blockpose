from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="True")

    # Launch configuration variables specific to simulation
    initial_pose = LaunchConfiguration("initial_pose")

    # Get Paths for Packages and combine them
    xacro_path = (
        PathSubstitution(FindPackageShare("epsilon_crane_description"))
        / "urdf"
        / "timber_loader_AIT.urdf.xacro"
    )

    # Get Paths for Packages and combine them
    tool_pkg_name = LaunchConfiguration("tool")

    # Write more often used parameters into a dictionary
    use_sim_time_param = {"use_sim_time": use_sim_time}

    # Process the xacro which contains the description of the robot
    # export for other simulators via
    # xacro src/epsilon_crane_description/urdf/timber_loader_AIT.urdf.xacro gazebo:=false tool:=epsilon_7040_description load_ros2_control:=false > crane.urdf
    robot_description_config = Command(
        [
            "xacro",
            " ",
            xacro_path,
            " ",
            "gazebo:=true",
            " ",
            "rigid:=true ",
            "sim_hydraulics:=true",
            " ",
            "enable_logs_sim:=",
            LaunchConfiguration("logs_sim"),
            " ",
            "enable_zed2i_sim:=",
            LaunchConfiguration("zed2i_sim"),
            " ",
            "enable_ait_stereo_sim:=",
            LaunchConfiguration("ait_stereo_sim"),
            " ",
            "enable_livox_sim:=",
            LaunchConfiguration("enable_livox_sim"),
            " ",
            "initial_pose:=",
            initial_pose,
            " ",
            "tool:=",
            tool_pkg_name,
            " ",
            "post_setup:=134 ",
        ]
    )  # create xml from xacro
    robot_description_rviz_config = Command(
        [
            "xacro",
            " ",
            xacro_path,
            " ",
            "gazebo:=false",
            " ",
            "load_ros2_control:=false",
            " ",
            "rigid:=true ",
            "sim_hydraulics:=false",
            " ",
            "initial_pose:=",
            initial_pose,
            " ",
            "tool:=",
            tool_pkg_name,
            " ",
            "post_setup:=134 ",
        ]
    )  # create xml from xacro
    robot_description_elastic_full_config = Command(
        [
            "xacro",
            " ",
            xacro_path,
            " ",
            "gazebo:=true ",
            "rigid:=false ",
            "sim_hydraulics:=true",
            " ",
            "initial_pose:=",
            initial_pose,
            " ",
            "tool:=",
            tool_pkg_name,
            " ",
            "tool_configuration:=",
            PathSubstitution(FindPackageShare(tool_pkg_name))
            / "config"
            / "gripper_parameter.yaml",
            " ",
            "post_setup:=134 ",
        ]
    )  # create xml from xacro

    # ---------------------------
    # start Launch Description
    # ---------------------------
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="logs_sim",
                default_value="False",
                description="publish log poses from gazebo simulation",
            ),
            DeclareLaunchArgument(
                name="zed2i_sim",
                default_value="False",
                description="Simulate zed2i camera within gazebo simulation",
            ),
            DeclareLaunchArgument(
                name="ait_stereo_sim",
                default_value="False",
                description="Simulate ait stereo camera within gazebo simulation",
            ),
            DeclareLaunchArgument(
                name="enable_livox_sim",
                default_value="",
                description="Simulate livox lidar within gazebo simulation ('', 'simple' or 'livox')",
            ),
            DeclareLaunchArgument(
                name="initial_pose",
                default_value=TextSubstitution(text="1"),
                description="Define initial pose of crane, see readme for further details",
            ),
            DeclareLaunchArgument(
                name="plot",
                default_value="False",
                description="Flag to start up plotjuggler too",
            ),
            DeclareLaunchArgument(
                name="tool",
                default_value="epsilon_7040_description",
                description=("Name of the package where the tool is defined"),
            ),
            DeclareLaunchArgument(
                name="controller_common_config_package",
                default_value="epsilon_crane_bringup_sim",
                description="Package containing the common controller config",
            ),
            DeclareLaunchArgument(
                name="controller_common_config",
                default_value="crane_controller_hydraulic_common.ros2_control.yaml",
                description="ros2_control common controller config file",
            ),
            DeclareLaunchArgument(
                name="controller_a2b_config_package",
                default_value="timber_crane_unconstrained_mpc_controller_cpp",
                description="Package containing the a2b controller config",
            ),
            DeclareLaunchArgument(
                name="controller_a2b_config",
                default_value="crane_controller_hydraulic_a2b_jtc_mpc.ros2_control.yaml",
                description="A2B controller config file (MPC, PID, or LQR)",
            ),
            DeclareLaunchArgument(
                name="mp_launch_file",
                default_value="mp_esdf.launch.py",
                description="Motion planning launch file (mp_esdf.launch.py or mp.launch.py)",
            ),
            DeclareLaunchArgument(
                name="mp_backend",
                default_value="timber",
                description="Motion planning backend: 'timber', 'ilqr', or 'cbs'",
            ),
            DeclareLaunchArgument(
                name="mp_param_file",
                default_value="mp_parameter.yaml",
                description="Motion-planning parameter config file",
            ),
            DeclareLaunchArgument(
                name="grip_traj_param_file",
                default_value="grip_traj_parameter.yaml",
                description="Grip trajectory parameter config file",
            ),
            DeclareLaunchArgument(
                name="collision_objects_file",
                default_value="collision_objects.yaml",
                description="Collision objects config file",
            ),
            DeclareLaunchArgument(
                name="estimators_param_file",
                default_value="estimators.yaml",
                description="Estimator parameter config file",
            ),
            DeclareLaunchArgument(
                name="enable_gripper_tip_publisher",
                default_value="True",
                description="Enable the gripper tip publisher helper node",
            ),
            DeclareLaunchArgument(
                name="enable_pump_flow_estimator",
                default_value="True",
                description="Enable the pump flow estimator",
            ),
            DeclareLaunchArgument(
                name="gui",
                default_value="True",
                description=("Flag to launch gazebo+rviz2 GUI"),
            ),
            DeclareLaunchArgument(
                name="gazebo_world_file",
                default_value="epsilon_crane.world",
                description="World file from testsite_description/worlds to load in Gazebo.",
            ),
            DeclareLaunchArgument(
                name="rviz_config",
                default_value=PathSubstitution(
                    FindPackageShare("epsilon_crane_bringup_hmi")
                )
                / "rviz"
                / "crane_data.rviz",
                description="Path to the RViz2 config file to load",
            ),
            DeclareLaunchArgument(
                name="joint_states_topic",
                default_value="joint_states",
                description=(
                    "Topic that robot_state_publisher subscribes to. Downstream "
                    "stacks that publish a transformed joint state (e.g. PZS100) "
                    "override this to point at the transformed topic."
                ),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[
                    {"robot_description": robot_description_config},
                    use_sim_time_param,
                    {"publish_frequency": 100.0},
                ],
                remappings=[
                    ("joint_states", LaunchConfiguration("joint_states_topic")),
                    ("robot_description", "robot_description_full"),
                    ("tf", "tf_gazebo"),
                    ("tf_static", "tf_static_gazebo"),
                ],
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher_rviz",
                parameters=[
                    {"robot_description": robot_description_rviz_config},
                    use_sim_time_param,
                    {"publish_frequency": 100.0},
                ],
                remappings=[
                    ("joint_states", LaunchConfiguration("joint_states_topic")),
                ],
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher_elastic_full",
                parameters=[
                    {"robot_description": robot_description_elastic_full_config},
                    use_sim_time_param,
                    # we don't need this TF tree
                    {"publish_frequency": 0.01},
                ],
                remappings=[
                    ("robot_description", "robot_description_elastic_full"),
                    ("tf", "tf_elastic_full"),
                    ("tf_static", "tf_static_elastic_full"),
                ],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    "WARN",
                ],  # do not output info msgs
            ),
            Node(
                package="crane_tools_description",
                executable="crane_tools_description_publisher",
                name="crane_tools_description_publisher",
                parameters=[
                    PathSubstitution(FindPackageShare(tool_pkg_name))
                    / "config"
                    / "gripper_parameter.yaml",
                    use_sim_time_param,
                ],
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_sim"))
                / "launch"
                / "gazebo_base.launch.py",
                launch_arguments={
                    "gui": LaunchConfiguration("gui"),
                    "gazebo_world_file": LaunchConfiguration("gazebo_world_file"),
                    "parallel_pile": LaunchConfiguration(
                        "parallel_pile", default="False"
                    ),
                    "parallel_logs": LaunchConfiguration(
                        "parallel_logs", default="False"
                    ),
                    "cross_pile": LaunchConfiguration("cross_pile", default="False"),
                    "chaos_pile": LaunchConfiguration("chaos_pile", default="False"),
                    "log_on_truck": LaunchConfiguration("log_on_truck", default="True"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_common"))
                / "launch"
                / "bt.launch.py",
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "keyboard_node": LaunchConfiguration(
                        "start_bt_action_server", default="True"
                    ),
                    "bt_action_server": LaunchConfiguration(
                        "start_bt_action_server", default="True"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_common"))
                / "launch"
                / "estimators.launch.py",
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "tool": tool_pkg_name,
                    "estimators_param_file": LaunchConfiguration(
                        "estimators_param_file"
                    ),
                    "estimators_param_path": LaunchConfiguration(
                        "estimators_param_path",
                        default=PathSubstitution(
                            FindPackageShare("epsilon_crane_bringup_common")
                        )
                        / "config"
                        / LaunchConfiguration("estimators_param_file"),
                    ),
                    "enable_gripper_tip_publisher": LaunchConfiguration(
                        "enable_gripper_tip_publisher"
                    ),
                    "enable_pump_flow_estimator": LaunchConfiguration(
                        "enable_pump_flow_estimator"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_common"))
                / "launch"
                / "common_controllers.launch.py",
                launch_arguments={
                    **use_sim_time_param,
                    "enable_log_state_estimator": LaunchConfiguration(
                        "enable_log_state_estimator", default="True"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_mp"))
                / "launch"
                / LaunchConfiguration("mp_launch_file"),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "mp_backend": LaunchConfiguration("mp_backend"),
                    "mp_param_file": LaunchConfiguration("mp_param_file"),
                    # Forward absolute path so downstream stacks can ship
                    # their own param files. Defaults to the file inside
                    # epsilon_crane_bringup_mp/config/motion_planning so
                    # existing callers keep working.
                    "mp_param_path": LaunchConfiguration(
                        "mp_param_path",
                        default=PathSubstitution(
                            FindPackageShare("epsilon_crane_bringup_mp")
                        )
                        / "config/motion_planning"
                        / LaunchConfiguration("mp_param_file"),
                    ),
                    "grip_traj_param_file": LaunchConfiguration("grip_traj_param_file"),
                    "grip_traj_param_path": LaunchConfiguration(
                        "grip_traj_param_path",
                        default=PathSubstitution(
                            FindPackageShare("epsilon_crane_bringup_mp")
                        )
                        / "config/motion_planning"
                        / LaunchConfiguration("grip_traj_param_file"),
                    ),
                    "collision_objects_file": LaunchConfiguration(
                        "collision_objects_file"
                    ),
                    "start_grip_traj_server": LaunchConfiguration(
                        "start_grip_traj_server", default="True"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("epsilon_crane_bringup_common"))
                / "launch"
                / "loadbed_frame.launch.py"
            ),
            # IncludeLaunchDescription(
            #     PathSubstitution(FindPackageShare("epsilon_crane_bringup_sim"))
            #     / "launch"
            #     / "temporal_log_aggregation.launch.py"
            # ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "trajectory_controller_a2b",
                    "--inactive",
                    "--controller-manager-timeout",
                    "20",
                    "-p",
                    PathSubstitution(
                        FindPackageShare(
                            LaunchConfiguration("controller_common_config_package")
                        )
                    )
                    / "config"
                    / "ros2_control"
                    / LaunchConfiguration("controller_common_config"),
                    "-p",
                    PathSubstitution(
                        FindPackageShare(
                            LaunchConfiguration("controller_a2b_config_package")
                        )
                    )
                    / "config"
                    / LaunchConfiguration("controller_a2b_config"),
                    "--controller-ros-args",
                    # remap to the same topic, as there can be only one controller active at a time
                    "-r /trajectory_controller_a2b/controller_state:=/trajectory_controllers/controller_state",
                ],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "trajectory_controller_grasping",
                    "--inactive",
                    "--controller-manager-timeout",
                    "20",
                    "-p",
                    PathSubstitution(FindPackageShare("epsilon_crane_bringup_sim"))
                    / "config"
                    / "ros2_control"
                    / "crane_controller_hydraulic_common.ros2_control.yaml",
                    "--controller-ros-args",
                    # remap to the same topic, as there can be only one controller active at a time
                    "-r /trajectory_controller_grasping/controller_state:=/trajectory_controllers/controller_state",
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                parameters=[use_sim_time_param],
                arguments=[
                    "-d",
                    LaunchConfiguration("rviz_config"),
                ],
                condition=IfCondition(LaunchConfiguration("gui")),
            ),
            Node(
                package="plotjuggler",
                executable="plotjuggler",
                arguments=[
                    "-l",
                    PathSubstitution(FindPackageShare("epsilon_crane_bringup_hmi"))
                    / "config"
                    / "pj"
                    / "Plot_juggler_layout_hydraulic_jtc.xml",
                ],
                condition=IfCondition(LaunchConfiguration("plot")),
            ),
        ]
    )
