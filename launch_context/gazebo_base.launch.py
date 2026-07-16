from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    cross_pile = LaunchConfiguration("cross_pile")
    log_on_truck = LaunchConfiguration("log_on_truck")
    parallel_pile = LaunchConfiguration("parallel_pile")
    parallel_logs = LaunchConfiguration("parallel_logs")
    chaos_pile = LaunchConfiguration("chaos_pile")

    # ---------------------------
    # start Launch Description
    # ---------------------------
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                name="gui",
                default_value="True",
                description="Flag to launch gazebo GUI",
            ),
            DeclareLaunchArgument(
                name="gazebo_world_file",
                default_value=PythonExpression(
                    [
                        "'empty.world' if '",
                        LaunchConfiguration("gui"),
                        "'.lower() in ('false', '0', 'no') else 'epsilon_crane.world'",
                    ]
                ),
                description="World file from testsite_description/worlds to load in Gazebo.",
            ),
            DeclareLaunchArgument(
                name="parallel_pile",
                default_value="False",
                description="Flag to spawn a pile of wood logs arranged in parallel",
            ),
            DeclareLaunchArgument(
                name="parallel_logs",
                default_value="False",
                description="Flag to spawn a set of wood logs arranged in parallel",
            ),
            DeclareLaunchArgument(
                name="cross_pile",
                default_value="False",
                description="Flag to spawn a pile of wood logs arranged orthogonally",
            ),
            DeclareLaunchArgument(
                name="chaos_pile",
                default_value="False",
                description="Flag to spawn a pile of wood logs arranged chaotically",
            ),
            DeclareLaunchArgument(
                name="log_on_truck",
                default_value="True",
                description="Flag to spawn a log on truck bed",
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("gazebo_ros"))
                / "launch"
                / "gazebo.launch.py",
                launch_arguments={
                    "world": PathSubstitution(FindPackageShare("testsite_description"))
                    / "worlds"
                    / LaunchConfiguration("gazebo_world_file"),
                    "pause": "False",
                    "gui": LaunchConfiguration("gui", default="True"),
                }.items(),
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-topic",
                    "robot_description_full",
                    "-entity",
                    "crane_model",
                    "-timeout",
                    "60",
                    "-x",
                    "-0.0",
                    "-y",
                    "-6.0",
                    "-z",
                    "0",
                    "-R",
                    "0",
                    "-Y",
                    "3.14",
                ],
            ),
            # ros2 run gazebo_ros spawn_entity.py -database Wood_Log_1 -entity Wood_Log_1 -x -0.6 -y 5 -z 0.4 -R 1.569727 -Y -0.7
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-database",
                    "Wood_Log_1",
                    "-entity",
                    "Wood_Log_1",
                    "-x",
                    "8.4",
                    "-y",
                    "0",
                    "-z",
                    "0.4",
                    "-R",
                    "1.569727",
                    "-Y",
                    "-0.7",
                    "-timeout",
                    "60",
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            chaos_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            parallel_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            parallel_logs,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            cross_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            log_on_truck,
                            "'.lower() not in ('true', '1', 'yes')",
                        ]
                    )
                ),
            ),
            # ros2 run gazebo_ros spawn_entity.py -database Wood_Log_1 -entity Wood_Log_1 -x -2.8 -y -0.6 -z 1.6 -R 1.569727 -Y 1.57
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-database",
                    "Wood_Log_1",
                    "-entity",
                    "Wood_Log_Truck",
                    "-x",
                    "3.0",
                    "-y",
                    "-5.6",
                    "-z",
                    "1.6",
                    "-R",
                    "1.569727",
                    "-Y",
                    "-1.57",
                    "-timeout",
                    "60",
                ],
                condition=IfCondition(log_on_truck),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("testsite_description"))
                / "launch"
                / "spawn_parallel_pile.launch.py",
                condition=IfCondition(parallel_pile),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("testsite_description"))
                / "launch"
                / "spawn_cross_pile.launch.py",
                condition=IfCondition(cross_pile),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("testsite_description"))
                / "launch"
                / "spawn_chaos_pile.launch.py",
                condition=IfCondition(chaos_pile),
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("testsite_description"))
                / "launch"
                / "spawn_parallel_logs.launch.py",
                condition=IfCondition(parallel_logs),
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-database",
                    "wood_log_pile_on_pallet",
                    "-entity",
                    "wood_log_pile_on_pallet",
                    "-x",
                    "13",
                    "-y",
                    "3",
                    "-z",
                    "0.01",
                    "-R",
                    "1.569727",
                    "-P",
                    "0",
                    "-Y",
                    "0",
                    "-timeout",
                    "60",
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            chaos_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            parallel_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            parallel_logs,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            cross_pile,
                            "'.lower() not in ('true', '1', 'yes') and '",
                            log_on_truck,
                            "'.lower() not in ('true', '1', 'yes')",
                        ]
                    )
                ),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["--frame-id", "world", "--child-frame-id", "floor"],
            ),
            Node(
                package="gazebo_grasp_plugin_ros",
                executable="grasp_event_republisher",
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
