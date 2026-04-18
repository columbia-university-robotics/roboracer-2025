import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    planner_share = get_package_share_directory("planning")
    web_share = get_package_share_directory("planner_web_ui")
    gym_share = get_package_share_directory("f1tenth_gym_ros")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="maps/levine",
                description="Simulator map path or built-in gym track name.",
            ),
            DeclareLaunchArgument(
                "planner_config",
                default_value=os.path.join(planner_share, "config", "sim.yaml"),
                description="Planner configuration YAML for sim mode.",
            ),
            DeclareLaunchArgument(
                "host",
                default_value="0.0.0.0",
                description="Host interface for the planner web UI server.",
            ),
            DeclareLaunchArgument(
                "web_port",
                default_value="8081",
                description="Port to serve the planner web UI on.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gym_share, "launch", "gym_bridge_launch.py")
                ),
                launch_arguments={
                    "map_path": LaunchConfiguration("map"),
                    "open_foxglove": "false",
                    "start_foxglove_bridge": "false",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(planner_share, "launch", "planning.launch.py")
                ),
                launch_arguments={
                    "config_file": LaunchConfiguration("planner_config"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(web_share, "launch", "web_ui.launch.py")
                ),
                launch_arguments={
                    "host": LaunchConfiguration("host"),
                    "web_port": LaunchConfiguration("web_port"),
                }.items(),
            ),
        ]
    )
