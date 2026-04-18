import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    localization_share = get_package_share_directory("localization")
    planner_share = get_package_share_directory("planning")
    web_share = get_package_share_directory("planner_web_ui")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_file",
                default_value="",
                description="Map YAML file for localization mode.",
            ),
            DeclareLaunchArgument(
                "planner_config",
                default_value=os.path.join(planner_share, "config", "real.yaml"),
                description="Planner configuration YAML for hardware mode.",
            ),
            DeclareLaunchArgument(
                "enable_rviz",
                default_value="false",
                description="Whether to launch RViz alongside localization.",
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
                    os.path.join(localization_share, "launch", "mapping.launch.py")
                ),
                launch_arguments={
                    "mode": "localization",
                    "map_file": LaunchConfiguration("map_file"),
                    "enable_rviz": LaunchConfiguration("enable_rviz"),
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
