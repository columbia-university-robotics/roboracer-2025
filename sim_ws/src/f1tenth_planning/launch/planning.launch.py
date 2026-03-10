import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("f1tenth_planning")
    default_config = os.path.join(package_share, "config", "sim.yaml")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Planner/follower parameter YAML.",
            ),
            Node(
                package="f1tenth_planning",
                executable="occupancy_grid_planner",
                name="occupancy_grid_planner",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="f1tenth_planning",
                executable="pure_pursuit_follower",
                name="pure_pursuit_follower",
                parameters=[config_file],
                output="screen",
            ),
        ]
    )
