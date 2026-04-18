from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "host",
                default_value="0.0.0.0",
                description="Host interface to bind the planner web UI server to.",
            ),
            DeclareLaunchArgument(
                "web_port",
                default_value="8081",
                description="Port to serve the planner web UI on.",
            ),
            Node(
                package="planner_web_ui",
                executable="planner_web_ui",
                name="planner_web_ui",
                output="screen",
                parameters=[
                    {
                        "host": LaunchConfiguration("host"),
                        "port": ParameterValue(
                            LaunchConfiguration("web_port"), value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
