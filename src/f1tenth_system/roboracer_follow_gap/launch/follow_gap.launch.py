from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='roboracer_follow_gap',
            executable='follow_gap',
            name='follow_gap_node',
            output='screen',
            parameters=[
                {'scan_topic': '/scan'},
                {'drive_topic': '/drive'},
                {'bubble_radius': 0.5},
                {'max_steering_angle': 0.4},
                {'max_speed': 3.0},
                {'min_speed': 0.7},
            ]
        )
    ])
