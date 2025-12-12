from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Intel RealSense D435i node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        output='screen',
        parameters=[{
            'enable_depth': True,
            'enable_color': True,
            'enable_gyro': True,
            'enable_accel': True,
            'align_depth.enable': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'unite_imu_method': 'copy',
            'publish_tf': True,
            'enable_sync': True
        }]
    )

    # Static TF base_link → camera_link
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=['0.1', '0.0', '0.25', '0', '0', '0', 'base_link', 'camera_link']
    )

    # Correct path to localization package
    localization_pkg_share = get_package_share_directory('localization')
    localization_launch = os.path.join(
        localization_pkg_share, 'launch', 'mapping.launch.py'
    )


    localization_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch)
    )

    return LaunchDescription([
        realsense_node,
        static_tf,
        localization_include
    ])
