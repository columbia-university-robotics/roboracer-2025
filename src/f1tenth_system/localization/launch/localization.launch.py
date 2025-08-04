from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    params_file = os.path.join(get_package_share_directory('localization'), 'params', 'slam_toolbox.yaml')
    ekf_file = os.path.join(get_package_share_directory('localization'), 'params', 'ekf.yaml')
    map_path_no_ext = PathJoinSubstitution([FindPackageShare('localization'), 'maps', 'map'])
    map_path_ext = f"{map_path_no_ext}.yaml"

    slam_node = Node(
        package='slam_toolbox', 
        executable='localization_slam_toolbox_node',
        name='slam_toolbox', 
        output='screen', 
        parameters=[params_file, {'map_file_name': map_path_no_ext}]
    )

    robot_localization_node = Node(
        package='robot_localization', 
        executable='ekf_node',
        name='ekf_filter_node', 
        output='screen', 
        parameters=[ekf_file]
    )

    # Load pre-built map
    nav2_node = Node( 
        package='nav2_map_server', 
        executable='map_server',
        name='map_server', 
        output='screen',
        parameters=[{'yaml_filename': map_path_ext}]
    ) 
    
    # RViz for visualization
    rviz_node = Node(
        package='rviz2', 
        executable='rviz2', 
        arguments=['-d', os.path.join(get_package_share_directory('localization'), 'rviz', 'localization.rviz')])

    return LaunchDescription([slam_node, robot_localization_node, nav2_node, rviz_node])

