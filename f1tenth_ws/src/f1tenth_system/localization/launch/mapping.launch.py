"""
Launch file for SLAM Toolbox mapping and localization modes.

This launch file supports two modes:
1. mapping: Creates a new map using slam_toolbox
2. localization: Localizes the robot on an existing map using AMCL and nav2_map_server

Usage:
  Mapping mode:
    ros2 launch localization mapping.launch.py mode:=mapping
  
  Localization mode:
    ros2 launch localization mapping.launch.py mode:=localization map_file:=/path/to/map.yaml
"""

import os
from glob import glob
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def find_map_files(context):
    """Search for map files in common locations."""
    search_paths = [
        '/home/curc/Desktop/f1tenth_ws/src/f1tenth_system/localization/maps',
        '/home/curc/Desktop/f1tenth_ws',
    ]
    
    map_files = []
    for path in search_paths:
        if os.path.exists(path):
            yaml_files = glob(os.path.join(path, '*.yaml'))
            map_files.extend(yaml_files)
    
    return map_files


def generate_launch_description_from_context(context):
    """Generate launch description based on mode."""
    mode = LaunchConfiguration('mode').perform(context)
    map_file = LaunchConfiguration('map_file').perform(context)
    
    nodes_to_launch = []
    
    # RViz2 node (common for both modes)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', '/opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz']
    )
    nodes_to_launch.append(rviz_node)
    
    if mode == 'mapping':
        # SLAM Toolbox for mapping
        slam_params_file = os.path.join(get_package_share_directory('localization'), 'params', 'slam_toolbox_mapping.yaml')
        
        slam_toolbox_dir = get_package_share_directory('slam_toolbox')
        
        slam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'False',
                'slam_params_file': slam_params_file
            }.items()
        )
        nodes_to_launch.append(slam_launch)
        
        nodes_to_launch.append(LogInfo(msg='Starting SLAM Toolbox in MAPPING mode'))

        # Nav2 Navigation
        navigation_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
            ),
        )
        nodes_to_launch.append(navigation_launch)
    
    elif mode == 'localization':
        # Validate map file
        if not map_file or map_file == '':
            available_maps = find_map_files(context)
            error_msg = 'ERROR: map_file argument is required for localization mode!\n'
            error_msg += 'Available maps found:\n'
            for m in available_maps:
                error_msg += f'  - {m}\n'
            error_msg += '\nUsage: ros2 launch localization mapping.launch.py mode:=localization map_file:=<path_to_map.yaml>'
            nodes_to_launch.append(LogInfo(msg=error_msg))
            return nodes_to_launch
        
        if not os.path.exists(map_file):
            nodes_to_launch.append(LogInfo(msg=f'ERROR: Map file not found: {map_file}'))
            return nodes_to_launch
        
        # Map Server node
        map_server_node = Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': map_file,
                'use_sim_time': False
            }]
        )
        nodes_to_launch.append(map_server_node)
        
        # Lifecycle bringup for map_server using nav2_util
        map_server_bringup = Node(
            package='nav2_util',
            executable='lifecycle_bringup',
            name='lifecycle_bringup_map_server',
            output='screen',
            arguments=['map_server']
        )
        nodes_to_launch.append(map_server_bringup)
        
        amcl_params_file = os.path.join(
            get_package_share_directory('localization'),
            'params',
            'amcl_config.yaml'
        )
        
        # AMCL node
        amcl_node = Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                amcl_params_file,
                { 'use_sim_time': False }
            ]
        )
        nodes_to_launch.append(amcl_node)
        
        # Lifecycle bringup for amcl using nav2_util
        amcl_bringup = Node(
            package='nav2_util',
            executable='lifecycle_bringup',
            name='lifecycle_bringup_amcl',
            output='screen',
            arguments=['amcl']
        )
        nodes_to_launch.append(amcl_bringup)
        
        nodes_to_launch.append(LogInfo(msg=f'Starting LOCALIZATION mode with map: {map_file}'))
    
    else:
        nodes_to_launch.append(LogInfo(msg=f'ERROR: Invalid mode "{mode}". Must be "mapping" or "localization"'))
    
    return nodes_to_launch


def generate_launch_description():
    """Generate the launch description."""
    
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='mapping',
        description='Mode: "mapping" to create a map, "localization" to localize on existing map'
    )
    
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='',
        description='Absolute path to map YAML file (required for localization mode). Example: /home/curc/Desktop/f1tenth_ws/src/f1tenth_system/localization/maps/my_map.yaml'
    )
    
    # Use OpaqueFunction to generate nodes based on context since it defers evaluation until launch args are parsed
    launch_nodes = OpaqueFunction(function=generate_launch_description_from_context)
    
    return LaunchDescription([
        mode_arg,
        map_file_arg,
        launch_nodes
    ])
