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
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def package_root() -> Path:
    return Path(get_package_share_directory("localization"))


def find_map_files() -> list[str]:
    """Search for map files in the package's maps directory."""
    maps_dir = package_root() / "maps"
    if not maps_dir.exists():
        return []
    return sorted(str(path) for path in maps_dir.glob("*.yaml"))


def resolve_map_file(map_file: str) -> Path:
    """Resolve absolute or package-relative map YAML paths."""
    candidate = Path(map_file).expanduser()
    if candidate.is_absolute():
        return candidate

    search_roots = (
        Path.cwd(),
        package_root(),
        package_root() / "maps",
    )
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def pose_relay_node(input_topic: str) -> Node:
    return Node(
        package="localization",
        executable="pose_relay.py",
        name="localization_pose_relay",
        output="screen",
        parameters=[
            {
                "input_topic": input_topic,
                "output_topic": "/localization/pose",
            }
        ],
    )


def generate_launch_description_from_context(context):
    """Generate launch description based on mode."""
    mode = LaunchConfiguration('mode').perform(context)
    map_file = LaunchConfiguration('map_file').perform(context)
    enable_rviz = LaunchConfiguration('enable_rviz').perform(context).lower() in (
        '1',
        'true',
        'yes',
        'on',
    )
    
    nodes_to_launch = []
    share_dir = package_root()
    
    if enable_rviz:
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
        slam_params_file = os.path.join(str(share_dir), 'params', 'slam_toolbox_mapping.yaml')
        
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
        nodes_to_launch.append(pose_relay_node('/pose'))
        
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
            available_maps = find_map_files()
            error_msg = 'ERROR: map_file argument is required for localization mode!\n'
            error_msg += 'Available maps found:\n'
            for m in available_maps:
                error_msg += f'  - {m}\n'
            error_msg += '\nUsage: ros2 launch localization mapping.launch.py mode:=localization map_file:=<path_to_map.yaml>'
            nodes_to_launch.append(LogInfo(msg=error_msg))
            return nodes_to_launch

        resolved_map_file = resolve_map_file(map_file)
        if not resolved_map_file.exists():
            nodes_to_launch.append(LogInfo(msg=f'ERROR: Map file not found: {map_file}'))
            return nodes_to_launch
        
        # Map Server node
        map_server_node = Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': str(resolved_map_file),
                'use_sim_time': False
            }]
        )
        nodes_to_launch.append(map_server_node)
        
        # Lifecycle bringup for map_server using nav2_lifecycle_manager
        map_server_bringup = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map_server',
            output='screen',
            parameters=[
                { 'use_sim_time': False },
                { 'autostart': True },
                { 'node_names': ['map_server'] },
            ]
        )
        nodes_to_launch.append(map_server_bringup)
        
        amcl_params_file = os.path.join(
            str(share_dir),
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
        nodes_to_launch.append(pose_relay_node('/amcl_pose'))
        
        # Lifecycle bringup for amcl using nav2_lifecycle_manager
        amcl_bringup = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_amcl',
            output='screen',
            parameters=[
                { 'use_sim_time': False },
                { 'autostart': True },
                { 'node_names': ['amcl'] },
            ]
        )
        nodes_to_launch.append(amcl_bringup)
        
        nodes_to_launch.append(
            LogInfo(msg=f'Starting LOCALIZATION mode with map: {resolved_map_file}')
        )
    
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
        description='Map YAML file for localization mode. Accepts an absolute path or a path relative to the localization package.'
    )
    enable_rviz_arg = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Whether to launch RViz alongside mapping/localization.'
    )
    
    # Use OpaqueFunction to generate nodes based on context since it defers evaluation until launch args are parsed
    launch_nodes = OpaqueFunction(function=generate_launch_description_from_context)
    
    return LaunchDescription([
        mode_arg,
        map_file_arg,
        enable_rviz_arg,
        launch_nodes
    ])
