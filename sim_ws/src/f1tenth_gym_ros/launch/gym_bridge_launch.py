# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import pathlib
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _resolve_yaml_path(base_path: pathlib.Path) -> pathlib.Path:
    if base_path.suffix in ('.yaml', '.yml'):
        return base_path
    candidates = (
        base_path.with_suffix('.yaml'),
        base_path.with_suffix('.yml'),
        base_path.parent / f"{base_path.stem}_map.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_path.with_suffix('.yaml')


def _resolve_map_yaml_path(map_path: str, package_share: str) -> pathlib.Path:
    if map_path.startswith(('/', '\\')):
        return _resolve_yaml_path(pathlib.Path(map_path))
    if '/' in map_path or '\\' in map_path:
        return _resolve_yaml_path(pathlib.Path(package_share) / map_path)
    try:
        from f1tenth_gym.envs.track.utils import find_track_dir
        track_dir = find_track_dir(map_path)
        yaml_path = track_dir / f"{track_dir.stem}.yaml"
        if not yaml_path.exists():
            yaml_path = track_dir / f"{track_dir.stem}_map.yaml"
        return yaml_path
    except Exception:
        return _resolve_yaml_path(pathlib.Path(package_share) / 'maps' / map_path)


def _bool_arg(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def launch_setup(context):
    package_share = get_package_share_directory('f1tenth_gym_ros')
    config = os.path.join(package_share, 'config', 'sim.yaml')
    with open(config, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1
    use_sim_time = config_dict['bridge']['ros__parameters']['use_sim_time']

    map_path = LaunchConfiguration('map_path').perform(context)
    foxglove_host = LaunchConfiguration('foxglove_host').perform(context)
    foxglove_port = LaunchConfiguration('foxglove_port').perform(context)
    foxglove_app_url = LaunchConfiguration('foxglove_app_url').perform(context)
    foxglove_layout = LaunchConfiguration('foxglove_layout').perform(context)
    foxglove_target = LaunchConfiguration('foxglove_target').perform(context).lower()
    open_foxglove = _bool_arg(LaunchConfiguration('open_foxglove').perform(context))
    start_foxglove_bridge = _bool_arg(
        LaunchConfiguration('start_foxglove_bridge').perform(context)
    )
    if foxglove_target not in ('browser', 'studio'):
        raise ValueError("config/foxglove/target must be either 'browser' or 'studio'.")

    foxglove_open_url = (
        f'foxglove://open?ds=foxglove-websocket&ds.url=ws://{foxglove_host}:{foxglove_port}'
        if foxglove_target == 'studio'
        else f'{foxglove_app_url}/?ds=foxglove-websocket&ds.url=ws://{foxglove_host}:{foxglove_port}'
    )

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config, {
            'map_path': map_path,
            'use_sim_time': False,  # Always use real time for the bridge node
            'use_sim_time_bridge': use_sim_time, # Whether to internally use and publish sim time
            }], 
    )

    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{'port': foxglove_port}],
        output='screen',
    )

    # Create custom yaml file for map server by copying the original yaml file and scaling the resolution.
    map_yaml_path = _resolve_map_yaml_path(map_path, package_share)
    with open(map_yaml_path, 'r') as file:
        map_yaml = yaml.safe_load(file)
    scale = config_dict['bridge']['ros__parameters']['scale']
    map_yaml['resolution'] *= scale
    origin = map_yaml['origin']
    scaled_origin = (
        origin[0] * scale,
        origin[1] * scale,
        origin[2],
    )
    map_yaml['origin'] = scaled_origin
    image_field = map_yaml.get('image')
    if not image_field:
        raise ValueError('map yaml is missing image field')
    image_path = pathlib.Path(image_field)
    if not image_path.is_absolute():
        image_path = map_yaml_path.parent / image_path
    map_img_ext = image_path.suffix
    map_yaml['image'] = 'scaled_map' + map_img_ext

    temp_yaml_path = None
    # Create a temporary directory to store the scaled map yaml and image
    # Create a temporary directory to store the scaled map yaml and image in the same location as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_yaml_path = os.path.join(temp_dir, 'scaled_map.yaml')
    temp_img_path = os.path.join(temp_dir, 'scaled_map' + map_img_ext)

    # Write the scaled map yaml to the temporary file
    with open(temp_yaml_path, 'w') as file:
        yaml.dump(map_yaml, file)

    # Copy the map image to the temporary directory
    map_image_path = image_path
    with open(temp_img_path, 'wb') as file:
        with open(map_image_path, 'rb') as img_file:
            file.write(img_file.read())

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': temp_yaml_path},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': use_sim_time}],
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )


    ego_xacro = None
    if config_dict['bridge']['ros__parameters']['vehicle_params'] == 'f1tenth':
        ego_xacro = "ego_racecar.xacro"
    elif config_dict['bridge']['ros__parameters']['vehicle_params'] == 'fullscale':
        ego_xacro = "ego_racecar_fullscale.xacro"
    elif config_dict['bridge']['ros__parameters']['vehicle_params'] == 'f1fifth':
        ego_xacro = "ego_racecar_f1fifth.xacro"
    else:
        raise ValueError('vehicle_params should be either f1tenth, fullscale, or f1fifth.')
    
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[
            {'robot_description': Command([
                'xacro ',
                os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'urdf', ego_xacro)
            ])},
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/robot_description', 'ego_robot_description')]
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[
            {'robot_description': Command([
                'xacro ',
                os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'urdf', 'opp_racecar.xacro')
            ])},
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/robot_description', 'opp_robot_description')]
    )

    actions = [bridge_node, nav_lifecycle_node, map_server_node, ego_robot_publisher]
    if has_opp:
        actions.append(opp_robot_publisher)

    if start_foxglove_bridge:
        actions.insert(0, foxglove_bridge_node)
        if open_foxglove:
            actions.insert(1, ExecuteProcess(cmd=['xdg-open', foxglove_open_url], output='screen'))
        actions.extend(
            [
                LogInfo(msg=f'\033[34mFoxglove app: {foxglove_app_url}\033[0m'),
                LogInfo(
                    msg=f'\033[34mFoxglove WebSocket: ws://{foxglove_host}:{foxglove_port}\033[0m'
                ),
                LogInfo(msg=f'\033[34mFoxglove target: {foxglove_target}\033[0m'),
                LogInfo(msg=f'\033[34mFoxglove layout: {foxglove_layout}\033[0m'),
            ]
        )

    return actions


def generate_launch_description():
    package_share = get_package_share_directory('f1tenth_gym_ros')
    config = os.path.join(package_share, 'config', 'sim.yaml')
    with open(config, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)
    foxglove_config = config_dict.get('foxglove', {})
    if 'ros__parameters' in foxglove_config:
        foxglove_config = foxglove_config['ros__parameters']
    open_foxglove_default = str(foxglove_config.get('open_foxglove', True)).lower()
    foxglove_target_default = str(foxglove_config.get('target', 'browser')).lower()
    map_path_default = str(config_dict['bridge']['ros__parameters']['map_path'])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'map_path',
                default_value=map_path_default,
                description='Map path or built-in track name for the simulator.',
            ),
            DeclareLaunchArgument(
                'foxglove_host',
                default_value='localhost',
                description='Host for foxglove_bridge websocket server.',
            ),
            DeclareLaunchArgument(
                'foxglove_port',
                default_value='8765',
                description='Port for foxglove_bridge websocket server.',
            ),
            DeclareLaunchArgument(
                'foxglove_app_url',
                default_value='https://app.foxglove.dev',
                description='Foxglove app URL to open in a browser.',
            ),
            DeclareLaunchArgument(
                'foxglove_layout',
                default_value=os.path.join(
                    package_share, 'config', 'foxglove', 'gym_bridge_foxglove.json'
                ),
                description='Path to Foxglove layout JSON.',
            ),
            DeclareLaunchArgument(
                'open_foxglove',
                default_value=open_foxglove_default,
                description='Whether to open Foxglove automatically.',
            ),
            DeclareLaunchArgument(
                'foxglove_target',
                default_value=foxglove_target_default,
                description="Where to open Foxglove: 'browser' or 'studio'.",
            ),
            DeclareLaunchArgument(
                'start_foxglove_bridge',
                default_value='true',
                description='Whether to start the foxglove_bridge node.',
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
