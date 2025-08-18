from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction
import os

def generate_launch_description():
	# Use localization-specific parameters
	pkg_localization = get_package_share_directory('localization')
	params_file = os.path.join(pkg_localization, 'params', 'slam_toolbox_localization.yaml')
	ekf_file = os.path.join(pkg_localization, 'params', 'ekf.yaml')
	
	# Map path configuration
	map_path_no_ext = PathJoinSubstitution([FindPackageShare('localization'), 'maps', 'map'])
	map_path_ext = f"{map_path_no_ext}.yaml"

	slam_node = Node(
		package='slam_toolbox', 
		executable='localization_slam_toolbox_node',
		name='slam_toolbox', 
		output='screen', 
		parameters=[params_file, {'map_file_name': map_path_no_ext, 'use_sim_time': False, 'tf_buffer_duration': 10.0}]
	)

	robot_localization_node = Node(
		package='robot_localization', 
		executable='ekf_node',
		name='ekf_filter_node', 
		output='screen', 
		parameters=[ekf_file, {'use_sim_time': False}]
	)

	# Lidar driver
	pkg_f1tenth_stack = FindPackageShare('f1tenth_stack')
	sensors_config = PathJoinSubstitution([pkg_f1tenth_stack, 'config', 'sensors.yaml'])
	lidar_node = Node(
		package='urg_node',
		executable='urg_node_driver',
		name='urg_node',
		output='screen',
		parameters=[sensors_config, {'calibrate_time': True}],
		remappings=[('scan', '/scan')]
	)

	# Load pre-built map
	nav2_node = Node( 
		package='nav2_map_server', 
		executable='map_server',
		name='map_server', 
		output='screen',
		parameters=[{'yaml_filename': map_path_ext, 'use_sim_time': False}]
	) 
	
	# RViz for visualization (use config if available, else default)
	rviz_config = os.path.join(pkg_localization, 'rviz', 'localization.rviz')
	if os.path.exists(rviz_config):
		rviz_args = ['-d', rviz_config]
	else:
		rviz_args = []
	rviz_node = Node(
		package='rviz2', 
		executable='rviz2', 
		arguments=rviz_args)

	# Delay SLAM by 2 seconds so TF and sensors are up
	delayed_slam = TimerAction(period=2.0, actions=[slam_node])

	return LaunchDescription([lidar_node, robot_localization_node, nav2_node, rviz_node, delayed_slam])

