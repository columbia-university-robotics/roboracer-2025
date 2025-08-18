from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction
import os

def generate_launch_description():
	# Get paths
	pkg_localization = get_package_share_directory('localization')
	params_file = os.path.join(pkg_localization, 'params', 'slam_toolbox_mapping.yaml')

	# Dynamically load sensors.yaml from f1tenth_stack
	pkg_f1tenth_stack = FindPackageShare('f1tenth_stack')
	sensors_config = PathJoinSubstitution([pkg_f1tenth_stack, 'config', 'sensors.yaml'])

	slam_node = Node(
		package='slam_toolbox',
		executable='async_slam_toolbox_node',
		name='slam_toolbox',
		output='screen',
		parameters=[params_file, {'use_sim_time': False}],
		remappings=[('scan', '/scan')]
	)

	robot_localization_node = Node(
		package='robot_localization',
		executable='ekf_node',
		name='ekf_filter_node',
		output='screen',
		parameters=[os.path.join(pkg_localization, 'params', 'ekf.yaml'), {'use_sim_time': False}]
	)

	lidar_node = Node(
		package='urg_node',
		executable='urg_node_driver',
		name='urg_node',
		output='screen',
		parameters=[sensors_config, {'calibrate_time': True}],  # ensure timestamps synchronized
		remappings=[('scan', '/scan')]  # Ensure slam_toolbox can subscribe
	)

	# Static transform (base_link to laser; adjust offsets based on your robot)
	static_tf_node = Node(
		package='tf2_ros',
		executable='static_transform_publisher',
		name='base_to_laser_tf',
		arguments=['0.1', '0', '0.2', '0', '0', '0', 'base_link', 'laser']  # x, y, z, yaw, pitch, roll
	)

	# RViz for visualization (no saved config available)
	rviz_node = Node(
		package='rviz2',
		executable='rviz2',
		arguments=[]
	)

	# Delay SLAM by 2 seconds so TF and sensors are up
	delayed_slam = TimerAction(period=2.0, actions=[slam_node])

	return LaunchDescription([robot_localization_node, lidar_node, static_tf_node, rviz_node, delayed_slam])

