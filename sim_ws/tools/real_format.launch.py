import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetRemap


def generate_launch_description():
    here = os.path.dirname(os.path.realpath(__file__))
    sim_launch = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'launch', 'gym_bridge_launch.py')   # verify this name, see below

    sim = GroupAction([
        # Remaps apply only inside this group, so the converter is unaffected
        SetRemap(src='/scan', dst='/sim/scan'),
        SetRemap(src='/ego_racecar/odom', dst='/sim/odom'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim_launch)),
    ])

    converter = ExecuteProcess(
        cmd=['python3', os.path.join(here, 'real_format_converter.py')],
        output='screen')

    return LaunchDescription([sim, converter])
