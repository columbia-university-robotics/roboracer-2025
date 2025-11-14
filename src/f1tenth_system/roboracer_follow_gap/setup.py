from setuptools import setup

package_name = 'roboracer_follow_gap'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/follow_gap.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hanse',
    maintainer_email='hje2113@columbia.edu',
    description='Follow-The-Gap autonomous driving controller for RoboRacer.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follow_gap = roboracer_follow_gap.follow_gap_node:main',
        ],
    },
)
