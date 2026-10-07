from setuptools import setup

setup(
    name='lidar_processing',
    version='0.1.0',
    packages=['lidar_processing'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/lidar_processing']),
        ('share/lidar_processing', ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    maintainer='curc',
    maintainer_email='naowal.ar@gmail.com',
    description='Short-range horn masks for LaserScan messages.',
    license='MIT',
    entry_points={'console_scripts': ['horn_mask = lidar_processing.horn_mask:main']},
)
