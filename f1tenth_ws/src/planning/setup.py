from glob import glob
import os

from setuptools import setup


package_name = "planning"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="curc",
    maintainer_email="curc@local",
    description="Shared planning and waypoint following for F1TENTH.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "occupancy_grid_planner = planning.occupancy_grid_planner:main",
            "pure_pursuit_follower = planning.pure_pursuit_follower:main",
        ],
    },
)
