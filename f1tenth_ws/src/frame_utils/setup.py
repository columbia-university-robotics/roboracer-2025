from setuptools import setup


package_name = "frame_utils"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="curc",
    maintainer_email="curc@local",
    description="Reusable Python helpers for transforming ROS poses and paths.",
    license="MIT",
)
