from glob import glob
import os

from setuptools import setup


package_name = "planner_web_ui"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "static"), glob("static/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="curc",
    maintainer_email="curc@local",
    description="Lightweight browser UI and launch glue for path planning workflows.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_web_ui = planner_web_ui.web_server:main",
        ],
    },
)
