# RoboRacer 2025
The RoboRacer team's 2025 autonomous vehicle code.

## Project Structure

In `src/f1tenth`, you'll find various folders that represent different *packages* used by our car. Some of those packages are provided by RoboRacer and enable the car's basic functionality, such as driving with a joystick and interfacing with the LIDAR. Others, like the localization package, are packages that we have created to serve different functionalities of the car. The idea is that RoboRacer provides a base set of packages that gets your car up and running synchronously (i.e. not autonomously), so that you can focus on doing the autonomous stuff.

The following packages are provided/required by RoboRacer:
* f1tenth_stack
* ackermann_mux
* teleop_tools
* vesc

So far, we have been working on the following packages:
* **localization**: used to create a map of the car's environment using `slam_toolbox`.

## Working With the Code

To build the project, including every package in it, run `colcon build` inside the root directory. `colcon` is the build tool provided by [ROS2](https://docs.ros.org/en/humble/index.html).

Once you've built the project, run a launch file to perform your desired task. For example, to run teleop and drive the car with a controller/joystick, you'd run

```sh
ros2 launch f1tenth_stack bringup_launch.py
```

The general format of a ROS2 launch command is `ros2 launch <package_name> <launch_file>`. In the case of teleop, we're launching the `bringup_launch.py` file in the `f1tenth_stack` package. 

Now, when do you need to run `colcon build`? Each package in the project consists of parameter files (`*.yaml`), launch files (`*launch.py`), and other code files (`*.py`, `*.cpp`). You do not need to run `colcon build` if you change a parameter file - you just need to relaunch your launch file. However, if you change any other file, you need to run `colcon build`. 

## Writing Code

To understand how to write code for the car, you need to understand the basics of ROS2. I could explain that here, but it'd be redundant because it's in the [documentation](https://docs.ros.org/en/humble/Tutorials.html). So, please read through the first two tutorials (CLI tools and client libraries)!
