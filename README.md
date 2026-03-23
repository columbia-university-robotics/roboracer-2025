# RoboRacer 2025

Autonomy code for the Columbia University Robotics Club F1TENTH platform.

## Repository layout

This repo has two ROS 2 workspaces:

- `f1tenth_ws`: hardware workspace for the real car
- `sim_ws`: simulation workspace for `f1tenth_gym_ros`

The guiding rule is:

- hardware- and simulator-specific packages live in their own workspace
- shared team-owned autonomy packages live once in `f1tenth_ws/src`
- `sim_ws/src` should symlink shared packages instead of duplicating them

Today that means:

```text
f1tenth_ws/src/
  f1tenth_system/   # vendor, driver, bringup, teleop, and hardware stack
  localization/     # mapping and map-frame localization output
  f1tenth_planning/ # shared planner and path follower

sim_ws/src/
  f1tenth_gym_ros/  # simulator bridge and simulator-specific launch/config
  f1tenth_planning  # symlink to ../../f1tenth_ws/src/f1tenth_planning
```

## Package philosophy

`f1tenth_system` is the low-level system stack. It owns packages such as:

- `f1tenth_stack`
- `ackermann_mux`
- `teleop_tools`
- `vesc`

These packages bring up the real car, publish raw sensors, and accept low-level drive commands.

`localization` and `f1tenth_planning` are team-owned autonomy packages. They sit beside `f1tenth_system` because they are consumers of the hardware stack, not part of the hardware stack itself.

## Standard autonomy interfaces

The main autonomy-facing pose contract is:

- `/localization/pose` as `geometry_msgs/PoseWithCovarianceStamped`

Both real and simulated flows publish that same topic:

- real car:
  - SLAM Toolbox mapping mode publishes `/pose`, which `localization` relays to `/localization/pose`
  - AMCL localization mode publishes `/amcl_pose`, which `localization` relays to `/localization/pose`
- simulation:
  - `f1tenth_gym_ros` publishes the exact ground-truth map pose directly to `/localization/pose`

`f1tenth_planning` subscribes to `/localization/pose` in both workspaces, so planning code does not need separate real/sim pose topic names anymore.

Lower-level motion topics still exist and are intentionally separate from the autonomy pose contract:

- real car: `/odom`
- sim: `/ego_racecar/odom`

## Build

Build each workspace from its own root:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
colcon build
```

```bash
cd ~/Desktop/sim_ws
source /opt/ros/humble/setup.bash
colcon build
```

If you change Python or C++ code, rebuild the affected workspace. If you only change launch or YAML files, relaunching is usually enough.

## Common workflows

Bring up the real car stack:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

Run mapping on the real car:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch localization mapping.launch.py mode:=mapping
```

Run map-based localization on the real car:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch localization mapping.launch.py mode:=localization map_file:=maps/map_1761949489.yaml
```

Run the simulator:

```bash
cd ~/Desktop/sim_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

Run planning in either workspace:

```bash
ros2 launch f1tenth_planning planning.launch.py
```

The planner package includes both `config/real.yaml` and `config/sim.yaml`. Both consume `/localization/pose`, so the same launch flow works in either workspace unless you explicitly override `config_file`.
