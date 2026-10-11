# Run mapping on the robot

Build a new map from raw `/scan` using SLAM Toolbox and view it in Foxglove.
Use the current robot IP from the gist linked in [SSH access](ssh-access.md).
Stop any existing AMCL/localization launch and standalone LiDAR driver before starting.

## 1. Start the hardware stack

In robot SSH terminal 0:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

Leave it running; do not start a second bringup instance.

## 2. Initialize odometry

Without a controller, the odometry node may wait indefinitely for its first steering command.
In another robot SSH terminal, send the configured steering-center value.
This moves the front wheels toward straight ahead but sends no throttle command.

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash

ros2 topic pub --once /commands/servo/position \
  std_msgs/msg/Float64 "{data: 0.4564}"

timeout 5s ros2 run tf2_ros tf2_echo odom base_link
```

Confirm it prints transforms before starting SLAM.
Zero translation and rotation are expected while stationary.
An initial waiting message followed by transforms is normal.
`0.4564` is this robot's configured steering center; recheck it if the VESC calibration changes.

## 3. Start SLAM Toolbox

In robot SSH terminal 2:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch localization mapping.launch.py mode:=mapping
```

This is the command verified on the car after initializing odometry and restarting mapping.
For Foxglove-only viewing, append `enable_rviz:=false` to avoid launching RViz.
This launch also starts Nav2 navigation and the localization pose relay.
Leave it running and allow about 10 seconds for the first map.

## 4. Connect Foxglove

In another robot SSH terminal, start the bridge unless it is already running:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

In a local WSL terminal on your laptop, replace `ROBOT_IP`:

```bash
ssh -N -L 8765:localhost:8765 curc@ROBOT_IP
```

Leave both terminals open.

1. Open [Foxglove](https://app.foxglove.dev).
2. Choose **Open connection → Foxglove WebSocket** and connect to `ws://localhost:8765`.
3. Add a **3D panel**, set **Display frame** to `map`, and enable `/map` and `/scan`.
4. Use a top-down view and zoom out to see the walls.

A stationary robot can show its initial surroundings.
To extend the map, drive slowly with the controller using [controller driving](controller-driving.md).
The configured map update interval is about five seconds.

## If no map appears

Repeated `queue is full` messages together with `odom ... frame does not exist` mean scans are waiting for a missing transform.
Complete step 2, then Ctrl+C only the mapping launch and rerun step 3.
Keep bringup running so the initialized odometry node stays alive.
This resolved the failure on our car.

Check map delivery from a sourced robot terminal:

```bash
timeout 10s ros2 topic echo /map --once --field info \
  --qos-durability transient_local --qos-reliability reliable
```

A publisher count of one does not prove a map has been generated.
If transforms are still missing, check the inputs and bringup errors:

```bash
ros2 param get /vesc_to_odom_node publish_tf
timeout 5s ros2 topic echo /sensors/core --once
timeout 5s ros2 topic echo /sensors/servo_position_command --once
```

Steering feedback is not latched; a late subscriber may see nothing after a one-shot command.
Use the `odom` transform check to confirm initialization succeeded.
If `vesc_msgs/msg/VescStateStamped` is reported as invalid, source the hardware workspace in that terminal and retry.
Do not substitute a static `odom` transform for real odometry.

## Save the map

Before stopping SLAM, run in another sourced robot terminal:

```bash
mkdir -p ~/maps
ros2 run nav2_map_server map_saver_cli -f ~/maps/my_map
```

This saves the occupancy map image and YAML for later localization, not the SLAM pose graph for continuing a mapping session.
Choose a new filename for each map you want to keep.
