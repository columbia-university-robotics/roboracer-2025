# Drive and build a live map

This guide starts the car, connects a PS4 controller, runs SLAM Toolbox, and displays the live map in Foxglove.
Use the current robot IP from the gist linked in [SSH access](ssh-access.md).

## 1. Connect the PS4 controller to the robot

Pair the controller to the robot's Bluetooth adapter, not to your laptop.
In an SSH terminal on the robot, run:

```bash
bluetoothctl
```

At the Bluetooth prompt, enter:

```text
power on
agent on
default-agent
scan on
```

Hold **SHARE + PS** until the controller light flashes rapidly.
When `Wireless Controller` appears, note its MAC address, then run:

```text
pair CONTROLLER_MAC
trust CONTROLLER_MAC
connect CONTROLLER_MAC
scan off
quit
```

Replace `CONTROLLER_MAC` with the address shown by the scan.
If pairing says `AlreadyExists`, run `trust` and `connect` for that address instead.
Verify the connection and Linux joystick device:

```bash
bluetoothctl info CONTROLLER_MAC
ls -l /dev/input/js*
```

The Bluetooth info should say `Connected: yes`, and `/dev/input/js0` should exist.
If needed, verify Linux receives stick/button changes with `sudo jstest /dev/input/js0` and stop it with Ctrl+C.

## 2. Start the hardware bringup once

Do not run a second bringup alongside an existing one; both may try to use the VESC.
Stop any old bringup with Ctrl+C, then use robot SSH terminal 0:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

Leave it running.
Confirm ROS sees the controller from another sourced robot terminal:

```bash
ros2 run joy joy_enumerate_devices
ros2 topic echo /joy
```

Move a stick and press buttons; `/joy` messages should change.
Stop the echo with Ctrl+C.
On this car, the launch-started joystick node did not publish even though Linux could read the controller.
Running `ros2 run joy joy_node --ros-args -p device_id:=0` made `/joy` work.
If you need that workaround, run it in its own sourced terminal and check `ros2 topic info /joy` for multiple publishers.
Avoid leaving duplicate joystick publishers active once input is working.

If the enumerator stays empty, check Bluetooth is connected and `js0` exists before troubleshooting ROS.

## 3. Initialize odometry before mapping

Without a controller input, odometry may wait for its first steering command.
In another robot SSH terminal, send the configured steering-center value.
This sets steering and sends no throttle command:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 topic pub --once /commands/servo/position \
  std_msgs/msg/Float64 "{data: 0.4564}"
timeout 5s ros2 run tf2_ros tf2_echo odom base_link
```

Confirm `tf2_echo` prints transforms; zero translation and rotation are expected while stationary.
The steering-center value is specific to this car and may need rechecking after VESC calibration changes.

## 4. Start mapping

In robot SSH terminal 2, start SLAM Toolbox:

```bash
cd ~/Desktop/f1tenth_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch localization mapping.launch.py mode:=mapping
```

For Foxglove-only viewing, append `enable_rviz:=false` to avoid starting RViz.
This launch also starts Nav2 navigation and the localization pose relay.
Leave it running and allow about 10 seconds for the initial map.

## 5. Connect Foxglove

Start the bridge in another robot SSH terminal unless it is already running:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

In a local WSL terminal on your laptop, replace `ROBOT_IP` with the current robot IP:

```bash
ssh -N -L 8765:localhost:8765 curc@ROBOT_IP
```

Leave the bridge and tunnel terminals open.
In [Foxglove](https://app.foxglove.dev), open a Foxglove WebSocket connection to `ws://localhost:8765`.
Add a 3D panel, set its display frame to `map`, and enable `/map` and `/scan`.
Use a top-down view and zoom out to see the surrounding walls.

## 6. Drive slowly to extend the map

Keep the car on the ground in a clear area, and be ready to stop it.
The repository's [controller configuration](../f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/joy_teleop.yaml) uses zero-based indices:

| Function | Input | Command range |
| --- | --- | --- |
| Hold to enable driving | `buttons[4]` | Enabled while held |
| Forward/reverse | `axes[1]` | Up to ±2 m/s |
| Steering | `axes[2]` | Up to ±0.34 rad |

Check `/joy` and verify the physical button and stick indices before moving.
Use small inputs and drive slowly; release the enable button or center controls to stop.
Do not assume `buttons[4]` is L1 because button ordering depends on the controller.

SLAM Toolbox uses LiDAR scans plus odometry/TF to place them in the map.
The current config sets `minimum_travel_distance: 0.5` m and `minimum_travel_heading: 0.5` rad, so drive at least that far or turn enough for SLAM to accept additional scans.
`map_update_interval: 5.0` controls map publication cadence; a fresh timestamp alone does not mean occupancy cells changed.

## If the map frame or map is missing

The key transforms are `odom → base_link` from VESC odometry, `base_link → laser` from the static transform, and `map → odom` from SLAM Toolbox.
Check the nodes, scan, and transforms from a sourced robot terminal:

```bash
ros2 node list | grep -E 'slam_toolbox|vesc_to_odom|urg_node'
ros2 topic hz /scan
ros2 param get /vesc_to_odom_node publish_tf
timeout 5s ros2 run tf2_ros tf2_echo odom base_link
timeout 5s ros2 run tf2_ros tf2_echo base_link laser
```

If `odom → base_link` is missing, initialize odometry as in step 3.
`Message Filter dropping ... queue is full` alongside an absent `odom` frame means scans are waiting for the transform.
Keep bringup running, Ctrl+C only the mapping launch, initialize odometry, then restart mapping.
Do not start another bringup to fix it.

Check that `/map` has data using transient-local QoS:

```bash
timeout 10s ros2 topic echo /map --once --field info \
  --qos-durability transient_local --qos-reliability reliable
```

If `/map` exists but Foxglove has no `map` frame, refresh the connection after confirming `map → odom` is present in `/tf`.

## Save the map

Before stopping SLAM, run in another sourced robot terminal:

```bash
mkdir -p ~/maps
ros2 run nav2_map_server map_saver_cli -f ~/maps/my_map
```

This saves the occupancy map image and YAML for later localization, not the SLAM pose graph for continuing a mapping session.
Choose a new filename for each map you want to keep.
