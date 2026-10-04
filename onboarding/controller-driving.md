# Drive with a controller

## 1. Connect the controller

Connect a PS4 controller to the robot with a USB data cable, or pair it with the robot over Bluetooth.
Run these commands over SSH on the robot:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 run joy joy_enumerate_devices
```

The current configuration selects device ID `0`.

## 2. Start manual control

For the first test, raise the drive wheels off the ground and stop any autonomous driving nodes.
Run only one hardware bringup instance:

```bash
ros2 launch f1tenth_stack no_lidar_bringup_launch.py
```

This starts the controller and motor stack and can run alongside the standalone LiDAR driver in [LiDAR visualization](lidar-visualization.md).
Leave it running.

## 3. Verify controls, then drive

In another SSH terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 topic echo /joy
```

The repository's [controller configuration](../f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/joy_teleop.yaml) uses zero-based indices:

| Function | Input | Command range |
| --- | --- | --- |
| Hold to enable driving | `buttons[4]` | Enabled while held |
| Forward/reverse | `axes[1]` | Up to ±2 m/s |
| Steering | `axes[2]` | Up to ±0.34 rad |

Verify which physical buttons and sticks change those entries before driving.
Do not assume button `4` is L1: [`joy_node` mappings depend on the controller](https://index.ros.org/p/joy/).

Stop the echo with Ctrl+C, then inspect drive commands:

```bash
ros2 topic echo /teleop
```

Check forward/reverse and steering with small stick movements while holding the enable button.
Center the sticks and release all buttons; verify `drive.speed` returns to `0.0` and the wheels stop before placing the robot on the ground.
To finish, stop the robot this way before pressing Ctrl+C in the bringup terminal.

## If the mapping differs

Copy the installed configuration:

```bash
cp "$(ros2 pkg prefix --share f1tenth_stack)/config/joy_teleop.yaml" ~/joy-teleop.yaml
nano ~/joy-teleop.yaml
```

Adjust `device_id`, `human_control.deadman_buttons`, and the speed/steering axis indices to match `/joy`.
For initial driving, reduce the `human_control` speed scale from `2.0` to `0.3`.
Stop the existing bringup and relaunch with your file:

```bash
ros2 launch f1tenth_stack no_lidar_bringup_launch.py joy_config:=$HOME/joy-teleop.yaml
```
