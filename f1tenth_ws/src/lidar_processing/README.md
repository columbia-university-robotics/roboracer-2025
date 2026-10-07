# Horn mask

Publishes `/scan_filtered` from `/scan`, replacing returns within 12 cm at angles -58° to -45° and 46° to 59° with `NaN`.
These defaults cover the two persistent close clusters in `horn_baseline`; verify them after changes to the sensor or mounting.
Farther returns, other angles, timestamps, intensities, and scan geometry remain unchanged.
The input frame must be `laser`; a different frame produces an error log and an unchanged output scan.

## Build and run

From the hardware workspace (on the robot, `~/Desktop/f1tenth_ws`):

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select lidar_processing
source install/setup.bash
ros2 run lidar_processing horn_mask
```

Leave the live LiDAR driver or bag replay running in another terminal.
In Foxglove, display `/scan` and `/scan_filtered` in different colors using the `laser` frame.
Downstream consumers must explicitly subscribe to `/scan_filtered` to use the mask.
Invalidated rays represent unknown space, not confirmed free space; the mask cannot recover obstacles hidden by the horns.

Override defaults at startup if needed:

```bash
ros2 run lidar_processing horn_mask --ros-args \
  -p max_range_m:=0.12 \
  -p 'right_angles_deg:=[-58.0, -45.0]' \
  -p 'left_angles_deg:=[46.0, 59.0]'
```
