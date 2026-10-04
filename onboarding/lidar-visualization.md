# View Hokuyo UST-10LX LiDAR

## 1. Start the LiDAR on the robot

In an SSH terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ping -c 3 -W 2 192.168.0.10

ros2 run urg_node urg_node_driver --ros-args \
  -r __node:=urg_node \
  -p ip_address:=192.168.0.10 \
  -p ip_port:=10940 \
  -p laser_frame_id:=laser \
  -p cluster:=1 \
  -p skip:=0
```

Leave it running.

## 2. Start the viewer bridge on the robot

In another SSH terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/Desktop/f1tenth_ws/install/setup.bash
ros2 topic echo /scan --once --field header --qos-reliability best_effort
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

If the bridge package is missing, install it and retry:

```bash
sudo apt update
sudo apt install ros-humble-foxglove-bridge
```

Leave the bridge running.

## 3. Connect from your laptop

Run locally, replacing `ROBOT_IP` with the robot's SSH address:

```bash
ssh -N -L 8765:localhost:8765 curc@ROBOT_IP
```

Leave the tunnel running.

1. Open [Foxglove](https://app.foxglove.dev).
2. Select **Open connection → Foxglove WebSocket**.
3. Connect to `ws://localhost:8765`.
4. Add a **3D panel** and set its **display frame** to `laser`.
5. Enable `/scan`, select a top-down view, and adjust the zoom.

Keep the robot stationary and move an object in front of the sensor to verify the live scan.
