# F1TENTH Localization System

This package provides SLAM and localization capabilities for the F1TENTH racing platform.

## Fixed Issues

The following issues have been resolved:
- **Message dropping**: Increased scan queue sizes and improved timing parameters
- **Configuration separation**: Separate configs for mapping vs localization
- **Timing optimization**: Better update rates and transform publishing
- **Queue management**: Improved message handling to prevent drops

## Usage

### 1. Mapping (Create a new map)

To create a new map of your environment:

```bash
ros2 launch localization mapping.launch.py
```

This will:
- Start SLAM toolbox in mapping mode
- Launch the LiDAR driver (URG node)
- Start robot localization (EKF)
- Publish static transforms

**Important**: Drive your robot around the environment to create the map. The map will be saved automatically.

### 2. Localization (Use existing map)

To localize your robot using an existing map:

```bash
ros2 launch localization localization.launch.py
```

This will:
- Start SLAM toolbox in localization mode
- Load the existing map
- Start robot localization (EKF)
- Launch RViz for visualization

### 3. Testing the Setup

To verify that everything is working correctly:

```bash
ros2 run localization test_setup.py
```

This will check:
- Laser scan messages are being received
- Odometry messages are being received
- TF transforms are properly configured

## Configuration Files

### `slam_toolbox.yaml`
- **Mapping mode** configuration
- Optimized for creating new maps
- Higher update rates and larger queues

### `slam_toolbox_localization.yaml`
- **Localization mode** configuration
- Optimized for racing with existing maps
- Faster response times for real-time localization

### `ekf.yaml`
- Robot localization parameters
- Fuses wheel odometry and IMU data
- Optimized timing parameters

## Key Parameters

### Scan Queue Size
- **Before**: 1 (causing message drops)
- **After**: 10 (preventing drops)

### Update Rates
- **Mapping**: 5 Hz (stable mapping)
- **Localization**: 10 Hz (fast response)

### Transform Timeout
- **Before**: Default (too aggressive)
- **After**: 0.2 seconds (more tolerant)

## Troubleshooting

### Still getting message drops?
1. Check if your LiDAR is publishing at the expected rate
2. Verify network connectivity (for network-based LiDARs)
3. Ensure your computer can handle the message rate

### TF errors?
1. Verify the static transform publisher is running
2. Check frame names in your configuration
3. Ensure all required nodes are running

### Poor localization performance?
1. Check if your map quality is good
2. Verify sensor calibration
3. Adjust EKF parameters if needed

## Dependencies

- `slam_toolbox`
- `robot_localization`
- `urg_node` (for Hokuyo LiDAR)
- `tf2_ros`
- `nav2_map_server`

## Notes

- The system automatically switches between mapping and localization modes
- Maps are saved in the `maps/` directory
- RViz configurations are provided for visualization
- All timing parameters have been optimized for F1TENTH racing
