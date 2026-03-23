# Localization

This package owns map building and map-frame pose publication for the real car.

## Inputs

- `/scan` from the LiDAR bringup stack
- `/odom` from the driver stack

## Output

- `/localization/pose` as the standard `PoseWithCovarianceStamped` topic used by autonomy code

The package normalizes two backends to that single output:

- Mapping mode: SLAM Toolbox publishes `/pose`, which is relayed to `/localization/pose`
- Localization mode: AMCL publishes `/amcl_pose`, which is relayed to `/localization/pose`

## Launch

### Mapping

```bash
ros2 launch localization mapping.launch.py mode:=mapping
```

This starts SLAM Toolbox, RViz, Nav2 navigation, and the pose relay.

### Localization

```bash
ros2 launch localization mapping.launch.py mode:=localization map_file:=maps/map_1761949489.yaml
```

`map_file` accepts either an absolute path or a path relative to this package.

This starts `nav2_map_server`, `nav2_amcl`, RViz, and the pose relay.

## Utilities

```bash
ros2 run localization test_setup.py
ros2 run localization print_scan.py
```

`test_setup.py` checks that scans, odometry, TF, and `/localization/pose` are all present.
