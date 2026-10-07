import math

from sensor_msgs.msg import LaserScan
from rclpy.serialization import serialize_message
from lidar_processing.horn_mask import mask_scan


def test_masks_close_horns_preserving_obstacles_and_metadata():
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp.sec = 123
    scan.angle_min = math.radians(-60)
    scan.angle_increment = math.radians(10)
    scan.angle_max = math.radians(60)
    scan.range_min = 0.02
    scan.range_max = 30.0
    scan.ranges = [1.0] * 13
    scan.ranges[1] = 0.05  # Right horn at -50 degrees.
    scan.ranges[11] = 0.06  # Left horn at +50 degrees.
    scan.ranges[6] = 0.05  # Nearby real obstacle outside horn sectors.
    scan.ranges[3] = float('inf')
    scan.ranges[4] = float('nan')
    scan.intensities = [10.0] * 13
    result = mask_scan(scan, [(-58, -45), (46, 59)], 0.12)
    assert math.isnan(result.ranges[1]) and math.isnan(result.ranges[11])
    assert result.ranges[6] == scan.ranges[6]
    assert math.isinf(result.ranges[3]) and math.isnan(result.ranges[4])
    assert scan.ranges[1] > 0  # Input remains untouched.
    result.ranges = scan.ranges
    assert serialize_message(result) == serialize_message(scan)
    scan.ranges[1] = 0.5
    scan.ranges[11] = 0.5
    result = mask_scan(scan, [(-58, -45), (46, 59)], 0.12)
    assert result.ranges[1] == result.ranges[11] == 0.5
