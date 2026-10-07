"""Invalidate short-range horn returns without changing raw scans or geometry."""
from copy import deepcopy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def mask_scan(scan, sectors, max_range):
    """Return a separate scan with valid returns inside the sectors set to NaN."""
    result = deepcopy(scan)
    for i, distance in enumerate(scan.ranges):
        if not (math.isfinite(distance) and scan.range_min <= distance <= scan.range_max
                and distance <= max_range):
            continue
        angle = math.degrees(scan.angle_min + i * scan.angle_increment)
        if any(low <= angle <= high for low, high in sectors):
            result.ranges[i] = float('nan')
    return result


class HornMask(Node):
    def __init__(self):
        super().__init__('horn_mask')
        self.declare_parameter('expected_frame', 'laser')
        self.declare_parameter('max_range_m', 0.12)
        self.declare_parameter('right_angles_deg', [-58.0, -45.0])
        self.declare_parameter('left_angles_deg', [46.0, 59.0])
        self.frame = self.get_parameter('expected_frame').value
        self.max_range = self.get_parameter('max_range_m').value
        self.sectors = [self.get_parameter(name).value for name in
                        ('right_angles_deg', 'left_angles_deg')]
        if not self.frame or not math.isfinite(self.max_range) or self.max_range <= 0:
            raise ValueError('Expected frame must be nonempty and max_range_m positive and finite')
        for sector in self.sectors:
            if (len(sector) != 2 or not all(math.isfinite(v) for v in sector)
                    or not -180 <= sector[0] < sector[1] <= 180):
                raise ValueError('Each angular sector must be two increasing degrees within [-180, 180]')
        self.publisher = self.create_publisher(LaserScan, 'scan_filtered', qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.get_logger().info(f'Masking {self.sectors} degrees within {self.max_range} m in {self.frame}')

    def on_scan(self, scan):
        if scan.header.frame_id != self.frame:
            self.get_logger().error(
                f'Expected frame {self.frame!r}, got {scan.header.frame_id!r}; passing scan unchanged',
                throttle_duration_sec=5.0)
            self.publisher.publish(scan)
            return
        self.publisher.publish(mask_scan(scan, self.sectors, self.max_range))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HornMask()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
