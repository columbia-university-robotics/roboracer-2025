#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

#
# This file subscribes to /scan topic and prints out the first 10 values of the LiDAR sensor (each value representing
# distance measurements in meters from the sensor to the nearest surrounding object at a given angle on the LiDAR sensor).
# It also provides a summary of the scan data including the total number of data measurements at a given scan
#
class PrintScan(Node):
    def __init__(self):
        super().__init__('print_scan')
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.count = 0
        self.get_logger().info('PrintScan node started, subscribing to /scan')

    def scan_callback(self, msg: LaserScan) -> None:
        self.count += 1

        raw_ranges = msg.ranges
        n = len(raw_ranges)

        # Extract the first 10 values safely
        first_10 = list(raw_ranges[:10])

        # Log the raw first 10 values
        self.get_logger().info(f"scan #{self.count}: first 10 ranges = {first_10}")

        # Compute summary
        valid_ranges = [r for r in raw_ranges if not (math.isnan(r) or math.isinf(r))]
        if valid_ranges:
            min_val = min(valid_ranges)
            min_idx = raw_ranges.index(min_val)
            self.get_logger().info(
                f"summary: length={n}, valid={len(valid_ranges)}, "
                f"min={min_val:.3f}m @ idx={min_idx}"
            )
        else:
            self.get_logger().warn(
                f"summary: length={n}, no valid ranges"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PrintScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
