#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


class FollowGapNode(Node):
    def __init__(self):
        super().__init__('follow_gap_node')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('bubble_radius', 0.5)        # meters
        self.declare_parameter('min_range', 0.05)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('max_steering_angle', 0.4)   # radians
        self.declare_parameter('max_speed', 3.0)            # m/s
        self.declare_parameter('min_speed', 0.7)            # m/s

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

        self.get_logger().info(
            f'FollowGapNode started. Listening on {scan_topic}, driving on {drive_topic}'
        )

    def scan_callback(self, scan: LaserScan):
        ranges = list(scan.ranges)

        min_range = self.get_parameter('min_range').get_parameter_value().double_value
        max_range = self.get_parameter('max_range').get_parameter_value().double_value

        # 1. Clip and clean ranges
        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r):
                ranges[i] = max_range
            else:
                ranges[i] = max(min_range, min(r, max_range))

        # 2. Smooth with a simple moving average
        smooth_ranges = self.moving_average(ranges, window_size=5)

        # 3. Create a safety bubble around the closest object
        closest_idx, closest_dist = min(
            enumerate(smooth_ranges),
            key=lambda x: x[1]
        )

        bubble_radius = self.get_parameter('bubble_radius').get_parameter_value().double_value

        if closest_dist <= 0.0:
            self.publish_drive(0.0, 0.0)
            return

        bubble_angle = math.atan2(bubble_radius, closest_dist)
        indices_per_rad = 1.0 / scan.angle_increment
        bubble_indices = int(bubble_angle * indices_per_rad)

        blocked = smooth_ranges[:]  # copy
        n = len(blocked)
        for i in range(-bubble_indices, bubble_indices + 1):
            idx = (closest_idx + i) % n
            blocked[idx] = 0.0  # mark as blocked

        # 4. Find the largest contiguous gap of non-blocked ranges
        gap_start, gap_end = self.find_largest_gap(blocked)

        if gap_start is None:
            self.publish_drive(0.0, 0.0)
            return

        # 5. Choose the best point (middle of the gap)
        best_idx = (gap_start + gap_end) // 2

        # 6. Convert best_idx into steering angle
        angle = scan.angle_min + best_idx * scan.angle_increment
        steering_angle = max(-self.max_steer, min(angle, self.max_steer))

        # 7. Pick speed based on steering (slower on sharp turns)
        speed = self.speed_for_steering(steering_angle)

        self.publish_drive(speed, steering_angle)

    @property
    def max_steer(self):
        return self.get_parameter('max_steering_angle').get_parameter_value().double_value

    @property
    def max_speed(self):
        return self.get_parameter('max_speed').get_parameter_value().double_value

    @property
    def min_speed(self):
        return self.get_parameter('min_speed').get_parameter_value().double_value

    def moving_average(self, data, window_size=5):
        if window_size <= 1:
            return data

        half = window_size // 2
        smoothed = []
        n = len(data)

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
        return smoothed

    def find_largest_gap(self, ranges):
        best_start = best_end = None
        cur_start = None

        for i, r in enumerate(ranges):
            if r > 0.0:
                if cur_start is None:
                    cur_start = i
            else:
                if cur_start is not None:
                    cur_end = i - 1
                    if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                        best_start, best_end = cur_start, cur_end
                    cur_start = None

        if cur_start is not None:
            cur_end = len(ranges) - 1
            if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end

        return best_start, best_end

    def speed_for_steering(self, steering_angle):
        if self.max_steer <= 0.0:
            return self.min_speed

        steering_norm = abs(steering_angle) / self.max_steer
        steering_norm = max(0.0, min(1.0, steering_norm))

        return self.max_speed - steering_norm * (self.max_speed - self.min_speed)

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FollowGapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
