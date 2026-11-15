import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

REAL_N = 1081
REAL_INC = 0.004363323096185923


class RealFormatConverter(Node):
    def __init__(self):
        super().__init__('real_format_converter')

        # sim side (remapped by the launch file)
        self.create_subscription(LaserScan, '/sim/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/sim/odom', self.odom_cb, 10)

        # real-car side
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.br = TransformBroadcaster(self)

        # base_link -> laser, same numbers as the real car's launch file
        self.static_br = StaticTransformBroadcaster(self)
        s = TransformStamped()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = 'base_link'
        s.child_frame_id = 'laser'
        s.transform.translation.x = 0.27
        s.transform.translation.z = 0.11
        s.transform.rotation.w = 1.0
        self.static_br.sendTransform(s)

        self.x = self.y = self.yaw = 0.0
        self.last_t = None

    def scan_cb(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if len(ranges) != REAL_N:  # skip if sim.yaml already says 1081
            idx = np.round(np.linspace(0, len(ranges) - 1, REAL_N)).astype(int)
            ranges = ranges[idx]

        out = LaserScan()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'laser'
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = REAL_INC
        out.scan_time = 0.025
        out.time_increment = 0.025 / 1440
        out.range_min = 0.02
        out.range_max = 30.0
        out.ranges = ranges.tolist()
        self.scan_pub.publish(out)

    def odom_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        if abs(v) < 0.05:
            v = 0.0

        if self.last_t is not None:
            dt = t - self.last_t
            if 0.0 < dt < 1.0:
                self.x += v * math.cos(self.yaw) * dt
                self.y += v * math.sin(self.yaw) * dt
                self.yaw += w * dt
        self.last_t = t

        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_link'
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance[0] = 0.2
        out.pose.covariance[7] = 0.2
        out.pose.covariance[35] = 0.4
        out.twist.twist.linear.x = v
        out.twist.twist.angular.z = w
        self.odom_pub.publish(out)

        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.br.sendTransform(tf)


def main():
    rclpy.init()
    rclpy.spin(RealFormatConverter())


if __name__ == '__main__':
    main()
