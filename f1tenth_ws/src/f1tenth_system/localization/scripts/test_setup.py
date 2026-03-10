#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import time

class LocalizationTestNode(Node):
    def __init__(self):
        super().__init__('localization_test_node')
        
        # Subscribe to topics
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info('Localization test node started')
        self.scan_count = 0
        self.odom_count = 0
        
    def scan_callback(self, msg):
        self.scan_count += 1
        if self.scan_count % 10 == 0:  # Log every 10th message
            self.get_logger().info(f'Received scan message #{self.scan_count}, '
                                 f'frame: {msg.header.frame_id}, '
                                 f'stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
    
    def odom_callback(self, msg):
        self.odom_count += 1
        if self.odom_count % 10 == 0:  # Log every 10th message
            self.get_logger().info(f'Received odom message #{self.odom_count}, '
                                 f'frame: {msg.header.frame_id}, '
                                 f'stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
    
    def check_tf(self):
        try:
            # Check if transform from base_link to laser exists
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'laser', rclpy.time.Time())
            self.get_logger().info(f'TF transform found: base_link -> laser')
            return True
        except Exception as e:
            self.get_logger().warn(f'TF transform not found: {e}')
            return False

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationTestNode()
    
    try:
        # Wait a bit for topics to start
        time.sleep(2.0)
        
        # Check TF
        node.check_tf()
        
        # Spin for a while to collect data
        rclpy.spin_once(node, timeout_sec=5.0)
        
        node.get_logger().info(f'Test completed. Received {node.scan_count} scan messages, {node.odom_count} odom messages')
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
