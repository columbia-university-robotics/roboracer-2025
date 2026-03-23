#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


class PoseRelay(Node):
    def __init__(self) -> None:
        super().__init__("pose_relay")

        self.declare_parameter("input_topic", "/pose")
        self.declare_parameter("output_topic", "/localization/pose")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.publisher = self.create_publisher(PoseWithCovarianceStamped, output_topic, 10)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            input_topic,
            self.pose_callback,
            10,
        )

        self.get_logger().info(
            f"Relaying PoseWithCovarianceStamped from {input_topic} to {output_topic}"
        )

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
