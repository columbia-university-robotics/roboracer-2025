import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PurePursuitFollower(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit_follower")

        self.declare_parameter("pose_topic", "/localization/pose")
        self.declare_parameter("path_topic", "/planner/path")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("lookahead_distance", 1.2)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("max_steering_angle", 0.4189)
        self.declare_parameter("desired_speed", 1.5)
        self.declare_parameter("min_speed", 0.8)
        self.declare_parameter("max_speed", 2.5)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("stop_at_goal", True)

        self.current_pose = None
        self.current_yaw = 0.0
        self.current_path = []
        self.target_index = 0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("drive_topic").value, 10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Path,
            self.get_parameter("path_topic").value,
            self.path_callback,
            latched_qos,
        )

        rate_hz = float(self.get_parameter("control_rate").value)
        self.timer = self.create_timer(1.0 / rate_hz, self.control_callback)
        self.get_logger().info(
            "Follower ready on pose=%s path=%s drive=%s"
            % (
                self.get_parameter("pose_topic").value,
                self.get_parameter("path_topic").value,
                self.get_parameter("drive_topic").value,
            )
        )

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.current_pose = msg.pose.pose.position
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def path_callback(self, msg: Path) -> None:
        self.current_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.target_index = 0
        self.get_logger().info(f"Received new path with {len(self.current_path)} poses.")

    def control_callback(self) -> None:
        if self.current_pose is None or not self.current_path:
            return

        current_xy = (self.current_pose.x, self.current_pose.y)
        goal_xy = self.current_path[-1]
        goal_distance = math.dist(current_xy, goal_xy)
        if goal_distance <= float(self.get_parameter("goal_tolerance").value):
            if bool(self.get_parameter("stop_at_goal").value):
                self.publish_drive(0.0, 0.0)
            return

        target_index = self.find_target_index(current_xy)
        self.target_index = target_index
        target_x, target_y = self.current_path[target_index]

        heading_to_target = math.atan2(target_y - current_xy[1], target_x - current_xy[0])
        alpha = normalize_angle(heading_to_target - self.current_yaw)
        lookahead = max(
            float(self.get_parameter("lookahead_distance").value),
            math.dist(current_xy, (target_x, target_y)),
        )
        wheelbase = float(self.get_parameter("wheelbase").value)
        steering = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
        max_steering = float(self.get_parameter("max_steering_angle").value)
        steering = max(-max_steering, min(max_steering, steering))

        desired_speed = float(self.get_parameter("desired_speed").value)
        min_speed = float(self.get_parameter("min_speed").value)
        max_speed = float(self.get_parameter("max_speed").value)
        curvature_ratio = min(1.0, abs(steering) / max_steering) if max_steering > 0.0 else 0.0
        speed = desired_speed - curvature_ratio * (desired_speed - min_speed)
        speed = max(min_speed, min(max_speed, speed))
        if goal_distance < 1.0:
            speed = min(speed, min_speed)

        self.publish_drive(steering, speed)

    def find_target_index(self, current_xy: tuple[float, float]) -> int:
        lookahead = float(self.get_parameter("lookahead_distance").value)
        nearest_index = min(
            range(len(self.current_path)),
            key=lambda i: math.dist(current_xy, self.current_path[i]),
        )

        target_index = nearest_index
        while (
            target_index + 1 < len(self.current_path)
            and math.dist(current_xy, self.current_path[target_index]) < lookahead
        ):
            target_index += 1

        return target_index

    def publish_drive(self, steering: float, speed: float) -> None:
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        self.drive_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
