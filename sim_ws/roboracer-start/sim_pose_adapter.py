# Adapted directly from Software_Setup_Guidance, Section 7.
# Run inside the initialized Ubuntu ROS 2 Humble environment.
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

class SimPoseAdapter(Node):
    def __init__(self):
        super().__init__('sim_pose_adapter')
        self.warned = False
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/pose', 10)
        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.convert, 10)

    def convert(self, msg):
        if msg.header.frame_id != 'map':
            if not self.warned:
                self.get_logger().error(
                    'Expected map-frame odometry; check simulator.')
                self.warned = True
            return
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose = msg.pose
        self.pub.publish(out)

def main():
    rclpy.init()
    node = SimPoseAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
