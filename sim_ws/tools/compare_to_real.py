import math
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


def load_ref(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    ref = {}
    while r.has_next() and len(ref) < 2:
        topic, data, _ = r.read_next()
        if topic in ('/scan', '/odom') and topic not in ref:
            ref[topic] = deserialize_message(data, get_message(types[topic]))
    return ref


def scan_fields(m):
    return {
        'frame_id': m.header.frame_id, 'angle_min': m.angle_min,
        'angle_max': m.angle_max, 'angle_increment': m.angle_increment,
        'n_ranges': len(m.ranges), 'scan_time': m.scan_time,
        'range_min': m.range_min, 'range_max': m.range_max,
        'intensities_len': len(m.intensities),
        'time_increment_nonzero': m.time_increment > 0,
    }


def odom_fields(m):
    c, t = m.pose.covariance, m.twist.covariance
    return {
        'frame_id': m.header.frame_id, 'child_frame_id': m.child_frame_id,
        'pose_cov_diag': (c[0], c[7], c[35]),
        'twist_cov_zero': not any(t),
        'linear_y': m.twist.twist.linear.y,
        'yaw_only_quat': m.pose.pose.orientation.x == 0 and m.pose.pose.orientation.y == 0,
    }


def same(a, b):
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-9)
    return a == b


class Compare(Node):
    def __init__(self, ref):
        super().__init__('compare_to_real')
        self.ref, self.done = ref, set()
        self.create_subscription(LaserScan, '/scan', lambda m: self.check('/scan', m, scan_fields), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', lambda m: self.check('/odom', m, odom_fields), 10)

    def check(self, topic, msg, fields):
        if topic in self.done:
            return
        self.done.add(topic)
        want, got = fields(self.ref[topic]), fields(msg)
        print(f'\n{topic}')
        for k in want:
            ok = same(want[k], got[k])
            print(f"  {'OK  ' if ok else 'DIFF'} {k}: real={want[k]}  sim={got[k]}")
        if len(self.done) == 2:
            raise SystemExit


def main():
    rclpy.init()
    try:
        rclpy.spin(Compare(load_ref(sys.argv[1])))
    except SystemExit:
        pass


main()
