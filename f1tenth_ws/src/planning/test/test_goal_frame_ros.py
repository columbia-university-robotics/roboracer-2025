"""Run with unittest discovery in an initialized ROS 2 Humble workspace."""
import math
from types import SimpleNamespace

import unittest

try:
    import rclpy
except ImportError:
    raise unittest.SkipTest("ROS 2 Humble environment required")
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformException
from planning.occupancy_grid_planner import OccupancyGridPlanner


class RealTFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_real_tf_rotation_translation_and_timestamp(self):
        buffer = Buffer()
        for sec, x in [(5, 10.), (6, 100.)]:
            transform = TransformStamped()
            transform.header.frame_id = 'map'
            transform.child_frame_id = 'ego_racecar/base_link'
            transform.header.stamp.sec = sec
            transform.transform.translation.x = x
            transform.transform.translation.y = 20.
            transform.transform.rotation.z = math.sin(math.pi / 4)
            transform.transform.rotation.w = math.cos(math.pi / 4)
            buffer.set_transform(transform, 'test')
        node = SimpleNamespace(map_msg=SimpleNamespace(header=SimpleNamespace(frame_id='map')),
                               tf_buffer=buffer)
        goal = PoseStamped()
        goal.header.frame_id = 'ego_racecar/base_link'
        goal.header.stamp.sec = 5
        goal.pose.position.x = 1.
        goal.pose.orientation.w = 1.
        result = OccupancyGridPlanner.goal_in_map(node, goal)
        self.assertEqual(result.header.frame_id, 'map')
        self.assertEqual(result.header.stamp.sec, 5)
        self.assertAlmostEqual(result.pose.position.x, 10.)
        self.assertAlmostEqual(result.pose.position.y, 21.)
        self.assertAlmostEqual(result.pose.orientation.z, math.sin(math.pi / 4))
        goal.header.stamp.sec = 7
        with self.assertRaises(TransformException):
            OccupancyGridPlanner.goal_in_map(node, goal)
