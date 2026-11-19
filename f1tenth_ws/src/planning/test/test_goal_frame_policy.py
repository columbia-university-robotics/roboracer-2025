"""Offline tests of actual callback bodies, with explicit ROS boundary doubles.

These verify frame/timestamp policy and invalidation, not ROS TF geometry or DDS.
Run test_goal_frame_ros.py inside ROS Humble to verify the actual TF library.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import math
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]

def methods(filename, classname, names, namespace):
    tree = ast.parse((ROOT / 'planning' / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in ast.walk(ast.Module(body=selected, type_ignores=[])):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.FunctionDef):
            node.returns = None
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, filename, 'exec'), namespace)
    return namespace

class TFError(Exception):
    pass

def pose(frame='map', sec=5):
    return NS(header=NS(frame_id=frame, stamp=NS(sec=sec, nanosec=123)),
              pose=NS(position=NS(x=1., y=0., z=0.),
                      orientation=NS(x=0., y=0., z=0., w=1.)))

class GoalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.transform = Mock(return_value=pose('map'))
        self.env = methods('occupancy_grid_planner.py', 'OccupancyGridPlanner',
                           {'goal_in_map', 'goal_callback'},
                           {'math': math, 'Time': NS(from_msg=lambda m: (m.sec, m.nanosec)),
                            'TransformException': TFError,
                            'do_transform_pose_stamped': self.transform})
        self.node = NS(map_msg=NS(header=NS(frame_id='map')), tf_buffer=Mock(),
                       occupied=True, current_pose=pose().pose, current_pose_frame='map',
                       publish_outputs=Mock(), get_logger=lambda: Mock(), world_to_grid=Mock())
        self.node.goal_in_map = lambda msg: self.env['goal_in_map'](self.node, msg)

    def test_same_frame_needs_no_tf(self):
        msg = pose()
        self.assertIs(self.node.goal_in_map(msg), msg)
        self.node.tf_buffer.lookup_transform.assert_not_called()

    def test_cross_frame_uses_message_time_and_preserves_stamp(self):
        msg = pose('ego_racecar/base_link')
        result = self.node.goal_in_map(msg)
        self.node.tf_buffer.lookup_transform.assert_called_once_with(
            'map', 'ego_racecar/base_link', (5, 123))
        self.transform.assert_called_once_with(msg, self.node.tf_buffer.lookup_transform.return_value)
        self.assertIs(result.header.stamp, msg.header.stamp)

    def test_empty_frames_rejected(self):
        for source, target in [('', 'map'), ('map', ''), ('', '')]:
            with self.subTest(source=source, target=target):
                self.node.map_msg.header.frame_id = target
                with self.assertRaises(ValueError):
                    self.node.goal_in_map(pose(source))

    def test_zero_cross_frame_time_rejected(self):
        msg = pose('base_link', 0)
        msg.header.stamp.nanosec = 0
        with self.assertRaises(ValueError):
            self.node.goal_in_map(msg)

    def test_invalid_numeric_input_rejected(self):
        for value in [float('nan'), float('inf')]:
            msg = pose()
            msg.pose.position.x = value
            with self.assertRaises(ValueError):
                self.node.goal_in_map(msg)
        msg = pose()
        msg.pose.orientation.w = 0.
        with self.assertRaises(ValueError):
            self.node.goal_in_map(msg)

    def test_tf_failure_clears_route_and_does_not_plan(self):
        self.node.tf_buffer.lookup_transform.side_effect = TFError('unavailable at timestamp')
        self.env['goal_callback'](self.node, pose('base_link'))
        self.node.publish_outputs.assert_called_once_with([])
        self.node.world_to_grid.assert_not_called()

    def test_localization_mismatch_does_not_plan(self):
        self.node.current_pose_frame = 'odom'
        self.env['goal_callback'](self.node, pose())
        self.node.publish_outputs.assert_called_once_with([])
        self.node.world_to_grid.assert_not_called()

    def test_converted_coordinates_reach_planner(self):
        converted = pose()
        converted.pose.position.x, converted.pose.position.y = 10., 21.
        self.transform.return_value = converted
        self.node.world_to_grid.side_effect = [(0, 0), None]
        self.env['goal_callback'](self.node, pose('base_link'))
        self.assertEqual(self.node.world_to_grid.call_args_list[-1].args, (10., 21.))

    def test_empty_route_stops_follower_immediately_and_on_timer(self):
        env = methods('pure_pursuit_follower.py', 'PurePursuitFollower',
                      {'path_callback', 'control_callback'}, {'math': math})
        node = NS(current_path=[(1., 2.)], target_index=3,
                  get_logger=lambda: Mock(), publish_drive=Mock(), current_pose=None)
        env['path_callback'](node, NS(poses=[]))
        env['control_callback'](node)
        self.assertEqual(node.current_path, [])
        self.assertEqual(node.publish_drive.call_count, 2)
        node.publish_drive.assert_called_with(0., 0.)

if __name__ == '__main__':
    unittest.main()
