"""Exercise the actual callback with mocked simulator boundaries; no ROS needed."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation

source = Path(__file__).resolve().parents[1] / 'f1tenth_gym_ros' / 'gym_bridge.py'
tree = ast.parse(source.read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'ego_reset_callback')
space = {'np': np, 'Rotation': Rotation}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), space)
callback = space['ego_reset_callback']

class ResetTests(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.node = NS(sim_paused=False, ego_pose=[9., 8., 0.], env=Mock(),
                       get_logger=lambda: self.logger, _update_sim_state=Mock())
        self.node._all_poses = lambda: [self.node.ego_pose]

    def message(self, frame):
        q = Rotation.from_euler('z', 0.6).as_quat()
        return NS(header=NS(frame_id=frame), pose=NS(pose=NS(
            position=NS(x=1.2, y=-3.4), orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))))

    def test_map_resets_to_requested_position_and_yaw(self):
        callback(self.node, self.message('map'))
        np.testing.assert_allclose(self.node.ego_pose, [1.2, -3.4, .6])
        self.node.env.reset.assert_called_once()
        np.testing.assert_allclose(self.node.env.reset.call_args.kwargs['options']['poses'], [[1.2, -3.4, .6]])
        self.node._update_sim_state.assert_called_once()
        self.logger.warning.assert_not_called()

    def test_other_frames_do_not_change_pose_or_reset_simulator(self):
        for frame in ('', 'ego_racecar/base_link', 'odom', '/map', ' map '):
            with self.subTest(frame=frame):
                self.setUp()
                callback(self.node, self.message(frame))
                self.assertEqual(self.node.ego_pose, [9., 8., 0.])
                self.node.env.reset.assert_not_called()
                self.node._update_sim_state.assert_not_called()
                self.logger.warning.assert_called_once()

    def test_paused_behavior_is_preserved(self):
        self.node.sim_paused = True
        callback(self.node, self.message('map'))
        self.node.env.reset.assert_not_called()
        self.assertEqual(self.node.ego_pose, [9., 8., 0.])

if __name__ == '__main__':
    unittest.main()
