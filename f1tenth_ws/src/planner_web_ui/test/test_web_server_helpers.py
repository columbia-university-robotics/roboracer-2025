import base64
import math

from planner_web_ui.web_server import default_goal_yaw, encode_occupancy_grid


def test_encode_occupancy_grid_clamps_unknown_and_high_values():
    encoded = encode_occupancy_grid([-1, 0, 42, 100, 150])
    assert list(base64.b64decode(encoded)) == [255, 0, 42, 100, 100]


def test_default_goal_yaw_points_from_pose_to_goal():
    pose = {"x": 1.0, "y": 2.0, "yaw": 1.23}
    assert math.isclose(default_goal_yaw(pose, 4.0, 6.0), math.atan2(4.0, 3.0))


def test_default_goal_yaw_reuses_pose_heading_for_same_point():
    pose = {"x": 1.0, "y": 2.0, "yaw": -0.75}
    assert math.isclose(default_goal_yaw(pose, 1.0, 2.0), pose["yaw"])
