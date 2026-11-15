from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener


class FrameTransformer:
    """Transform poses and paths at their message timestamps without blocking."""

    def __init__(self, node) -> None:
        self.buffer = Buffer(node=node)
        self.listener = TransformListener(self.buffer, node)
        self._logger = node.get_logger()

    def transform_pose(self, msg: PoseStamped, target_frame: str) -> PoseStamped | None:
        if not msg.header.frame_id or not target_frame:
            self._logger.warning(
                "Cannot transform a pose with an empty frame_id.",
                throttle_duration_sec=5.0,
            )
            return None
        if msg.header.frame_id == target_frame:
            return msg

        try:
            transform = self.buffer.lookup_transform(
                target_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp)
            )
            return do_transform_pose_stamped(msg, transform)
        except TransformException as exc:
            self._logger.warning(
                f"Cannot transform {msg.header.frame_id} to {target_frame}: {exc}",
                throttle_duration_sec=5.0,
            )
            return None

    def transform_path(self, msg: Path, target_frame: str) -> Path | None:
        if not msg.header.frame_id or not target_frame:
            self._logger.warning(
                "Cannot transform a path with an empty frame_id.",
                throttle_duration_sec=5.0,
            )
            return None

        result = Path()
        result.header.frame_id = target_frame
        result.header.stamp = msg.header.stamp
        for pose in msg.poses:
            # A pose without its own frame inherits the path's header.
            stamped_pose = pose
            if not pose.header.frame_id:
                stamped_pose = PoseStamped(header=msg.header, pose=pose.pose)
            transformed = self.transform_pose(stamped_pose, target_frame)
            if transformed is None:
                return None
            result.poses.append(transformed)
        return result
