import base64
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float]:
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def clamp_occupancy_value(value: int) -> int:
    if value < 0:
        return 255
    return max(0, min(100, value))


def encode_occupancy_grid(data: list[int]) -> str:
    payload = bytes(clamp_occupancy_value(value) for value in data)
    return base64.b64encode(payload).decode("ascii")


def default_goal_yaw(current_pose: dict[str, float] | None, goal_x: float, goal_y: float) -> float:
    if current_pose is None:
        return 0.0
    dx = goal_x - current_pose["x"]
    dy = goal_y - current_pose["y"]
    if math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9):
        return current_pose["yaw"]
    return math.atan2(dy, dx)


class PlannerWebUi(Node):
    def __init__(self) -> None:
        super().__init__("planner_web_ui")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8081)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("pose_topic", "/localization/pose")
        self.declare_parameter("path_topic", "/planner/path")
        self.declare_parameter("goal_topic", "/planner/goal_pose")

        self._lock = threading.Lock()
        self._map_version = 0
        self._map_snapshot: dict[str, Any] | None = None
        self._pose_snapshot: dict[str, float] | None = None
        self._goal_snapshot: dict[str, float] | None = None
        self._path_snapshot: list[list[float]] = []

        self._static_dir = Path(get_package_share_directory("planner_web_ui")) / "static"

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        map_topic = str(self.get_parameter("map_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        path_topic = str(self.get_parameter("path_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)

        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, 10)
        self.create_subscription(OccupancyGrid, map_topic, self.map_callback, latched_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self.pose_callback, 10
        )
        self.create_subscription(NavPath, path_topic, self.path_callback, latched_qos)
        self.create_subscription(PoseStamped, goal_topic, self.goal_callback, 10)

        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._http_server = ThreadingHTTPServer((host, port), self.build_handler())
        self._http_server.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, name="planner-web-ui-http", daemon=True
        )
        self._http_thread.start()

        self.get_logger().info(
            "Planner web UI listening on http://%s:%d using map=%s pose=%s path=%s goal=%s"
            % (host, port, map_topic, pose_topic, path_topic, goal_topic)
        )

    def build_handler(self):
        node = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                node.handle_get(self)

            def do_POST(self) -> None:  # noqa: N802
                node.handle_post(self)

            def do_OPTIONS(self) -> None:  # noqa: N802
                node.send_empty(self, HTTPStatus.NO_CONTENT)

            def log_message(self, format: str, *args) -> None:
                return

        return RequestHandler

    def map_callback(self, msg: OccupancyGrid) -> None:
        self._map_version += 1
        with self._lock:
            self._map_snapshot = {
                "version": self._map_version,
                "frameId": msg.header.frame_id or "map",
                "width": msg.info.width,
                "height": msg.info.height,
                "resolution": msg.info.resolution,
                "origin": {
                    "x": msg.info.origin.position.x,
                    "y": msg.info.origin.position.y,
                    "yaw": yaw_from_quaternion(msg.info.origin.orientation),
                },
                "data": encode_occupancy_grid(list(msg.data)),
            }
        self.get_logger().info(
            f"Cached map version {self._map_version} ({msg.info.width}x{msg.info.height})."
        )

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        with self._lock:
            self._pose_snapshot = {
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "yaw": yaw_from_quaternion(msg.pose.pose.orientation),
            }

    def path_callback(self, msg: NavPath) -> None:
        with self._lock:
            self._path_snapshot = [
                [pose.pose.position.x, pose.pose.position.y] for pose in msg.poses
            ]

    def goal_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            self._goal_snapshot = {
                "x": msg.pose.position.x,
                "y": msg.pose.position.y,
                "yaw": yaw_from_quaternion(msg.pose.orientation),
            }

    def snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            map_snapshot = self._map_snapshot
            pose_snapshot = self._pose_snapshot
            goal_snapshot = self._goal_snapshot
            path_snapshot = list(self._path_snapshot)

        return {
            "ready": {
                "map": map_snapshot is not None,
                "pose": pose_snapshot is not None,
                "path": bool(path_snapshot),
            },
            "mapVersion": map_snapshot["version"] if map_snapshot else 0,
            "frameId": map_snapshot["frameId"] if map_snapshot else "map",
            "pose": pose_snapshot,
            "goal": goal_snapshot,
            "path": path_snapshot,
        }

    def snapshot_map(self) -> dict[str, Any] | None:
        with self._lock:
            if self._map_snapshot is None:
                return None
            return dict(self._map_snapshot)

    def publish_goal(self, goal_x: float, goal_y: float, yaw: float | None) -> dict[str, float]:
        with self._lock:
            current_pose = dict(self._pose_snapshot) if self._pose_snapshot else None
            frame_id = (
                self._map_snapshot["frameId"] if self._map_snapshot is not None else "map"
            )

        goal_yaw = yaw if yaw is not None else default_goal_yaw(current_pose, goal_x, goal_y)
        qz, qw = quaternion_from_yaw(goal_yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = goal_x
        msg.pose.position.y = goal_y
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.goal_pub.publish(msg)

        goal_snapshot = {"x": goal_x, "y": goal_y, "yaw": goal_yaw}
        with self._lock:
            self._goal_snapshot = goal_snapshot
        return goal_snapshot

    def handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlparse(handler.path).path
        if path in ("/", "/index.html"):
            self.serve_static(handler, "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self.serve_static(handler, "app.js", "application/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self.serve_static(handler, "styles.css", "text/css; charset=utf-8")
            return
        if path == "/api/healthz":
            self.send_json(handler, {"ok": True})
            return
        if path == "/api/state":
            self.send_json(handler, self.snapshot_state())
            return
        if path == "/api/map":
            map_snapshot = self.snapshot_map()
            if map_snapshot is None:
                self.send_json(
                    handler,
                    {"ready": False, "message": "No occupancy grid received yet."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            self.send_json(handler, map_snapshot)
            return
        self.send_json(
            handler,
            {"ok": False, "message": f"Unknown route: {path}"},
            status=HTTPStatus.NOT_FOUND,
        )

    def handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlparse(handler.path).path
        if path != "/api/goal":
            self.send_json(
                handler,
                {"ok": False, "message": f"Unknown route: {path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            raw_body = handler.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            goal_x = float(payload["x"])
            goal_y = float(payload["y"])
            yaw_value = payload.get("yaw")
            goal_yaw = float(yaw_value) if yaw_value is not None else None
            if not math.isfinite(goal_x) or not math.isfinite(goal_y):
                raise ValueError("Goal coordinates must be finite.")
            if goal_yaw is not None and not math.isfinite(goal_yaw):
                raise ValueError("Goal yaw must be finite when provided.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json(
                handler,
                {"ok": False, "message": f"Invalid goal payload: {exc}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        goal_snapshot = self.publish_goal(goal_x, goal_y, goal_yaw)
        self.send_json(handler, {"ok": True, "goal": goal_snapshot})

    def serve_static(
        self, handler: BaseHTTPRequestHandler, file_name: str, content_type: str
    ) -> None:
        file_path = self._static_dir / file_name
        if not file_path.exists():
            self.send_json(
                handler,
                {"ok": False, "message": f"Missing static asset: {file_name}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        self.send_bytes(handler, file_path.read_bytes(), content_type)

    def send_json(
        self,
        handler: BaseHTTPRequestHandler,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_bytes(
            handler,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def send_empty(
        self, handler: BaseHTTPRequestHandler, status: HTTPStatus = HTTPStatus.NO_CONTENT
    ) -> None:
        handler.send_response(status)
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()

    def send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        payload: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Content-Type")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(payload)

    def destroy_node(self) -> bool:
        if hasattr(self, "_http_server"):
            self._http_server.shutdown()
            self._http_server.server_close()
        if hasattr(self, "_http_thread") and self._http_thread.is_alive():
            self._http_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerWebUi()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
