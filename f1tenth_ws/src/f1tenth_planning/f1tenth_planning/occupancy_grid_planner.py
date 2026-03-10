import heapq
import math
from typing import Iterable

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy import ndimage


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OccupancyGridPlanner(Node):
    def __init__(self) -> None:
        super().__init__("occupancy_grid_planner")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_topic", "/ego_racecar/odom")
        self.declare_parameter("goal_topic", "/planner/goal_pose")
        self.declare_parameter("path_topic", "/planner/path")
        self.declare_parameter("waypoints_topic", "/planner/waypoints")
        self.declare_parameter("occupancy_threshold", 50)
        self.declare_parameter("allow_unknown", False)
        self.declare_parameter("inflation_radius", 0.20)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("max_goal_search_radius", 1.50)
        self.declare_parameter("waypoint_spacing", 0.50)

        self.map_msg: OccupancyGrid | None = None
        self.occupied: np.ndarray | None = None
        self.current_pose: Pose | None = None

        self.path_pub = self.create_publisher(
            Path, self.get_parameter("path_topic").value, 10
        )
        self.waypoints_pub = self.create_publisher(
            PoseArray, self.get_parameter("waypoints_topic").value, 10
        )

        self.create_subscription(
            OccupancyGrid, self.get_parameter("map_topic").value, self.map_callback, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.odom_callback, 10
        )
        self.create_subscription(
            PoseStamped, self.get_parameter("goal_topic").value, self.goal_callback, 10
        )

        self.get_logger().info(
            "Planner ready on map=%s odom=%s goal=%s"
            % (
                self.get_parameter("map_topic").value,
                self.get_parameter("odom_topic").value,
                self.get_parameter("goal_topic").value,
            )
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg
        grid = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        occupied = grid >= int(self.get_parameter("occupancy_threshold").value)
        if not bool(self.get_parameter("allow_unknown").value):
            occupied |= grid < 0

        inflation_radius = float(self.get_parameter("inflation_radius").value)
        if inflation_radius > 0.0 and msg.info.resolution > 0.0:
            inflation_cells = max(1, math.ceil(inflation_radius / msg.info.resolution))
            occupied = ndimage.binary_dilation(
                occupied,
                structure=ndimage.generate_binary_structure(2, 2),
                iterations=inflation_cells,
            )

        self.occupied = occupied
        self.get_logger().info(
            f"Loaded occupancy grid {msg.info.width}x{msg.info.height} at "
            f"{msg.info.resolution:.3f} m/cell."
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg: PoseStamped) -> None:
        if self.map_msg is None or self.occupied is None:
            self.get_logger().warning("Cannot plan: no occupancy grid received.")
            return
        if self.current_pose is None:
            self.get_logger().warning("Cannot plan: no odometry received.")
            return

        start_cell = self.world_to_grid(
            self.current_pose.position.x, self.current_pose.position.y
        )
        goal_cell = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        if start_cell is None or goal_cell is None:
            self.get_logger().warning("Start or goal pose is outside the map.")
            return

        search_radius = float(self.get_parameter("max_goal_search_radius").value)
        start_cell = self.find_nearest_free(start_cell, search_radius)
        goal_cell = self.find_nearest_free(goal_cell, search_radius)
        if start_cell is None or goal_cell is None:
            self.get_logger().warning("No free start/goal cell found near request.")
            return

        cell_path = self.a_star(start_cell, goal_cell)
        if not cell_path:
            self.get_logger().warning("No path found to the requested goal.")
            return

        pruned_path = self.prune_path(cell_path)
        waypoints = self.cells_to_waypoints(pruned_path)
        self.publish_outputs(waypoints)
        self.get_logger().info(
            f"Published {len(waypoints)} waypoints from {len(cell_path)} planned cells."
        )

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        if self.map_msg is None:
            return None

        info = self.map_msg.info
        yaw = yaw_from_quaternion(info.origin.orientation)
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y

        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        gx = int(math.floor(local_x / info.resolution))
        gy = int(math.floor(local_y / info.resolution))
        if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        assert self.map_msg is not None
        info = self.map_msg.info
        yaw = yaw_from_quaternion(info.origin.orientation)
        local_x = (gx + 0.5) * info.resolution
        local_y = (gy + 0.5) * info.resolution
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = info.origin.position.x + local_x * cos_yaw - local_y * sin_yaw
        y = info.origin.position.y + local_x * sin_yaw + local_y * cos_yaw
        return x, y

    def is_free(self, cell: tuple[int, int]) -> bool:
        if self.occupied is None:
            return False
        gx, gy = cell
        if gx < 0 or gy < 0 or gy >= self.occupied.shape[0] or gx >= self.occupied.shape[1]:
            return False
        return not bool(self.occupied[gy, gx])

    def neighbors(self, cell: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float]]:
        gx, gy = cell
        for dx, dy, step_cost in (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ):
            next_cell = (gx + dx, gy + dy)
            if self.is_free(next_cell):
                yield next_cell, step_cost

    @staticmethod
    def heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> float:
        return math.hypot(goal[0] - cell[0], goal[1] - cell[1])

    def a_star(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.neighbors(current):
                new_cost = g_score[current] + step_cost
                if new_cost >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = new_cost
                heapq.heappush(
                    open_heap,
                    (new_cost + self.heuristic(neighbor, goal), neighbor),
                )

        return []

    def reconstruct_path(
        self, came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_nearest_free(
        self, cell: tuple[int, int], max_radius_m: float
    ) -> tuple[int, int] | None:
        if self.map_msg is None:
            return None
        if self.is_free(cell):
            return cell

        radius_cells = max(1, math.ceil(max_radius_m / self.map_msg.info.resolution))
        cx, cy = cell
        nearest = None
        nearest_distance = float("inf")
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                candidate = (cx + dx, cy + dy)
                if not self.is_free(candidate):
                    continue
                distance = math.hypot(dx, dy)
                if distance < nearest_distance:
                    nearest = candidate
                    nearest_distance = distance
        return nearest

    def prune_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(path) < 3:
            return path

        pruned = [path[0]]
        anchor = 0
        probe = 2
        while probe < len(path):
            if self.has_line_of_sight(path[anchor], path[probe]):
                probe += 1
                continue
            pruned.append(path[probe - 1])
            anchor = probe - 1
            probe = anchor + 2

        if pruned[-1] != path[-1]:
            pruned.append(path[-1])
        return pruned

    def has_line_of_sight(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        x0, y0 = start
        x1, y1 = goal
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            error = dx / 2.0
            while x != x1:
                if not self.is_free((x, y)):
                    return False
                error -= dy
                if error < 0:
                    y += sy
                    error += dx
                x += sx
        else:
            error = dy / 2.0
            while y != y1:
                if not self.is_free((x, y)):
                    return False
                error -= dx
                if error < 0:
                    x += sx
                    error += dy
                y += sy

        return self.is_free((x1, y1))

    def cells_to_waypoints(self, cells: list[tuple[int, int]]) -> list[tuple[float, float]]:
        points = [self.grid_to_world(gx, gy) for gx, gy in cells]
        if not points:
            return []

        spacing = float(self.get_parameter("waypoint_spacing").value)
        goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        if spacing <= 0.0:
            return points

        spaced = [points[0]]
        last = points[0]
        for point in points[1:-1]:
            if math.dist(last, point) >= spacing:
                spaced.append(point)
                last = point

        if math.dist(spaced[-1], points[-1]) > goal_tolerance or len(spaced) == 1:
            spaced.append(points[-1])
        return spaced

    def publish_outputs(self, waypoints: list[tuple[float, float]]) -> None:
        if self.map_msg is None:
            return

        frame_id = self.map_msg.header.frame_id or "map"
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = frame_id
        waypoint_msg = PoseArray()
        waypoint_msg.header.stamp = stamp
        waypoint_msg.header.frame_id = frame_id

        for idx, (x, y) in enumerate(waypoints):
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y

            heading = 0.0
            if idx + 1 < len(waypoints):
                nx, ny = waypoints[idx + 1]
                heading = math.atan2(ny - y, nx - x)
            elif idx > 0:
                px, py = waypoints[idx - 1]
                heading = math.atan2(y - py, x - px)
            pose.pose.orientation.z = math.sin(heading / 2.0)
            pose.pose.orientation.w = math.cos(heading / 2.0)

            path_msg.poses.append(pose)
            waypoint_msg.poses.append(pose.pose)

        self.path_pub.publish(path_msg)
        self.waypoints_pub.publish(waypoint_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancyGridPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
