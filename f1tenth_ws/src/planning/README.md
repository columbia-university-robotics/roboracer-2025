# Planning

**Note to self: Update with a better explanation of the planning algorithms later**


This package provides map-based path planning and path following for the F1TENTH stack.

## Nodes

### `occupancy_grid_planner`

Builds a collision-free path from the current localized pose to a requested goal pose.

**Inputs**

- `/map` as `nav_msgs/OccupancyGrid`
- `/localization/pose` as `PoseWithCovarianceStamped`
- `/planner/goal_pose` as `PoseStamped`

**Outputs**

- `/planner/path` as `nav_msgs/Path`
- `/planner/waypoints` as `geometry_msgs/PoseArray`

**How it works**

- Converts the occupancy grid into a boolean obstacle map using `occupancy_threshold`
- Optionally treats unknown cells as blocked and inflates obstacles by `inflation_radius`
- Converts the robot pose and goal pose from world coordinates into grid cells
- If either point lands in an occupied cell, searches nearby up to `max_goal_search_radius` for the nearest free cell
- Runs 8-connected A* with a Euclidean heuristic to find a path through free cells
- Prunes redundant cells using line-of-sight checks, converts the result back to world coordinates, and spaces waypoints by `waypoint_spacing`
- Publishes waypoint headings aligned with the next segment of the path

### `pure_pursuit_follower`

Consumes `/planner/path` and publishes `/drive` commands using a pure-pursuit controller.
