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


## Goal coordinate frames

The planner requires localization and occupancy-grid headers to name the same
nonempty frame. Goals in that frame are used directly. Other goals are transformed
with TF at their message timestamp; cross-frame goals require a nonzero timestamp.
Empty frames, non-finite poses, non-unit quaternions and unavailable TF are rejected.
TF lookup does not block: if TF has not arrived yet, submit the goal again.
Every new goal clears the old path first. The follower commands zero speed on an
empty path, including when a replacement goal cannot be planned.

Run the policy tests and ROS geometry/timestamp test in a sourced Humble workspace:

```bash
PYTHONPATH=f1tenth_ws/src/planning:$PYTHONPATH python3 -m unittest discover -s f1tenth_ws/src/planning/test -p 'test_goal_frame_*.py' -v
```

Without ROS, the geometry test is skipped; offline tests alone do not verify TF.
The deployed patch passed all 10 tests on Ubuntu on 2026-10-18. Planning-only
simulation on 2026-10-25 confirmed that a car at map (-12, 0) receiving car-frame
(2, 0) produced map endpoints near (-10, 0) at yaw 0 degrees and (-10.05, 0.35)
at yaw 10 degrees. The latter differs from the expected (-10.0304, 0.3473) by
about 2 cm, consistent with the 5 cm map grid. User-observed low-speed checks
also covered route clearing/stopping for an invalid goal and recovery with a valid
goal. These checks do not establish moving-frame time synchronization or real-car
performance. Localization transformation and full follower frame/freshness checks
are outside this change.
