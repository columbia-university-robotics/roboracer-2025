# planner_web_ui

`planner_web_ui` is a lightweight browser alternative to Foxglove and RViz for map-based path planning workflows.

It serves a small HTTP app that:

- renders the current `/map`
- overlays `/localization/pose`, `/planner/path`, and the latest goal
- publishes `/planner/goal_pose` when you click on the map

## Launch files

- `ros2 launch planner_web_ui web_ui.launch.py`
- `ros2 launch planner_web_ui sim_pathplanning.launch.py map:=Spielberg`
- `ros2 launch planner_web_ui real_pathplanning.launch.py map_file:=/path/to/map.yaml`

By default the UI is served on `http://localhost:8081`.

The map's `header.frame_id` is the display frame. The shared `frame_utils`
package transforms pose, goal, and path messages into that frame using TF2 at
their message timestamps. Path poses with an empty frame inherit the path's
header. Messages with an empty source frame or unavailable TF are hidden, and
pending transforms are retried when the browser polls. Successful transforms
are cached. Maps with an empty frame are ignored.

The repo-level wrapper scripts `scripts/pathplanning_sim.sh` and `scripts/pathplanning_real.sh` auto-detect native ROS versus Docker so the same command can be used on a Jetson or in the project container.
