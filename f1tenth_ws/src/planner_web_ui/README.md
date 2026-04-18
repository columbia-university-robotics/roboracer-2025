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
