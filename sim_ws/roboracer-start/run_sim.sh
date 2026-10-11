#!/usr/bin/env bash
# Keep this directory directly inside a built simulation workspace.
set -e
launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec python3 "$launcher_dir/run_sim.py" --help
fi
roboracer_ws="${ROBORACER_SIM_WS:-$(dirname -- "$launcher_dir")}"
roboracer_setup="${ROBORACER_ROS_SETUP:-/opt/ros/humble/setup.bash}"
roboracer_venv="${ROBORACER_VENV:-$roboracer_ws/.venv}"
for required in "$roboracer_setup" "$roboracer_venv/bin/activate" "$roboracer_ws/install/local_setup.bash"; do
    if [[ ! -f "$required" ]]; then
        printf 'ERROR: Required environment file not found: %s\nSee README.md for workspace setup and path overrides.\n' "$required" >&2
        exit 1
    fi
done
cd "$roboracer_ws"
source "$roboracer_setup"
source "$roboracer_venv/bin/activate"
source "$roboracer_ws/install/local_setup.bash"
exec python3 "$launcher_dir/run_sim.py" "$@"
