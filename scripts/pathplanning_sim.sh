#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./pathplanning_common.sh
source "${SCRIPT_DIR}/pathplanning_common.sh"

print_usage() {
  cat <<'EOF'
Usage: ./scripts/pathplanning_sim.sh --map <map> [options]

Options:
  --map <map>              Simulator map path or built-in track name.
  --planner-config <file>  Optional planner YAML override.
  --no-build               Skip colcon build before launch.
  --no-open-browser        Do not auto-open the planner UI in a browser.
  -h, --help               Show this help text.
EOF
}

MAP=""
PLANNER_CONFIG=""
DO_BUILD=1
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map)
      MAP="${2:-}"
      shift 2
      ;;
    --planner-config)
      PLANNER_CONFIG="${2:-}"
      shift 2
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --no-open-browser)
      OPEN_BROWSER=0
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MAP" ]]; then
  echo "--map is required." >&2
  print_usage >&2
  exit 1
fi

ensure_ros_container
maybe_open_browser "$OPEN_BROWSER"

BUILD_COMMAND="colcon build --packages-select f1tenth_gym_ros planning planner_web_ui"
SIM_ENV_COMMAND="if [[ ! -d /workspace/sim_ws/.venv ]]; then python3 -m venv --system-site-packages /workspace/sim_ws/.venv; fi && source /workspace/sim_ws/.venv/bin/activate && python -m pip install -U pip >/dev/null && (python -c 'import gymnasium, numba, PIL, OpenGL, cv2, yamldataclassconfig, pandas, requests' >/dev/null 2>&1 || python -m pip install -e /workspace/sim_ws/src/f1tenth_gym_ros/f1tenth_gym)"
LAUNCH_COMMAND="${SIM_ENV_COMMAND} && ros2 launch planner_web_ui sim_pathplanning.launch.py map:=$(quote_for_bash "$(to_container_path "$MAP")") web_port:=8081"

if [[ -n "$PLANNER_CONFIG" ]]; then
  LAUNCH_COMMAND+=" planner_config:=$(quote_for_bash "$(to_container_path "$PLANNER_CONFIG")")"
fi

run_launch_in_container "/workspace/sim_ws" "$BUILD_COMMAND" "$LAUNCH_COMMAND" "$DO_BUILD"
