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
  --runtime <mode>         Choose `auto`, `native`, or `docker` runtime.
  --no-build               Skip colcon build before launch.
  --no-open-browser        Do not auto-open the planner UI in a browser.
  -h, --help               Show this help text.
EOF
}

MAP=""
PLANNER_CONFIG=""
RUNTIME_OVERRIDE="${ROBORACER_RUNTIME:-auto}"
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
    --runtime)
      RUNTIME_OVERRIDE="${2:-}"
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

select_runtime "$RUNTIME_OVERRIDE"
maybe_open_browser "$OPEN_BROWSER"

SIM_WORKSPACE_ROOT="$(workspace_root_for_runtime "sim_ws")"
SIM_VENV_ROOT="${SIM_WORKSPACE_ROOT}/.venv"
SIM_GYM_ROOT="${SIM_WORKSPACE_ROOT}/src/f1tenth_gym_ros/f1tenth_gym"

SIM_PREPARE_COMMAND="if [[ ! -d $(quote_for_bash "$SIM_VENV_ROOT") ]]; then python3 -m venv --system-site-packages $(quote_for_bash "$SIM_VENV_ROOT"); fi && source $(quote_for_bash "$SIM_VENV_ROOT/bin/activate") && python -m pip install -U pip >/dev/null && (python -c 'import gymnasium, numba, PIL, OpenGL, cv2, yamldataclassconfig, pandas, requests' >/dev/null 2>&1 || python -m pip install -e $(quote_for_bash "$SIM_GYM_ROOT"))"
BUILD_COMMAND="colcon build --packages-select f1tenth_gym_ros planning planner_web_ui"
LAUNCH_COMMAND="ros2 launch planner_web_ui sim_pathplanning.launch.py map:=$(quote_for_bash "$(runtime_path "$MAP")") web_port:=8081"

if [[ -n "$PLANNER_CONFIG" ]]; then
  LAUNCH_COMMAND+=" planner_config:=$(quote_for_bash "$(runtime_path "$PLANNER_CONFIG")")"
fi

run_launch_command "$SIM_WORKSPACE_ROOT" "$SIM_PREPARE_COMMAND" "$BUILD_COMMAND" "$LAUNCH_COMMAND" "$DO_BUILD"
