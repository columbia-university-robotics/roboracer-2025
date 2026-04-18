#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./pathplanning_common.sh
source "${SCRIPT_DIR}/pathplanning_common.sh"

print_usage() {
  cat <<'EOF'
Usage: ./scripts/pathplanning_real.sh --map <map.yaml> [options]

Options:
  --map <map.yaml>         Localization map YAML.
  --planner-config <file>  Optional planner YAML override.
  --runtime <mode>         Choose `auto`, `native`, or `docker` runtime.
  --no-build               Skip colcon build before launch.
  --no-open-browser        Do not auto-open the planner UI in a browser.
  -h, --help               Show this help text.

Note:
  This script launches localization, planning, and the lightweight planner UI.
  Bring up the real hardware stack separately before using it on the car.
EOF
}

MAP_FILE=""
PLANNER_CONFIG=""
RUNTIME_OVERRIDE="${ROBORACER_RUNTIME:-auto}"
DO_BUILD=1
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map)
      MAP_FILE="${2:-}"
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

if [[ -z "$MAP_FILE" ]]; then
  echo "--map is required." >&2
  print_usage >&2
  exit 1
fi

select_runtime "$RUNTIME_OVERRIDE"
maybe_open_browser "$OPEN_BROWSER"

REAL_WORKSPACE_ROOT="$(workspace_root_for_runtime "f1tenth_ws")"
BUILD_COMMAND="colcon build --packages-select localization planning planner_web_ui"
LAUNCH_COMMAND="ros2 launch planner_web_ui real_pathplanning.launch.py map_file:=$(quote_for_bash "$(runtime_path "$MAP_FILE")") web_port:=8081 enable_rviz:=false"

if [[ -n "$PLANNER_CONFIG" ]]; then
  LAUNCH_COMMAND+=" planner_config:=$(quote_for_bash "$(runtime_path "$PLANNER_CONFIG")")"
fi

run_launch_command "$REAL_WORKSPACE_ROOT" "" "$BUILD_COMMAND" "$LAUNCH_COMMAND" "$DO_BUILD"
