#!/usr/bin/env bash

set -euo pipefail

ROBORACER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROBORACER_ROOT
readonly WEB_UI_URL="http://localhost:8081"
ROBORACER_RUNTIME_MODE=""

ensure_ros_container() {
  (
    cd "$ROBORACER_ROOT"
    docker compose up -d ubuntu >/dev/null
  )
}

native_ros_available() {
  [[ -f "/opt/ros/humble/setup.bash" ]]
}

docker_runtime_available() {
  [[ -f "$ROBORACER_ROOT/docker-compose.yml" ]] &&
  command -v docker >/dev/null 2>&1 && (
    cd "$ROBORACER_ROOT"
    docker compose version >/dev/null 2>&1
  )
}

require_runtime_selection() {
  if [[ -z "$ROBORACER_RUNTIME_MODE" ]]; then
    echo "Runtime not selected. Call select_runtime before using runtime-aware helpers." >&2
    exit 1
  fi
}

select_runtime() {
  local requested_runtime="${1:-${ROBORACER_RUNTIME:-auto}}"

  case "$requested_runtime" in
    auto)
      if native_ros_available; then
        ROBORACER_RUNTIME_MODE="native"
      elif docker_runtime_available; then
        ROBORACER_RUNTIME_MODE="docker"
      else
        echo "Could not find a native ROS 2 Humble install or a usable Docker Compose setup." >&2
        exit 1
      fi
      ;;
    native)
      if ! native_ros_available; then
        echo "Native runtime requested, but /opt/ros/humble/setup.bash was not found." >&2
        exit 1
      fi
      ROBORACER_RUNTIME_MODE="native"
      ;;
    docker)
      if ! docker_runtime_available; then
        echo "Docker runtime requested, but docker compose is not available for this repo." >&2
        exit 1
      fi
      ROBORACER_RUNTIME_MODE="docker"
      ;;
    *)
      echo "Unsupported runtime '$requested_runtime'. Use auto, native, or docker." >&2
      exit 1
      ;;
  esac

  if [[ "$ROBORACER_RUNTIME_MODE" == "docker" ]]; then
    ensure_ros_container
  fi

  echo "Using ${ROBORACER_RUNTIME_MODE} runtime." >&2
}

quote_for_bash() {
  printf '%q' "$1"
}

runtime_path() {
  require_runtime_selection

  local candidate="$1"
  if [[ "$candidate" != /* ]]; then
    local repo_relative_candidate="$ROBORACER_ROOT/$candidate"
    if [[ -e "$repo_relative_candidate" ]]; then
      candidate="$repo_relative_candidate"
    fi
  fi
  if [[ "$ROBORACER_RUNTIME_MODE" == "docker" && "$candidate" == "$ROBORACER_ROOT"* ]]; then
    printf '/workspace%s' "${candidate#$ROBORACER_ROOT}"
    return
  fi
  printf '%s' "$candidate"
}

workspace_root_for_runtime() {
  require_runtime_selection

  local workspace_name="$1"
  if [[ "$ROBORACER_RUNTIME_MODE" == "docker" ]]; then
    printf '/workspace/%s' "$workspace_name"
    return
  fi
  printf '%s/%s' "$ROBORACER_ROOT" "$workspace_name"
}

maybe_open_browser() {
  local should_open="$1"
  if [[ "$should_open" != "1" ]]; then
    return
  fi

  (
    for _ in $(seq 1 30); do
      if curl --silent --fail "${WEB_UI_URL}/api/healthz" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done

    if command -v open >/dev/null 2>&1; then
      open "$WEB_UI_URL" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$WEB_UI_URL" >/dev/null 2>&1 || true
    fi
  ) &
}

run_launch_command() {
  require_runtime_selection

  local workspace_root="$1"
  local prepare_command="$2"
  local build_command="$3"
  local launch_command="$4"
  local do_build="$5"

  local command="set -eo pipefail"
  command+=" && set +u && source /opt/ros/humble/setup.bash && set -u"
  command+=" && cd $(quote_for_bash "$workspace_root")"
  if [[ -n "$prepare_command" ]]; then
    command+=" && ${prepare_command}"
  fi
  if [[ "$do_build" == "1" ]]; then
    command+=" && ${build_command}"
  fi
  command+=" && set +u && source install/setup.bash && set -u"
  command+=" && ${launch_command}"

  (
    cd "$ROBORACER_ROOT"
    if [[ "$ROBORACER_RUNTIME_MODE" == "docker" ]]; then
      exec docker compose exec ubuntu bash -lc "$command"
    fi
    exec bash -lc "$command"
  )
}
