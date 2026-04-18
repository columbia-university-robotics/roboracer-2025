#!/usr/bin/env bash

set -euo pipefail

ROBORACER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROBORACER_ROOT
readonly WEB_UI_URL="http://localhost:8081"

ensure_ros_container() {
  (
    cd "$ROBORACER_ROOT"
    docker compose up -d ubuntu >/dev/null
  )
}

quote_for_bash() {
  printf '%q' "$1"
}

to_container_path() {
  local candidate="$1"
  if [[ "$candidate" == "$ROBORACER_ROOT"* ]]; then
    printf '/workspace%s' "${candidate#$ROBORACER_ROOT}"
    return
  fi
  printf '%s' "$candidate"
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

run_launch_in_container() {
  local workspace_root="$1"
  local build_command="$2"
  local launch_command="$3"
  local do_build="$4"

  local command="set -eo pipefail"
  command+=" && set +u"
  command+=" && source /opt/ros/humble/setup.bash"
  command+=" && cd ${workspace_root}"
  if [[ "$do_build" == "1" ]]; then
    command+=" && ${build_command}"
  fi
  command+=" && source install/setup.bash"
  command+=" && set -u"
  command+=" && ${launch_command}"

  (
    cd "$ROBORACER_ROOT"
    exec docker compose exec ubuntu bash -lc "$command"
  )
}
