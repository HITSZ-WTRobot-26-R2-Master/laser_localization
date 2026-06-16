#!/usr/bin/env bash
set -euo pipefail

source_relaxed() {
  local setup_file="$1"
  local nounset_was_enabled=0

  case "$-" in
    *u*) nounset_was_enabled=1 ;;
  esac

  set +u
  # shellcheck disable=SC1090
  source "$setup_file"

  if [[ "$nounset_was_enabled" -eq 1 ]]; then
    set -u
  fi
}

source_relaxed "/opt/ros/${ROS_DISTRO}/setup.bash"
source_relaxed "${WORKSPACE_DIR}/install/setup.bash"

exec "$@"
