#!/usr/bin/env bash
# Stop and remove this project's Docker containers/images.
# Does NOT touch ./data or ./artifacts (they are host directories, not
# Docker volumes, so `docker compose down` never removes them).
#
# Usage:
#   scripts/cleanup.sh            # stop + remove containers
#   scripts/cleanup.sh --images   # also remove the built image
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found." >&2
  exit 1
fi

docker compose down

if [ "${1:-}" = "--images" ]; then
  read -r -p "Remove the local-ai-cli:dev image too? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    docker image rm local-ai-cli:dev || true
  fi
fi
