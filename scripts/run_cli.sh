#!/usr/bin/env bash
# Run the CLI from the local virtualenv, forwarding all arguments.
# Example: scripts/run_cli.sh models
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/local-ai ]; then
  echo "Error: .venv not set up. Run scripts/setup_mac.sh first." >&2
  exit 1
fi

exec .venv/bin/local-ai "$@"
