#!/usr/bin/env bash
# One-time local dev setup on macOS (Apple Silicon). Creates a virtualenv
# and installs the CLI in editable mode with dev dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.13}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.11-3.13, or set PYTHON_BIN." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then try: local-ai --help"
