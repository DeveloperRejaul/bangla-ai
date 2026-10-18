# Troubleshooting

This grows as each milestone adds new failure modes. Milestone 1 covers
setup issues only — inference/Ollama/training troubleshooting will be
added in Milestones 2 and 5.

## `python3.13: command not found`

The project targets Python 3.11-3.13. Install one (e.g.
`brew install python@3.13`), or set `PYTHON_BIN` before running
`scripts/setup_mac.sh`:

```bash
PYTHON_BIN=python3.14 scripts/setup_mac.sh
```

## `local-ai: command not found`

The virtualenv isn't activated. Either run
`source .venv/bin/activate` first, or use `scripts/run_cli.sh <args>`.

## `ModuleNotFoundError: No module named 'local_ai'`

The package wasn't installed. Run `scripts/setup_mac.sh`, or manually:
`pip install -e ".[dev]"`.

## Docker: image won't build / `docker: command not found`

Docker wasn't installed on this machine when Milestone 1 was written.
Install Docker Desktop for Apple Silicon, then retry
`docker compose build`. See [DOCKER.md](DOCKER.md).
