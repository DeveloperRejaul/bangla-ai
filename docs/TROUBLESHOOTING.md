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

## `local-ai ask`/`chat`/`models`: "Could not reach Ollama"

Ollama isn't running, or `LOCAL_AI_OLLAMA_HOST` points somewhere wrong.

```bash
brew services start ollama      # start it as a background service
# or, to run it in the foreground and see its logs:
ollama serve
```

Then confirm it answers: `curl http://localhost:11434/` should print
`Ollama is running`.

## `local-ai models` says the model "is not pulled yet"

Pull it (check the current tag/size/license at
https://ollama.com/library first if you're changing `LOCAL_AI_MODEL_NAME`):

```bash
ollama pull qwen3:4b
```

This is a multi-GB download; it was ~2.5GB for `qwen3:4b` as of this
writing. It runs in the foreground and shows progress — don't assume
it's done until `ollama list` shows the model.

## Memory pressure / slow or stalled responses

On a 16GB Apple Silicon Mac, close other memory-heavy apps before
running a 4B+ model, and avoid very long prompts/contexts. If responses
are consistently too slow, try a smaller tag (e.g. `qwen3:1.7b`) before
assuming something is broken.
