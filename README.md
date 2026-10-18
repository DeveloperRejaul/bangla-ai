# local-ai

A local-first, command-line-only AI project: no frontend, no hosted AI
API, no public server. You type a prompt in the terminal; a model
running on your own machine answers.

This is Milestone 1 of a staged build (see [Project status](#project-status)
below). Local inference (`ask`/`chat` actually talking to a model) lands
in Milestone 2.

## Quick start (macOS, Apple Silicon)

Requires Python 3.11-3.13. This machine had 3.13 and 3.14 installed; 3.13
is used below as the closer match to what's broadly tested with the ML
libraries this project will add later.

```bash
scripts/setup_mac.sh
source .venv/bin/activate
local-ai --help
```

`scripts/setup_mac.sh` creates a `.venv`, installs the package in
editable mode with dev dependencies, and prints next steps.

## CLI usage

```bash
local-ai --help
local-ai models              # shows configured model/host (no live check yet)
local-ai ask "hello"         # validates config; reports "not implemented yet" (Milestone 2)
local-ai chat                # same, for the interactive loop (Milestone 2)
local-ai dataset validate ./data/raw/example.jsonl   # Milestone 3
local-ai train                                       # always experimental, never runs silently
local-ai evaluate                                     # Milestone 4
```

Configuration comes from environment variables (see
[.env.example](.env.example)): `LOCAL_AI_OLLAMA_HOST`,
`LOCAL_AI_MODEL_NAME`, `LOCAL_AI_TIMEOUT_SECONDS`, `LOCAL_AI_DEBUG`.
Copy `.env.example` to `.env` and adjust if needed; `local-ai` reads
real environment variables directly (export them, or `source .env`, or
use the Docker Compose `env_file` path below).

## Docker

See [docs/DOCKER.md](docs/DOCKER.md) for the full design and commands.
Short version:

```bash
docker compose build
docker compose run --rm cli --help
docker compose down
```

Docker was not installed on the development machine as of this
milestone, so the build/run commands above are documented but
**not yet verified** — see [Project status](#project-status).

## Local model (Milestone 2 preview)

This project will use [Ollama](https://ollama.com) running on the Mac
host, with a small model (candidate: `qwen3:4b` — confirm the current
tag and license at https://ollama.com/library before pulling it; model
tags change). No prompts, datasets, or weights are sent to a hosted AI
API.

## Datasets and fine-tuning

Not implemented yet. See [local_ai_cli_project_spec.md](local_ai_cli_project_spec.md)
for the full plan: JSONL dataset validation (Milestone 3), evaluation
(Milestone 4), and an optional LoRA/PEFT fine-tuning workflow
(Milestone 5) gated on verifying hardware/framework support first.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

## Project status

**Implemented and tested:**
- Project structure, packaging (`pyproject.toml`), CLI skeleton (Typer).
- Config loading/validation from environment variables.
- `local-ai --help`, `local-ai models`.
- `ask`/`chat`/`dataset validate`/`dataset prepare`/`train`/`evaluate`
  exist, validate config where applicable, and clearly report that
  their real functionality is not implemented yet (no command silently
  pretends to work or starts a job).
- Unit tests for config validation and CLI behavior (`pytest`).
  Verified: `python -m pytest` → 14 passed. `local-ai --help`, `local-ai
  models`, `local-ai ask "hello"` (exit 2, as designed), and `local-ai
  train` (exit 2, as designed) were run manually and behaved as
  documented above.

**Written but not tested:**
- `Dockerfile` / `compose.yaml` — Docker is not installed on this
  machine yet. Build/run commands in [docs/DOCKER.md](docs/DOCKER.md)
  need to be run once Docker Desktop is available.

**Not implemented (future milestones):**
- Actual local inference via Ollama (`ask`, `chat`, live `models` check) — Milestone 2.
- Dataset validation/preparation — Milestone 3.
- Evaluation — Milestone 4.
- LoRA/PEFT fine-tuning — Milestone 5.

**Known limitations:**
- Python 3.11/3.12 (as originally planned) were not present on this
  machine; 3.13 is used instead. Revisit if a dependency needs an exact
  minor version.
- `typer<0.14` (as initially pinned) is incompatible with current
  `click` releases (`TypeError: Parameter.make_metavar() missing 1
  required positional argument: 'ctx'` on `--help`). Fixed by pinning
  `typer>=0.15,<0.28`; re-verified working after the fix.
- GPU-accelerated training inside Docker on Apple Silicon is not
  supported by Docker itself; later milestones will document running
  training natively on macOS or on a separate Linux/NVIDIA machine
  instead.
