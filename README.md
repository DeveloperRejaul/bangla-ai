# local-ai

A local-first, command-line-only AI project: no frontend, no hosted AI
API, no public server. You type a prompt in the terminal; a model
running on your own machine answers.

This is Milestone 2 of a staged build (see [Project status](#project-status)
below): local inference via Ollama is wired up and working.

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
local-ai models                      # shows configured model/host + whether it's pulled
local-ai ask "Explain photosynthesis in simple Bengali."
local-ai chat                        # interactive loop; 'exit'/'quit' or Ctrl+C to leave
local-ai dataset validate ./data/raw/example.jsonl   # Milestone 3
local-ai train                                       # always experimental, never runs silently
local-ai evaluate                                     # Milestone 4
```

`ask`/`chat`/`models` require Ollama to be running locally
(`brew services start ollama` or `ollama serve`) and reachable at
`LOCAL_AI_OLLAMA_HOST` (default `http://localhost:11434`). If it isn't,
you get a clear "Could not reach Ollama" error, not a hang or a crash.

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

## Local model

Uses [Ollama](https://ollama.com) running on the Mac host (installed via
`brew install ollama`, started with `brew services start ollama`), with
`qwen3:4b` (2.5GB, Apache-2.0 license, confirmed at
https://ollama.com/library and https://huggingface.co/Qwen/Qwen3-4B as
of 2026-10-19 — re-check if much time has passed, tags/licenses can
change). No prompts, datasets, or weights are sent to a hosted AI API;
the only network call is to `LOCAL_AI_OLLAMA_HOST`.

`qwen3:4b` is a "thinking" model — it reasons before answering, which is
slow on this hardware (a single response took over two minutes in
testing on this MacBook Air M4). The default timeout was raised to 180s
accordingly; see `.env.example`.

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
- Ollama installed (`brew install ollama`, running as a `brew services`
  background service) and `qwen3:4b` pulled.
- `ask`, `chat`, `models` wired to Ollama's HTTP API
  (`src/local_ai/inference.py`; stdlib `urllib`, no extra HTTP
  dependency). Verified end-to-end with real prompts, including a
  Bengali prompt producing a correct Bengali answer, and a real
  "Ollama unreachable" error path (stopped the service, confirmed the
  error message, restarted it).
- `dataset`/`train`/`evaluate` still exist as placeholders, validate
  config where applicable, and clearly report their target milestone
  (no command silently pretends to work or starts a job).
- Unit tests for config validation, CLI behavior (mocked inference, so
  tests don't need Ollama running), and inference error handling.
  Verified: `python -m pytest` → 20 passed.

**Written but not tested:**
- `Dockerfile` / `compose.yaml` — Docker is not installed on this
  machine yet. Build/run commands in [docs/DOCKER.md](docs/DOCKER.md)
  need to be run once Docker Desktop is available. In particular, the
  `host.docker.internal` networking path to reach the host's Ollama
  from inside a container is documented but not yet verified.

**Not implemented (future milestones):**
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
- Ollama was installed via Homebrew and registered as a `brew services`
  background service, so it now auto-starts at every login on this
  machine (not scoped to this project). Disable with
  `brew services stop ollama` if you don't want that.
- `qwen3:4b` responses are slow on this hardware (~2 minutes for a short
  prompt, cold or warm) because it's a reasoning/"thinking" model. If
  that's too slow for interactive use, a smaller tag such as
  `qwen3:1.7b` is worth trying instead — not yet benchmarked here.
