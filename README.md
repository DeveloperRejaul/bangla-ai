# local-ai

A local-first, command-line-only AI project: no frontend, no hosted AI
API, no public server. You type a prompt in the terminal; a model
running on your own machine answers.

This is Milestone 4 of a staged build (see [Project status](#project-status)
below): local inference via Ollama, dataset validation/preparation, and
evaluation are wired up and working.

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
local-ai dataset validate ./data/raw/my_dataset.jsonl
local-ai dataset prepare ./data/raw/my_dataset.jsonl --output ./data/processed/train.jsonl \
  --val-ratio 0.1 --test-ratio 0.1   # fixed-seed split, see docs/DATASETS.md
local-ai evaluate ./data/eval/my_eval.jsonl --output-dir artifacts/evaluations \
  --pass-threshold 0.5              # see docs/EVALUATION.md for the metrics
local-ai train                                       # always experimental, never runs silently
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

## Datasets and evaluation

JSONL validation/preparation and evaluation are implemented — see
[docs/DATASETS.md](docs/DATASETS.md) (schema, commands, provenance/
license/consent) and [docs/EVALUATION.md](docs/EVALUATION.md) (metrics,
how to keep eval data separate from training data, and a real result
observed with `qwen3:4b`). An optional LoRA/PEFT fine-tuning workflow
(Milestone 5, gated on verifying hardware/framework support) is not
implemented yet; see
[local_ai_cli_project_spec.md](local_ai_cli_project_spec.md) for the full plan.

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
- `dataset validate`/`dataset prepare` (`src/local_ai/dataset.py`,
  stdlib `json`/`random`, no new dependency): JSONL schema validation
  (required `messages`, valid roles, non-empty content, at least one
  user + one assistant message), normalization, deduplication, and
  fixed-seed train/val/test splitting with no leakage between splits.
  Error messages report line numbers and structural reasons only,
  never the line's actual content. Verified against committed synthetic
  fixtures (`tests/fixtures/*.jsonl`) both via the test suite and by
  hand: confirmed raw input is never modified, duplicate records are
  deduplicated, and splits with the same seed are byte-identical across
  runs.
- `evaluate` (`src/local_ai/evaluation.py`, stdlib `difflib`/`json`/
  `re`, no new dependency): scores the local model against a JSONL
  reference Q&A dataset (reuses the dataset.py schema; the last message
  per record is the reference answer) using three documented metrics —
  `exact_match`, `token_f1`, `char_similarity` (see
  [docs/EVALUATION.md](docs/EVALUATION.md)) — none of which is exact-match-only.
  Writes a new timestamped run under `artifacts/evaluations/` each time
  (`results.jsonl` + `summary.json`), never overwriting a previous run.
  Verified both with mocked inference (unit tests) and for real against
  the live `qwen3:4b` model on `tests/fixtures/eval_valid.jsonl` —
  confirmed the artifacts were written correctly and investigated the
  (low, honestly reported) scores by reading the raw predictions; see
  docs/EVALUATION.md for what that run actually showed.
- `train` still exists as a placeholder, clearly reporting its target
  milestone (no command silently pretends to work or starts a job).
- Unit tests for config validation, CLI behavior (mocked inference, so
  tests don't need Ollama running), inference error handling, dataset
  validation/preparation, and evaluation (metrics, example loading,
  run/summarize/write, all with mocked inference).
  Verified: `python -m pytest` → 47 passed.

**Written but not tested:**
- `Dockerfile` / `compose.yaml` — Docker is not installed on this
  machine yet. Build/run commands in [docs/DOCKER.md](docs/DOCKER.md)
  need to be run once Docker Desktop is available. In particular, the
  `host.docker.internal` networking path to reach the host's Ollama
  from inside a container is documented but not yet verified.

**Not implemented (future milestones):**
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
- `token_f1`/`char_similarity` score verbose-but-correct answers low
  when the reference answer is a bare word/phrase, because `qwen3:4b`
  answers in full sentences by default. Always check `results.jsonl`,
  not just the summary — see the real example in
  [docs/EVALUATION.md](docs/EVALUATION.md).
