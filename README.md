# local-ai

A local-first, command-line-only AI project: no frontend, no hosted AI
API, no public server. You type a prompt in the terminal; a model
running on your own machine answers.

This is Milestone 5 (in progress) of a staged build (see
[Project status](#project-status) below): local inference via Ollama,
dataset validation/preparation, evaluation, and LoRA fine-tuning
scaffolding via Apple MLX are implemented. **No real training run has
been executed yet** — that step needs an explicit, separate
confirmation (see [docs/TRAINING.md](docs/TRAINING.md)).

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
local-ai train run ./data/raw/my_train.jsonl --checkpoint models/qwen3-4b-mlx-4bit
  # ^ dry-run plan only; add --yes to actually launch training. See docs/TRAINING.md
local-ai train compare my_eval.jsonl --checkpoint models/qwen3-4b-mlx-4bit --adapter artifacts/lora/<run>
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

## Datasets, evaluation, and fine-tuning

JSONL validation/preparation and evaluation are implemented — see
[docs/DATASETS.md](docs/DATASETS.md) (schema, commands, provenance/
license/consent) and [docs/EVALUATION.md](docs/EVALUATION.md) (metrics,
how to keep eval data separate from training data, and a real result
observed with `qwen3:4b`).

LoRA fine-tuning scaffolding is implemented (`local-ai train run`/
`train compare`, via Apple's MLX framework — see
[docs/TRAINING.md](docs/TRAINING.md) for the hardware compatibility
assessment and why MLX was chosen over `transformers`/`peft`/`trl`).
**A real training run has not been executed** — it needs a ~2.3GB
checkpoint download and an explicit `--yes`, both gated on confirmation
first.

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
- `train run`/`train compare` (`src/local_ai/training.py`): LoRA
  fine-tuning via Apple's MLX framework (`mlx`/`mlx-lm`, actually
  installed and its real CLI/Python API inspected — not guessed from
  memory) and base-vs-tuned comparison reusing `evaluation.py`'s
  metrics. `train run` reuses `dataset.prepare_dataset` (Milestone 3)
  to build the data directory, writes an explicit, reproducible YAML
  training config, and **only runs with `--yes`** — without it, it
  validates the dataset/checkpoint and prints a dry-run plan (rough
  memory/time estimates) and stops. **Never downloads the base
  checkpoint itself** — if missing, it prints the exact manual command
  and size and exits. Verified by hand: dry-run against the real
  `tests/fixtures/train_bangla_v1.jsonl` fixture (16 synthetic Bengali
  examples, topics disjoint from the eval fixtures) correctly prepared
  14/2 train/val records and then correctly refused to train since no
  checkpoint has been downloaded yet.
- Unit tests for config validation, CLI behavior (mocked inference, so
  tests don't need Ollama running), inference error handling, dataset
  validation/preparation, evaluation (metrics, example loading,
  run/summarize/write), and training (data prep, precondition checks,
  config/metadata writing, mocked `mlx_lm`/`subprocess`).
  Verified: `python -m pytest` → 62 passed.

**Written but not tested:**
- `Dockerfile` / `compose.yaml` — Docker is not installed on this
  machine yet. Build/run commands in [docs/DOCKER.md](docs/DOCKER.md)
  need to be run once Docker Desktop is available. In particular, the
  `host.docker.internal` networking path to reach the host's Ollama
  from inside a container is documented but not yet verified.
- **A real `train run --yes` has not been executed.** It needs an MLX
  checkpoint (~2.3GB, per the model card) that hasn't been downloaded,
  and that download plus the actual training run both need explicit
  confirmation first (see [docs/TRAINING.md](docs/TRAINING.md), which
  also has a full pre-training verification of checkpoint/dataset/
  config compatibility against the actually-installed `mlx-lm` version).

**Not implemented (future milestones):**
- None remaining from the original 5-milestone plan; Docker portability
  verification and final review are the only items left.

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
- `qwen3:4b`'s answer length relative to any single reference string is
  unpredictable (sometimes far more verbose, sometimes more terse than
  the reference), so `token_f1`/`passed` systematically undercounts
  correct answers for this model regardless of how references are
  worded — confirmed across two real evaluation runs (4/7 correct in a
  v2 run where only 1/7 "passed"). Always read `results.jsonl`, not
  just the summary; see the full investigation in
  [docs/EVALUATION.md](docs/EVALUATION.md). A `system` message
  explicitly instructing a Bengali response did reliably work, though,
  for datasets that need Bengali output.
- `qwen3:4b` has a couple of confirmed, repeatable factual gaps on
  Bangladesh-specific trivia (got the national flower and national bird
  wrong in testing, with different wrong answers across repeated runs
  on the flower question) — a real model limitation, not a scoring bug.
- `train` extras were switched from `transformers`/`peft`/`trl` to
  `mlx`/`mlx-lm` after verifying real framework support on this
  hardware (see [docs/TRAINING.md](docs/TRAINING.md)); `mlx-lm` itself
  pulls in `transformers` as its own transitive dependency (for
  tokenizer handling), which is fine — it's not something this
  project's code depends on directly.
- Fine-tuning does not guarantee factual accuracy, and on a dataset this
  small it mostly nudges style/format, not durable knowledge — it is
  not expected to fix `qwen3:4b`'s confirmed factual gaps above unless
  those exact facts are added to the training data.
