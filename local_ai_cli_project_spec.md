# Local-First AI CLI Project --- Implementation Specification

## 1. Project Goal

Build a self-contained, command-line-only AI project. A user runs a
command in the terminal, enters a prompt, and receives a generated
answer.

This is an AI/ML development project, not a website or web application.

### Required outcomes

-   A local CLI chat/ask command.
-   Run an open-weight language model locally; do not depend on OpenAI,
    Gemini, or another hosted AI API.
-   Support future fine-tuning or training experiments using the user's
    own datasets.
-   Package the environment and supporting services with Docker so it
    can be started, stopped, removed, and reproduced on another machine.
-   Provide clear setup, usage, dataset preparation, training,
    evaluation, and cleanup instructions.
-   Work on Apple Silicon macOS (the initial development machine is a
    MacBook Air M4 with 16 GB unified memory), while documenting
    platform limitations.

## 2. Important Scope and Constraints

-   No frontend, website, browser UI, public API server, authentication
    system, or database is required for the initial version.
-   The primary interface is a terminal CLI.
-   Do not expose model ports publicly.
-   Do not download or train a large model automatically without asking
    first; model and dataset downloads can be large.
-   Use open-weight models only after checking their license and usage
    restrictions.
-   Do not claim a model is "trained from scratch" when the project only
    performs inference or fine-tuning.
-   Prefer a small model and parameter-efficient fine-tuning (LoRA/QLoRA
    where supported) over full training.
-   Apple Silicon GPU support inside Docker can be limited. The initial
    Mac setup should run the CLI and local inference reliably; document
    that GPU-based training may require native macOS tools or a separate
    Linux machine with an NVIDIA GPU. Do not promise CUDA in Docker on a
    Mac.
-   Avoid unnecessary dependencies and keep the first version easy to
    remove.

## 3. Recommended Initial Stack

-   **CLI:** Python 3.11 or 3.12, using Typer (or argparse if a
    dependency-free CLI is preferred).
-   **Local inference:** Ollama as the simplest first option, with a
    model such as Qwen3 4B as an initial candidate. Confirm the
    currently available model tag and its license before use.
-   **Alternative inference path:** llama.cpp for GGUF models if a more
    self-contained deployment is needed.
-   **Dataset and experiment utilities:** Hugging Face `datasets` and
    `transformers` only when needed.
-   **Fine-tuning:** Hugging Face `peft` and `trl` for supported
    LoRA/QLoRA workflows; choose the actual training path based on
    hardware.
-   **Containerization:** Docker Compose for reproducible project
    services. Keep host-specific model storage in a named volume or a
    documented mounted directory.
-   **Tests:** pytest.
-   **Configuration:** `.env.example` plus validated environment
    variables. Never commit secrets.

Do not install every ML library in the initial container by default.
Split dependencies into minimal inference and optional training
requirements so the Docker image stays manageable.

## 4. Proposed Repository Layout

``` text
local-ai-cli/
├── README.md
├── pyproject.toml
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── .env.example
├── src/
│   └── local_ai/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── inference.py
│       ├── dataset.py
│       ├── training.py
│       └── evaluation.py
├── tests/
│   ├── test_cli.py
│   ├── test_dataset.py
│   └── test_config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── artifacts/
│   └── README.md
├── scripts/
│   ├── setup_mac.sh
│   ├── run_cli.sh
│   └── cleanup.sh
└── docs/
    ├── DATASETS.md
    ├── TRAINING.md
    ├── DOCKER.md
    └── TROUBLESHOOTING.md
```

Keep `data/` and model artifacts out of Git. Add suitable `.gitignore`
entries.

## 5. CLI Behavior

Implement commands similar to:

``` bash
local-ai ask "Explain photosynthesis in simple Bengali."
local-ai chat
local-ai models
local-ai dataset validate ./data/raw/my_dataset.jsonl
local-ai dataset prepare ./data/raw/my_dataset.jsonl --output ./data/processed/train.jsonl
local-ai train --config ./configs/train.yaml
local-ai evaluate --dataset ./data/processed/eval.jsonl
```

The `ask` command should: 1. Validate configuration. 2. Check that the
configured inference engine is available. 3. Send the prompt to the
local model. 4. Print the answer to stdout. 5. Return a helpful error if
the model is missing or unavailable.

The `chat` command should provide a simple interactive loop, allow
`exit`/`quit`, and handle Ctrl+C cleanly. Conversation history can
remain in memory for the initial version; persistent history is not
required.

Training commands must initially be clearly marked as experimental until
implemented and tested. Never make `local-ai train` silently run a
costly job.

## 6. Local Model and Hardware Plan

### First milestone

Use Ollama on the Mac host for the simplest local inference experience.
The CLI can call Ollama's local endpoint at `http://localhost:11434`
from the host process. If the CLI itself runs inside Docker, `localhost`
refers to the container, not the Mac host; handle this explicitly with
documented platform-specific configuration. Do not assume
`host.docker.internal` works identically on every platform without
testing.

Example host setup:

``` bash
# Install Ollama using its official installation instructions:
# https://ollama.com/download

ollama pull qwen3:4b
ollama run qwen3:4b
```

Model tags can change. Confirm the exact tag with the official Ollama
library before documenting a release.

### MacBook Air M4, 16 GB unified memory

-   Start with a small quantized model (around 4B parameters) and check
    actual memory use and response speed.
-   An 8B model may be possible with suitable quantization, but
    performance and available memory depend on context length and other
    running applications.
-   Do not promise that full fine-tuning of large models will work on
    this machine.
-   For real training experiments, first inspect the chosen framework's
    current Apple Silicon support. If the required training stack is
    unsupported or too slow, use a separate Linux/NVIDIA machine or a
    suitable rented GPU environment. This does not change the
    requirement that normal inference should not rely on a hosted AI
    API.

## 7. Docker Design

Use Docker Compose to make the Python CLI environment reproducible. Keep
the model runtime choice explicit.

Preferred initial design: - `cli` service: Python CLI container. -
Ollama: either run on the host for the Mac-first development workflow,
or define an optional `ollama` service for environments where
containerized inference is supported and tested. - Use named volumes for
downloaded model files only when Ollama runs as a container. - Mount
`./data` and `./artifacts` as local directories so datasets and outputs
remain accessible outside the container. - Do not bake model weights or
private datasets into the image. - Add health checks where supported. -
Pin dependency versions and use a non-root user in the application
container where practical.

Provide separate documented commands for host-Ollama mode and
container-Ollama mode. Verify networking on macOS and Linux rather than
assuming one Compose configuration works everywhere.

Example workflow (final commands must be tested by Claude on the target
environment):

``` bash
docker compose build
docker compose run --rm cli --help
docker compose run --rm cli ask "Write a short greeting in Bengali."
docker compose down
```

If the model runtime is a separate Compose service, include its startup
and health-check steps before running inference.

## 8. Dataset Format and Data Safety

Start with JSONL. Each line should be one valid JSON object. For
supervised fine-tuning, support a documented format such as:

``` json
{"messages":[{"role":"user","content":"বাংলাদেশের রাজধানী কী?"},{"role":"assistant","content":"বাংলাদেশের রাজধানী ঢাকা।"}]}
```

Requirements: - Validate JSON syntax, required fields, message roles,
and empty records. - Report invalid line numbers without printing
sensitive data. - Keep raw data unchanged; write normalized output to
`data/processed/`. - Document dataset provenance, license, consent, and
personal-data considerations. - Do not scrape private or copyrighted
material without permission. - Add train/validation/test split support
with a fixed random seed. - Avoid data leakage between splits and
duplicates where feasible. - Include a small synthetic sample for tests,
not a large real dataset. - Never start a training run unless the user
explicitly requests it and confirms the configuration.

## 9. Training and Fine-Tuning Strategy

The project should be structured so dataset-driven adaptation can be
added safely, but the initial deliverable is local inference plus
validated dataset tooling.

Implement training in stages: 1. Dataset validation and preprocessing.
2. Baseline evaluation before training. 3. A documented LoRA/PEFT
fine-tuning workflow for a model/framework/hardware combination that is
actually supported. 4. Save adapters and metadata under `artifacts/`. 5.
Evaluate the adapted model against a held-out test set and compare it
with the baseline. 6. Document how to load the adapter and how to revert
to the base model.

Do not imply that adding documents to a dataset automatically changes
model weights. If the goal is question-answering over a changing
collection of documents, explain retrieval-augmented generation (RAG) as
a separate optional future feature; it is not required for the first
version.

Include a training config with explicit values for model name, dataset
paths, output directory, epochs, batch size, learning rate, maximum
sequence length, and random seed. Validate resource requirements and
fail early if the selected configuration is incompatible.

## 10. Security, Privacy, and Reliability

-   Keep all inference local by default.
-   Do not send prompts, datasets, or weights to a hosted AI API.
-   Do not log full prompts or answers by default; provide an opt-in
    debug mode.
-   Validate user input and handle long prompts gracefully.
-   Never execute model-generated shell commands automatically.
-   Do not mount the Docker socket into the application container.
-   Avoid privileged containers.
-   Keep secrets out of logs and Git.
-   Provide clear errors, timeouts, and a way to interrupt a request.
-   Include license notices for code, model weights, and datasets as
    applicable.

## 11. Testing and Acceptance Criteria

The project is not complete until these are verified: -
`python -m pytest` passes. - The CLI help command works. - Dataset
validation tests include valid and invalid JSONL. - Configuration
validation tests catch missing settings. - A mocked inference test
verifies CLI behavior without requiring a model download. - An
end-to-end local inference smoke test is documented and run when
Ollama/model access is available. - Docker image builds successfully on
the target platform. - A container can run `--help`; the documented
inference workflow works on the target platform. - `docker compose down`
stops project services; `docker compose down -v` is documented as
destructive because it removes named volumes. - Cleanup instructions
remove containers/images/volumes only when the user explicitly chooses
to do so. - Documentation distinguishes tests that passed from tests not
run.

## 12. Implementation Milestones

### Milestone 1 --- Clean project foundation

Create repository structure, Python package, CLI skeleton, config
handling, tests, Dockerfile, Compose file, `.gitignore`, and README. No
training yet.

### Milestone 2 --- Local inference

Integrate the local Ollama runtime, implement `ask`, `chat`, and
`models`, handle errors and timeouts, and test a real prompt.

### Milestone 3 --- Dataset tools

Implement JSONL validation, preprocessing, fixed-seed splitting, sample
dataset, and documentation.

### Milestone 4 --- Evaluation

Add a repeatable small evaluation command and baseline report.

### Milestone 5 --- Fine-tuning

Implement one supported LoRA/PEFT path only after verifying hardware and
framework compatibility. Keep training dependencies optional.

### Milestone 6 --- Reproducibility and portability

Test Docker workflows on macOS and Linux where available. Document
host-versus-container networking and platform-specific limitations.

### Milestone 7 --- Final quality check

Run tests, build images, check lint/type issues if configured, review
licenses, verify cleanup, and update README with exact commands and
actual test results.

Do not start a later milestone until the current milestone's tests and
acceptance criteria pass.

## 13. Documentation Required

The repository must include: - Quick start for macOS Apple Silicon. -
Docker setup and exact commands. - CLI usage examples. - Model selection
and license notes. - Dataset schema and sample data. - Fine-tuning
prerequisites, limitations, and commands. - Evaluation instructions. -
Troubleshooting for memory pressure, missing model, connection refused,
Docker networking, and unsupported training hardware. - Cleanup/removal
instructions. - A section listing what has been implemented, tested, not
tested, and known limitations.

## 14. Copy-Paste Build Prompt for Claude

You are a senior ML infrastructure and Python engineer. Build the
project described in this specification from an empty directory. The
user has a MacBook Air M4 with 16 GB unified memory and currently has no
repository or code.

Follow these rules strictly:

1.  Start from zero. Do not assume files, dependencies, or a repository
    already exist.
2.  This is a command-line-only local AI/ML project. Do not build a
    website, frontend, public API server, login system, or unrelated
    services.
3.  Do not use OpenAI, Gemini, or any other hosted AI inference API. Use
    a locally controlled open-weight model through a local inference
    runtime.
4.  Docker must be used to reproduce and manage the Python project
    environment. Explicitly account for Apple Silicon/macOS networking
    and the limitations of GPU training in Docker on a Mac.
5.  The initial goal is a working local CLI (`ask` and interactive
    `chat`). Dataset validation and future fine-tuning support come
    after local inference works.
6.  Use a small model first. Check the current official model tag and
    license before relying on it. Do not automatically download large
    weights or start training.
7.  Do not claim that the model is trained from scratch. Implement a
    realistic, hardware-compatible fine-tuning path only in a later
    milestone.
8.  Before each milestone, state the objective, files to create/change,
    commands to run, and expected results. Then implement only that
    milestone.
9.  Provide complete file contents or make the actual repository changes
    if your environment supports it. Never provide pseudocode as if it
    were working code.
10. After each milestone, run the relevant tests and commands. Report
    the exact commands run and actual results. Clearly identify anything
    you could not test.
11. Never overwrite user data silently. In this initial empty project,
    create files cleanly and explain each command. Ask before
    destructive cleanup or deleting volumes.
12. Keep dependencies minimal, pin versions where reasonable, use
    `.gitignore` for datasets and model artifacts, and do not put model
    weights into the Docker image.
13. Do not expose model ports publicly or execute model-generated shell
    commands.
14. If a technical choice depends on current versions or hardware
    support, verify it against official documentation instead of
    guessing.
15. If a command requires the user to do something on their Mac, give
    one copy-paste-ready command at a time, explain what it does, and
    wait for the result before proceeding when the result affects the
    next step.

Begin with Milestone 1 only. Give the exact steps to create the project
directory and files, then implement and test the CLI skeleton. Do not
jump ahead to training or deployment. The project is complete only when
the acceptance criteria have actually been verified.
