# Fine-tuning (Milestone 5)

## Fine-tuning vs. inference vs. RAG

- **Inference** (`ask`/`chat`): weights are fixed; a prompt goes in, a
  response comes out using only what's already in the weights plus the
  current conversation.
- **Fine-tuning** (this doc): further trains the model (here, a small
  LoRA adapter on top of it) on example input/output pairs to nudge
  behavior — tone, format, instruction-following, light domain
  adaptation. **It does not reliably inject new facts from a handful of
  examples, and it does not guarantee correctness on anything** —
  including facts it was trained on.
- **RAG** (not implemented, not needed for this project's current
  scope): weights stay completely fixed; at query time, relevant
  documents are retrieved and stuffed into the prompt as context. It's
  the right tool for "answer questions about a changing document
  collection" — fine-tuning is the wrong tool for that.

## Hardware compatibility assessment (MacBook Air M4, 16GB unified memory)

| Path | Verdict | Why |
|---|---|---|
| Full fine-tuning | **Ruled out** | Needs the full model + full optimizer state + gradients simultaneously — far beyond 16GB for any useful model. |
| `transformers` + `peft` (LoRA) on PyTorch's MPS backend | **Not recommended** | MPS is officially "beta": missing ops fall back to CPU, `device_map="auto"` layer offloading isn't supported (the whole model must fit in unified memory), no distributed training. Classic **QLoRA needs `bitsandbytes`, whose Apple Silicon/MPS support is partial/experimental** (an MPS backend exists as an unmerged/early PR, not a stable release) — so the 4-bit path doesn't have a reliable story here. Plain (non-quantized) LoRA would need the full bf16 4B model (~8GB) resident plus training overhead: tight, and far less tested on Mac than the option below. |
| **Apple MLX (`mlx-lm`) + a 4-bit MLX-format checkpoint + LoRA** | **Chosen** | MLX is Apple's own ML framework, built natively for Metal/unified memory (not emulating CUDA). MIT-licensed. There's a working real-world example of exactly this combination (a LoRA adapter trained on `mlx-community/Qwen3-4B-4bit`, with a documented perplexity improvement). `mlx_lm.lora`'s own built-in default config even points at a Qwen model, corroborating support for this family. |

**This project's `train` extras were changed from `transformers`/`peft`/`trl` to `mlx`/`mlx-lm`** for this reason — verified by actually installing `mlx-lm` and inspecting its real CLI/Python API (not assumed from memory), since framework support on Apple Silicon changes over time.

## What this project implements

- `local-ai train run DATA_PATH --checkpoint PATH [options]` — prepares
  the data (reuses `dataset.prepare_dataset`, Milestone 3) and, **only
  with `--yes`**, launches `mlx_lm`'s LoRA trainer via a generated YAML
  config (so LoRA rank, layers, iterations, batch size, learning rate,
  and seed are all explicit and reproducible, per the spec's training
  config requirement). Without `--yes` it only validates the dataset
  and checkpoint and prints a dry-run plan with rough memory/time
  estimates — it never trains silently.
- **The base model checkpoint is never downloaded automatically.** If
  `--checkpoint` doesn't exist on disk, the command stops and prints
  the exact manual command to fetch it (and its size), e.g.:
  ```bash
  huggingface-cli download mlx-community/Qwen3-4B-4bit --local-dir models/qwen3-4b-mlx-4bit
  ```
  This is a separate, deliberate step you run yourself (~2.3GB
  download) — not something `local-ai train` can trigger.
- `local-ai train compare EVAL_PATH --checkpoint PATH --adapter PATH` —
  runs the same held-out eval set through the base checkpoint and the
  base+adapter checkpoint, **both via the same `mlx_lm` runtime**, so
  the comparison isolates the fine-tuning effect rather than mixing in
  a quantization/runtime difference (comparing Ollama's GGUF base
  against an MLX-tuned adapter would confound those two things).
  Reuses `evaluation.py`'s `token_f1`/`char_similarity` metrics
  directly — see [docs/EVALUATION.md](EVALUATION.md).
- Every run writes to a new timestamped directory under
  `artifacts/lora/` (training) or `artifacts/lora-comparisons/`
  (comparison) — never overwrites a previous run, same pattern as
  `artifacts/evaluations/`.

## Training data

A small, synthetic Bengali dataset ships at
`tests/fixtures/train_bangla_v1.jsonl` (16 examples: greetings, simple
facts, and — reusing the technique validated in the Milestone 4
investigation — a few examples with an explicit "respond in Bengali"
`system` message). Its topics are deliberately disjoint from
`tests/fixtures/eval_valid.jsonl`/`eval_bangla_v2.jsonl` (different
facts entirely, not just a different file) so there's no train/eval
overlap — see [docs/EVALUATION.md](EVALUATION.md#keeping-eval-data-separate-from-training-data).

## Pre-training verification (2026-10-19)

Checked before downloading anything or training, per the project's own
rule that "additional large model files" and "resource-intensive
training" both need confirmation first:

**1. Checkpoint compatibility.** Fetched the actual
`mlx-community/Qwen3-4B-4bit` model card: **2.26GB**, Apache-2.0,
converted from `Qwen/Qwen3-4B` using `mlx-lm` 0.24.0 (this project has
0.32.2 installed — the safetensors weight format is stable across that
range; if it weren't, `mlx_lm.load()` would fail loudly and immediately
on the first call, before any training starts, not partway through).
Also inspected `mlx_lm/tuner/utils.py`'s `linear_to_lora_layers`:
it generically wraps any `nn.Linear`/`nn.QuantizedLinear`/`nn.Embedding`
submodule under `model.layers`, with no per-architecture allowlist —
and it explicitly handles `QuantizedLinear`, which is exactly what
makes LoRA-on-a-4-bit-base ("QLoRA" in spirit) work natively here,
unlike the bitsandbytes situation on PyTorch/MPS. This confirms the
checkpoint is structurally compatible with this project's LoRA workflow,
not just "probably fine."

**2. Config/dataset/tokenizer/output-path compatibility**, verified
against `mlx_lm/tuner/datasets.py`'s actual source (not assumed):
- `ChatDataset` expects exactly `{"messages": [...]}` per line — this
  project's schema already matches, with zero transformation needed.
- `load_local_dataset` looks for exactly `train.jsonl`/`valid.jsonl`/
  `test.jsonl` in the `--data` directory — confirms
  `prepare_mlx_data_dir`'s `val.jsonl` → `valid.jsonl` mirroring is the
  exactly correct filename (not a guess).
- **New finding, now documented**: `mask_prompt=True` (set by this
  project by default) computes the loss-masking offset based on
  whether the record's *last* message is `role: "assistant"`. This
  project's `dataset.py` schema only requires *at least one* assistant
  message anywhere in the record, not that it's necessarily last. All
  of `train_bangla_v1.jsonl`'s 16 examples do end with the assistant's
  reply (by construction), so this isn't an active bug here — but if
  you add training examples by hand, **make sure each record's last
  message is the assistant's response**, or prompt-masking will
  silently compute the wrong offset for that record (not a crash — a
  quietly lower-quality training signal for that example). Worth
  promoting to an explicit check in a future milestone if hand-written
  training data becomes common.
- The `lora_parameters` config key must be named `scale`, not `alpha`
  (mlx_lm raises a clear `KeyError` otherwise, with a pointer to this
  exact distinction) — this project's generated YAML already uses
  `scale`, confirmed correct by reading the source, not by luck.
- `tokenizer.apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True)`, used by `training._mlx_chat` for the
  base-vs-tuned comparison, is the identical pattern `mlx_lm`'s own
  `generate` CLI uses internally — confirmed by reading
  `mlx_lm/generate.py`, not assumed from general HF conventions.

**3. Limitations of 16 synthetic examples.** This is enough to smoke-test
that the pipeline runs end-to-end, and nothing more. Concretely:
- At the default `--iters 200` with `--batch-size 1` on 14 training
  records (after the 85/15 val split), the model sees each example
  roughly **14 times** over the run — a real memorization/overfitting
  risk on this few, distinct examples, not a generalization signal.
  **Recommendation: drop to `--iters 40-60` for a first real run on
  this dataset size**, to limit repetition, and treat any result as a
  pipeline smoke test, not a meaningful fine-tune.
- A *meaningful* style/format-adaptation experiment (what LoRA is
  actually good for) realistically needs on the order of **100-300
  examples** with real topic/phrasing variety — not just more of the
  same few patterns repeated. Quality bar for that dataset: every
  example factually correct if it makes factual claims (fine-tuning
  will happily teach the model to confidently repeat a wrong answer in
  your training data), consistent instruction formatting, varied
  sentence structure and length, and — per
  [docs/EVALUATION.md](EVALUATION.md) — topics disjoint from whatever
  you'll evaluate on. A "knowledge-injection" experiment realistically
  needs far more than that and is not what this milestone is built for
  (see "Fine-tuning vs. inference vs. RAG" above).

**4. Resource requirements, confirmed on this machine right now:**
- Disk: 53GB free (`df -h /`). Checkpoint download: 2.26GB, one-time.
  LoRA adapter output: tiny (rank 8 across 8 layers of a 4B model is on
  the order of a few MB of weights, plus a small metadata/config file)
  — not a meaningful addition to disk usage per run.
- Memory: 16GB total unified memory (`sysctl hw.memsize`). Estimated
  peak during training: **~4-6GB** (2.3GB quantized base + LoRA adapter
  + activations for short Bengali sequences with `max_seq_length=512`
  and `grad_checkpoint` enabled) — comfortably under 16GB on paper, but
  this is still an estimate, not a measurement on this exact machine.
- **Likely limitations on this hardware**: no dedicated GPU/VRAM means
  training runs on the same unified memory and Metal GPU the rest of
  macOS is using — expect it to be slower than a discrete-GPU machine,
  and expect to close memory-heavy apps before a longer run (same
  caution as `qwen3:4b` inference speed in [README.md](../README.md)).
  **Verify the memory/time estimates empirically with a short run
  (`--iters 10-20`) before committing to a longer one** — this is
  already the plan's own advice, now reinforced by the fact that these
  numbers are still estimates, not measurements.

## Before running this for real

Confirm with yourself (and, in this project's workflow, explicitly with
whoever's driving) before:
1. **Downloading the checkpoint** (~2.3GB disk, one-time).
2. **Running `train run --yes`** (compute/time; get an estimate from
   the dry-run plan first, then verify empirically with a short trial —
   e.g. `--iters 20` — before committing to a longer run).

Fine-tuning will not fix factual gaps unless the exact facts are in the
training data, and even then a handful of examples mostly teaches
style/format, not durable knowledge — see the Milestone 4 investigation
in [docs/EVALUATION.md](EVALUATION.md) for two confirmed, repeatable
factual gaps this model has that fine-tuning on this small dataset is
not expected to fix.

## Tests

`tests/test_training.py` and the `train`-related tests in
`tests/test_cli.py` cover data preparation, checkpoint/mlx-lm
precondition checks, config/metadata writing, and the comparison logic
— all with `subprocess.run` and `mlx_lm.load`/`generate` mocked, so the
suite never needs the real checkpoint or a real training run.
