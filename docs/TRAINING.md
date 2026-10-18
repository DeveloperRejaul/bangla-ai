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
  This is a separate, deliberate step you run yourself (~2.5GB
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

## Before running this for real

Confirm with yourself (and, in this project's workflow, explicitly with
whoever's driving) before:
1. **Downloading the checkpoint** (~2.5GB disk, one-time).
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
