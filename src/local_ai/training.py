"""LoRA fine-tuning via Apple's MLX framework (mlx-lm), and base-vs-tuned
comparison on a held-out evaluation set.

Why MLX and not transformers/peft/trl + bitsandbytes: full fine-tuning
and classic QLoRA need far more than 16GB unified memory, and
bitsandbytes' Apple Silicon/MPS support is partial/experimental (no
dedicated VRAM, no CUDA). MLX is Apple's own ML framework, built
natively for Metal/unified memory, and is the realistic path on this
hardware. See docs/TRAINING.md for the full assessment.

Design rules:
- This module never downloads the base model checkpoint. That's a
  multi-GB file the caller must fetch explicitly first (see
  check_checkpoint_available's error message).
- run_training() never asks for confirmation itself - the caller
  (cli.py) must already have gated the call behind an explicit user
  confirmation (the spec requires `local-ai train` never run silently).
- Fine-tuning on a handful of examples does not reliably inject new
  facts or guarantee accuracy; nothing here claims otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from local_ai import dataset as dataset_module
from local_ai import evaluation


class TrainingError(Exception):
    """Raised for any precondition that blocks a training run."""


@dataclass(frozen=True)
class TrainingConfig:
    checkpoint_path: Path
    data_dir: Path
    output_dir: Path
    rank: int = 8
    num_layers: int = 8
    iters: int = 200
    batch_size: int = 1
    learning_rate: float = 1e-5
    seed: int = 42
    max_seq_length: int = 512


@dataclass
class ComparisonResult:
    line_number: int
    question: str
    reference: str
    base_prediction: str
    tuned_prediction: str
    base_token_f1: float
    tuned_token_f1: float
    base_char_similarity: float
    tuned_char_similarity: float


def check_mlx_lm_available() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise TrainingError(
            "mlx-lm is not installed. Install the training extras with: pip install -e '.[train]'"
        ) from exc


def check_checkpoint_available(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise TrainingError(
            f"Base checkpoint not found at {checkpoint_path}. This project never "
            "downloads it automatically - fetch it explicitly first (see "
            "docs/TRAINING.md), e.g.:\n"
            f"  huggingface-cli download mlx-community/Qwen3-4B-4bit --local-dir {checkpoint_path}\n"
            "(~2.3GB download, per the model card; make sure you want to spend that bandwidth/disk first.)"
        )


def prepare_mlx_data_dir(
    raw_jsonl: Path, data_dir: Path, val_ratio: float = 0.15, seed: int = 42
) -> dataset_module.PrepareResult:
    """Reuse dataset.prepare_dataset (Milestone 3), then mirror val.jsonl to
    valid.jsonl - the exact filename mlx_lm.lora's --data directory expects
    (it looks for train.jsonl/valid.jsonl/test.jsonl)."""
    train_path = data_dir / "train.jsonl"
    result = dataset_module.prepare_dataset(
        raw_jsonl, train_path, val_ratio=val_ratio, test_ratio=0.0, seed=seed
    )
    if result.val_path is not None:
        (data_dir / "valid.jsonl").write_text(
            result.val_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return result


def estimate_requirements(config: TrainingConfig, num_examples: int) -> dict[str, str]:
    """Rough, clearly-labeled estimates - not guarantees. Verify on the
    actual machine with a short trial run before trusting these."""
    minutes_low = max(1, round(config.iters * max(num_examples, 1) / 4000))
    minutes_high = max(minutes_low + 1, round(config.iters * max(num_examples, 1) / 1500))
    return {
        "estimated_peak_memory_gb": (
            "~4-6GB (4-bit base + a small LoRA adapter + short-sequence activations; "
            "not measured on this machine yet)"
        ),
        "estimated_duration_minutes": f"~{minutes_low}-{minutes_high} (rough estimate only)",
        "trainable_params": (
            f"LoRA rank {config.rank} on the last {config.num_layers} layers only; "
            "the 4B base model stays frozen"
        ),
    }


def build_plan_summary(config: TrainingConfig, num_examples: int) -> str:
    est = estimate_requirements(config, num_examples)
    return "\n".join(
        [
            f"Base checkpoint: {config.checkpoint_path}",
            f"Training examples: {num_examples}",
            f"LoRA rank={config.rank}, layers={config.num_layers}, iters={config.iters}, "
            f"batch_size={config.batch_size}, lr={config.learning_rate}, seed={config.seed}",
            f"Output: {config.output_dir}",
            "Estimates (not guarantees - verify empirically on this machine):",
            f"  peak memory: {est['estimated_peak_memory_gb']}",
            f"  duration: {est['estimated_duration_minutes']}",
            f"  trainable params: {est['trainable_params']}",
            "Fine-tuning on this few examples will NOT reliably inject new facts "
            "or guarantee factual accuracy - it mainly nudges style/format.",
        ]
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def new_run_dir(base_dir: Path, slug: str) -> Path:
    """Timestamped, collision-safe run directory - never overwrites a
    previous run (same pattern as evaluation.write_run)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{timestamp}_{_slug(slug)}"
    run_dir = base_dir / base_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base_dir / f"{base_name}-{suffix}"
    return run_dir


def run_training(config: TrainingConfig) -> Path:
    """Actually invoke mlx_lm's LoRA trainer as a subprocess.

    The caller (cli.py) must have already gated this behind an explicit
    user confirmation - this function does not ask again, and never
    downloads the checkpoint itself.
    """
    check_mlx_lm_available()
    check_checkpoint_available(config.checkpoint_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    mlx_config = {
        "model": str(config.checkpoint_path),
        "train": True,
        "fine_tune_type": "lora",
        "data": str(config.data_dir),
        "seed": config.seed,
        "num_layers": config.num_layers,
        "batch_size": config.batch_size,
        "iters": config.iters,
        "learning_rate": config.learning_rate,
        "adapter_path": str(config.output_dir),
        "max_seq_length": config.max_seq_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "lora_parameters": {"rank": config.rank, "dropout": 0.0, "scale": 20.0},
    }
    config_path = config.output_dir / "mlx_lora_config.yaml"
    config_path.write_text(yaml.safe_dump(mlx_config, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)],
        check=True,
    )

    metadata = {
        "checkpoint_path": str(config.checkpoint_path),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "rank": config.rank,
        "num_layers": config.num_layers,
        "iters": config.iters,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "max_seq_length": config.max_seq_length,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config.output_dir


def _mlx_chat(model, tokenizer, messages: list[dict[str, str]], max_tokens: int = 200) -> str:
    from mlx_lm import generate as mlx_generate

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return mlx_generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)


def compare_base_and_tuned(
    checkpoint_path: Path,
    adapter_path: Path,
    eval_examples: list[evaluation.EvalExample],
) -> list[ComparisonResult]:
    """Run the same held-out eval examples through the base checkpoint and
    the base+adapter checkpoint, using the same mlx_lm runtime for both so
    the comparison isolates the fine-tuning effect (not a quantization or
    runtime difference). Reuses evaluation.py's metric functions directly."""
    check_mlx_lm_available()
    check_checkpoint_available(checkpoint_path)
    if not adapter_path.exists():
        raise TrainingError(f"Adapter not found at {adapter_path}")

    from mlx_lm import load as mlx_load

    base_model, base_tokenizer = mlx_load(str(checkpoint_path))
    tuned_model, tuned_tokenizer = mlx_load(str(checkpoint_path), adapter_path=str(adapter_path))

    results: list[ComparisonResult] = []
    for example in eval_examples:
        base_pred = _mlx_chat(base_model, base_tokenizer, example.context_messages)
        tuned_pred = _mlx_chat(tuned_model, tuned_tokenizer, example.context_messages)
        results.append(
            ComparisonResult(
                line_number=example.line_number,
                question=example.question,
                reference=example.reference,
                base_prediction=base_pred,
                tuned_prediction=tuned_pred,
                base_token_f1=evaluation.token_f1(base_pred, example.reference),
                tuned_token_f1=evaluation.token_f1(tuned_pred, example.reference),
                base_char_similarity=evaluation.char_similarity(base_pred, example.reference),
                tuned_char_similarity=evaluation.char_similarity(tuned_pred, example.reference),
            )
        )
    return results


def write_comparison(output_dir: Path, results: list[ComparisonResult]) -> Path:
    path = output_dir / "comparison.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return path
