"""Command-line interface for local-ai.

`ask`, `chat`, and `models` talk to a local Ollama instance (Milestone 2).
`dataset validate`/`dataset prepare` work on local JSONL files
(Milestone 3). `evaluate` scores the local model against a reference
JSONL dataset (Milestone 4). `train run`/`train compare` do LoRA
fine-tuning via Apple MLX and base-vs-tuned comparison (Milestone 5;
see docs/TRAINING.md). No command here calls a hosted AI API. `train run`
never launches training without the --yes flag, and never downloads a
model checkpoint itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console

from local_ai import dataset, evaluation, inference, training
from local_ai.config import Config, ConfigError, load_config

app = typer.Typer(
    name="local-ai",
    help="Local-first AI CLI: run prompts against a locally hosted model.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset validation and preparation (Milestone 3).")
app.add_typer(dataset_app, name="dataset")
train_app = typer.Typer(help="LoRA fine-tuning via Apple MLX and base-vs-tuned comparison (Milestone 5).")
app.add_typer(train_app, name="train")

console = Console()


def _load_config_or_exit() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def ask(prompt: str = typer.Argument(..., help="The question or instruction to send to the model.")) -> None:
    """Send a single prompt to the local model and print the answer."""
    config = _load_config_or_exit()
    try:
        with console.status(f"[cyan]Thinking ({config.model_name})...", spinner="dots"):
            answer = inference.generate(config, prompt)
    except inference.OllamaError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(answer)


@app.command()
def chat() -> None:
    """Start an interactive chat loop with the local model. Type 'exit' or 'quit' to leave."""
    config = _load_config_or_exit()
    history: list[dict[str, str]] = []
    typer.echo(f"Chatting with {config.model_name} (Ctrl+C or 'exit' to quit)")
    while True:
        try:
            user_input = typer.prompt("you")
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nExiting.")
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            typer.echo("Exiting.")
            break
        if not user_input.strip():
            continue
        history.append({"role": "user", "content": user_input})
        try:
            with console.status(f"[cyan]Thinking ({config.model_name})...", spinner="dots"):
                reply = inference.chat(config, history)
        except inference.OllamaError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            history.pop()
            continue
        history.append({"role": "assistant", "content": reply})
        typer.echo(f"model: {reply}")


@app.command()
def models() -> None:
    """Show the configured model and whether it's pulled locally in Ollama."""
    config = _load_config_or_exit()
    typer.echo(f"Configured model: {config.model_name}")
    typer.echo(f"Configured inference host: {config.ollama_host}")
    typer.echo(f"Request timeout: {config.request_timeout_seconds}s")
    try:
        local_models = inference.list_local_models(config)
    except inference.OllamaError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if config.model_name in local_models:
        typer.secho(f"'{config.model_name}' is pulled and ready.", fg=typer.colors.GREEN)
    else:
        typer.secho(
            f"'{config.model_name}' is not pulled yet. Run: ollama pull {config.model_name}",
            fg=typer.colors.YELLOW,
        )
    if local_models:
        typer.echo("Locally available models: " + ", ".join(local_models))


@dataset_app.command("validate")
def dataset_validate(path: str = typer.Argument(..., help="Path to a JSONL dataset file.")) -> None:
    """Validate a JSONL dataset file. Does not modify it."""
    input_path = Path(path)
    if not input_path.is_file():
        typer.secho(f"File not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    result = dataset.validate_jsonl(input_path)
    typer.echo(
        f"Checked {result.total_lines} line(s): {result.valid_count} valid, "
        f"{len(result.errors)} invalid."
    )
    max_shown = 50
    for error in result.errors[:max_shown]:
        typer.secho(f"  line {error.line_number}: {error.reason}", fg=typer.colors.YELLOW)
    if len(result.errors) > max_shown:
        typer.echo(f"  ... and {len(result.errors) - max_shown} more")

    if not result.is_valid:
        raise typer.Exit(code=1)
    typer.secho("Dataset is valid.", fg=typer.colors.GREEN)


@dataset_app.command("prepare")
def dataset_prepare(
    path: str = typer.Argument(..., help="Path to a raw JSONL dataset file. Never modified."),
    output: str = typer.Option(
        ..., "--output", help="Where to write the normalized train split, e.g. data/processed/train.jsonl."
    ),
    val_ratio: float = typer.Option(
        0.0, "--val-ratio", help="Fraction of deduplicated valid records to hold out for validation."
    ),
    test_ratio: float = typer.Option(
        0.0, "--test-ratio", help="Fraction of deduplicated valid records to hold out for testing."
    ),
    seed: int = typer.Option(
        42, "--seed", help="Random seed for the shuffle before splitting (fixed by default for reproducibility)."
    ),
) -> None:
    """Validate, normalize, deduplicate, and optionally split a raw JSONL dataset.

    Invalid lines are skipped and reported; the raw input file is never
    modified. If --val-ratio/--test-ratio are set, val.jsonl/test.jsonl
    are written next to --output in the same directory.
    """
    input_path = Path(path)
    if not input_path.is_file():
        typer.secho(f"File not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        result = dataset.prepare_dataset(
            input_path, Path(output), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"Read {result.total_lines} line(s): {result.valid_records} valid "
        f"({result.duplicate_records} duplicate, skipped), {result.invalid_records} invalid."
    )
    if result.invalid_records:
        typer.secho(
            f"{result.invalid_records} invalid line(s) were skipped. "
            f"Run 'local-ai dataset validate {path}' to see details.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"Wrote {result.train_count} train record(s) to {result.train_path}")
    if result.val_path:
        typer.echo(f"Wrote {result.val_count} val record(s) to {result.val_path}")
    if result.test_path:
        typer.echo(f"Wrote {result.test_count} test record(s) to {result.test_path}")

    if result.train_count == 0 and not result.val_path and not result.test_path:
        typer.secho("No valid records to write.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@train_app.command("run")
def train_run(
    data_path: str = typer.Argument(..., help="Path to a raw JSONL training dataset (same schema as dataset/evaluate)."),
    checkpoint: str = typer.Option(
        None, "--checkpoint", help="Local path to the MLX-format base checkpoint. Defaults to LOCAL_AI_MLX_CHECKPOINT."
    ),
    output_dir: str = typer.Option("artifacts/lora", "--output-dir", help="Base directory for a new timestamped run."),
    rank: int = typer.Option(8, "--rank", help="LoRA rank."),
    layers: int = typer.Option(8, "--layers", help="Number of (final) model layers to fine-tune."),
    iters: int = typer.Option(200, "--iters", help="Training iterations."),
    batch_size: int = typer.Option(1, "--batch-size", help="Minibatch size."),
    learning_rate: float = typer.Option(1e-5, "--lr", help="Learning rate."),
    val_ratio: float = typer.Option(0.15, "--val-ratio", help="Fraction of data held out for validation during training."),
    seed: int = typer.Option(42, "--seed", help="Random seed for the data split and training."),
    yes: bool = typer.Option(False, "--yes", help="Actually launch training. Without this flag, only prints the plan."),
) -> None:
    """Prepare data and (with --yes) LoRA fine-tune the local MLX checkpoint.

    Without --yes, this only validates the dataset/checkpoint and prints
    a dry-run plan (sizes, rough memory/time estimates) - it never trains
    silently. Never downloads the base checkpoint itself; see
    docs/TRAINING.md. Fine-tuning here does not guarantee factual accuracy.
    """
    config = _load_config_or_exit()
    checkpoint_path = Path(checkpoint) if checkpoint else Path(config.mlx_checkpoint_path)
    input_path = Path(data_path)
    if not input_path.is_file():
        typer.secho(f"File not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        training.check_mlx_lm_available()
    except training.TrainingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    run_dir = training.new_run_dir(Path(output_dir), checkpoint_path.name)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    prep_result = training.prepare_mlx_data_dir(input_path, data_dir, val_ratio=val_ratio, seed=seed)
    typer.echo(
        f"Prepared training data: {prep_result.train_count} train, {prep_result.val_count} val "
        f"({prep_result.invalid_records} invalid line(s) skipped, {prep_result.duplicate_records} duplicate)."
    )
    if prep_result.train_count == 0:
        typer.secho("No valid training records - nothing to train on.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    train_config = training.TrainingConfig(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=run_dir,
        rank=rank,
        num_layers=layers,
        iters=iters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    typer.echo(training.build_plan_summary(train_config, prep_result.train_count))

    try:
        training.check_checkpoint_available(checkpoint_path)
    except training.TrainingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if not yes:
        typer.secho(
            "Dry run only (no --yes passed) - no training was started.",
            fg=typer.colors.YELLOW,
        )
        return

    typer.secho(f"Starting LoRA training (adapter -> {run_dir}) ...", fg=typer.colors.CYAN)
    try:
        training.run_training(train_config)
    except subprocess.CalledProcessError as exc:
        typer.secho(f"mlx_lm training process failed (exit code {exc.returncode}).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Training complete. Adapter + metadata written to {run_dir}", fg=typer.colors.GREEN)


@train_app.command("compare")
def train_compare(
    eval_path: str = typer.Argument(..., help="Path to a held-out JSONL evaluation dataset (same schema as `evaluate`)."),
    adapter: str = typer.Option(..., "--adapter", help="Path to a trained adapter directory (output of `train run`)."),
    checkpoint: str = typer.Option(
        None, "--checkpoint", help="Local path to the MLX-format base checkpoint. Defaults to LOCAL_AI_MLX_CHECKPOINT."
    ),
    output_dir: str = typer.Option(
        "artifacts/lora-comparisons", "--output-dir", help="Base directory for a new timestamped comparison run."
    ),
    max_tokens: int = typer.Option(
        400,
        "--max-tokens",
        help="Max generation tokens per side. This model's <think> reasoning can run past 200 tokens before the answer even appears, so the default is more generous than that - but may still truncate; check output for cut-off text before trusting scores.",
    ),
) -> None:
    """Compare the base checkpoint against base+adapter on the same held-out eval set.

    Both sides run through the same MLX runtime so the comparison isolates
    the fine-tuning effect. Reuses evaluate's token_f1/char_similarity
    metrics - see docs/EVALUATION.md and docs/TRAINING.md.
    """
    config = _load_config_or_exit()
    checkpoint_path = Path(checkpoint) if checkpoint else Path(config.mlx_checkpoint_path)
    adapter_path = Path(adapter)
    input_path = Path(eval_path)
    if not input_path.is_file():
        typer.secho(f"File not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    examples, total_lines, skipped = evaluation.load_eval_examples(input_path)
    if not examples:
        typer.secho("No valid evaluation examples found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        with console.status(f"[cyan]Comparing base vs. tuned on {len(examples)} example(s)...", spinner="dots"):
            results = training.compare_base_and_tuned(checkpoint_path, adapter_path, examples, max_tokens=max_tokens)
    except training.TrainingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    run_dir = training.new_run_dir(Path(output_dir), checkpoint_path.name)
    run_dir.mkdir(parents=True)
    comparison_path = training.write_comparison(run_dir, results)

    mean_base_f1 = sum(r.base_token_f1 for r in results) / len(results)
    mean_tuned_f1 = sum(r.tuned_token_f1 for r in results) / len(results)
    typer.echo(f"Compared {len(results)} example(s):")
    typer.echo(f"  base mean_token_f1:  {mean_base_f1:.2f}")
    typer.echo(f"  tuned mean_token_f1: {mean_tuned_f1:.2f}")
    typer.secho(
        "This is a small sample - read the per-example rows before concluding anything; "
        "fine-tuning does not guarantee factual accuracy.",
        fg=typer.colors.YELLOW,
    )
    typer.secho(f"Comparison written to {comparison_path}", fg=typer.colors.GREEN)


@app.command()
def evaluate(
    dataset_path: str = typer.Argument(
        ...,
        help=(
            "Path to a JSONL evaluation dataset (same schema as training data; "
            "the last message in each record must be the reference 'assistant' answer)."
        ),
    ),
    output_dir: str = typer.Option(
        "artifacts/evaluations",
        "--output-dir",
        help="Directory to write a new timestamped evaluation run under. Never overwrites a previous run.",
    ),
    pass_threshold: float = typer.Option(
        0.5,
        "--pass-threshold",
        help="token_f1 score at/above which an example counts as a 'pass' in the summary (heuristic, not a correctness guarantee).",
    ),
) -> None:
    """Evaluate the local model against a JSONL dataset of reference Q&A pairs.

    Does not use any hosted AI API and does not change model weights.
    See docs/EVALUATION.md for the metrics and dataset schema.
    """
    config = _load_config_or_exit()
    input_path = Path(dataset_path)
    if not input_path.is_file():
        typer.secho(f"File not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    examples, total_lines, skipped_invalid = evaluation.load_eval_examples(input_path)
    if skipped_invalid:
        typer.secho(
            f"{skipped_invalid} of {total_lines} line(s) were skipped "
            "(invalid schema, or last message wasn't an 'assistant' reference answer).",
            fg=typer.colors.YELLOW,
        )
    if not examples:
        typer.secho("No valid evaluation examples found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        with console.status(
            f"[cyan]Evaluating {len(examples)} example(s) against {config.model_name}...", spinner="dots"
        ):
            results = evaluation.run_evaluation(config, examples, pass_threshold)
    except inference.OllamaError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    summary = evaluation.summarize(
        results, config.model_name, input_path, total_lines, skipped_invalid, pass_threshold
    )
    run_dir = evaluation.write_run(Path(output_dir), config.model_name, results, summary)

    typer.echo(f"Evaluated {summary.evaluated} example(s) with {config.model_name}:")
    typer.echo(f"  exact_match_rate:    {summary.exact_match_rate:.2f}")
    typer.echo(f"  mean_token_f1:       {summary.mean_token_f1:.2f}")
    typer.echo(f"  mean_char_similarity:{summary.mean_char_similarity:.2f}")
    typer.echo(f"  passed (f1 >= {pass_threshold:.2f}): {summary.passed}/{summary.evaluated}")
    typer.secho(f"Results written to {run_dir}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
