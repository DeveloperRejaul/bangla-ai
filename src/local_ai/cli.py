"""Command-line interface for local-ai.

`ask`, `chat`, and `models` talk to a local Ollama instance (Milestone 2).
`dataset validate`/`dataset prepare` work on local JSONL files
(Milestone 3). `evaluate` scores the local model against a reference
JSONL dataset (Milestone 4). `train` is still a placeholder for
Milestone 5. No command here calls a hosted AI API, and no command
starts a training job.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from local_ai import dataset, evaluation, inference
from local_ai.config import Config, ConfigError, load_config

app = typer.Typer(
    name="local-ai",
    help="Local-first AI CLI: run prompts against a locally hosted model.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset validation and preparation (Milestone 3).")
app.add_typer(dataset_app, name="dataset")

console = Console()


def _load_config_or_exit() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def _not_implemented(feature: str, milestone: int) -> None:
    typer.secho(
        f"'{feature}' is not implemented yet (planned for Milestone {milestone}). "
        "Configuration was validated successfully.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


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


@app.command()
def train() -> None:
    """Run a LoRA/PEFT fine-tuning job (experimental, not yet implemented)."""
    typer.secho(
        "EXPERIMENTAL: fine-tuning is not implemented yet (planned for Milestone 5). "
        "No training job has been started.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


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
