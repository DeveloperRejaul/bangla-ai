"""Command-line interface for local-ai.

`ask`, `chat`, and `models` talk to a local Ollama instance (Milestone 2).
`dataset`, `train`, and `evaluate` are still placeholders for Milestones
3-5. No command here calls a hosted AI API, and no command starts a
training job.
"""

from __future__ import annotations

import typer

from local_ai import inference
from local_ai.config import Config, ConfigError, load_config

app = typer.Typer(
    name="local-ai",
    help="Local-first AI CLI: run prompts against a locally hosted model.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Dataset validation and preparation (Milestone 3).")
app.add_typer(dataset_app, name="dataset")


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
    """Validate a JSONL dataset file."""
    _not_implemented("dataset validate", milestone=3)


@dataset_app.command("prepare")
def dataset_prepare(
    path: str = typer.Argument(..., help="Path to a raw JSONL dataset file."),
    output: str = typer.Option(..., "--output", help="Where to write the normalized dataset."),
) -> None:
    """Normalize a raw dataset into the processed dataset format."""
    _not_implemented("dataset prepare", milestone=3)


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
def evaluate() -> None:
    """Evaluate a model against a dataset (not yet implemented)."""
    _not_implemented("evaluate", milestone=4)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
