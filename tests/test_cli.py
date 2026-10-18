from typer.testing import CliRunner

from local_ai import inference
from local_ai.cli import app

runner = CliRunner()


def test_help_exits_zero_and_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "chat", "models", "dataset", "train", "evaluate"):
        assert command in result.stdout


def test_ask_prints_model_response(monkeypatch):
    monkeypatch.setattr(inference, "generate", lambda config, prompt: f"echo: {prompt}")
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 0
    assert "echo: hello" in result.stdout


def test_ask_reports_ollama_error(monkeypatch):
    def _raise(config, prompt):
        raise inference.OllamaError("Could not reach Ollama at http://localhost:11434.")

    monkeypatch.setattr(inference, "generate", _raise)
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 1
    assert "Could not reach Ollama" in result.output


def test_models_reports_pulled_model(monkeypatch):
    monkeypatch.setattr(inference, "list_local_models", lambda config: ["qwen3:4b"])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "is pulled and ready" in result.output


def test_models_reports_missing_model(monkeypatch):
    monkeypatch.setattr(inference, "list_local_models", lambda config: [])
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "not pulled yet" in result.output
    assert "ollama pull" in result.output


def test_models_reports_ollama_unreachable(monkeypatch):
    def _raise(config):
        raise inference.OllamaError("Could not reach Ollama at http://localhost:11434.")

    monkeypatch.setattr(inference, "list_local_models", _raise)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "Could not reach Ollama" in result.output


def test_chat_exits_cleanly_on_exit_keyword(monkeypatch):
    result = runner.invoke(app, ["chat"], input="exit\n")
    assert result.exit_code == 0
    assert "Exiting." in result.output


def test_chat_round_trip(monkeypatch):
    monkeypatch.setattr(inference, "chat", lambda config, history: "hi there")
    result = runner.invoke(app, ["chat"], input="hello\nexit\n")
    assert result.exit_code == 0
    assert "hi there" in result.output


def test_dataset_validate_is_not_yet_implemented():
    result = runner.invoke(app, ["dataset", "validate", "some.jsonl"])
    assert result.exit_code == 2
    assert "Milestone 3" in result.output


def test_train_never_runs_silently():
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 2
    assert "EXPERIMENTAL" in result.output
    assert "No training job has been started" in result.output


def test_evaluate_is_not_yet_implemented():
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 2
    assert "Milestone 4" in result.output


def test_invalid_config_blocks_ask():
    result = runner.invoke(app, ["ask", "hello"], env={"LOCAL_AI_TIMEOUT_SECONDS": "not-a-number"})
    assert result.exit_code == 1
    assert "Configuration error" in result.output
