from typer.testing import CliRunner

from local_ai.cli import app

runner = CliRunner()


def test_help_exits_zero_and_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "chat", "models", "dataset", "train", "evaluate"):
        assert command in result.stdout


def test_models_shows_configuration():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Configured model:" in result.stdout
    assert "Configured inference host:" in result.stdout


def test_ask_is_not_yet_implemented_but_validates_config():
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output
    assert "Milestone 2" in result.output


def test_chat_is_not_yet_implemented():
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 2
    assert "Milestone 2" in result.output


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
