import pytest

from local_ai.config import Config
from local_ai.inference import OllamaError, generate


def _config(host: str) -> Config:
    return Config(
        ollama_host=host,
        model_name="qwen3:4b",
        request_timeout_seconds=2,
        debug=False,
        mlx_checkpoint_path="models/qwen3-4b-mlx-4bit",
    )


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        generate(_config("http://localhost:11434"), "   ")


def test_generate_reports_unreachable_host():
    # Port 1 is reserved and nothing listens there, so this fails fast
    # with connection-refused instead of actually waiting on a network call.
    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        generate(_config("http://localhost:1"), "hello")
