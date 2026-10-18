"""Local inference via Ollama's HTTP API.

Only ever talks to the configured local Ollama host (see config.py).
No hosted AI API is contacted from this module.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from local_ai.config import Config


class OllamaError(Exception):
    """Raised when the local Ollama runtime is unreachable or returns an error."""


def _post(config: Config, path: str, payload: dict) -> dict:
    url = f"{config.ollama_host}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {config.ollama_host}. "
            "Is it running? Try: ollama serve (or `brew services start ollama`)."
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(
            f"Request to {config.ollama_host} timed out after {config.request_timeout_seconds}s."
        ) from exc


def _get(config: Config, path: str) -> dict:
    url = f"{config.ollama_host}{path}"
    try:
        with urllib.request.urlopen(url, timeout=config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {config.ollama_host}. "
            "Is it running? Try: ollama serve (or `brew services start ollama`)."
        ) from exc
    except TimeoutError as exc:
        raise OllamaError(
            f"Request to {config.ollama_host} timed out after {config.request_timeout_seconds}s."
        ) from exc


def list_local_models(config: Config) -> list[str]:
    """Return the names of models Ollama currently has pulled locally."""
    result = _get(config, "/api/tags")
    return [model["name"] for model in result.get("models", [])]


def is_model_available(config: Config) -> bool:
    return config.model_name in list_local_models(config)


def generate(config: Config, prompt: str) -> str:
    """Send a single prompt and return the model's full response text."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    result = _post(
        config,
        "/api/generate",
        {"model": config.model_name, "prompt": prompt, "stream": False},
    )
    if "error" in result:
        raise OllamaError(result["error"])
    return result.get("response", "")


def chat(config: Config, messages: list[dict[str, str]]) -> str:
    """Send a running conversation (list of {role, content}) and return the assistant's reply."""
    result = _post(
        config,
        "/api/chat",
        {"model": config.model_name, "messages": messages, "stream": False},
    )
    if "error" in result:
        raise OllamaError(result["error"])
    return result.get("message", {}).get("content", "")
