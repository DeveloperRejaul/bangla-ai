"""Configuration loading and validation for the local-ai CLI.

All settings come from environment variables (optionally loaded from a
.env file by the shell or docker compose's env_file). Nothing here talks
to a hosted AI API: the only network target is the local inference
runtime (Ollama), configured via LOCAL_AI_OLLAMA_HOST.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    ollama_host: str
    model_name: str
    request_timeout_seconds: int
    debug: bool


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL_NAME = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 60


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config(env: dict[str, str] | None = None) -> Config:
    """Load and validate configuration from environment variables.

    Args:
        env: Optional mapping to read from instead of os.environ
            (used by tests to avoid mutating real process environment).

    Raises:
        ConfigError: if a provided value is present but invalid.
    """
    source = os.environ if env is None else env

    ollama_host = source.get("LOCAL_AI_OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    if not ollama_host.startswith(("http://", "https://")):
        raise ConfigError(
            f"LOCAL_AI_OLLAMA_HOST must start with http:// or https://, got: {ollama_host!r}"
        )

    model_name = source.get("LOCAL_AI_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    if not model_name:
        raise ConfigError("LOCAL_AI_MODEL_NAME must not be empty")

    timeout_raw = source.get("LOCAL_AI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        timeout_seconds = int(timeout_raw)
    except ValueError as exc:
        raise ConfigError(
            f"LOCAL_AI_TIMEOUT_SECONDS must be an integer, got: {timeout_raw!r}"
        ) from exc
    if timeout_seconds <= 0:
        raise ConfigError(
            f"LOCAL_AI_TIMEOUT_SECONDS must be a positive integer, got: {timeout_seconds}"
        )

    debug = _parse_bool(source.get("LOCAL_AI_DEBUG", "false"))

    return Config(
        ollama_host=ollama_host,
        model_name=model_name,
        request_timeout_seconds=timeout_seconds,
        debug=debug,
    )
