from local_ai.config import ConfigError, load_config


def test_defaults_when_env_empty():
    config = load_config(env={})
    assert config.ollama_host == "http://localhost:11434"
    assert config.model_name == "qwen3:4b"
    assert config.request_timeout_seconds == 180
    assert config.debug is False


def test_env_overrides_defaults():
    config = load_config(
        env={
            "LOCAL_AI_OLLAMA_HOST": "http://localhost:9999",
            "LOCAL_AI_MODEL_NAME": "some-model:1b",
            "LOCAL_AI_TIMEOUT_SECONDS": "15",
            "LOCAL_AI_DEBUG": "true",
        }
    )
    assert config.ollama_host == "http://localhost:9999"
    assert config.model_name == "some-model:1b"
    assert config.request_timeout_seconds == 15
    assert config.debug is True


def test_rejects_host_without_scheme():
    try:
        load_config(env={"LOCAL_AI_OLLAMA_HOST": "localhost:11434"})
        assert False, "expected ConfigError"
    except ConfigError:
        pass


def test_rejects_empty_model_name():
    try:
        load_config(env={"LOCAL_AI_MODEL_NAME": "  "})
        assert False, "expected ConfigError"
    except ConfigError:
        pass


def test_rejects_non_integer_timeout():
    try:
        load_config(env={"LOCAL_AI_TIMEOUT_SECONDS": "soon"})
        assert False, "expected ConfigError"
    except ConfigError:
        pass


def test_rejects_non_positive_timeout():
    try:
        load_config(env={"LOCAL_AI_TIMEOUT_SECONDS": "0"})
        assert False, "expected ConfigError"
    except ConfigError:
        pass
