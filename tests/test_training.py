import json
from pathlib import Path

import pytest

from local_ai import training
from local_ai.evaluation import EvalExample

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN_FIXTURE = FIXTURES / "train_bangla_v1.jsonl"


def test_prepare_mlx_data_dir_writes_train_and_valid_jsonl(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = training.prepare_mlx_data_dir(TRAIN_FIXTURE, data_dir, val_ratio=0.15, seed=42)

    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "valid.jsonl").exists()  # mlx_lm's exact expected filename, not val.jsonl
    assert not (data_dir / "test.jsonl").exists()
    assert result.train_count + result.val_count == 16  # no duplicates/invalid in this fixture


def test_check_checkpoint_available_raises_actionable_error(tmp_path):
    missing = tmp_path / "no-such-checkpoint"
    with pytest.raises(training.TrainingError, match="never downloads it automatically"):
        training.check_checkpoint_available(missing)


def test_check_checkpoint_available_passes_when_present(tmp_path):
    present = tmp_path / "checkpoint"
    present.mkdir()
    training.check_checkpoint_available(present)  # must not raise


def test_check_mlx_lm_available_does_not_raise():
    training.check_mlx_lm_available()  # mlx-lm is actually installed in this environment


def test_build_plan_summary_never_overclaims():
    config = training.TrainingConfig(
        checkpoint_path=Path("models/fake"),
        data_dir=Path("data/fake"),
        output_dir=Path("artifacts/fake"),
    )
    summary = training.build_plan_summary(config, num_examples=14)
    assert "NOT reliably inject new facts" in summary
    assert "rank 8" in summary


def test_new_run_dir_never_collides(tmp_path):
    first = training.new_run_dir(tmp_path, "qwen3-4b")
    first.mkdir(parents=True)
    second = training.new_run_dir(tmp_path, "qwen3-4b")
    assert first != second


def test_run_training_writes_config_and_metadata_without_a_real_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "run"

    calls = []
    monkeypatch.setattr(training.subprocess, "run", lambda cmd, check: calls.append(cmd))

    config = training.TrainingConfig(
        checkpoint_path=checkpoint_path, data_dir=data_dir, output_dir=output_dir, iters=10
    )
    result_dir = training.run_training(config)

    assert result_dir == output_dir
    assert len(calls) == 1
    assert "--config" in calls[0]
    assert (output_dir / "mlx_lora_config.yaml").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["iters"] == 10


def test_run_training_never_invokes_subprocess_without_a_checkpoint(tmp_path, monkeypatch):
    missing_checkpoint = tmp_path / "no-checkpoint"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    calls = []
    monkeypatch.setattr(training.subprocess, "run", lambda cmd, check: calls.append(cmd))

    config = training.TrainingConfig(
        checkpoint_path=missing_checkpoint, data_dir=data_dir, output_dir=tmp_path / "run"
    )
    with pytest.raises(training.TrainingError):
        training.run_training(config)
    assert calls == []


def test_compare_base_and_tuned_uses_mocked_mlx_lm(tmp_path, monkeypatch):
    import mlx_lm

    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

    def fake_load(path, adapter_path=None, **kwargs):
        return ("tuned" if adapter_path is not None else "base"), FakeTokenizer()

    def fake_generate(model, tokenizer, prompt, max_tokens=200, verbose=False):
        return "tuned-answer" if model == "tuned" else "base-answer"

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    monkeypatch.setattr(mlx_lm, "generate", fake_generate)

    examples = [
        EvalExample(
            line_number=1,
            question="Q?",
            context_messages=[{"role": "user", "content": "Q?"}],
            reference="tuned-answer",
        )
    ]
    results = training.compare_base_and_tuned(checkpoint_path, adapter_path, examples)

    assert len(results) == 1
    assert results[0].base_prediction == "base-answer"
    assert results[0].tuned_prediction == "tuned-answer"
    assert results[0].tuned_token_f1 == 1.0
    assert results[0].base_token_f1 == 0.0


def test_compare_base_and_tuned_requires_adapter_to_exist(tmp_path):
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    missing_adapter = tmp_path / "no-adapter"
    with pytest.raises(training.TrainingError):
        training.compare_base_and_tuned(checkpoint_path, missing_adapter, [])
