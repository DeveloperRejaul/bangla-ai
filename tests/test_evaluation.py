import json
from pathlib import Path

from local_ai import inference
from local_ai.config import Config
from local_ai.evaluation import (
    char_similarity,
    exact_match,
    load_eval_examples,
    run_evaluation,
    summarize,
    token_f1,
    write_run,
)

FIXTURES = Path(__file__).parent / "fixtures"
VALID_FIXTURE = FIXTURES / "eval_valid.jsonl"
INVALID_FIXTURE = FIXTURES / "eval_invalid.jsonl"


def _config() -> Config:
    return Config(ollama_host="http://localhost:11434", model_name="qwen3:4b", request_timeout_seconds=5, debug=False)


# --- metric functions -------------------------------------------------


def test_exact_match_is_case_and_whitespace_insensitive():
    assert exact_match("  Dhaka  ", "dhaka")
    assert not exact_match("Chittagong", "Dhaka")


def test_exact_match_handles_bengali_text():
    assert exact_match("বাংলা", "  বাংলা  ")
    assert not exact_match("বাংলা", "ইংরেজি")


def test_token_f1_partial_overlap():
    score = token_f1("The capital is Dhaka", "Dhaka is the capital of Bangladesh")
    assert 0.0 < score < 1.0


def test_token_f1_perfect_and_zero():
    assert token_f1("Dhaka", "Dhaka") == 1.0
    assert token_f1("completely unrelated", "Dhaka") == 0.0


def test_char_similarity_rewards_close_rewording():
    close = char_similarity("The capital of Bangladesh is Dhaka.", "Dhaka is the capital of Bangladesh.")
    far = char_similarity("xyz", "The capital of Bangladesh is Dhaka.")
    assert close > far


# --- load_eval_examples -------------------------------------------------


def test_load_eval_examples_accepts_all_valid_records():
    examples, total_lines, skipped = load_eval_examples(VALID_FIXTURE)
    assert total_lines == 4
    assert skipped == 0
    assert len(examples) == 4
    assert examples[0].reference == "Dhaka"
    assert examples[0].question == "What is the capital of Bangladesh?"
    assert examples[0].context_messages == [{"role": "user", "content": "What is the capital of Bangladesh?"}]


def test_load_eval_examples_skips_non_assistant_endings_and_invalid_schema():
    examples, total_lines, skipped = load_eval_examples(INVALID_FIXTURE)
    assert total_lines == 5
    assert skipped == 3  # ends-with-user, missing 'messages', and no-user-message lines
    assert len(examples) == 2
    assert {e.reference for e in examples} == {"Hello!", "Goodbye!"}


# --- run_evaluation (mocked inference) -------------------------------------------------


def test_run_evaluation_uses_mocked_inference_only(monkeypatch):
    calls = []

    def fake_chat(config, messages):
        calls.append(messages)
        return "Dhaka"  # pretend the model always answers "Dhaka"

    monkeypatch.setattr(inference, "chat", fake_chat)
    examples, _, _ = load_eval_examples(VALID_FIXTURE)
    results = run_evaluation(_config(), examples, pass_threshold=0.5)

    assert len(calls) == len(examples)  # one inference call per example, no network/model download
    assert len(results) == len(examples)
    dhaka_result = next(r for r in results if r.reference == "Dhaka")
    assert dhaka_result.exact_match is True
    assert dhaka_result.token_f1 == 1.0
    assert dhaka_result.passed is True


def test_run_evaluation_propagates_ollama_error(monkeypatch):
    def fake_chat(config, messages):
        raise inference.OllamaError("Could not reach Ollama at http://localhost:11434.")

    monkeypatch.setattr(inference, "chat", fake_chat)
    examples, _, _ = load_eval_examples(VALID_FIXTURE)
    try:
        run_evaluation(_config(), examples, pass_threshold=0.5)
        assert False, "expected OllamaError"
    except inference.OllamaError:
        pass


# --- summarize + write_run -------------------------------------------------


def test_summarize_computes_aggregate_metrics(monkeypatch):
    monkeypatch.setattr(inference, "chat", lambda config, messages: "Dhaka")
    examples, total_lines, skipped = load_eval_examples(VALID_FIXTURE)
    results = run_evaluation(_config(), examples, pass_threshold=0.5)
    summary = summarize(results, "qwen3:4b", VALID_FIXTURE, total_lines, skipped, pass_threshold=0.5)

    assert summary.evaluated == len(examples)
    assert summary.total_lines == total_lines
    assert summary.skipped_invalid == skipped
    assert 0.0 <= summary.mean_token_f1 <= 1.0
    assert 0.0 <= summary.exact_match_rate <= 1.0


def test_write_run_never_overwrites_a_previous_run(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "chat", lambda config, messages: "Dhaka")
    examples, total_lines, skipped = load_eval_examples(VALID_FIXTURE)
    results = run_evaluation(_config(), examples, pass_threshold=0.5)
    summary = summarize(results, "qwen3:4b", VALID_FIXTURE, total_lines, skipped, pass_threshold=0.5)

    run_dir_1 = write_run(tmp_path, "qwen3:4b", results, summary)
    run_dir_2 = write_run(tmp_path, "qwen3:4b", results, summary)

    assert run_dir_1 != run_dir_2
    assert run_dir_1.exists() and run_dir_2.exists()
    assert (run_dir_1 / "results.jsonl").exists()
    assert (run_dir_1 / "summary.json").exists()

    written_summary = json.loads((run_dir_1 / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["model_name"] == "qwen3:4b"

    results_lines = (run_dir_1 / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(results_lines) == len(results)
