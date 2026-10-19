import json
from pathlib import Path

from local_ai import inference
from local_ai.config import Config
from local_ai.evaluation import (
    METRICS_VERSION,
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
BANGLA_V2_FIXTURE = FIXTURES / "eval_bangla_v2.jsonl"


def _config() -> Config:
    return Config(
        ollama_host="http://localhost:11434",
        model_name="qwen3:4b",
        request_timeout_seconds=5,
        debug=False,
        mlx_checkpoint_path="models/qwen3-4b-mlx-4bit",
    )


# --- metric functions -------------------------------------------------


def test_exact_match_is_case_and_whitespace_insensitive():
    assert exact_match("  Dhaka  ", "dhaka")
    assert not exact_match("Chittagong", "Dhaka")


def test_exact_match_handles_bengali_text():
    assert exact_match("বাংলা", "  বাংলা  ")
    assert not exact_match("বাংলা", "ইংরেজি")


def test_markdown_bold_english_answer_matches_bare_reference():
    # The exact bug found investigating Milestone 5's real comparison run:
    # qwen3:4b wraps short factual answers in **bold**, and the metric's
    # tokenizer never stripped markdown, so a correct answer scored as if
    # it were wrong.
    assert exact_match("**Dhaka**.", "Dhaka")
    assert token_f1("**Dhaka**.", "Dhaka") == 1.0
    assert char_similarity("**Dhaka**.", "Dhaka") == 1.0


def test_markdown_bold_bengali_answer_matches_bare_reference():
    # Same bug, Bengali script - and specifically must NOT corrupt the
    # text: Bengali vowel signs (e.g. the two া marks in শাপলা) are
    # Unicode combining marks, which a naive \w-based "keep word chars"
    # filter actually strips, silently mangling the word. This asserts
    # the fix preserves it exactly.
    assert exact_match("**শাপলা**।", "শাপলা")
    assert token_f1("**শাপলা**।", "শাপলা") == 1.0
    assert char_similarity("**শাপলা**।", "শাপলা") == 1.0


def test_surrounding_punctuation_does_not_block_a_match():
    assert exact_match("Dhaka.", "Dhaka")
    assert exact_match("Dhaka!", "Dhaka")
    assert exact_match('"Dhaka"', "Dhaka")
    assert exact_match("(Dhaka)", "Dhaka")
    assert exact_match("বাংলা।", "বাংলা")


def test_punctuation_normalization_does_not_fuse_separate_words():
    # Deleting punctuation outright (instead of replacing it with a
    # space) can silently merge two unrelated words into one token,
    # which would make them NOT match two separate reference words.
    assert token_f1("Dhaka—Bangladesh", "Dhaka Bangladesh") == 1.0


def test_genuinely_wrong_answers_still_score_low_after_normalization():
    # The fix must not accidentally make incorrect answers look correct.
    assert not exact_match("**Chittagong**.", "Dhaka")
    assert token_f1("**Chittagong**.", "Dhaka") == 0.0
    assert not exact_match("**জাসমিন**।", "শাপলা")
    assert token_f1("**জাসমিন**।", "শাপলা") == 0.0


def test_sentence_wrapped_markdown_answer_is_not_exact_match_but_scores_well():
    # A full sentence containing the bolded answer should NOT count as an
    # exact match (it isn't one), but token_f1 should still reward the
    # overlap - this was already true before the fix and must stay true.
    prediction = "The capital of Bangladesh is **Dhaka**."
    assert not exact_match(prediction, "Dhaka")
    assert 0.0 < token_f1(prediction, "Dhaka") < 1.0


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


def test_load_eval_examples_parses_the_v2_bengali_fixture():
    # v2 was written to fix the reference-style issue found investigating the
    # v1 real-model run (docs/EVALUATION.md): full-sentence references
    # matching the model's natural phrasing, plus system messages that
    # explicitly instruct a Bengali response. This just checks it parses
    # cleanly with no skipped lines and includes both system-message
    # examples.
    examples, total_lines, skipped = load_eval_examples(BANGLA_V2_FIXTURE)
    assert total_lines == 7
    assert skipped == 0
    assert len(examples) == 7
    with_system_message = [e for e in examples if any(m["role"] == "system" for m in e.context_messages)]
    assert len(with_system_message) == 2


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
    assert summary.metrics_version == METRICS_VERSION


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
