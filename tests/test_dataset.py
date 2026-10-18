import json
from pathlib import Path

from local_ai.dataset import prepare_dataset, validate_jsonl

FIXTURES = Path(__file__).parent / "fixtures"
VALID_FIXTURE = FIXTURES / "sample_valid.jsonl"
INVALID_FIXTURE = FIXTURES / "sample_invalid.jsonl"


def test_validate_accepts_all_valid_records():
    result = validate_jsonl(VALID_FIXTURE)
    assert result.total_lines == 6
    assert result.valid_count == 6
    assert result.errors == []
    assert result.is_valid


def test_validate_reports_each_invalid_line_with_a_reason():
    result = validate_jsonl(INVALID_FIXTURE)
    assert result.total_lines == 9
    assert result.valid_count == 2
    assert len(result.errors) == 7
    error_lines = {error.line_number for error in result.errors}
    assert error_lines == {2, 3, 4, 5, 6, 7, 8}
    assert not result.is_valid


def test_validate_error_messages_never_echo_raw_content():
    # "only user here" is the sensitive-looking payload on line 7;
    # error messages must describe the problem structurally, never quote it.
    result = validate_jsonl(INVALID_FIXTURE)
    for error in result.errors:
        assert "only user here" not in error.reason
        assert "Hi" not in error.reason


def test_validate_rejects_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    try:
        validate_jsonl(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_prepare_normalizes_and_deduplicates(tmp_path):
    output = tmp_path / "train.jsonl"
    result = prepare_dataset(VALID_FIXTURE, output)

    assert result.total_lines == 6
    assert result.duplicate_records == 1  # line 3 duplicates line 1 after whitespace normalization
    assert result.invalid_records == 0
    assert result.train_count == 5
    assert result.val_path is None
    assert result.test_path is None

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        for message in record["messages"]:
            assert message["content"] == message["content"].strip()


def test_prepare_never_modifies_the_input_file(tmp_path):
    original = VALID_FIXTURE.read_text(encoding="utf-8")
    prepare_dataset(VALID_FIXTURE, tmp_path / "train.jsonl")
    assert VALID_FIXTURE.read_text(encoding="utf-8") == original


def test_prepare_skips_invalid_lines_and_reports_them(tmp_path):
    output = tmp_path / "train.jsonl"
    result = prepare_dataset(INVALID_FIXTURE, output)
    assert result.invalid_records == 7
    assert result.train_count == 2


def test_prepare_splits_with_fixed_seed_is_reproducible(tmp_path):
    out_a = tmp_path / "a" / "train.jsonl"
    out_b = tmp_path / "b" / "train.jsonl"
    result_a = prepare_dataset(VALID_FIXTURE, out_a, val_ratio=0.2, test_ratio=0.2, seed=7)
    result_b = prepare_dataset(VALID_FIXTURE, out_b, val_ratio=0.2, test_ratio=0.2, seed=7)

    assert result_a.train_count == result_b.train_count
    assert result_a.val_count == result_b.val_count
    assert result_a.test_count == result_b.test_count
    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")
    assert result_a.val_path is not None
    assert result_a.test_path is not None

    # No overlap between splits (no data leakage).
    train_records = out_a.read_text(encoding="utf-8").splitlines()
    val_records = result_a.val_path.read_text(encoding="utf-8").splitlines()
    test_records = result_a.test_path.read_text(encoding="utf-8").splitlines()
    assert set(train_records).isdisjoint(val_records)
    assert set(train_records).isdisjoint(test_records)
    assert set(val_records).isdisjoint(test_records)
    assert len(train_records) + len(val_records) + len(test_records) == 5


def test_prepare_rejects_invalid_ratios(tmp_path):
    try:
        prepare_dataset(VALID_FIXTURE, tmp_path / "train.jsonl", val_ratio=0.6, test_ratio=0.6)
        assert False, "expected ValueError"
    except ValueError:
        pass
