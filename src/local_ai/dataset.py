"""JSONL dataset validation and preparation.

Format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Design rules (see docs/DATASETS.md):
- Raw input files are never modified or moved.
- Error messages report line numbers and structural reasons only, never
  the line's actual content (it may contain sensitive data).
- Splitting uses a fixed seed by default for reproducibility.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass
class LineError:
    line_number: int
    reason: str


@dataclass
class ValidationResult:
    total_lines: int
    valid_count: int
    errors: list[LineError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors and self.valid_count > 0


@dataclass
class PrepareResult:
    total_lines: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    train_count: int
    val_count: int
    test_count: int
    train_path: Path
    val_path: Path | None
    test_path: Path | None


def _check_record(record: object) -> tuple[dict | None, str | None]:
    """Validate and normalize a single parsed JSON record.

    Returns (normalized_record, None) if valid, or (None, reason) if not.
    Never includes the record's own content in the reason string.
    """
    if not isinstance(record, dict):
        return None, "record is not a JSON object"

    messages = record.get("messages")
    if messages is None:
        return None, "missing 'messages' field"
    if not isinstance(messages, list) or len(messages) == 0:
        return None, "'messages' must be a non-empty list"

    normalized_messages = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            return None, f"message {i} is not a JSON object"
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            return None, f"message {i} has invalid role (must be one of {sorted(ALLOWED_ROLES)})"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None, f"message {i} has empty or missing content"
        normalized_messages.append({"role": role, "content": content.strip()})

    if not any(m["role"] == "user" for m in normalized_messages):
        return None, "'messages' must include at least one 'user' message"
    if not any(m["role"] == "assistant" for m in normalized_messages):
        return None, "'messages' must include at least one 'assistant' message"

    return {"messages": normalized_messages}, None


def _check_line(raw_line: str) -> tuple[dict | None, str | None]:
    if not raw_line.strip():
        return None, "empty line"
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON syntax ({exc.msg})"
    return _check_record(record)


def validate_jsonl(path: Path) -> ValidationResult:
    """Validate a JSONL dataset file without modifying it."""
    errors: list[LineError] = []
    total_lines = 0
    valid_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            normalized, reason = _check_line(raw_line)
            if reason is not None:
                errors.append(LineError(line_number=line_number, reason=reason))
            else:
                valid_count += 1
    return ValidationResult(total_lines=total_lines, valid_count=valid_count, errors=errors)


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    val_ratio: float = 0.0,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> PrepareResult:
    """Normalize, deduplicate, and optionally split a raw JSONL dataset.

    Writes the train split to `output_path`. If val_ratio/test_ratio are
    greater than 0, also writes `val.jsonl`/`test.jsonl` next to it, in
    the same directory. The input file is never modified.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1, and neither may be negative")

    total_lines = 0
    invalid_records = 0
    seen: set[str] = set()
    duplicate_records = 0
    records: list[dict] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            total_lines += 1
            normalized, reason = _check_line(raw_line)
            if reason is not None:
                invalid_records += 1
                continue
            canonical = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
            if canonical in seen:
                duplicate_records += 1
                continue
            seen.add(canonical)
            records.append(normalized)

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    test_n = round(n * test_ratio)
    val_n = round(n * val_ratio)
    train_n = n - val_n - test_n

    test_records = records[:test_n]
    val_records = records[test_n:test_n + val_n]
    train_records = records[test_n + val_n:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, train_records)

    val_path = None
    test_path = None
    if val_n > 0:
        val_path = output_path.parent / "val.jsonl"
        _write_jsonl(val_path, val_records)
    if test_n > 0:
        test_path = output_path.parent / "test.jsonl"
        _write_jsonl(test_path, test_records)

    return PrepareResult(
        total_lines=total_lines,
        valid_records=len(records) + duplicate_records,
        invalid_records=invalid_records,
        duplicate_records=duplicate_records,
        train_count=train_n,
        val_count=val_n,
        test_count=test_n,
        train_path=output_path,
        val_path=val_path,
        test_path=test_path,
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
