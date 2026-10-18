"""Evaluation of the local model against a JSONL reference Q&A dataset.

Reuses the same JSONL schema as training data (local_ai.dataset): each
line is a JSON object with a "messages" list of {role, content} pairs.
For evaluation, the LAST message in each record must have role
"assistant" (it's the reference answer); everything before it is sent
to the model as the conversation it must answer. This works for both
single-turn QA ([user, assistant]) and multi-turn context.

Metrics (see docs/EVALUATION.md for full definitions and rationale):
- exact_match: case-folded, whitespace-normalized exact string equality.
- token_f1: precision/recall F1 over whitespace-separated token overlap
  (the metric classically used for SQuAD-style QA evaluation).
- char_similarity: difflib.SequenceMatcher ratio in [0, 1] — a
  longest-matching-block similarity, more forgiving of rewording,
  punctuation, or small typos than exact/token matching.

No metric here relies only on exact string matching. Inference goes
through local_ai.inference, which only ever talks to the configured
local Ollama host — no hosted AI API is contacted.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from local_ai import dataset as dataset_module
from local_ai import inference
from local_ai.config import Config


@dataclass
class EvalExample:
    line_number: int
    question: str
    context_messages: list[dict[str, str]]
    reference: str


@dataclass
class ExampleResult:
    line_number: int
    question: str
    reference: str
    prediction: str
    exact_match: bool
    token_f1: float
    char_similarity: float
    passed: bool


@dataclass
class EvalSummary:
    model_name: str
    dataset_path: str
    generated_at: str
    total_lines: int
    skipped_invalid: int
    evaluated: int
    pass_threshold: float
    passed: int
    exact_match_rate: float
    mean_token_f1: float
    mean_char_similarity: float


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> bool:
    return _normalize_text(prediction) == _normalize_text(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-words precision/recall F1 over whitespace-separated tokens."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    remaining = dict(ref_counts)
    for token in pred_tokens:
        if remaining.get(token, 0) > 0:
            overlap += 1
            remaining[token] -= 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def char_similarity(prediction: str, reference: str) -> float:
    """difflib ratio on normalized text; robust to word-boundary differences
    that disadvantage token_f1 for morphologically rich languages."""
    return difflib.SequenceMatcher(
        None, _normalize_text(prediction), _normalize_text(reference)
    ).ratio()


def load_eval_examples(path: Path) -> tuple[list[EvalExample], int, int]:
    """Load and validate eval examples from a JSONL file.

    Returns (examples, total_lines, skipped_invalid). A line is skipped
    (counted, not raised) if it fails the normal dataset schema checks,
    or if its last message isn't from the assistant (eval-specific: the
    last message is the reference answer, so there must be at least one
    message before it, and that final one must be "assistant").
    """
    examples: list[EvalExample] = []
    total_lines = 0
    skipped_invalid = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            normalized, reason = dataset_module.parse_record(raw_line)
            if reason is not None:
                skipped_invalid += 1
                continue
            messages = normalized["messages"]
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                skipped_invalid += 1
                continue
            context_messages = messages[:-1]
            question = next(
                (m["content"] for m in reversed(context_messages) if m["role"] == "user"),
                "",
            )
            examples.append(
                EvalExample(
                    line_number=line_number,
                    question=question,
                    context_messages=context_messages,
                    reference=messages[-1]["content"],
                )
            )
    return examples, total_lines, skipped_invalid


def run_evaluation(
    config: Config, examples: list[EvalExample], pass_threshold: float
) -> list[ExampleResult]:
    """Run each example through the local model and score it.

    Raises inference.OllamaError immediately if the model can't be
    reached - there's no point scoring an empty run.
    """
    results: list[ExampleResult] = []
    for example in examples:
        prediction = inference.chat(config, example.context_messages)
        f1 = token_f1(prediction, example.reference)
        results.append(
            ExampleResult(
                line_number=example.line_number,
                question=example.question,
                reference=example.reference,
                prediction=prediction,
                exact_match=exact_match(prediction, example.reference),
                token_f1=f1,
                char_similarity=char_similarity(prediction, example.reference),
                passed=f1 >= pass_threshold,
            )
        )
    return results


def summarize(
    results: list[ExampleResult],
    model_name: str,
    dataset_path: Path,
    total_lines: int,
    skipped_invalid: int,
    pass_threshold: float,
) -> EvalSummary:
    evaluated = len(results)
    passed = sum(1 for r in results if r.passed)
    exact_matches = sum(1 for r in results if r.exact_match)
    mean_f1 = sum(r.token_f1 for r in results) / evaluated if evaluated else 0.0
    mean_sim = sum(r.char_similarity for r in results) / evaluated if evaluated else 0.0
    return EvalSummary(
        model_name=model_name,
        dataset_path=str(dataset_path),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total_lines=total_lines,
        skipped_invalid=skipped_invalid,
        evaluated=evaluated,
        pass_threshold=pass_threshold,
        passed=passed,
        exact_match_rate=exact_matches / evaluated if evaluated else 0.0,
        mean_token_f1=mean_f1,
        mean_char_similarity=mean_sim,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "model"


def write_run(
    output_dir: Path,
    model_name: str,
    results: list[ExampleResult],
    summary: EvalSummary,
) -> Path:
    """Write a new timestamped evaluation run directory. Never overwrites
    a previous run: if the timestamp collides, a numeric suffix is added."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{timestamp}_{_slug(model_name)}"
    run_dir = output_dir / base_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_dir / f"{base_name}-{suffix}"
    run_dir.mkdir(parents=True)

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    (run_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir
