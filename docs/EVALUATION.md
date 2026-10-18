# Evaluation

## Dataset format

Same JSONL schema as training data (see [DATASETS.md](DATASETS.md)):
one JSON object per line, with a `messages` list. The **last** message
in each record must have `role: "assistant"` — that's the reference
answer. Everything before it is sent to the model as the question/context
it must answer.

```json
{"messages":[{"role":"user","content":"বাংলাদেশের জাতীয় ভাষা কী?"},{"role":"assistant","content":"বাংলা"}]}
```

Multi-turn context works too — anything before the final assistant
message is sent as conversation history.

A line is skipped (counted, not fatal) if it fails the normal dataset
schema checks, or if its last message isn't from the assistant.

## Keeping eval data separate from training data

Put evaluation files in `data/eval/`, not `data/raw/` or
`data/processed/`. An example the model is scored on must never also
have been in its training/fine-tuning data — if it was, the score is
meaningless (the model may have memorized the answer rather than
generalized to it). This project doesn't yet track which examples were
used for training automatically, so keeping the directories physically
separate is the safeguard for now.

## Command

```bash
local-ai evaluate ./data/eval/my_eval.jsonl
local-ai evaluate ./data/eval/my_eval.jsonl \
  --output-dir artifacts/evaluations \
  --pass-threshold 0.5
```

Requires Ollama running locally, same as `ask`/`chat` — no hosted AI API
is ever contacted, and no model weights are changed.

## Metrics

None of these rely on exact string matching alone (deliberately: model
phrasing legitimately varies even when the answer is correct).

- **`exact_match`** — case-folded, whitespace-normalized exact string
  equality. Strict; mostly useful for single-word/short-fact answers
  (e.g. "what's the capital of X").
- **`token_f1`** — precision/recall F1 over whitespace-separated token
  overlap (bag-of-words), the metric classically used for SQuAD-style
  QA evaluation. Rewards partial overlap even when word order or
  surrounding phrasing differs.
- **`char_similarity`** — `difflib.SequenceMatcher` ratio (0-1) on the
  normalized text. A longest-matching-block similarity; more forgiving
  of rewording, punctuation, and minor typos than token overlap.

A per-example `passed` flag is `token_f1 >= --pass-threshold` (default
`0.5`). **This is a heuristic cutoff for a quick summary, not a
correctness guarantee** — always look at a few actual
question/reference/prediction rows in `results.jsonl`, especially before
trusting a borderline score.

### Known limitation: Bengali tokenization

`token_f1` splits on whitespace. Bengali words are space-separated, so
this works reasonably for word-level comparison, but it doesn't account
for morphological variants (e.g. a suffixed form not matching the
reference's base form) the way a language-aware tokenizer would.
`char_similarity` is less sensitive to this since it compares raw
normalized text rather than tokens — when `token_f1` looks
surprisingly low on a Bengali example, check `char_similarity` and the
raw prediction before concluding the answer was wrong.

## Output

Each run writes a **new, timestamped** directory under `--output-dir`
(default `artifacts/evaluations/`) — it never overwrites a previous
run:

```
artifacts/evaluations/20261019T120000Z_qwen3-4b/
├── results.jsonl   # one line per example: question, reference, prediction, all three scores, passed
└── summary.json    # aggregate: model, dataset path, timestamp, counts, mean scores
```

`results.jsonl` contains the actual question/reference/prediction text
(unlike `dataset validate`'s error messages, which deliberately omit
content) — that's the point of an evaluation report. Treat it with the
same care as the dataset it was generated from; `artifacts/` is
gitignored by default.

## Tests

`tests/test_evaluation.py` covers the metric functions, `load_eval_examples`
(including the skip rules), and the full run/summarize/write_run flow —
all with `local_ai.inference.chat` mocked, so the suite never needs
Ollama running or a model downloaded. `tests/test_cli.py` adds the same
coverage at the CLI level, also mocked.

For a **real** end-to-end check (not part of the automated suite — it
needs Ollama running with the configured model pulled):

```bash
local-ai evaluate tests/fixtures/eval_valid.jsonl --output-dir /tmp/eval-smoke-test
cat /tmp/eval-smoke-test/*/summary.json
```

### Real result observed with `qwen3:4b`

Run against `tests/fixtures/eval_valid.jsonl` on 2026-10-19:
`exact_match_rate 0.00`, `mean_token_f1 0.00`, `mean_char_similarity 0.01`,
`0/4 passed`. Inspecting `results.jsonl` showed this is **not a metrics
bug** — it's two real properties of this model/dataset combination:

1. `qwen3:4b` answers in full explanatory sentences/paragraphs (e.g.
   "The capital of **Bangladesh** is **Dhaka**. Dhaka is not only the
   political..."), while this fixture's references are bare single
   words ("Dhaka", "4"). Every metric here correctly scores that as a
   poor match on overlap/similarity, even though a human would call the
   answer correct. **Lesson: write reference answers in the style you
   expect the model to actually respond in** (a full sentence, not just
   the bare fact), or add a system message asking for a short answer,
   or treat a low `token_f1` with a high `char_similarity` as "probably
   correct, phrased differently" rather than "wrong" — read
   `results.jsonl`, don't trust the summary alone.
2. One answer was genuinely wrong (asked for Bangladesh's national
   flower, reference "শাপলা" [water lily], model answered "জাসমিন"
   [jasmine]), and one answered in English ("Bengali") instead of the
   Bengali script the reference used ("বাংলা") despite the question
   being asked in Bengali — both real model behavior, not artifacts of
   the scoring code.

This is exactly why the summary's `passed` count is documented as a
heuristic: on this run it would have been misleading to report "0/4
correct" without the context above.
