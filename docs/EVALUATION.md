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

- **`exact_match`** — case-folded, normalized exact string equality.
  Strict; mostly useful for single-word/short-fact answers (e.g. "what's
  the capital of X").
- **`token_f1`** — precision/recall F1 over normalized token overlap
  (bag-of-words), the metric classically used for SQuAD-style QA
  evaluation. Rewards partial overlap even when word order or
  surrounding phrasing differs.
- **`char_similarity`** — `difflib.SequenceMatcher` ratio (0-1) on the
  normalized text. A longest-matching-block similarity; more forgiving
  of rewording, punctuation, and minor typos than token overlap.

**Normalization (`METRICS_VERSION = "2-markdown-punctuation-normalized"`,
recorded in every `summary.json`)**: lowercases, strips Markdown
emphasis markers (`**`/`__`/`` ` ``/`~`), and replaces punctuation/symbol
characters with a space (by Unicode category — not a fixed ASCII list,
and deliberately not a `\w`-based allowlist, since Python's `\w`
excludes the Unicode combining-mark category that Bengali vowel signs
use and would silently corrupt Bengali text). Punctuation is replaced
with a space rather than deleted, so two unrelated words never fuse into
one token. **Found and fixed after the Milestone 5 training comparison
surfaced it**: before this, `**Dhaka**.` never matched the reference
`Dhaka` at all — every model answer wrapped in Markdown bold scored as
if it were wrong, deflating scores throughout earlier evaluation runs
(the "v1"/"v2" results earlier in this doc predate the fix and are kept
as-is below; don't compare their raw numbers against new runs without
accounting for this).

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
└── summary.json    # aggregate: model, dataset path, timestamp, counts, mean scores, metrics_version
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

### Investigation + v2 dataset (2026-10-19)

Following up on the v1 result above, each of the 4 v1 failures was
re-examined individually:

| Question (v1) | Verdict |
|---|---|
| Capital of Bangladesh | Correct answer; low score purely from verbose-answer-vs-bare-word reference mismatch. |
| 2 + 2 | Correct answer; same verbosity mismatch. |
| National language | Correct in substance ("Bengali"); answered in English rather than the Bengali script the reference used. |
| National flower | **Genuinely wrong** — model said "জাসমিন" (jasmine); the actual national flower is "শাপলা" (shapla/water lily). |

So 3 of 4 v1 failures were metric/reference-style artifacts, and 1 was
a real model error.

**`tests/fixtures/eval_bangla_v2.jsonl`** (7 examples) was built to test
whether better-calibrated references fix the artifact cases, while
re-testing the one real error:
- References written as full short sentences matching the model's
  natural phrasing (not bare words), for factual questions.
- Two examples add a `system` message explicitly instructing a Bengali
  response (`"সংক্ষেপে বাংলায় উত্তর দিন।"` / `"Always respond in
  Bengali, even if the question is in English."`) — the schema already
  supports `system` role messages, so this needed no code change.
- One example explicitly asks for a brief answer, to test whether an
  instruction (rather than reference calibration) controls verbosity.
- One example is a short-explanation type (photosynthesis), not a bare
  fact.
- The national-flower question is repeated (with a fuller reference
  this time) specifically to check if the wrong answer was consistent.
- A national-bird question was added for additional factual coverage.

**Real run against `qwen3:4b`** (`artifacts/evaluations/20261018T193828Z_qwen3-4b/`):
`exact_match_rate 0.14`, `mean_token_f1 0.21`, `mean_char_similarity 0.35`,
`1/7 passed` (threshold 0.5). Per-example review:

| # | Question | Verdict |
|---|---|---|
| 1 | Capital of Bangladesh | **Correct**, verbatim match to the full-sentence reference (f1=1.0). Confirms full-sentence references work *when the model happens to answer at that length*. |
| 2 | National language (Bengali-instructed) | **Correct** ("বাংলা") — but *more terse* than my full-sentence reference, so it scored low (f1=0.0) anyway. The Bengali-instruction fixed the v1 language problem; the reference-length problem just flipped direction. |
| 3 | National flower (repeated) | **Still wrong** — this time said "লটাস" (lotus), not "জাসমিন" like v1, but still not "শাপলা". Two different wrong answers across two runs is good evidence this is a genuine, repeatable gap in the model's knowledge of this specific fact, not a one-off. |
| 4 | 2+2 (brevity-instructed) | **Correct** ("4") — the brevity instruction worked perfectly, but the model used an Arabic numeral instead of the Bengali-digit reference ("৪"), so it scored 0 anyway. |
| 5 | Photosynthesis (short explanation) | Content **correct**, but answered **entirely in English** despite a Bengali question — this example had no Bengali-instruction system message (unlike #2 and #6), so this is as much a dataset-design gap as a model one. |
| 6 | Currency (English question, Bengali-instructed) | **Correct** ("টাকা"/Taka, with bonus ISO-code detail) and correctly answered in Bengali despite the English question — the Bengali-instruction technique worked here too. |
| 7 | National bird | **Wrong** — said "Brahminy kite"; the actual national bird is "দোয়েল" (Oriental magpie-robin). A second genuine factual gap. |

**Conclusions (small sample — 7 examples, not a benchmark):**
- 4 of 7 answers were correct in substance; only 1 of 7 "passed" the
  default threshold. The gap between those two numbers is the real
  finding: **`token_f1`/`passed` systematically undercounts correctness
  for this model**, because its answer length relative to any single
  reference string is unpredictable — sometimes more verbose (v1),
  sometimes more terse (#2, #4 here) than the reference. Rewriting
  references to match one direction doesn't fix the other.
- The explicit Bengali-instruction system message **did reliably work**
  (#2, #6) — a validated, reusable technique, not just better reference
  wording.
- 2 of 7 were genuine, repeatable factual errors, both specifically
  about Bangladesh national symbols (flower, bird) — a real, narrow
  knowledge gap worth knowing about if this model is used for Bangladesh
  trivia, separate from the scoring-methodology issue entirely.
- **Net methodology change:** none to the metrics themselves (kept as
  documented above, per design — the metrics aren't wrong, they're
  measuring exactly what they say they measure; the mismatch is between
  "token/string overlap" and "human judgment of correctness" for a
  verbose model). The actionable change is procedural: for this model,
  treat `results.jsonl` + manual read-through as the primary signal and
  the summary `passed` count as, at best, a rough triage filter — not a
  pass/fail verdict. This conclusion is now validated twice (v1 and v2),
  not assumed.
