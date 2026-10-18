# Datasets

## Format

One JSON object per line (JSONL):

```json
{"messages":[{"role":"user","content":"বাংলাদেশের রাজধানী কী?"},{"role":"assistant","content":"বাংলাদেশের রাজধানী ঢাকা।"}]}
```

- `messages` is required and must be a non-empty list.
- Each message needs `role` (one of `system`, `user`, `assistant`) and a
  non-empty `content` string.
- Each record needs at least one `user` and one `assistant` message.
- Blank lines and malformed JSON are rejected, not silently skipped.

## Commands

```bash
# Check a raw file without modifying it
local-ai dataset validate ./data/raw/my_dataset.jsonl

# Normalize + write to data/processed/ (raw input is never modified)
local-ai dataset prepare ./data/raw/my_dataset.jsonl --output ./data/processed/train.jsonl

# With a train/val/test split (fixed seed by default, for reproducibility)
local-ai dataset prepare ./data/raw/my_dataset.jsonl \
  --output ./data/processed/train.jsonl \
  --val-ratio 0.1 --test-ratio 0.1 --seed 42
```

`prepare` always:
- Validates every line (invalid lines are skipped and counted, never crash the run).
- Normalizes records (strips whitespace from `content`, drops unknown fields).
- **Deduplicates** identical normalized records before splitting.
- Shuffles with the given `--seed` before splitting, so the same input +
  seed always produces the same split (reproducible, and train/val/test
  never overlap — no data leakage).
- Writes `val.jsonl`/`test.jsonl` next to `--output`, in the same
  directory, only if their ratio is greater than 0.

`validate` and `prepare` never print a line's actual content in error
messages — only the line number and a structural reason (e.g. "missing
'messages' field", "message 0 has invalid role") — because dataset
content may be sensitive.

## Where files live

- `data/raw/` — your original files, exactly as provided. Nothing here
  is ever written to by this project.
- `data/processed/` — output of `dataset prepare`. Safe to delete and
  regenerate at any time; nothing here is a source of truth.
- Neither directory is committed to Git (see `.gitignore`); only
  `tests/fixtures/*.jsonl` (small synthetic examples, not real data) is
  tracked, so tests don't depend on you having any dataset at all.

## Provenance, license, and consent

Before adding a real dataset to `data/raw/`:
- Document where it came from and under what license/terms.
- Don't scrape or include private or copyrighted material without
  permission.
- If it contains personal data, confirm you have a lawful basis to use
  it, and avoid committing it anywhere (it's already gitignored here,
  but double-check before changing that).

## Try it now

A small synthetic example ships in the repo for exactly this purpose:

```bash
local-ai dataset validate tests/fixtures/sample_valid.jsonl
local-ai dataset prepare tests/fixtures/sample_valid.jsonl \
  --output data/processed/train.jsonl --val-ratio 0.2 --test-ratio 0.2
```

`tests/fixtures/sample_invalid.jsonl` is the companion fixture with
deliberately broken lines, used by the test suite to check that every
kind of validation error is caught and reported correctly.
