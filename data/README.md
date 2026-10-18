# data/

- `raw/` — your original training dataset files, unmodified. Never edited in place.
- `processed/` — normalized train/val/test output produced by `local-ai dataset prepare` (Milestone 3).
- `eval/` — evaluation (reference Q&A) datasets used by `local-ai evaluate` (Milestone 4).
  **Keep files here out of `raw/`/`processed/`**: an example a model is
  evaluated on must not also have been used to train/fine-tune it, or
  the evaluation score becomes meaningless (data leakage). See
  [docs/EVALUATION.md](../docs/EVALUATION.md).

Nothing in this directory is committed to Git (see `.gitignore`). Document
the provenance, license, and consent basis of any dataset you place here
before using it.
