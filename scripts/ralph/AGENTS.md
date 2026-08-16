# Ralph Agent Instructions

You are an autonomous coding agent working on the 100M Chinese LLM Pretraining Lab.

## Required reading

1. Read `tasks/prd-100m-llm-pretrain.md`, root `prd.json`, and `progress.txt`.
2. Treat the PRD's Decisions, Non-Goals, Validation, and Risks as accepted. Do not reopen them.
3. Work on branch `ralph/100m-llm-pretrain` and preserve unrelated changes.

## One-story loop

1. Pick the lowest-priority-number story where `passes` is false.
2. Implement exactly that story; do not opportunistically start later stories.
3. Use `apply_patch` for hand-written file edits.
4. Run focused tests, then `uv run ruff check .` and `uv run pyright`.
5. Never download real training data or start the multi-day pretraining run.
6. Never use unsafe/bypass flags and never commit data, credentials, logs, tokenizers, or weights.
7. Mark the story passing in both the PRD checklist and `prd.json` only after evidence exists.
8. Append the changed files, exact validation, failures, and durable lessons to `progress.txt`.
9. Commit as `feat: US-NNN - Story title` and push the feature branch.

## Aggregate validation

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run pyright
uv run pytest -q
uv run python scripts/audit_tracked_files.py
```

GPU-marked tests run only after CPU checks pass and must not start a long training job.

## Stop condition

When every story has `passes: true` and aggregate validation passes, output the following token on
its own line:

```text
<promise>COMPLETE</promise>
```

