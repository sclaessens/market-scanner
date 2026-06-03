# RESET-10D — Fundamentals Raw-to-Normalized Contract Closeout

## Purpose

RESET-10D defined the v2 fundamentals raw-to-normalized contract boundary before any provider ingestion, SEC or EDGAR ingestion, real raw-data loading, runtime normalization, financial scoring, reporting runtime, Telegram delivery, or Decision Engine behavior is introduced.

## Contract Files Added or Updated

Added:

- `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`

The new contract module defines:

- raw source capture dataset role;
- normalized fundamentals input dataset role;
- source-data readiness dataset role;
- generated fundamentals output roles;
- reporting display input role;
- raw source required provenance fields;
- normalized fundamentals required traceability fields;
- source-readiness required status fields;
- forbidden normalized fundamentals authority fields;
- side-effect-free contract validation helpers.

## Tests Added

Added:

- `tests/contract/test_v2_fundamentals_normalization_contracts.py`

The tests prove:

- raw source capture is separate from normalized input;
- normalized input is separate from generated output;
- source-data readiness is not investment quality;
- generated fundamentals outputs and reporting display input are not source-of-truth;
- raw records require provenance;
- normalized records require source traceability;
- missing metric values and missing count fields remain explicit and are not converted to zero;
- partial, stale, source-missing, unavailable, and invalid readiness states remain explicit;
- forbidden authority fields are reported explicitly;
- imports and helpers are side-effect-free.

## Active Docs Added or Updated

Added:

- `docs/active/fundamentals_raw_to_normalized.md`

Updated:

- `docs/active/data_contracts.md`
- `docs/active/data_lifecycle.md`
- `docs/active/source_data_strategy.md`

The documentation now explicitly separates raw source capture, normalized fundamentals input, source-data readiness, generated fundamentals outputs, and reporting display input.

## Scope Confirmation

No real raw, normalized, generated, processed, portfolio, watchlist, log, report, or CSV data files were modified.

No legacy scripts or legacy tests were modified.

No workflow files were modified.

No report or Telegram artifact was generated.

No `reports/daily/telegram_message.txt` file was created by this sprint.

No provider, SEC, EDGAR, broker, network, Telegram API, live data, production pipeline, or legacy runtime call was run.

RESET-10D did not implement provider ingestion, SEC parsing, real normalization, file loading, file writing, financial scoring, target price calculation, threshold calculation, Decision Engine behavior, reporting runtime, Telegram delivery, or production runtime behavior.

## Validation Commands and Results

Focused contract tests:

```bash
.venv/bin/python -m pytest tests/contract/test_v2_fundamentals_normalization_contracts.py
```

Result: passed, 16 tests.

Full validation:

```bash
.venv/bin/python -m pytest
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- tests scripts src data reports .github/workflows || true
git ls-files reports/daily/telegram_message.txt
git status --short --ignored reports/daily/telegram_message.txt || true
```

Results:

- `.venv/bin/python -m pytest`: passed, 591 tests.
- `git diff --check`: passed with no whitespace errors.
- `git status --short --untracked-files=all`: showed only intended RESET-10D docs, contract module, and contract tests.
- `git diff --stat`: showed active documentation changes before staging; newly added files were visible in `git status`.
- `git diff --name-only`: showed active documentation changes before staging; newly added files were visible in `git status`.
- `git diff -- tests scripts src data reports .github/workflows`: showed only intended v2 source/test diffs and no legacy script, data, report, or workflow diffs.
- Explicit guardrail checks for `scripts/`, `data/raw/`, `data/normalized/`, `data/generated/`, `data/processed/`, `data/portfolio/`, `data/watchlist/`, `data/logs/`, `reports/`, and `.github/workflows/`: passed with no forbidden changed paths.
- `git ls-files reports/daily/telegram_message.txt`: no tracked file.
- `git status --short --ignored reports/daily/telegram_message.txt || true`: reported ignored `reports/daily/`.
- `ls -l reports/daily/telegram_message.txt 2>/dev/null || true`: showed an ignored local file dated May 24, 2026; it was not created, modified, staged, or deleted by RESET-10D.
- `git check-ignore -v reports/daily/telegram_message.txt 2>/dev/null || true`: confirmed the file is ignored by `.gitignore`.

## Recommended Next Action

Recommended next action:

```text
RESET-10J — V2 Reporting Synthetic End-to-End Contract Test
```

Alternative next action:

```text
RESET-10K — V2 Fundamentals Synthetic Normalization Adapter
```
