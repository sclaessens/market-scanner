# RESET-10C — Portfolio Source-of-Truth Contract Closeout

## Purpose

RESET-10C defines and tests the v2 Portfolio Source-of-Truth Contract.

The sprint formalizes the boundary between manual portfolio source records, normalized portfolio records, generated portfolio outputs, Decision Engine action/status fields, and Reporting/Telegram display fields.

## Contract Files Added

- `src/market_scanner/portfolio/portfolio_source_contracts.py`

The contract defines:

- portfolio source dataset roles;
- manual source dataset roles;
- generated dataset roles;
- reporting display dataset roles;
- required manual position fields;
- required manual transaction fields;
- required portfolio display input fields;
- forbidden source authority fields;
- forbidden reporting-display authority fields;
- side-effect-free shape validation helpers.

## Tests Added

- `tests/contract/test_v2_portfolio_source_of_truth_contracts.py`

The tests prove:

- manual source positions are the source of truth for holdings;
- manual source transactions are source records, not generated output;
- generated portfolio review and intelligence are not source-of-truth;
- reporting display input and Telegram output are not source-of-truth;
- required manual source and display fields are explicit;
- display-ready profit/loss, current price, and target price are supplied upstream and not calculated by Telegram;
- missing values remain explicit and are not converted to zero;
- forbidden authority fields are reported explicitly;
- no legacy `scripts` imports are used;
- imports and helpers create no files or report artifacts.

## Active Docs Added or Updated

- Added `docs/active/portfolio_source_of_truth.md`
- Updated `docs/active/data_contracts.md`
- Updated `docs/active/data_lifecycle.md`

## Scope Confirmation

No real portfolio CSVs were modified.

No legacy scripts were modified.

No legacy tests were modified.

No files were modified under:

- `scripts/`
- `data/portfolio/`
- `data/processed/`
- `data/generated/`
- `data/raw/`
- `data/normalized/`
- `data/watchlist/`
- `data/logs/`
- `reports/`
- `.github/workflows/`

No report artifact was generated.

No Telegram artifact was generated.

No `reports/daily/telegram_message.txt` file was created, modified, staged, or tracked by RESET-10C.

No broker, provider, SEC, EDGAR, network, Telegram API, or live data call was run.

No production pipeline, portfolio legacy script, reporting script, or Telegram script was run.

## Validation Commands and Results

- `git diff --check` passed.
- `git status --short` showed only RESET-10C intended changes.
- `git diff --stat` showed tracked documentation updates before staging.
- `git diff --name-only` showed tracked documentation updates before staging.
- `git diff -- tests scripts src data reports .github/workflows` showed no tracked diff before staging because the new source and test files were untracked.
- `.venv/bin/python -m pytest tests/contract/test_v2_portfolio_source_of_truth_contracts.py` passed: 17 tests passed.
- `.venv/bin/python -m pytest` passed: 543 tests passed.
- `git status --short --untracked-files=all` showed only:
  - `docs/active/data_contracts.md`
  - `docs/active/data_lifecycle.md`
  - `docs/active/portfolio_source_of_truth.md`
  - `docs/resets/reset_10c_portfolio_source_of_truth_contract_closeout.md`
  - `src/market_scanner/portfolio/portfolio_source_contracts.py`
  - `tests/contract/test_v2_portfolio_source_of_truth_contracts.py`
- Explicit status guardrails found no changes under `scripts/`, `data/portfolio/`, `data/processed/`, `data/generated/`, `data/raw/`, `data/normalized/`, `data/watchlist/`, `data/logs/`, `reports/`, or `.github/workflows/`.
- Existing-test guardrail found no modified pre-existing test files. The only test file added was `tests/contract/test_v2_portfolio_source_of_truth_contracts.py`.
- `git ls-files reports/daily/telegram_message.txt` returned no tracked file.
- `git status --short --ignored reports/daily/telegram_message.txt` showed ignored local `reports/daily/` state only.
- `ls -l reports/daily/telegram_message.txt` showed that an ignored local file already exists from before RESET-10C. RESET-10C did not create, track, modify, or write that file.

## Recommended Next Action

RESET-10H — V2 Reporting Input Aggregation Contract.
