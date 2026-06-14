# RESET-10H — V2 Reporting Input Aggregation Contract Closeout

## Purpose

RESET-10H defines and tests the v2 Reporting Input Aggregation Contract.

The sprint formalizes how approved upstream display records may come together as reporting input without making Reporting or Telegram a source-of-truth layer, decision layer, portfolio engine, price calculator, threshold calculator, or report generator.

## Contract Files Added

- `src/market_scanner/reporting/reporting_input_contracts.py`

The contract defines:

- reporting input roles;
- aggregation boundary roles;
- required portfolio display input fields;
- required candidate display input fields;
- required Decision Engine status input fields;
- required source-data status input fields;
- required data warning input fields;
- required aggregation trace fields;
- forbidden reporting aggregation authority fields;
- side-effect-free shape validation helpers.

## Tests Added

- `tests/contract/test_v2_reporting_input_aggregation_contracts.py`

The tests prove:

- reporting input aggregation roles exist;
- portfolio display input contains Telegram UX fields;
- candidate display input contains Telegram UX fields;
- source-data status input contains data status and review reason;
- every reporting input requires source role, source reference, and aggregation contract version;
- reporting input aggregation does not define source-of-truth roles;
- portfolio source-of-truth remains upstream of reporting input aggregation;
- Telegram renderer input remains downstream of reporting input aggregation;
- display values are supplied upstream and not calculated by reporting aggregation;
- missing values remain explicit and are not converted to zero;
- forbidden authority fields are reported explicitly;
- no legacy `scripts` imports are used;
- imports and helpers create no files or report artifacts.

## Active Docs Added or Updated

- Added `docs/active/reporting_input_aggregation.md`
- Updated `docs/active/reporting_contract.md`
- Updated `docs/active/data_contracts.md`

## Scope Confirmation

No real data files were modified.

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

No `reports/daily/telegram_message.txt` file was created, modified, staged, or tracked by RESET-10H.

No broker, provider, SEC, EDGAR, network, Telegram API, or live data call was run.

No production pipeline, reporting legacy script, or Telegram script was run.

## Validation Commands and Results

- `git diff --check` passed.
- `git status --short --untracked-files=all` showed only RESET-10H intended changes.
- `git diff --stat` showed tracked documentation updates before staging.
- `git diff --name-only` showed tracked documentation updates before staging.
- `git diff -- tests scripts src data reports .github/workflows` showed no tracked diff before staging because the new source and test files were untracked.
- `.venv/bin/python -m pytest tests/contract/test_v2_reporting_input_aggregation_contracts.py` passed: 18 tests passed.
- `.venv/bin/python -m pytest` passed: 561 tests passed.
- Explicit status guardrails found no changes under `scripts/`, `data/portfolio/`, `data/processed/`, `data/generated/`, `data/raw/`, `data/normalized/`, `data/watchlist/`, `data/logs/`, `reports/`, or `.github/workflows/`.
- Existing-test guardrail found no modified pre-existing test files. The only test file added was `tests/contract/test_v2_reporting_input_aggregation_contracts.py`.
- `git ls-files reports/daily/telegram_message.txt` returned no tracked file.
- `git status --short --ignored reports/daily/telegram_message.txt` showed ignored local `reports/daily/` state only.
- `ls -l reports/daily/telegram_message.txt` showed that an ignored local file already exists from before RESET-10H. RESET-10H did not create, track, modify, or write that file.

## Recommended Next Action

RESET-10I — V2 Reporting Input Aggregation Synthetic Adapter.

If source-data readiness input needs more formalization first, run RESET-10D — Fundamentals Raw-to-Normalized Contract before RESET-10I.
