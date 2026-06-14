# RESET-9C2D: Translate Reporting and Telegram UX Requirements to V2 Contracts Closeout

## Purpose

RESET-9C2D translated the approved portfolio-first Telegram/reporting UX specification into minimal v2 reporting/Telegram contract metadata and v2 contract tests.

This sprint did not implement Telegram delivery, inbound Telegram command handling, full reporting runtime, or real report generation.

## UX Requirements Translated

The translated UX requirements came from `docs/active/reporting_telegram_ux.md`.

The v2 contract now captures:

- portfolio-first Telegram section order;
- compact message sections: Header, Portfolio, Buy now, Buy on pullback, Buy on breakout, and Data status;
- required portfolio display fields for ticker, profit/loss, current price, target price, action/status, and currency;
- required candidate display fields for ticker, display group, threshold price, threshold direction, action/status, and currency;
- required data-status fields for data status and review reason;
- pullback display semantics requiring upstream `below` threshold direction;
- breakout display semantics requiring upstream `above` threshold direction;
- buy-now display semantics requiring no threshold direction;
- compact empty-state text for candidate sections;
- explicit missing-value handling for target price and profit/loss;
- forbidden reporting/Telegram authority fields for allocation, execution, conviction, urgency, tradeability, ranking, score, recommendation, and threshold calculation semantics.

The contract intentionally does not directly translate legacy Telegram output into v2.

## V2 Contracts and Tests Added

Added:

- `src/market_scanner/reporting/telegram_contracts.py`
- `tests/contract/test_v2_telegram_contracts.py`

Updated:

- `docs/active/reporting_contract.md`

The new v2 Telegram contract module defines:

- `TelegramSection`;
- `CandidateDisplayGroup`;
- `ThresholdDirection`;
- `TelegramContractIssueCode`;
- `TelegramContractIssue`;
- portfolio display field metadata;
- candidate display field metadata;
- data-status display field metadata;
- candidate-section empty-state metadata;
- forbidden Telegram authority field metadata;
- side-effect-free portfolio, candidate, and data-status shape validation helpers.

The new tests verify:

- section order is portfolio-first after the header;
- portfolio display fields include the approved UX concepts;
- candidate groups include buy now, buy on pullback, and buy on breakout;
- pullback, breakout, and buy-now threshold directions are explicit;
- candidate empty-state text exists;
- data status remains explicit;
- reporting/Telegram contracts do not include authority fields;
- forbidden authority fields are reported as contract issues;
- missing target price and profit/loss remain explicit and are not converted to zero;
- invalid candidate groups are reported without reclassification;
- the contract module does not import legacy `scripts` or delivery/network modules;
- contract helpers create no files.

## Files Changed

- `docs/active/reporting_contract.md`
- `docs/resets/reset_9c2d_translate_reporting_telegram_ux_requirements_to_v2_contracts_closeout.md`
- `src/market_scanner/reporting/telegram_contracts.py`
- `tests/contract/test_v2_telegram_contracts.py`

## Scope Confirmation

Confirmed:

- no legacy reporting scripts were modified;
- no legacy Telegram scripts were modified;
- no legacy tests were modified;
- no legacy files were deleted, moved, archived, renamed, or refactored;
- no existing legacy CSV contents were modified;
- no files under `data/processed/`, `data/portfolio/`, `data/watchlist/`, `data/logs/`, or `reports/` were modified;
- no workflow files were modified;
- no report artifact was generated;
- `reports/daily/telegram_message.txt` was not created;
- no Telegram message was sent;
- no Telegram delivery was implemented;
- no inbound Telegram command handling was implemented;
- no full reporting runtime was implemented;
- no report generation from real data was implemented;
- Reporting and Telegram were not allowed to create decisions;
- Reporting and Telegram were not allowed to calculate target prices, buy thresholds, or breakout thresholds;
- no ranking, scoring, prioritization, urgency, conviction, tradeability, or allocation authority was added;
- missing values were not converted to zero;
- source-data readiness was not turned into investment quality.

## Runtime and External Call Confirmation

Confirmed:

- no production pipeline was run;
- `scripts/run_scan.py` was not run;
- Telegram scripts were not run;
- `scripts/reporting/send_telegram.py` was not run;
- `scripts/telegram/process_telegram_commands.py` was not run;
- SEC diagnostics were not run;
- SEC calls were not made;
- EDGAR calls were not made;
- provider calls were not made;
- broker calls were not made;
- network calls were not made;
- Telegram API calls were not made;
- live data calls were not made;
- the daily scan schedule was not re-enabled.

## Validation Commands and Results

Commands run:

```bash
.venv/bin/python -m pytest tests/contract/test_v2_telegram_contracts.py
.venv/bin/python -m pytest
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- tests scripts src data reports .github/workflows || true
```

Results:

- `.venv/bin/python -m pytest tests/contract/test_v2_telegram_contracts.py` passed: `16 passed`.
- `.venv/bin/python -m pytest` passed: `513 passed`.
- `git diff --check` passed.
- `git status --short` showed only intended RESET-9C2D changes before closeout creation.
- `git diff --stat` showed only the tracked reporting-contract documentation update before new files were staged.
- `git diff --name-only` showed only the tracked reporting-contract documentation update before new files were staged.
- guardrail diff showed no unintended legacy test, script, data, report, or workflow diffs.

Final validation was run after this closeout was added and before commit.

## Recommended Next Action

RESET-10G - V2 Telegram Renderer Design and Synthetic Before/After Example.
