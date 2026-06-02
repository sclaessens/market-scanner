# RESET-9C2B: Translate Legacy Portfolio Tests to V2 Contracts Closeout

## Purpose

RESET-9C2B translated safe legacy portfolio-test knowledge into minimal v2 portfolio contract metadata and v2 contract tests.

This sprint did not port legacy portfolio runtime code. It did not modify, move, delete, archive, rename, or refactor legacy tests or legacy scripts.

## Legacy Portfolio Knowledge Extracted

The legacy portfolio tests encode these safe contract expectations:

- manual portfolio transactions need explicit identity, account, symbol, transaction kind, quantity, cash, currency, occurrence date, and source-reference fields;
- normalized portfolio positions need explicit account and symbol identity;
- manual portfolio source records must remain separate from generated portfolio review, generated portfolio intelligence, and generated classification outputs;
- missing required portfolio fields must be reported explicitly;
- missing numeric portfolio values must remain missing and must not be treated as zero;
- invalid numeric portfolio values must be reported without running transaction ingestion;
- descriptive portfolio metadata may be useful later, but it must not create allocation, execution, tradeability, conviction, urgency, ranking, or final-action authority;
- generated portfolio outputs must not overwrite or become manual source records;
- portfolio contract helpers must be deterministic and side-effect free.

Legacy portfolio transaction rebuilding, position calculation, portfolio intelligence generation, metadata prefill, and portfolio review behavior remain protected by existing legacy tests and were not reimplemented in v2.

## V2 Contracts and Tests Added

Added:

- `src/market_scanner/portfolio/portfolio_contracts.py`
- `tests/contract/test_v2_portfolio_contracts.py`

Updated:

- `docs/active/data_contracts.md`

The new v2 portfolio contract module defines:

- `PortfolioDatasetType`;
- `PortfolioContractIssueCode`;
- `PortfolioContractIssue`;
- manual transaction required field metadata;
- normalized position required field metadata;
- transaction and position identity field metadata;
- source dataset type metadata;
- generated dataset type metadata;
- forbidden portfolio authority field metadata;
- `validate_portfolio_transaction_shape()` for side-effect-free manual transaction shape checks;
- `validate_portfolio_position_shape()` for side-effect-free normalized position shape checks.

The new tests verify:

- manual transaction required fields match the approved synthetic portfolio fixture;
- normalized position required fields are explicit;
- transaction and position identity fields are explicit;
- portfolio source dataset roles are separated from generated dataset roles;
- missing required fields and missing values are reported explicitly;
- missing numeric values are not converted to zero;
- invalid numeric values are reported without ingestion;
- forbidden portfolio authority fields are reported as contract issues;
- portfolio contract output metadata excludes final-action, allocation, execution, tradeability, conviction, and urgency authority;
- the portfolio contract module does not import legacy `scripts`;
- portfolio contract helpers create no files.

## Files Changed

- `docs/active/data_contracts.md`
- `docs/resets/reset_9c2b_translate_legacy_portfolio_tests_to_v2_contracts_closeout.md`
- `src/market_scanner/portfolio/portfolio_contracts.py`
- `tests/contract/test_v2_portfolio_contracts.py`

## Scope Confirmation

Confirmed:

- no legacy tests were modified;
- no legacy tests were deleted, moved, archived, renamed, or refactored;
- no legacy scripts were modified;
- no legacy scripts were deleted, moved, archived, renamed, or refactored;
- no real portfolio CSVs were modified;
- no legacy CSV contents were modified;
- no files under `data/portfolio/`, `data/processed/`, `data/watchlist/`, `data/logs/`, or `reports/` were modified;
- no workflow files were modified;
- no production runtime behavior was changed;
- no full portfolio runtime was implemented;
- no transaction ingestion was implemented;
- no broker integration was implemented;
- no generated portfolio outputs were added;
- no portfolio allocation decisions were added;
- no portfolio classification was allowed to become final-action authority.

## Runtime and External Call Confirmation

Confirmed:

- no production pipeline was run;
- `scripts/run_scan.py` was not run;
- portfolio legacy scripts were not run;
- Telegram scripts were not run;
- SEC diagnostics were not run;
- provider calls were not made;
- broker calls were not made;
- network calls were not made;
- Telegram API calls were not made;
- live data calls were not made;
- the daily scan schedule was not re-enabled.

## Validation Commands and Results

Commands run:

```bash
.venv/bin/python -m pytest
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- tests scripts src data reports .github/workflows || true
```

Results:

- `.venv/bin/python -m pytest` passed: `483 passed`.
- `git diff --check` passed.
- `git status --short` showed only intended RESET-9C2B changes before closeout creation.
- `git diff --stat` showed the tracked documentation update before new files were staged.
- `git diff --name-only` showed only the tracked documentation update before new files were staged.
- guardrail diff showed no unintended legacy test, script, data, report, or workflow diffs.

Final validation was run after this closeout was added and before commit.

## Recommended Next Action

RESET-9C2C - Translate Legacy Fundamentals and Source-Data Tests to V2 Contracts.
