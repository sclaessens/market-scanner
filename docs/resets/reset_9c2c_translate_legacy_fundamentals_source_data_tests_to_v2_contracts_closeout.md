# RESET-9C2C: Translate Legacy Fundamentals and Source-Data Tests to V2 Contracts Closeout

## Purpose

RESET-9C2C translated safe legacy fundamentals and source-data test knowledge into minimal v2 fundamentals/source-data contract metadata and v2 contract tests.

This sprint did not port legacy fundamentals runtime code. It did not modify, move, delete, archive, rename, or refactor legacy tests or legacy scripts.

## Legacy Fundamentals and Source-Data Knowledge Extracted

The legacy fundamentals and source-data tests encode these safe contract expectations:

- source-data readiness states must remain explicit, including available, missing, source-missing, row-missing, partial, stale, invalid, unavailable, and review-required states;
- source-data readiness is a governance state and must not imply investment quality;
- raw source capture, normalized source-readiness input, normalized fundamentals history input, generated classification output, generated analysis output, and local-only review artifacts must remain separate dataset roles;
- normalized source-readiness records need explicit source identity, symbol, source name, metric name, metric value, metric unit, date, missing-value policy, and review-reason fields;
- normalized fundamentals history records need explicit ticker, fiscal year, fiscal period, period dates, currency, source provenance, freshness, extraction, and notes fields;
- missing required fields must be reported explicitly;
- missing numeric values must remain missing and must not be treated as zero;
- invalid numeric values must be reported without computing metrics;
- invalid readiness states must be reported without quality inference;
- source provenance and period metadata must remain visible;
- fundamentals/source-data contracts must not create final actions, allocation fields, execution semantics, tradeability, conviction, urgency, rankings, or quality scores;
- contract helpers must be deterministic and side-effect free.

Legacy SEC fact selection, provider-assisted prefill, fundamentals history validation, metric calculation, quality classification, analysis generation, and data coverage diagnostics remain protected by existing legacy tests and were not reimplemented in v2.

## V2 Contracts and Tests Added

Added:

- `src/market_scanner/fundamentals/fundamental_contracts.py`
- `tests/contract/test_v2_fundamental_contracts.py`

Updated:

- `docs/active/data_contracts.md`
- `docs/active/source_data_strategy.md`

The new v2 fundamentals contract module defines:

- `FundamentalDatasetRole`;
- `SourceDataReadinessState`;
- `FundamentalContractIssueCode`;
- `FundamentalContractIssue`;
- source-readiness required field metadata;
- fundamentals-history required field metadata;
- source-readiness and fundamentals-history identity field metadata;
- source dataset role metadata;
- generated dataset role metadata;
- forbidden fundamentals authority field metadata;
- `validate_source_readiness_shape()` for side-effect-free source-readiness shape checks;
- `validate_fundamental_history_shape()` for side-effect-free fundamentals-history shape checks.

The new tests verify:

- source-data readiness states exist and are not quality states;
- raw source, normalized input, generated output, and local-only review roles remain distinct;
- source-readiness required fields match the approved synthetic fixture;
- fundamentals-history fields include period and provenance metadata;
- identity fields are explicit;
- missing fields and missing values are reported explicitly;
- missing numeric values are not converted to zero;
- invalid numeric values are reported without scoring;
- forbidden fundamentals authority fields are reported as contract issues;
- contract metadata excludes final-action and investment-quality authority;
- the contract module does not import legacy `scripts` or network/provider modules;
- contract helpers create no files.

## Files Changed

- `docs/active/data_contracts.md`
- `docs/active/source_data_strategy.md`
- `docs/resets/reset_9c2c_translate_legacy_fundamentals_source_data_tests_to_v2_contracts_closeout.md`
- `src/market_scanner/fundamentals/fundamental_contracts.py`
- `tests/contract/test_v2_fundamental_contracts.py`

## Scope Confirmation

Confirmed:

- no legacy tests were modified;
- no legacy tests were deleted, moved, archived, renamed, or refactored;
- no legacy scripts were modified;
- no legacy scripts were deleted, moved, archived, renamed, or refactored;
- no real data CSVs were modified;
- no legacy CSV contents were modified;
- no files under `data/processed/`, `data/portfolio/`, `data/watchlist/`, `data/logs/`, or `reports/` were modified;
- no workflow files were modified;
- no production runtime behavior was changed;
- no full fundamentals runtime was implemented;
- no SEC/provider ingestion was implemented;
- no raw-to-normalized transformation was implemented;
- no generated fundamentals outputs were added;
- no financial scoring model was implemented;
- no investment-quality model was implemented;
- source-data readiness was not turned into investment quality;
- missing numeric values were not treated as zero.

## Runtime and External Call Confirmation

Confirmed:

- no production pipeline was run;
- `scripts/run_scan.py` was not run;
- fundamentals legacy scripts were not run;
- data source legacy scripts were not run;
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
.venv/bin/python -m pytest tests/contract/test_v2_fundamental_contracts.py
.venv/bin/python -m pytest
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- tests scripts src data reports .github/workflows || true
```

Results:

- `.venv/bin/python -m pytest tests/contract/test_v2_fundamental_contracts.py` passed: `14 passed`.
- `.venv/bin/python -m pytest` passed: `497 passed`.
- `git diff --check` passed.
- `git status --short` showed only intended RESET-9C2C changes before closeout creation.
- `git diff --stat` showed only tracked documentation updates before new files were staged.
- `git diff --name-only` showed only tracked documentation updates before new files were staged.
- guardrail diff showed no unintended legacy test, script, data, report, or workflow diffs.

Final validation was run after this closeout was added and before commit.

## Recommended Next Action

RESET-9C2D - Translate Legacy Reporting and Telegram Tests to V2 Contracts.
