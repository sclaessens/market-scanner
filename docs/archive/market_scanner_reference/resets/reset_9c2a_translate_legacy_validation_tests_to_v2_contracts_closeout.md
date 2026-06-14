# RESET-9C2A: Translate Legacy Validation Tests to V2 Contracts Closeout

## Purpose

RESET-9C2A translated safe legacy validation-test knowledge into minimal v2 validation contract metadata and v2 contract tests.

This sprint did not port legacy validation runtime code. It did not modify, move, delete, archive, rename, or refactor legacy tests or legacy scripts.

## Legacy Validation Knowledge Extracted

The legacy validation tests encode these safe contract expectations:

- candidate validation requires stable identity fields, especially `ticker` and `date`;
- validation input shape requires explicit candidate structure fields such as setup, price, moving average, range, risk/reward, volume, and extension metadata;
- missing required fields must be reported explicitly;
- missing values must remain missing and must not be treated as zero;
- non-positive required metrics such as `atr14` are data-coherence issues;
- validation must preserve row identity conceptually and must not silently filter rows;
- validation must not create final actions, allocation fields, execution fields, ranking, urgency, conviction, or tradeability semantics;
- validation contract helpers must be deterministic and side-effect free.

Legacy entry-quality and validation-layer runtime behavior remains protected by the existing legacy tests and was not reimplemented in v2.

## V2 Contracts and Tests Added

Added:

- `src/market_scanner/validation/validation_contracts.py`
- `tests/contract/test_v2_validation_contracts.py`

Updated:

- `docs/active/data_contracts.md`

The new v2 validation contract module defines:

- `ValidationState`;
- `ValidationIssueCode`;
- `ValidationIssue`;
- required candidate field metadata;
- row identity field metadata;
- metadata-only validation classification fields;
- forbidden upstream decision field metadata;
- `validate_candidate_record_shape()` for side-effect-free shape checks.

The new tests verify:

- validation lifecycle states exist;
- required candidate fields are defined;
- row identity fields are preserved as contract metadata;
- validation output fields exclude final-action semantics;
- missing fields and missing values are reported explicitly;
- missing numeric values are not converted to zero;
- invalid numeric and non-positive required values are reported as contract issues;
- forbidden upstream decision fields are reported as contract issues;
- the validation contract module does not import legacy `scripts`;
- validation contract helpers create no files.

## Files Changed

- `docs/active/data_contracts.md`
- `docs/resets/reset_9c2a_translate_legacy_validation_tests_to_v2_contracts_closeout.md`
- `src/market_scanner/validation/validation_contracts.py`
- `tests/contract/test_v2_validation_contracts.py`

## Scope Confirmation

Confirmed:

- no legacy tests were modified;
- no legacy tests were deleted, moved, archived, renamed, or refactored;
- no legacy scripts were modified;
- no legacy scripts were deleted, moved, archived, renamed, or refactored;
- no legacy CSV contents were modified;
- no files under `data/processed/`, `data/portfolio/`, `data/watchlist/`, `data/logs/`, or `reports/` were modified;
- no workflow files were modified;
- no production runtime behavior was changed;
- no full validation runtime was implemented;
- no scanner runtime was implemented;
- no hidden filtering was introduced;
- no missing values were treated as zero;
- no source-data readiness was treated as investment quality.

## Runtime and External Call Confirmation

Confirmed:

- no production pipeline was run;
- `scripts/run_scan.py` was not run;
- Telegram scripts were not run;
- SEC diagnostics were not run;
- provider calls were not made;
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

- `.venv/bin/python -m pytest` passed: `472 passed`.
- `git diff --check` passed.
- `git status --short` showed only intended RESET-9C2A changes before closeout creation.
- `git diff --stat` showed the tracked documentation update before new files were staged.
- `git diff --name-only` showed only the tracked documentation update before new files were staged.
- guardrail diff showed no unintended legacy test, script, data, report, or workflow diffs.

Final validation was run after this closeout was added and before commit.

## Recommended Next Action

RESET-9C2B - Translate Legacy Portfolio Tests to V2 Contracts.
