# Numerical Fundamentals Standard Batch 1 Upstream Refresh Validation

## Status and Scope

This document records a timeout-safe upstream refresh validation for Numerical Fundamentals Standard Batch 1.

This was a validation-only task. It was not a coding sprint, runtime-logic change, new source-data collection task, new fundamentals update task, automated ingestion implementation, or provider/API integration task.

No code, tests, source CSV values, runtime behavior, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, watchlist files, portfolio metadata, or raw fundamentals data were changed by this task.

## References

Preflight validation reference:

- `docs/sprints/numerical_fundamentals_standard_batch_1_preflight_validation.md`

Source batch reference:

- `docs/sprints/numerical_fundamentals_standard_batch_1.md`

Scaling contract reference:

- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`

Protocol reference:

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`

## Local Ignored Source-Data Confirmation

Local ignored raw fundamentals artifact:

- `data/raw/fundamentals.csv`

Confirmation:

- The file exists locally.
- The file remains ignored under `data/raw/`.
- The file was inspected only.
- The file was not edited.
- The file was not staged or committed.

Selected batch tickers:

- `ANET`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`

Writable MVP metrics:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`

Local raw batch coverage:

- All 10 selected tickers have exactly one local raw row.
- `ENPH` and `FTNT` have all five writable MVP metrics present locally.
- The other eight tickers have partial local numerical data.
- No candidate/future metrics were written.
- No duplicate conflicting local raw rows were found.
- No forbidden decision semantics were found in selected source or notes fields.

## Upstream Context Mismatch Summary

The preflight validation found that current generated downstream artifacts contained the selected tickers, but direct Fundamental Layer input did not:

- `data/processed/timing_state_layer.csv`: selected tickers present, stale provenance-only fundamental fields.
- `data/processed/portfolio_intelligence.csv`: selected tickers present, stale provenance-only fundamental fields.
- `data/processed/context_strength.csv`: only 6 rows, no selected tickers.
- `data/processed/fundamental_quality.csv`: only 6 rows, no selected tickers.

This task tested whether the existing focused upstream builder sequence could refresh context broadly enough for the Fundamental Layer to validate the batch without running the full pipeline.

## Selected Minimal Builder Sequence and Rationale

Selected sequence:

1. `PYTHONPATH=. .venv/bin/python scripts/core/build_context_layer.py`
2. `PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py`

Rationale:

- `scripts/core/build_context_layer.py` reads existing `data/processed/scanner_ranked.csv` and optional `data/processed/sector_relative_strength.csv`, then writes `data/processed/context_strength.csv`.
- This step can refresh context from the already-generated scanner output without running the scanner or making provider calls.
- `scripts/core/build_fundamental_layer.py` reads `data/processed/context_strength.csv` and local ignored `data/raw/fundamentals.csv`, then writes `data/processed/fundamental_quality.csv`.
- Running these two focused builders is the minimum safe sequence needed to determine whether refreshed context reaches the Fundamental Layer.

Downstream builders were not run because the Fundamental Layer result revealed a source-date alignment blocker before downstream validation would add useful evidence.

The full pipeline was not run.

## Command Results

Timeout-safe execution was performed through a local Python wrapper with a 120-second timeout per focused builder command.

| command | purpose | result | timeout_used | output_artifact | row_count_after | batch_tickers_present_count | notes |
|---|---|---|---:|---|---:|---:|---|
| `PYTHONPATH=. .venv/bin/python scripts/core/build_context_layer.py` | Refresh `context_strength.csv` from existing scanner output | success in 0.65 seconds | 120 seconds | `data/processed/context_strength.csv` | 291 | 10 | Context refresh succeeded without scanner/provider execution. |
| `PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py` | Rebuild `fundamental_quality.csv` from refreshed context and local ignored raw fundamentals | success in 0.25 seconds | 120 seconds | `data/processed/fundamental_quality.csv` | 291 | 10 | Fundamental Layer output included the selected tickers, but classified them as `row_missing`. |

## Context Refresh Results

Post-refresh `data/processed/context_strength.csv`:

- Row count: 291
- Selected batch tickers present: 10
- `context_strength` distribution:
  - `LEADING`: 30
  - `STRONG`: 43
  - `NEUTRAL`: 102
  - `WEAK`: 116

Post-refresh `data/processed/fundamental_quality.csv`:

- Row count: 291
- Selected batch tickers present: 10
- `quality_state` distribution:
  - `SUFFICIENT_DATA`: 4
  - `PARTIAL_DATA`: 2
  - `INSUFFICIENT_DATA`: 285
- `quality_metadata_status` distribution:
  - `complete`: 4
  - `partial`: 17
  - `row_missing`: 270
- `source_data_status` distribution:
  - `source_available`: 4
  - `partial_data`: 17
  - `row_missing`: 270

## Fundamental Layer Validation Results

The Fundamental Layer output included all 10 selected batch tickers after context refresh. However, all 10 were classified as `INSUFFICIENT_DATA` with `quality_metadata_status = row_missing` and `source_data_status = row_missing`.

Short inspection found the reason:

- Refreshed context rows for all selected batch tickers use opportunity `date = 2026-05-22`.
- Local ignored raw fundamentals rows for all selected batch tickers use `as_of_date = 2026-05-24`.
- `scripts/core/build_fundamental_layer.py` selects raw fundamentals candidates only when `_as_of_date <= opportunity_date`.
- Because `2026-05-24` is later than `2026-05-22`, the selected local raw rows are intentionally excluded and the output remains `row_missing`.

This is not a provider failure, not a missing local raw row problem, and not a context refresh failure. It is a raw fundamentals date-alignment contract issue.

## Expected vs Actual Sufficiency Behavior

Expected behavior based on local metric coverage:

- `ENPH` and `FTNT`: expected `SUFFICIENT_DATA` if all five writable MVP metrics are recognized.
- Other eight selected tickers: expected `PARTIAL_DATA` because at least one writable MVP metric remains blank.

Actual behavior after refresh:

- All 10 selected tickers remained `INSUFFICIENT_DATA`.
- All 10 selected tickers had `quality_metadata_status = row_missing`.
- All 10 selected tickers had `source_data_status = row_missing`.

Discrepancy:

- Local raw rows are present, but their `as_of_date` is later than the scanner opportunity date, so the Fundamental Layer treats them as unavailable for that opportunity.

## Batch Validation Table

| ticker | raw_row_present | writable_mvp_metrics_present_count | missing_writable_mvp_metrics | appears_in_context_strength | appears_in_fundamental_quality | quality_state_after_refresh | source_data_status_after_refresh | expected_sufficiency_state | actual_sufficiency_state | validation_state |
|---|---|---:|---|---|---|---|---|---|---|---|
| ANET | YES | 4 | `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| DELL | YES | 3 | `gross_margin`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| ENPH | YES | 5 | none | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `SUFFICIENT_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| EOG | YES | 3 | `gross_margin`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| EQIX | YES | 4 | `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| EW | YES | 3 | `eps_growth_yoy`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| EXPD | YES | 3 | `gross_margin`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| FDX | YES | 3 | `gross_margin`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| FTNT | YES | 5 | none | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `SUFFICIENT_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |
| HAL | YES | 3 | `gross_margin`; `debt_to_equity` | YES | YES | `INSUFFICIENT_DATA` | `row_missing` | `PARTIAL_DATA` | `INSUFFICIENT_DATA` | PARTIAL_VALIDATION |

## Optional Downstream Observations

No downstream builders were run.

Reason:

- The focused upstream refresh already reached `fundamental_quality.csv` with all selected tickers.
- The validation blocker was identified at the Fundamental Layer row-matching step.
- Running timing, portfolio intelligence, decision, or reporting builders would only propagate the current `row_missing` state and would not clarify the date-alignment issue.

## Generated Artifact Handling

Generated artifacts changed during validation:

- `data/processed/context_strength.csv`
- `data/logs/context_layer_log.csv`

`data/processed/fundamental_quality.csv` and `data/logs/fundamental_layer_log.csv` were written by the builder and inspected as evidence, but no tracked changes remained for those files.

Tracked generated artifacts were restored before documentation commit.

No generated artifacts were staged or committed.

Ignored generated or local artifacts were not committed.

## Validation Limitations

- This task did not run the full pipeline.
- This task did not run scanner/provider downloads.
- This task did not update local raw fundamentals.
- This task did not test an alternate `as_of_date` policy.
- This task did not modify Fundamental Layer logic.
- This task did not validate downstream Decision Engine or Reporting output because the Fundamental Layer row-matching blocker was found first.

## Backlog Impact Assessment

Backlog impact assessment:
- Added `BL-0020 — Define raw fundamentals as-of-date alignment policy for opportunity-date validation`.

## Recommended Next Step

Run a governed data-contract alignment task for `BL-0020` before changing raw data or code.

That task should decide how local ignored raw fundamentals `as_of_date`, `source_freshness_date`, and scanner opportunity `date` should interact. It should explicitly decide whether Standard Batch 1 rows should be re-keyed to the scanner opportunity date, whether the Fundamental Layer should match by ticker with freshness rules, or whether a separate validation fixture should be used.

No raw fundamentals edits should be made until that date-alignment policy is approved.
