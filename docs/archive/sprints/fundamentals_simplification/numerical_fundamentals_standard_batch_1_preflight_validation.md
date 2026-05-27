# Numerical Fundamentals Standard Batch 1 Preflight Validation

## Status and Scope

This document records a targeted preflight validation for Numerical Fundamentals Standard Batch 1.

This was a validation-only task. It was not a coding sprint, runtime-logic change, new source-data collection task, new fundamentals update task, full pipeline validation task, or automated ingestion implementation.

No code, tests, source CSV values, runtime behavior, generated artifacts, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, watchlist files, portfolio source data, or raw fundamentals values were changed by this task.

## Source Batch Reference

Source batch documentation:

- `docs/sprints/numerical_fundamentals_standard_batch_1.md`

Batch tickers:

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

## Reason for Targeted Preflight Validation

A previous full-context validation attempt was stopped because the run exceeded a reasonable interaction window before useful validation started. That attempt confirmed that `main` was clean and that `data/raw/` remained ignored, but no full pipeline run had started.

This preflight determines whether the local ignored raw fundamentals values are present and whether current generated upstream context can validate the 10 batch tickers without running the full pipeline.

## Sync and Safety Check

Short commands only were used.

Repository state:

- Branch before documentation branch creation: `main`
- `main` was updated from `origin/main`.
- PR #86 content is present on `main` through `docs/sprints/numerical_fundamentals_standard_batch_1.md`.
- Working tree was clean before creating this documentation artifact.
- `data/raw/fundamentals.csv` exists locally.
- `data/raw/` remains ignored by Git.
- No full pipeline command was run.
- No GitHub CLI preflight command was run.

## Local Ignored Raw Fundamentals Confirmation

Local raw fundamentals artifact:

- `data/raw/fundamentals.csv`

Raw row count:

- 36

Raw columns:

- `ticker`
- `as_of_date`
- `source_name`
- `source_reference`
- `source_freshness_date`
- `currency`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`
- `fundamental_notes`

Candidate/future metrics were not present as writable raw columns and were not written:

- `net_margin`
- `return_on_equity`
- `free_cash_flow_margin`

No duplicate raw rows were found for the selected batch tickers. No forbidden decision semantics were found in the selected batch source or notes fields.

## Raw Batch Metric Coverage

| ticker | raw_row_present | revenue_growth_yoy_present | eps_growth_yoy_present | gross_margin_present | operating_margin_present | debt_to_equity_present | missing_writable_mvp_metrics | duplicate_conflict | validation_state |
|---|---|---|---|---|---|---|---|---|---|
| ANET | YES | YES | YES | YES | YES | NO | `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| DELL | YES | YES | YES | NO | YES | NO | `gross_margin`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| ENPH | YES | YES | YES | YES | YES | YES | none | NO | READY_FOR_FULL_CONTEXT_VALIDATION |
| EOG | YES | YES | YES | NO | YES | NO | `gross_margin`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| EQIX | YES | YES | YES | YES | YES | NO | `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| EW | YES | YES | NO | YES | YES | NO | `eps_growth_yoy`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| EXPD | YES | YES | YES | NO | YES | NO | `gross_margin`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| FDX | YES | YES | YES | NO | YES | NO | `gross_margin`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |
| FTNT | YES | YES | YES | YES | YES | YES | none | NO | READY_FOR_FULL_CONTEXT_VALIDATION |
| HAL | YES | YES | YES | NO | YES | NO | `gross_margin`; `debt_to_equity` | NO | PARTIAL_LOCAL_DATA |

Local metric coverage summary:

- All 10 selected tickers have exactly one local raw row.
- 36 of 50 writable MVP metric cells are present locally.
- 14 of 50 writable MVP metric cells remain blank because they were review-required in Standard Batch 1.
- `ENPH` and `FTNT` have all five writable MVP metrics locally and appear potentially eligible for `SUFFICIENT_DATA` once a full-context validation run refreshes the appropriate upstream universe.
- The other eight tickers remain partial based on local raw metric coverage.

## Upstream Context Inspection Results

| artifact | exists | row_count | batch_tickers_present_count | notes |
|---|---|---:|---:|---|
| `data/processed/context_strength.csv` | YES | 6 | 0 | This appears to be the direct Fundamental Layer upstream input. It does not include any selected batch ticker. |
| `data/processed/timing_state_layer.csv` | YES | 289 | 10 | All selected batch tickers are present, but embedded fundamental fields still show old provenance-only status from `source_name = provenance_only_manual_seed`. |
| `data/processed/portfolio_intelligence.csv` | YES | 289 | 10 | All selected batch tickers are present, but embedded fundamental fields still show old provenance-only status from `source_name = provenance_only_manual_seed`. |
| `data/processed/fundamental_quality.csv` | YES | 6 | 0 | Current direct Fundamental Layer output does not include selected batch tickers. |

Relevant observed status for selected tickers in `timing_state_layer.csv` and `portfolio_intelligence.csv`:

- `quality_state = INSUFFICIENT_DATA`
- `quality_reason = fundamental data insufficient`
- `quality_metadata_status = partial`
- `source_data_status = partial_data`
- `source_name = provenance_only_manual_seed`

These values appear stale relative to the local ignored raw fundamentals update from Standard Batch 1. They indicate that the current generated downstream artifacts have not been refreshed in a context that consumes the updated local raw values for the selected batch.

## Fundamental Layer Builder Decision

The Fundamental Layer builder was not run in this preflight.

Reason:

- `data/processed/context_strength.csv` currently contains 6 rows and 0 selected batch tickers.
- `data/processed/fundamental_quality.csv` currently contains 6 rows and 0 selected batch tickers.
- Running `PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py` against this context would not validate the selected batch tickers.

This matches the known limitation from Standard Batch 1: the direct Fundamental Layer builder can complete quickly but does not prove full-context recognition unless upstream generated context includes the selected batch tickers.

## Can Current Upstream Context Validate the Batch?

No.

The current upstream context cannot validate the selected batch through the direct Fundamental Layer builder because the relevant immediate upstream artifact contains only 6 rows and no selected batch tickers.

The repository does contain later generated artifacts with all 10 selected tickers, but those artifacts still reflect older provenance-only fundamental fields. They are evidence that the broader opportunity universe contains the batch tickers, not evidence that the updated local raw numerical metrics have been consumed.

## Generated Artifact Handling

No builder, test, or full pipeline command was run.

Generated artifacts were inspected only as existing files:

- `data/processed/context_strength.csv`
- `data/processed/timing_state_layer.csv`
- `data/processed/portfolio_intelligence.csv`
- `data/processed/fundamental_quality.csv`

No generated artifacts were modified, restored, staged, or committed.

## Validation Limitations

- This preflight did not run the full pipeline.
- This preflight did not run the Fundamental Layer builder because the current direct upstream context lacks the selected batch tickers.
- This preflight did not update `data/raw/fundamentals.csv`.
- This preflight did not collect new source values.
- This preflight cannot confirm whether the full pipeline will move `ENPH` or `FTNT` to `SUFFICIENT_DATA`.
- Full-context validation still requires an explicit upstream-refresh validation run with a timeout-safe execution plan.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

## Recommended Next Step

Run a separate timeout-safe full-context validation task that explicitly refreshes the upstream opportunity universe and then verifies whether the updated local ignored raw fundamentals values are consumed for the 10 selected batch tickers.

Recommended guardrails for that later task:

- keep `data/raw/fundamentals.csv` ignored and uncommitted;
- use a bounded timeout for the full pipeline command;
- capture the last completed pipeline stage if timeout occurs;
- restore tracked generated artifacts after inspection;
- commit only a documentation report.
