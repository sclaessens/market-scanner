# Portfolio Metadata Update 1 Post-Merge Validation

## Status and Scope

This document records post-merge validation for the governed Portfolio Metadata CSV Update batch 1.

This is a validation-only artifact. It is not a coding sprint, runtime-logic change, source-data expansion task, automated ingestion implementation, provider integration, scraping task, or credential task.

No source-data CSV values were added, edited, or removed during this validation.

## Protocol References

Governance and source-data references:

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/portfolio_metadata_expansion_preview_1.md`
- `docs/sprints/portfolio_metadata_source_lookup_preview_1.md`
- `docs/sprints/portfolio_metadata_csv_update_1.md`

## Source Update Reference

Merged source-data update:

- PR #78, `data: add approved portfolio metadata batch 1`
- Updated source artifact: `data/portfolio/portfolio_metadata.csv`

The update added 15 approved rows:

- `AMAT`
- `ANET`
- `ASML`
- `COST`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`
- `HLT`
- `HPE`

## Repository State Before Validation

Validation started from `main`.

`main` was updated from `origin/main` and fast-forwarded to include PR #78.

The working tree was clean before validation.

## Metadata CSV Validation Results

`data/portfolio/portfolio_metadata.csv` validation:

- Row count: 36
- Required columns present: yes
- Duplicate ticker rows: none
- Approved batch tickers present: 15 of 15
- Asset class distribution:
  - `Equity`: 33
  - `REIT`: 3
- Metadata source distribution:
  - `Yahoo Finance`: 36

Approved descriptive `asset_class` values used by this validation:

- `Equity`
- `REIT`
- `ETF`

## Approved Batch Coverage Table

| ticker | metadata_present | sector_present | industry_present | asset_class | currency_present | metadata_source_present | metadata_last_updated_valid | decision_semantics_absent | validation_state |
|---|---|---|---|---|---|---|---|---|---|
| AMAT | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| ANET | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| ASML | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| COST | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| DELL | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| ENPH | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| EOG | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| EQIX | yes | yes | yes | REIT | yes | yes | yes | yes | VALIDATED |
| EW | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| EXPD | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| FDX | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| FTNT | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| HAL | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| HLT | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |
| HPE | yes | yes | yes | Equity | yes | yes | yes | yes | VALIDATED |

All 15 approved tickers passed metadata CSV validation.

## Commands Run

Repository synchronization and status:

```bash
git checkout main
git pull origin main
git status --short
```

Metadata CSV validation:

```bash
.venv/bin/python - <<'PY'
import csv
from collections import Counter
from datetime import date
from pathlib import Path

path = Path('data/portfolio/portfolio_metadata.csv')
approved = ['AMAT','ANET','ASML','COST','DELL','ENPH','EOG','EQIX','EW','EXPD','FDX','FTNT','HAL','HLT','HPE']
required = ['ticker','sector','industry','asset_class','currency','metadata_source','metadata_last_updated']

with path.open(newline='') as f:
    rows = list(csv.DictReader(f))

print('row_count', len(rows))
print('required_missing', [c for c in required if c not in rows[0].keys()])
print('duplicates', sorted([ticker for ticker, count in Counter(r['ticker'] for r in rows).items() if count > 1]))
print('asset_class_distribution', Counter(r['asset_class'] for r in rows))
print('metadata_source_distribution', Counter(r['metadata_source'] for r in rows))
PY
```

Patch validation:

```bash
git diff --check
```

Portfolio Intelligence builder:

```bash
PYTHONPATH=. .venv/bin/python scripts/core/build_portfolio_intelligence.py
```

Full pipeline:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_full_pipeline.py
```

Focused portfolio tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/core -k "portfolio"
PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio
```

Generated artifact cleanup:

```bash
git restore data/logs/context_layer_log.csv data/portfolio/portfolio_positions.csv data/portfolio/portfolio_review.csv data/processed/context_strength.csv data/processed/scanner_ranked.csv
```

## Builder and Pipeline Results

Portfolio Intelligence builder result:

- Command succeeded.
- `data/processed/portfolio_intelligence.csv` rows: 291
- Portfolio metadata status distribution after builder:
  - `COMPLETE`: 21
  - `MISSING`: 255
  - `PARTIAL`: 15

The direct builder run before full pipeline used the pre-existing scanner opportunity date and reported the 15 new rows as `PARTIAL` because `metadata_last_updated = 2026-05-23` was after that opportunity date.

Full pipeline result:

- Command succeeded.
- Scanner output rows after duplicate removal: 291
- Portfolio Intelligence output rows: 291
- Final Decisions output rows: 291
- Reporting output rows: 291
- Telegram summary generation completed.

After the full pipeline refreshed the scanner opportunity date to `2026-05-23`, Portfolio Intelligence recognized all 15 approved batch rows as `COMPLETE`.

## Downstream Artifact Observations

Post-pipeline `data/processed/portfolio_intelligence.csv`:

- Rows: 291
- `portfolio_metadata_status` distribution:
  - `COMPLETE`: 36
  - `MISSING`: 255

Approved batch Portfolio Intelligence status:

| ticker | portfolio_metadata_status | portfolio_metadata_reason |
|---|---|---|
| AMAT | COMPLETE | portfolio metadata complete |
| ANET | COMPLETE | portfolio metadata complete |
| ASML | COMPLETE | portfolio metadata complete |
| COST | COMPLETE | portfolio metadata complete |
| DELL | COMPLETE | portfolio metadata complete |
| ENPH | COMPLETE | portfolio metadata complete |
| EOG | COMPLETE | portfolio metadata complete |
| EQIX | COMPLETE | portfolio metadata complete |
| EW | COMPLETE | portfolio metadata complete |
| EXPD | COMPLETE | portfolio metadata complete |
| FDX | COMPLETE | portfolio metadata complete |
| FTNT | COMPLETE | portfolio metadata complete |
| HAL | COMPLETE | portfolio metadata complete |
| HLT | COMPLETE | portfolio metadata complete |
| HPE | COMPLETE | portfolio metadata complete |

Post-pipeline `data/processed/fundamental_quality.csv`:

- Rows: 291
- `quality_state` distribution:
  - `INSUFFICIENT_DATA`: 285
  - `PARTIAL_DATA`: 2
  - `SUFFICIENT_DATA`: 4

Post-pipeline `data/processed/final_decisions.csv`:

- Rows: 291
- `final_action` distribution:
  - `NO_ACTION`: 5
  - `REVIEW`: 285
  - `WAIT`: 1
- `arbitration_state` distribution:
  - `MISSING_METADATA`: 285
  - `NO_CONFLICT`: 5
  - `TIMING_CONFLICT`: 1
- `portfolio_metadata_status` distribution:
  - `COMPLETE`: 36
  - `MISSING`: 255

For the 15 approved batch tickers, `portfolio_metadata_status = COMPLETE` in final decisions, but `final_action = REVIEW` remains because `quality_state = INSUFFICIENT_DATA` and `opportunity_decision_state = INSUFFICIENT_DECISION_METADATA`.

The current validation evidence indicates that the approved batch is no longer blocked by Portfolio Intelligence metadata coverage. The remaining blocker for these 15 rows is fundamentals/decision metadata coverage, not missing portfolio metadata.

## Generated Artifact Handling

The builder and full pipeline modified generated or runtime artifacts during validation.

Tracked runtime artifacts changed during validation:

- `data/logs/context_layer_log.csv`
- `data/portfolio/portfolio_positions.csv`
- `data/portfolio/portfolio_review.csv`
- `data/processed/context_strength.csv`
- `data/processed/scanner_ranked.csv`

These files were restored with `git restore` before committing this validation document.

No generated outputs are included in the validation commit.

## Test Results

Focused portfolio metadata / portfolio intelligence tests:

- `PYTHONPATH=. .venv/bin/python -m pytest tests/core -k "portfolio"`
- Result: 37 passed, 142 deselected

Portfolio source contract tests:

- Initial run after full pipeline: 1 failed, 2 passed
- Cause: generated `data/portfolio/portfolio_positions.csv` had been refreshed by the full pipeline with descriptive market enrichment values.
- Action: restored generated/tracked runtime artifacts.
- Rerun result: 3 passed

## Validation Limitations

This validation did not modify source data or runtime logic.

The direct Portfolio Intelligence builder result depends on the currently available upstream scanner opportunity date. Before the full pipeline refreshed scanner output to `2026-05-23`, the builder classified the 15 new rows as `PARTIAL` because their approved `metadata_last_updated` date was later than the older opportunity date.

The full pipeline refreshed scanner output and validated that all 15 approved rows classify as `COMPLETE` under the current end-to-end run.

No full pytest suite was run in this validation task.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0011 — Define and repair authoritative active portfolio source`

The remaining blocker is covered by existing fundamentals/source-data coverage workstreams.

## Recommended Next Step

Launch a separate governed fundamentals source-data expansion or validation task for the metadata-complete A-grade rows that still have `quality_state = INSUFFICIENT_DATA`.

Do not change Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, Portfolio Intelligence, or runtime behavior as part of that next step unless separately authorized.
