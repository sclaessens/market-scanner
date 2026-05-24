# Fundamentals Source-Data Expansion Preview 1

## Status and Scope

This document is the first governed fundamentals source-data expansion preview for metadata-complete tickers that remain blocked by insufficient fundamental data.

This is a preview-only source-data planning artifact. It is not a coding sprint, runtime-logic change, automated ingestion implementation, provider integration, scraping task, credential task, pipeline run, or source-data update task.

No fundamentals source-data values were added, edited, collected, or approved by this document.

No numerical fundamentals were collected.

`data/raw/fundamentals.csv` was inspected only. It was not modified.

No CSV files, generated files, runtime files, scripts, tests, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, watchlist files, portfolio source CSV values, or raw fundamentals files were modified.

## Protocol Reference

This preview follows `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`.

The Automated Source Data Steward protocol allows candidate identification and preview classification before any governed source-data update. Because this task does not authorize source lookup or source-data writes, all selected rows remain review-required for a later fundamentals source lookup task.

## Portfolio Metadata Validation Reference

Recent metadata validation reference:

- `docs/sprints/portfolio_metadata_update_1_post_merge_validation.md`

That validation confirmed that the 15 Portfolio Metadata Update batch 1 tickers are metadata-complete after the full pipeline refresh, but remain blocked downstream by insufficient fundamentals or decision metadata.

## Current Fundamentals Coverage Summary

Current inspected artifact:

- `data/processed/fundamental_quality.csv`

Observed row count: 291

`quality_state` distribution:

| quality_state | count |
|---|---:|
| INSUFFICIENT_DATA | 285 |
| PARTIAL_DATA | 2 |
| SUFFICIENT_DATA | 4 |

`quality_metadata_status` distribution:

| quality_metadata_status | count |
|---|---:|
| complete | 4 |
| partial | 17 |
| row_missing | 270 |

`source_data_status` distribution:

| source_data_status | count |
|---|---:|
| source_available | 4 |
| partial_data | 17 |
| row_missing | 270 |

Rows with `quality_state = INSUFFICIENT_DATA`: 285

Rows with visible source/provenance support: 21

Rows still source-missing based on `row_missing` source status: 270

Current local ignored raw fundamentals artifact:

- `data/raw/fundamentals.csv` exists locally.
- It remains local ignored source data.
- It has 21 rows.
- It uses the approved MVP schema.
- It includes 6 manually maintained rows and 15 provenance-only rows from a prior validation batch.
- It does not include any of the selected preview tickers in this document.

## Metadata-Complete Candidate Summary

Current inspected artifact:

- `data/processed/portfolio_intelligence.csv`

Observed row count: 291

`portfolio_metadata_status` distribution:

| portfolio_metadata_status | count |
|---|---:|
| COMPLETE | 36 |
| MISSING | 255 |

Metadata-complete tickers that still have `quality_state = INSUFFICIENT_DATA`: 30

Candidate tickers:

- `AAPL`
- `ALL`
- `AMAT`
- `AMD`
- `ANET`
- `ASML`
- `BKR`
- `BMY`
- `CB`
- `CL`
- `COST`
- `CRWD`
- `CSCO`
- `CSX`
- `D`
- `DDOG`
- `DELL`
- `DVN`
- `DXCM`
- `EBAY`
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

Priority observations:

- Current portfolio holding candidate: `COST`
- Recently metadata-complete batch 1 candidates: `AMAT`, `ANET`, `ASML`, `COST`, `DELL`, `ENPH`, `EOG`, `EQIX`, `EW`, `EXPD`, `FDX`, `FTNT`, `HAL`, `HLT`, `HPE`
- Existing source-supported or provenance-supported fundamentals candidates from prior local ignored raw rows: `AAPL`, `ALL`, `AMD`, `BKR`, `BMY`, `CB`, `CL`, `CRWD`, `CSCO`, `CSX`, `D`, `DDOG`, `DVN`, `DXCM`, `EBAY`

The selected preview batch follows the protocol priority order by selecting the recently metadata-complete batch 1 set, including `COST`, the current portfolio holding candidate.

## Selected Preview Batch

Selected preview tickers:

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

Batch size: 15

This does not expand the batch beyond the approved standard controlled preview size.

## Fundamentals Expansion Preview Table

| ticker | metadata_status | current_quality_state | current_quality_metadata_status | current_source_data_status | current_missing_fundamentals_status | candidate_reason | proposed_source_method | proposed_action | steward_state | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| AMAT | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| ANET | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| ASML | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| COST | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Current portfolio holding and recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| DELL | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| ENPH | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| EOG | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| EQIX | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| EW | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| EXPD | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| FDX | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| FTNT | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| HAL | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| HLT | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |
| HPE | COMPLETE | INSUFFICIENT_DATA | row_missing | row_missing | source row unavailable | Recently metadata-complete batch 1 ticker lacks a raw fundamentals row. | Governed manual fundamentals source review | PREVIEW_FUNDAMENTALS_PROVENANCE_LOOKUP | REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP | No fundamentals values collected; later task may evaluate provenance-only local ignored row first. |

## Steward Classification

All selected preview rows are classified as:

- `REVIEW_REQUIRED_FOR_FUNDAMENTALS_SOURCE_LOOKUP`

No row is classified as `APPROVED` in this preview because no fundamentals source lookup, provenance validation, metric definition validation, period validation, unit validation, or numerical value validation was authorized or performed.

Approval can happen only in a later source lookup or provenance task using an approved source method and explicit governance for the intended update type.

## Source-Method Recommendation

### Provenance-Only Fundamentals Expansion

Recommended next method for the selected batch:

- A separate governed local ignored source-data update may add provenance-only rows to `data/raw/fundamentals.csv` for the 15 selected tickers.
- That later task should use the approved MVP raw fundamentals schema.
- That later task should not add numerical metric values.
- That later task should keep `data/raw/fundamentals.csv` ignored and uncommitted.

Purpose:

- Create local ignored source/provenance support.
- Make the absence of numerical metrics explicit.
- Validate whether rows move from pure `row_missing` toward `partial` or `partial_data` under the current Fundamental Layer contract.

Expected limitation:

- Provenance-only rows are likely to remain `INSUFFICIENT_DATA` unless enough numerical metrics are sourced later.
- This method is useful for validation of source/provenance flow, not for final `SUFFICIENT_DATA` classification.

### Numerical Fundamentals Expansion

Numerical fundamentals expansion should remain a stricter later task.

It should not start until governance explicitly approves:

- source method;
- metric definitions;
- reporting period or as-of date;
- currency and unit conventions;
- parseability rules;
- allowed calculations, if any;
- validation rules for missing, stale, partial, or contradictory values.

Potential metric categories include revenue growth, EPS growth, margins, debt/equity, and other approved MVP optional metrics. No numerical values are collected or approved in this preview.

## Explicit Non-Approval Statements

No fundamentals values were collected or approved.

No numerical fundamentals were collected.

`data/raw/fundamentals.csv` was not modified.

No CSV files were modified.

No generated files were modified.

No provider APIs were called.

No scraping was performed.

No credentials or secrets were created.

No pipeline was run.

No allocation, tradeability, urgency, conviction, ranking, scoring, eligibility, hidden filtering, buy/sell advice, or Decision Engine bypass semantics were introduced.

## Validation Notes

Validation was documentation-only.

Inspected artifacts:

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/portfolio_metadata_update_1_post_merge_validation.md`
- `docs/sprints/project_backlog.md`
- `data/portfolio/portfolio_metadata.csv`
- `data/processed/fundamental_quality.csv`
- `data/processed/portfolio_intelligence.csv`
- local ignored `data/raw/fundamentals.csv`

No runtime tests were run because this task did not change code, tests, runtime logic, source CSV values, generated outputs, or local ignored raw fundamentals.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## Recommended Next Step

Launch a separate governed provenance-only fundamentals source-data task for the selected 15 tickers if the objective is to validate local ignored raw fundamentals source/provenance flow.

Launch a stricter numerical fundamentals source lookup task only after the source method, metric definitions, periods, units, freshness rules, and validation rules are explicitly governed.
