# Portfolio Metadata Source Lookup Preview 1

## Status and Scope

This document is a governed source-data lookup preview and steward classification artifact for the first Portfolio Metadata expansion batch selected under the Automated Source Data Steward protocol.

This is not a coding sprint, runtime-logic change, automated ingestion implementation, or CSV update task.

This document does not modify scripts, tests, source CSV files, generated files, runtime behavior, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, watchlist files, or raw fundamentals files.

No sprint is closed or certified complete by this document.

## Protocol Reference

This preview follows `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`.

The protocol permits a governed preview and steward classification step before any source-data CSV update. This artifact records preview values and steward states only. It does not authorize an automatic write to `data/portfolio/portfolio_metadata.csv`.

## Previous Preview Reference

The selected batch comes from `docs/sprints/portfolio_metadata_expansion_preview_1.md`.

The previous preview selected the next controlled batch from missing A-grade scanner tickers, limited to 15 tickers:

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

## Approved Source Method

The approved source method for this preview is Yahoo Finance.

No paid or restricted APIs were used. No credentials or secrets were created. No provider integration was implemented. No prohibited scraping was performed.

## Current Metadata State

`data/portfolio/portfolio_metadata.csv` currently contains 21 metadata rows.

The required metadata columns are present:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

None of the 15 selected tickers currently have metadata rows in `data/portfolio/portfolio_metadata.csv`.

The existing metadata file uses descriptive asset classes including `Equity` and `REIT`. Under the current protocol and runtime contract, `Equity`, `REIT`, and `ETF` are accepted descriptive asset classes. These values do not imply eligibility, priority, tradeability, urgency, conviction, allocation, ranking, scoring, or Decision Engine bypass.

## Source Lookup Preview Table

| ticker | sector | industry | asset_class | currency | metadata_source | metadata_last_updated | steward_state | notes |
|---|---|---|---|---|---|---|---|---|
| AMAT | Technology | Semiconductor Equipment & Materials | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists AMAT as Semiconductor Equipment & Materials / Technology. |
| ANET | Technology | Computer Hardware | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists ANET as Computer Hardware / Technology. |
| ASML | Technology | Semiconductor Equipment & Materials | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists ASML as Semiconductor Equipment & Materials / Technology. |
| COST | Consumer Defensive | Discount Stores | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists COST as Discount Stores / Consumer Defensive. |
| DELL | Technology | Computer Hardware | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists DELL as Computer Hardware / Technology. |
| ENPH | Technology | Solar | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists ENPH as Solar / Technology. |
| EOG | Energy | Oil & Gas E&P | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists EOG as Oil & Gas E&P / Energy. |
| EQIX | Real Estate | REIT - Specialty | REIT | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists EQIX as REIT - Specialty / Real Estate. |
| EW | Healthcare | Medical Devices | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists EW as Medical Devices / Healthcare. |
| EXPD | Industrials | Integrated Freight & Logistics | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists EXPD as Integrated Freight & Logistics / Industrials. |
| FDX | Industrials | Integrated Freight & Logistics | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists FDX as Integrated Freight & Logistics / Industrials. |
| FTNT | Technology | Software - Infrastructure | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists FTNT as Software - Infrastructure / Technology. |
| HAL | Energy | Oil & Gas Equipment & Services | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists HAL as Oil & Gas Equipment & Services / Energy. |
| HLT | Consumer Cyclical | Lodging | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists HLT as Lodging / Consumer Cyclical. |
| HPE | Technology | Communication Equipment | Equity | USD | Yahoo Finance | 2026-05-23 | APPROVED | Yahoo Finance lists HPE as Communication Equipment / Technology. |

## Steward Classification Table

| ticker | steward_state | reason | eligible_for_csv_update |
|---|---|---|---|
| AMAT | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| ANET | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| ASML | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| COST | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| DELL | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| ENPH | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| EOG | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| EQIX | APPROVED | Selected ticker; Yahoo Finance source values present; REIT asset class is valid for REIT - Specialty; no ambiguity detected. | YES |
| EW | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| EXPD | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| FDX | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| FTNT | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| HAL | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| HLT | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |
| HPE | APPROVED | Selected ticker; Yahoo Finance source values present; descriptive asset class is valid; no ambiguity detected. | YES |

## Approval State Distribution

| steward_state | count |
|---|---:|
| APPROVED | 15 |
| REVIEW_REQUIRED | 0 |
| REJECTED | 0 |

## Rows Eligible For Later CSV Update

The following rows are eligible for a later explicit `data/portfolio/portfolio_metadata.csv` update task:

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

Eligibility in this document means the steward preview found source-supported metadata values suitable for a later governed CSV update. No CSV update is performed or authorized automatically by this document.

## Rows Requiring Human Review

No rows require human review in this preview.

## Rejected Rows

No rows were rejected in this preview.

## Non-Change Statement

No source CSV files were modified.

No generated files were modified.

No runtime behavior was changed.

No Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, or Portfolio Intelligence logic was changed.

No source-data CSV values were written in this task.

## Validation Notes

Validation was documentation-only.

The selected batch was limited to the 15 tickers approved in `docs/sprints/portfolio_metadata_expansion_preview_1.md`.

`data/portfolio/portfolio_metadata.csv` was inspected only to confirm required columns and current row presence. The file was not edited.

The full pipeline was not run. Runtime tests were not run because this task only creates a documentation preview artifact.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0011 — Define and repair authoritative active portfolio source`

## Recommended Next Step

After human review, launch a separate governed source-data CSV update task for the 15 rows classified as `APPROVED`.

That future task should update only `data/portfolio/portfolio_metadata.csv`, preserve the existing schema and column order, avoid provider/API integration, avoid credentials and secrets, avoid generated artifact commits, and preserve all Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, and Portfolio Intelligence governance boundaries.
