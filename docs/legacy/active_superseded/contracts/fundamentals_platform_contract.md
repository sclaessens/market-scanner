# Fundamentals Platform Contract

Status: ACTIVE TARGET CONTRACT

## Status and authority

This document is the active target contract for the future fundamentals redesign.

Older sprint-level fundamentals previews, pilots, validations, provenance notes, source lookup documents, and source-data expansion notes become historical evidence once this contract is approved. They may explain why decisions were made, but they do not act as active implementation instructions unless this contract explicitly references them.

This document does not implement code, change runtime behavior, modify source data, authorize provider/API usage, update generated artifacts, or approve pipeline execution.

## Design principle

The simplified fundamentals platform must follow this target chain:

```text
raw fundamentals history
-> calculated fundamental metrics
-> fundamental quality classification
-> fundamental analysis classification
-> downstream portfolio/decision layers
```

The platform must first collect raw historical data, then calculate metrics, then classify quality, then classify business and fundamental characteristics. Downstream layers consume descriptive outputs only.

Do not continue the per-3-ticker metric pilot workflow as the main operating model. The project should first build enough raw historical data coverage for the relevant ticker universe, then build small deterministic calculation and analysis algorithms.

## Documentation-only confirmation

This contract confirms that the fundamentals simplification work is documentation-only until a later approved implementation sprint.

No code changes are authorized. No data changes are authorized. This sprint exists to reduce complexity and consolidate active doctrine.

## Raw fundamentals history layer

Target artifact:

```text
data/raw/fundamentals_history.csv
```

Purpose:

- local ignored source artifact unless repository policy changes;
- source of truth for raw financial statement history;
- store multiple fiscal years per ticker;
- capture period and source metadata separately from calculated metrics.

Required columns:

| Column | Meaning |
|---|---|
| `ticker` | Security ticker. |
| `fiscal_year` | Fiscal year of the reported period. |
| `fiscal_period` | Annual, quarterly, trailing, or other approved period label. |
| `period_end_date` | Fiscal period end date. |
| `report_date` | Company report, filing, or publication date. |
| `currency` | Reporting currency. |
| `revenue` | Raw reported revenue. |
| `gross_profit` | Raw reported gross profit. |
| `operating_income` | Raw reported operating income. |
| `net_income` | Raw reported net income. |
| `diluted_eps` | Raw reported diluted earnings per share. |
| `total_debt` | Raw reported total debt. |
| `total_equity` | Raw reported total equity. |
| `free_cash_flow` | Raw reported or source-supported free cash flow. |
| `source_name` | Source name. |
| `source_reference` | Specific source reference, URL, filing name, or evidence pointer. |
| `source_freshness_date` | Date the source was checked. |
| `extraction_date` | Local extraction date. |
| `notes` | Data-steward notes, caveats, or review comments. |

Rules:

- raw reported data only;
- no ratios unless directly reported and explicitly approved;
- no quality states;
- no analysis states;
- no scoring;
- no buy/sell fields;
- no allocation semantics;
- no Decision Engine semantics;
- no ranking, tradeability, urgency, conviction, eligibility, or hidden filtering semantics.

## Date semantics

The overloaded `as_of_date` concept is replaced by explicit date semantics:

| Date field | Meaning |
|---|---|
| `period_end_date` | Fiscal period end. |
| `report_date` | Company report or filing date. |
| `source_freshness_date` | Date the source was checked. |
| `extraction_date` | Local extraction date. |

This separation prevents source freshness, fiscal period identity, local extraction timing, and opportunity-date validation from being mixed into one ambiguous field.

## Data collection direction

Raw history must be collected before analysis.

Collection should target all relevant scanner and metadata tickers, not only 3-ticker pilots. The initial coverage target is at least 3 completed fiscal years per ticker. Prefer 5 fiscal years when available.

Every row must have source reference and period metadata. Values without source evidence remain review-required and must not be inferred.

## Fundamental metrics layer

Target artifact:

```text
data/processed/fundamental_metrics.csv
```

Purpose:

- calculate metrics from raw history;
- keep formulas deterministic;
- make calculations testable and reproducible;
- isolate metric formulas from raw source data and analysis interpretation.

Rules:

- calculations only;
- no scoring;
- no ranking;
- no allocation;
- no tradeability;
- no buy/sell semantics;
- no hidden filtering;
- no Decision Engine authority.

Detailed formula definitions belong in `docs/active/contracts/fundamental_calculations_technical_spec.md`.

## Fundamental quality layer

Target artifact:

```text
data/processed/fundamental_quality.csv
```

Purpose:

- classify data completeness and reliability;
- separate source/data readiness from business interpretation;
- identify whether raw history and metric inputs are usable;
- preserve descriptive classification only.

Suggested states:

- `SOURCE_MISSING`
- `RAW_HISTORY_INCOMPLETE`
- `RAW_HISTORY_PARTIAL`
- `RAW_HISTORY_READY`
- `METRICS_PARTIAL`
- `METRICS_READY`
- `REVIEW_REQUIRED`

Rules:

- no business interpretation;
- no buy/sell advice;
- no allocation authority;
- no ranking or scoring authority;
- no eligibility, conviction, urgency, tradeability, or filtering semantics.

## Fundamental analysis layer

Target artifact:

```text
data/processed/fundamental_analysis.csv
```

Purpose:

Classify business and fundamental characteristics descriptively from raw history, calculated metrics, and quality classifications.

Suggested outputs:

- `growth_state`
- `margin_state`
- `profitability_state`
- `debt_state`
- `cashflow_state`
- `consistency_state`
- `trend_state`

Rules:

- descriptive classification only;
- no allocation;
- no ranking;
- no scoring;
- no tradeability;
- no urgency;
- no conviction;
- no eligibility;
- no buy/sell;
- no hidden filtering.

## Downstream contract

Portfolio Intelligence and the Decision Engine may consume descriptive fundamentals outputs.

Portfolio Intelligence remains descriptive. The Decision Engine remains the only allocation, execution, arbitration, and final-action authority.

No fundamentals layer may pre-empt, duplicate, loosen, or bypass Decision Engine authority.

## Current MVP compatibility

The existing `data/raw/fundamentals.csv` and current Fundamental Layer behavior may remain temporary MVP or compatibility surfaces until a future approved implementation sprint replaces or wraps them.

A later implementation specification must define migration, compatibility, validation, and deprecation behavior explicitly.

## Out of scope

This contract does not authorize:

- code changes;
- tests;
- CSV edits;
- raw fundamentals changes;
- generated file changes;
- provider/API integration;
- scraping;
- credentials or secrets;
- pipeline runs;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- runtime implementation.