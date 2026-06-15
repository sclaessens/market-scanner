# ME11 — SEC Field Mapping And Fundamental Source Context Audit

Owner role: Technical Architect / Financial Analyst / Data Steward / QA / Test Lead / Governance Auditor

Status: COMPLETED BY ME11

## Purpose

ME11 implements the approved ME10 SEC CompanyFacts field mapping contract and creates the first Market Engine fundamental source context.

ME11 remains source/fundamental-context work only. It does not implement financial analysis, scoring, ranking, recommendations, Decision Engine behavior, reporting, Telegram delivery, portfolio mutation, or watchlist mutation.

## Files Created

- `src/market_engine/source_intake/sec_companyfacts_fields.py`
- `src/market_engine/fundamentals/__init__.py`
- `src/market_engine/fundamentals/source_context.py`
- `tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py`
- `tests/market_engine/fundamentals/test_source_context.py`
- `docs/market_engine/audits/me11_sec_field_mapping_fundamental_source_context_audit.md`
- `docs/market_engine/architecture/fundamental_source_context.md`

## Files Updated

- `src/market_engine/source_intake/sec_companyfacts_provider.py`
- `tests/market_engine/source_intake/test_sec_companyfacts_provider.py`
- `docs/market_engine/source_contracts/sec_companyfacts_field_mapping_contract.md`
- `docs/market_engine/backlog/market_engine_backlog.md`

## ME10 Contract Implemented

ME11 implements deterministic mapping for the four approved canonical SEC CompanyFacts source fields:

- `revenue`
- `net_income`
- `operating_cash_flow`
- `capital_expenditures`

Alias priority follows the ME10 contract. The mapper selects one approved SEC tag per canonical field and does not sum, combine, infer, or derive source values.

## Alias Priority Implemented

Implemented alias priority:

- `revenue`: `Revenues`, then `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`, `SalesRevenueGoodsNet`, `SalesRevenueServicesNet`.
- `net_income`: `NetIncomeLoss`, then `ProfitLoss`.
- `operating_cash_flow`: `NetCashProvidedByUsedInOperatingActivities`, then `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations`.
- `capital_expenditures`: `PaymentsToAcquirePropertyPlantAndEquipment`, then `PaymentsToAcquireProductiveAssets`.

Unapproved substitutions remain unselected.

## Provenance Preserved

For each mapped field, ME11 preserves:

- canonical field name;
- selected SEC tag;
- provider name;
- taxonomy namespace;
- unit;
- raw value;
- fiscal year;
- fiscal period;
- filing form;
- filing date;
- period start date;
- period end date;
- accession number;
- frame when available;
- selection reason;
- fallback alias when used.

## Missing-Data Behavior

Missing source values remain missing.

ME11 does not convert missing values to zero, false, estimated values, derived values, or previous-period fallbacks.

## Source Context Behavior

ME11 adds a source-only fundamental context that exposes:

- ticker;
- provider;
- source readiness;
- canonical source fields;
- missing canonical fields;
- provenance;
- period metadata;
- controlled provider error category and message when available.

The context consumes already-fetched provider evidence and does not run live provider calls.

## Readiness Behavior

Readiness behavior:

- `AVAILABLE`: all four canonical fields are present.
- `PARTIAL`: one or more canonical fields are present and one or more are missing.
- `MISSING`: no approved canonical fields are present.
- `UNSUPPORTED`: unsupported ticker result remains unsupported.
- `INVALID_TICKER`: invalid ticker result remains invalid.
- `PROVIDER_ERROR`: controlled provider failure result remains provider error.

## Tests Added

ME11 adds tests for:

- approved SEC tag mapping;
- alias priority;
- forbidden substitutions;
- no alias summing or combining;
- missing value preservation;
- selected tag preservation;
- unit, filing, and period metadata preservation;
- source context readiness behavior;
- source context provenance;
- source context boundary exclusions;
- no legacy runtime imports.

## Tests Run

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_intake tests/market_engine/fundamentals -q
```

Result:

```text
68 passed
```

## Boundary Confirmations

Confirmed:

- No live provider calls were used in automated tests.
- No analysis was implemented.
- No free cash flow, growth, margins, valuation metrics, scoring, ranking, recommendation, BUY / SELL / HOLD, allocation, conviction, urgency, tradeability, position sizing, or execution behavior was added.
- No Decision Engine, reporting, Telegram, portfolio, or watchlist behavior was added.
- No `src/market_scanner/` files were modified.
- No `scripts/` files were modified.
- `src/market_engine/` and `tests/market_engine/` do not import `market_scanner` or `scripts`.
- No old data or report paths were changed.

## Known Limitations

ME11 does not approve financial analysis.

ME11 does not calculate derived metrics.

ME11 does not broaden SEC ticker coverage or approve a production ticker-to-CIK owner model.

ME11 does not commit smoke artifacts.

## Recommended Next Sprint

Recommended next sprint:

```text
ME12 — Build first non-decision fundamental analysis pass
```

ME12 may begin source-only financial observations from the approved context, but it must not emit recommendations, scores, rankings, BUY / SELL / HOLD, portfolio changes, Telegram output, or Decision Engine behavior.
