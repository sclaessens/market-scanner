# ME10 — SEC CompanyFacts Field Mapping Contract Audit

Owner role: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Status: COMPLETED BY ME10

## Purpose

This audit records the ME10 documentation/governance contract work that converts the ME09 bounded SEC CompanyFacts coverage evidence into an approved SEC field mapping and source coverage contract.

ME10 does not implement analysis. It defines the approved source contract that a later implementation sprint may encode and test.

## Files Created

- `docs/market_engine/source_contracts/sec_companyfacts_field_mapping_contract.md`
- `docs/market_engine/audits/me10_sec_companyfacts_field_mapping_contract_audit.md`

## Files Updated

- None.

The dedicated contract document is the authoritative ME10 deliverable. Existing source-intake and backlog documents can be updated in a follow-up housekeeping commit if desired, but ME10's core governance decision is captured here and in the source contract.

## ME09 Evidence Used

ME09 ran a bounded SEC CompanyFacts coverage review for:

```text
NVDA AMD META COST AAPL MSFT GOOGL AMZN TSLA AVGO
```

Required fields:

```text
revenue
net_income
operating_cash_flow
capital_expenditures
```

ME09 result:

```text
AVAILABLE=10
missing_fields=none
provider_errors=0
provider_error_categories=none
```

ME09 source-data owner decision:

```text
APPROVED_FOR_BOUNDED_SEC_FIELD_MAPPING_CONTRACT
```

## Approved Fields

ME10 approves the first SEC CompanyFacts field mapping contract for these canonical source-intake fields:

- `revenue`
- `net_income`
- `operating_cash_flow`
- `capital_expenditures`

These are approved as source-intake fields only. They are not analysis metrics in ME10.

## Approved Alias Summary

### `revenue`

Primary:

- `Revenues`

Approved fallbacks, in priority order:

- `RevenueFromContractWithCustomerExcludingAssessedTax`
- `SalesRevenueNet`
- `SalesRevenueGoodsNet`
- `SalesRevenueServicesNet`

### `net_income`

Primary:

- `NetIncomeLoss`

Conditional fallback:

- `ProfitLoss`

Explicitly not approved as primary net income without later approval:

- `NetIncomeLossAvailableToCommonStockholdersBasic`
- `NetIncomeLossAvailableToCommonStockholdersDiluted`
- `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest`
- `IncomeLossFromContinuingOperations`
- `ComprehensiveIncomeNetOfTax`

### `operating_cash_flow`

Primary:

- `NetCashProvidedByUsedInOperatingActivities`

Conditional fallback:

- `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations`

### `capital_expenditures`

Primary:

- `PaymentsToAcquirePropertyPlantAndEquipment`

Approved fallback:

- `PaymentsToAcquireProductiveAssets`

Future-review aliases:

- `PaymentsToAcquirePropertyPlantAndEquipmentAndOtherProductiveAssets`
- `PaymentsToAcquireProductiveAssetsAndBusinesses`

## Source-Data Owner Decision

ME10 decision:

```text
APPROVED_FOR_SEC_FIELD_MAPPING_IMPLEMENTATION
```

Meaning:

SEC CompanyFacts may now be implemented as a formal field mapping contract in Market Engine code and tests.

Still not approved:

```text
APPROVED_FOR_ANALYSIS
```

## Contract Rules Established

ME10 establishes:

- deterministic alias priority;
- latest annual fiscal-year source coverage preference;
- preferred forms `10-K` and `10-K/A`;
- preferred unit `USD`;
- explicit missing-data behavior;
- raw evidence and provenance preservation requirements;
- readiness semantics for `AVAILABLE`, `PARTIAL`, `MISSING`, `UNSUPPORTED`, `INVALID_TICKER`, and `PROVIDER_ERROR`;
- ticker-to-CIK ownership constraints;
- smoke artifact retention policy;
- data isolation rules.

## Missing Data Decision

Missing source values must remain missing.

Forbidden behavior:

```text
missing -> 0
missing -> false
missing -> estimated value
missing -> derived value
missing -> previous period fallback
```

A ticker with one or more missing required fields is not fully source-ready under this contract.

## Data Isolation Decision

Any future SEC coverage artifacts must remain isolated under:

```text
data/market_engine/smokes/source_intake/sec_companyfacts/<run_id>/
```

Old paths must not be written:

```text
data/processed/
data/generated/
data/logs/
data/normalized/
reports/
data/portfolio/
data/watchlist/
```

Smoke artifacts remain source coverage evidence only and are not source truth.

## Boundary Confirmations

ME10 changed documentation only.

Confirmed boundaries:

- No `src/market_scanner/` files were changed.
- No `scripts/` files were changed.
- No `src/market_engine/` code files were changed.
- No tests were changed.
- No data, CSV, or report files were changed.
- No provider calls were run.
- No production writes were introduced.
- No Telegram, reporting, portfolio, watchlist, or Decision Engine behavior changed.
- No BUY / SELL / HOLD, recommendation, allocation, ranking, score, conviction, urgency, tradeability, position sizing, or execution behavior was added.

## Known Limitations

ME10 does not implement the field mapping contract in code.

ME10 does not validate broader ticker coverage.

ME10 does not approve free cash flow, growth rates, margins, valuation metrics, scoring, or recommendations.

ME10 does not resolve long-term ticker-to-CIK source ownership. It confirms that missing CIK mapping must remain explicit and that production ticker mapping requires a later Data Steward decision.

## Readiness Implication

Market Engine is now ready for an implementation sprint that encodes the approved SEC field mapping contract in code and tests.

The next implementation sprint must remain source-intake only and must not enter analysis.

## Recommended Next Sprint

```text
ME11 — Implement SEC CompanyFacts field mapping contract and tests
```

ME11 should:

- implement field mapping constants or configuration;
- implement deterministic alias priority;
- preserve provenance metadata;
- add tests for approved mappings;
- add tests for forbidden substitutions;
- prove missing values remain missing;
- keep automated tests fake/mocked;
- avoid analysis, scoring, ranking, recommendations, reporting, Telegram, portfolio, and watchlist behavior.
