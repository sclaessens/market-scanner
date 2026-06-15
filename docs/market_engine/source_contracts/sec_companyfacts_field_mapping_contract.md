# ME10 — SEC CompanyFacts Field Mapping and Source Coverage Contract

Owner role: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Status: ME10 CONTRACT

## Purpose

This document defines the approved SEC CompanyFacts field mapping and source coverage contract for the first Market Engine fundamental source-intake layer.

The contract defines which SEC CompanyFacts concepts may be used for the first approved source coverage fields, how aliases are prioritized, how missing data must be handled, and what remains forbidden before any analysis layer is built.

ME10 is a source-data contract sprint. It does not approve financial analysis, scoring, ranking, recommendations, Decision Engine behavior, reporting, Telegram delivery, portfolio mutation, or watchlist mutation.

## Scope

This contract covers the first SEC CompanyFacts source fields proven by the ME09 bounded coverage review:

- `revenue`
- `net_income`
- `operating_cash_flow`
- `capital_expenditures`

The ME09 bounded coverage review tested:

- `NVDA`
- `AMD`
- `META`
- `COST`
- `AAPL`
- `MSFT`
- `GOOGL`
- `AMZN`
- `TSLA`
- `AVGO`

ME09 result:

- `AVAILABLE=10`
- `missing_fields=none`
- `provider_errors=0`
- `provider_error_categories=none`

This supports moving from bounded SEC smoke evidence to an approved bounded SEC field mapping contract.

## Strategic Decision

SEC CompanyFacts is approved for:

```text
APPROVED_FOR_SEC_FIELD_MAPPING_IMPLEMENTATION
```

SEC CompanyFacts is not yet approved for:

- production all-ticker intake;
- financial analysis;
- scoring;
- ranking;
- BUY / SELL / HOLD;
- recommendation output;
- Decision Engine use;
- reporting;
- Telegram delivery;
- portfolio mutation;
- watchlist mutation.

## Approved Source

Provider:

```text
SEC CompanyFacts
```

Provider namespace:

```text
us-gaap
```

Approved source endpoint pattern:

```text
https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
```

CIK handling:

- CIK must be represented as a 10-digit string.
- Leading zeroes must be preserved.
- Missing CIK mapping must not become a generic provider failure if a more precise status is possible.
- Missing CIK should be classified as `UNSUPPORTED` or `INVALID_TICKER`, depending on ticker validity and mapping evidence.

## Approved Core Fields

The first approved Market Engine SEC field contract contains four canonical fields:

```text
revenue
net_income
operating_cash_flow
capital_expenditures
```

These are source-intake fields, not analysis metrics.

They may be used to determine whether a ticker has sufficient SEC source coverage for a future fundamental context.

They may not yet be used to emit analysis conclusions.

## Field Mapping Contract

### 1. `revenue`

Canonical Market Engine field:

```text
revenue
```

Primary approved SEC tag:

```text
Revenues
```

Approved fallback aliases, in priority order:

```text
RevenueFromContractWithCustomerExcludingAssessedTax
SalesRevenueNet
SalesRevenueGoodsNet
SalesRevenueServicesNet
```

Mapping rule:

Use the first available approved tag according to priority order for the selected period.

Do not sum fallback aliases unless a later sprint explicitly approves a segment-combination rule.

Do not combine goods and services revenue automatically.

Do not treat operating income, gross profit, net income, or comprehensive income as revenue.

Decision:

```text
revenue = approved if one approved revenue tag is available for the selected period
```

Missing behavior:

If no approved revenue tag is available for the selected period, set:

```text
revenue = missing
```

Do not convert missing revenue to `0`.

### 2. `net_income`

Canonical Market Engine field:

```text
net_income
```

Primary approved SEC tag:

```text
NetIncomeLoss
```

Conditional fallback alias:

```text
ProfitLoss
```

Mapping rule:

Use `NetIncomeLoss` as the preferred tag.

Use `ProfitLoss` only when `NetIncomeLoss` is unavailable and the fact is clearly an entity-level profit/loss measure for the selected reporting period.

Do not use the following as primary `net_income` without a later explicit approval:

```text
NetIncomeLossAvailableToCommonStockholdersBasic
NetIncomeLossAvailableToCommonStockholdersDiluted
IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest
IncomeLossFromContinuingOperations
ComprehensiveIncomeNetOfTax
```

Reason:

These may represent common-shareholder income, continuing operations, pre-tax income, or comprehensive income rather than full net income.

Decision:

```text
net_income = approved if NetIncomeLoss or validated ProfitLoss is available for the selected period
```

Missing behavior:

If no approved net income tag is available for the selected period, set:

```text
net_income = missing
```

Do not convert missing net income to `0`.

### 3. `operating_cash_flow`

Canonical Market Engine field:

```text
operating_cash_flow
```

Primary approved SEC tag:

```text
NetCashProvidedByUsedInOperatingActivities
```

Conditional fallback alias:

```text
NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
```

Mapping rule:

Use `NetCashProvidedByUsedInOperatingActivities` as the preferred tag.

Use `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations` only when the primary tag is unavailable and the limitation is explicitly preserved in source metadata.

Do not use investing cash flow, financing cash flow, net income, EBITDA, operating income, or free cash flow as operating cash flow.

Decision:

```text
operating_cash_flow = approved if the primary tag or approved conditional fallback is available for the selected period
```

Missing behavior:

If no approved operating cash flow tag is available for the selected period, set:

```text
operating_cash_flow = missing
```

Do not convert missing operating cash flow to `0`.

### 4. `capital_expenditures`

Canonical Market Engine field:

```text
capital_expenditures
```

Primary approved SEC tag:

```text
PaymentsToAcquirePropertyPlantAndEquipment
```

Approved fallback alias:

```text
PaymentsToAcquireProductiveAssets
```

Conditional future-review aliases:

```text
PaymentsToAcquirePropertyPlantAndEquipmentAndOtherProductiveAssets
PaymentsToAcquireProductiveAssetsAndBusinesses
```

Mapping rule:

Use `PaymentsToAcquirePropertyPlantAndEquipment` as the preferred tag.

Use `PaymentsToAcquireProductiveAssets` only if the primary tag is unavailable.

Do not use acquisition-related tags as capital expenditures unless a later source contract explicitly approves that interpretation.

Do not use depreciation, amortization, investing cash flow, or cash used in investing activities as capital expenditures.

Decision:

```text
capital_expenditures = approved if primary or approved fallback capex tag is available for the selected period
```

Sign policy:

SEC facts must preserve raw reported value.

For future normalized financial context, Market Engine may define:

```text
capital_expenditures_outflow
```

as a positive cash outflow value.

ME10 does not approve free cash flow calculation.

Missing behavior:

If no approved capital expenditure tag is available for the selected period, set:

```text
capital_expenditures = missing
```

Do not convert missing capital expenditures to `0`.

## Period Selection Contract

ME10 approves the following period-selection rules for source coverage only.

Preferred period:

```text
latest annual fiscal year
```

Preferred form types:

```text
10-K
10-K/A
```

Quarterly forms such as `10-Q` are not approved for the first annual field mapping contract.

Required period metadata to preserve:

- `fy`
- `fp`
- `form`
- `filed`
- `start`
- `end`
- `accn`
- `frame` if available
- SEC tag used
- unit
- raw value

Preferred unit:

```text
USD
```

Do not mix units for the same canonical field.

Do not silently select a non-USD fact for these first four fields.

If multiple facts exist for the same canonical field and period, selection must be deterministic and metadata must preserve the selected SEC tag and accession.

## Source Readiness Contract

A ticker may receive `AVAILABLE` for the ME10 SEC field contract only when all four canonical fields are present from approved tags for the selected period:

```text
revenue
net_income
operating_cash_flow
capital_expenditures
```

A ticker must receive `PARTIAL` when the SEC provider responds successfully but one or more required canonical fields are missing.

A ticker must receive `MISSING` when SEC CompanyFacts is available but no approved relevant facts are found for the selected period.

A ticker must receive `UNSUPPORTED` when the ticker is outside the supported SEC CompanyFacts coverage model or no supported CIK mapping exists.

A ticker must receive `INVALID_TICKER` when the ticker input is malformed or invalid.

A ticker must receive `PROVIDER_ERROR` only for controlled provider/network/HTTP/JSON failures, not for normal missing facts.

## Missing Data Policy

Missing data must remain missing.

Forbidden behavior:

```text
missing -> 0
missing -> false
missing -> estimated value
missing -> derived value
missing -> previous period fallback
```

A missing required field must block full source readiness for the ME10 contract.

A partial ticker remains useful as source coverage evidence but is not approved for fundamental analysis.

## Raw Evidence and Provenance Contract

For each selected canonical field, Market Engine must preserve:

- canonical field name;
- SEC tag used;
- provider name;
- taxonomy namespace;
- unit;
- raw value;
- normalized value if created;
- fiscal year;
- fiscal period;
- filing form;
- filing date;
- period start date;
- period end date;
- accession number;
- source endpoint or safe source reference;
- selection reason;
- fallback alias used, if any.

The system must be able to explain why a tag was selected and why other aliases were not selected.

## Alias Priority Rule

Alias priority is deterministic.

For each canonical field:

1. Try the primary approved tag.
2. If missing, try approved fallback aliases in order.
3. If still missing, mark the field missing.
4. Do not combine aliases.
5. Do not infer the value from another financial statement line.
6. Do not derive a metric to fill the source field.

## Ticker-to-CIK Ownership Decision

Ticker-to-CIK mapping remains source-intake infrastructure, not financial analysis.

ME10 approves continued bounded smoke use of the existing in-code mapping for the proven ticker sample and adjacent contract implementation tests.

ME10 does not approve a production ticker master, automatic ticker universe expansion, or broad all-ticker SEC ingestion.

A later sprint must decide whether ticker-to-CIK ownership belongs in:

- a manually approved Market Engine source input file;
- a bounded SEC submissions/company-tickers lookup workflow;
- an external provider-backed mapping;
- or a hybrid Data Steward-controlled mapping.

Missing CIK mapping must remain explicit and must not be silently filled by assumptions.

## Artifact Retention Policy

ME09 generated smoke artifacts under:

```text
data/market_engine/smokes/source_intake/sec_companyfacts/20260615T103333Z/
```

Those artifacts were intentionally not committed.

ME10 confirms:

- smoke artifacts are evidence only;
- smoke artifacts are not source truth;
- smoke artifacts are not production reports;
- smoke artifacts should remain uncommitted unless a later sprint explicitly approves retention;
- documentation and audit records are the committed evidence for ME09 and ME10.

## Not Approved in ME10

ME10 does not approve:

- free cash flow calculation;
- growth rates;
- margins;
- profitability scoring;
- valuation metrics;
- quality scores;
- ranking;
- BUY / SELL / HOLD;
- recommendation behavior;
- position sizing;
- allocation;
- urgency;
- conviction;
- tradeability;
- execution advice;
- portfolio mutation;
- watchlist mutation;
- Telegram output;
- production reporting.

## Future Fundamental Context Readiness

ME10 prepares but does not implement the first fundamental context.

A future sprint may build a fundamental source context when:

- SEC field mapping is implemented as code;
- tests prove alias priority behavior;
- tests prove missing values remain missing;
- tests prove metadata/provenance is preserved;
- bounded coverage evidence supports field availability;
- source-data owner approves the field contract for source context construction.

## QA/Test Requirements

Future implementation tests must prove:

1. `Revenues` maps to `revenue`.
2. `RevenueFromContractWithCustomerExcludingAssessedTax` maps to `revenue` only when primary revenue is missing.
3. Revenue aliases are not summed automatically.
4. `NetIncomeLoss` maps to `net_income`.
5. `ProfitLoss` is conditional fallback only.
6. `NetIncomeLossAvailableToCommonStockholdersBasic` is not accepted as primary `net_income`.
7. `NetCashProvidedByUsedInOperatingActivities` maps to `operating_cash_flow`.
8. `PaymentsToAcquirePropertyPlantAndEquipment` maps to `capital_expenditures`.
9. `PaymentsToAcquireProductiveAssets` maps to `capital_expenditures` only when primary capex is missing.
10. Missing fields remain missing and are not converted to zero.
11. `AVAILABLE` requires all four required fields.
12. Missing one required field produces `PARTIAL`.
13. Provider/network failures produce controlled `PROVIDER_ERROR`.
14. Unsupported tickers do not become generic provider errors.
15. No forbidden authority fields are emitted.

Automated tests must not call live SEC endpoints.

## Data Isolation Contract

Any future SEC coverage artifacts must be written only under:

```text
data/market_engine/smokes/source_intake/sec_companyfacts/<run_id>/
```

Old paths must not be used:

```text
data/processed/
data/generated/
data/logs/
data/normalized/
reports/
data/portfolio/
data/watchlist/
```

Smoke artifacts are not source truth.

Smoke artifacts are source coverage evidence only.

## Source-Data Owner Decision After ME10

Decision:

```text
APPROVED_FOR_SEC_FIELD_MAPPING_IMPLEMENTATION
```

Meaning:

SEC CompanyFacts may now be implemented as a formal field mapping contract in Market Engine code and tests.

Still not approved:

```text
APPROVED_FOR_ANALYSIS
```

The next sprint must implement the contract and tests, not build analysis conclusions.

## Recommended Next Sprint

Recommended next sprint:

```text
ME11 — Implement SEC CompanyFacts field mapping contract and tests
```

ME11 should:

- add field mapping constants or configuration under `src/market_engine/source_intake/`;
- implement deterministic alias priority;
- preserve provenance metadata;
- add tests for all approved mappings and forbidden substitutions;
- keep all tests fake/mocked;
- avoid analysis, scoring, ranking, recommendations, reporting, Telegram, portfolio, and watchlist behavior.
