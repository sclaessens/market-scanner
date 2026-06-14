# SEC Derived Formula Policy

Status: ACTIVE POLICY SPECIFICATION
Backlog driver: BL-0015
Sprint: SEC-6B — Derived Formula Policy Finalization

## 1. Purpose

SEC-6B finalizes the policy direction for derived SEC fundamentals fields before runtime implementation.

The focus fields are:

- `total_debt`
- `free_cash_flow`

SEC-6B does not implement formulas. It defines policy boundaries, evidence requirements, component handling expectations, and handoff criteria for a later implementation sprint.

## 2. Scope Boundary

This sprint is documentation-only.

It does not include:

- code changes;
- tests;
- SEC calls;
- SEC downloads;
- generated data;
- pipeline integration;
- SEC-to-fundamentals runtime changes;
- metrics/quality/analysis runtime changes;
- Decision Engine or Reporting changes.

## 3. Source Inputs

SEC-6B uses these source documents:

- SEC XBRL mapping investigation: `docs/active/source_data/sec_xbrl_mapping_investigation.md`
- SEC fundamental analysis rationalization: `docs/active/source_data/sec_fundamental_analysis_rationalization.md`
- SEC source architecture: `docs/active/source_data/sec_edgar_source_architecture.md`
- SEC-6A implementation note: `docs/sprints/sec_6a_direct_sec_fundamentals_transform.md`
- current fundamentals platform contract: `docs/active/contracts/fundamentals_platform_contract.md`
- current fundamental calculations technical spec: `docs/active/contracts/fundamental_calculations_technical_spec.md`

## 4. Derived Field Governance Principles

Derived values are allowed only from traceable source-supported components.

Derived values must:

- retain evidence for every component;
- keep missing components missing or review-required;
- never treat missing components as zero;
- never mix annual and quarterly facts;
- never mix different units;
- never mix amended and non-amended facts silently;
- never create ranking, scoring, eligibility, tradeability, urgency, conviction, buy/sell, final-action, allocation, or hidden filtering semantics;
- remain upstream descriptive source-data enrichment only.

## 5. Approved Investigation Policy For total_debt

`total_debt` is a derived field, not a simple direct single-tag field.

It may be derived only from non-overlapping source-supported debt components. Current and noncurrent components may be summed only when tags are semantically compatible.

Lease-inclusive and lease-exclusive debt tags must not be mixed unless each component is explicitly classified and the resulting formula prevents double-counting. Short-term borrowings may be included only when clearly debt-like and non-overlapping. Finance lease liabilities require explicit classification before inclusion.

Missing components must not be inferred. If compatible components are insufficient, `total_debt` must remain missing or review-required. If component overlap is possible, the derived value must be blocked or review-required.

Candidate component families from SEC-4C:

```text
us-gaap:DebtCurrent
us-gaap:LongTermDebtCurrent
us-gaap:LongTermDebtNoncurrent
us-gaap:LongTermDebtAndFinanceLeaseObligationsCurrent
us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent
us-gaap:ShortTermBorrowings
us-gaap:FinanceLeaseLiabilityCurrent
us-gaap:FinanceLeaseLiabilityNoncurrent
```

Component family classifications:

| candidate_component | component_classification | policy_direction |
|---|---|---|
| us-gaap:DebtCurrent | PRIMARY_DEBT_COMPONENT | May be considered a current debt component when source-supported and non-overlapping. |
| us-gaap:LongTermDebtCurrent | PRIMARY_DEBT_COMPONENT | May be considered the current portion of long-term debt when not duplicated by broader current debt tags. |
| us-gaap:LongTermDebtNoncurrent | PRIMARY_DEBT_COMPONENT | May be considered a noncurrent debt component when source-supported. |
| us-gaap:LongTermDebtAndFinanceLeaseObligationsCurrent | ALTERNATE_DEBT_COMPONENT | May be considered only as a lease-inclusive alternate to compatible current debt components. |
| us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent | ALTERNATE_DEBT_COMPONENT | May be considered only as a lease-inclusive alternate to compatible noncurrent debt components. |
| us-gaap:ShortTermBorrowings | BORROWING_COMPONENT_REVIEW_REQUIRED | Requires review before inclusion because overlap with current debt concepts is possible. |
| us-gaap:FinanceLeaseLiabilityCurrent | LEASE_COMPONENT_REVIEW_REQUIRED | Requires explicit lease treatment classification before inclusion. |
| us-gaap:FinanceLeaseLiabilityNoncurrent | LEASE_COMPONENT_REVIEW_REQUIRED | Requires explicit lease treatment classification before inclusion. |

Runtime formula implementation is not approved by SEC-6B. Any later implementation must first define deterministic component selection and overlap-blocking rules.

## 6. Approved Investigation Policy For free_cash_flow

`free_cash_flow` is a derived field, not a direct SEC XBRL field.

It may be derived only as operating cash flow minus capital expenditures after sign conventions are approved. Operating cash flow must come from source-supported cash-flow facts. Capital expenditure must come from source-supported investment or capex facts.

Capex sign conventions must be normalized explicitly. Missing capex must not be treated as zero. Missing operating cash flow must block the derived value.

If sign convention is ambiguous, the derived value must be review-required or blocked. Formula implementation belongs to a later implementation sprint.

Candidate component families from SEC-4C:

```text
us-gaap:NetCashProvidedByUsedInOperatingActivities
us-gaap:PaymentsToAcquirePropertyPlantAndEquipment
```

Component family classifications:

| candidate_component | component_classification | policy_direction |
|---|---|---|
| us-gaap:NetCashProvidedByUsedInOperatingActivities | OPERATING_CASH_FLOW_COMPONENT | May be considered the operating cash flow component when source-supported and period-compatible. |
| us-gaap:PaymentsToAcquirePropertyPlantAndEquipment | CAPEX_COMPONENT | May be considered the capital expenditure component only after sign handling is explicit. |
| us-gaap:PaymentsToAcquirePropertyPlantAndEquipment | CAPEX_SIGN_REVIEW_REQUIRED | Requires review because SEC facts may represent cash outflows and formula sign normalization must be deterministic. |

Runtime formula implementation is not approved by SEC-6B. Any later implementation must first define deterministic sign normalization and component-blocking rules.

## 7. Formula Readiness Decision

| derived_field | readiness_status | approved_policy_direction | implementation_allowed_next | blocking_conditions |
|---|---|---|---|---|
| total_debt | POLICY_READY_IMPLEMENTATION_BLOCKED | Treat as a derived field from non-overlapping, source-supported debt components only. | No runtime implementation in SEC-6B; a later approved sprint may implement fixture-only support after deterministic overlap rules are specified. | Missing components, possible overlap, mixed lease-inclusive and lease-exclusive tags, unit mismatch, period mismatch, amended fact conflict, or unsupported extension tags. |
| free_cash_flow | POLICY_READY_IMPLEMENTATION_BLOCKED | Treat as operating cash flow minus capital expenditures only after source support and sign normalization are explicit. | No runtime implementation in SEC-6B; a later approved sprint may implement fixture-only support after deterministic capex sign rules are specified. | Missing operating cash flow, missing capex, ambiguous capex sign, unit mismatch, period mismatch, amended fact conflict, or unsupported extension tags. |

## 8. Evidence Requirements For Derived Values

Future implementation must retain:

- source tag for each component;
- unit for each component;
- fiscal year;
- fiscal period;
- period end date;
- report/filing date where available;
- source reference;
- source freshness date;
- extraction date;
- derivation formula version;
- review notes;
- missing component notes.

## 9. Missing, Ambiguous, and Conflicting Component Policy

Required behavior:

- missing required component -> derived value missing/review-required;
- ambiguous component overlap -> block or review-required;
- unit mismatch -> block or review-required;
- period mismatch -> block or review-required;
- amended fact conflict -> review-required unless deterministic rule is approved;
- company extension tag -> review-required;
- conflicting values -> no silent winner without deterministic rule.

## 10. SEC-6C Handoff

If this document is accepted, SEC-6C should:

- keep runtime implementation blocked unless explicit approval is granted for fixture-only derived formula support;
- add tests for `total_debt` and `free_cash_flow` component handling if implementation is approved;
- keep no live SEC calls;
- keep no generated operational data commits;
- keep no pipeline integration unless separately approved;
- keep `total_debt` and `free_cash_flow` review-required where components are insufficient;
- preserve evidence for every component in any future fixture implementation.

## 11. SEC-7 / Real Data Review Handoff

Real data review should happen only after:

- formula policy is accepted;
- implementation exists with fixtures;
- generated output policy is confirmed;
- no live-data output is committed;
- review tickers are selected explicitly;
- results are reviewed as source-data quality, not investment advice or allocation.

## 12. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0015 covers fundamentals source-data and quality classification. BL-0017 covers future governed automated ingestion strategy. The SEC sprint sequence already covers immediate work.

## 13. No-Runtime-Change Confirmation

- no scripts changed;
- no tests changed;
- no data changed;
- no reports changed;
- no CSV files changed;
- no generated files changed;
- no workflow files changed;
- no runtime behavior changed;
- no SEC/network calls performed;
- no SEC data downloaded;
- no scraping performed;
- no Decision Engine behavior changed;
- no Reporting behavior changed;
- no Telegram behavior changed;
- no portfolio behavior changed.
