# ME-DATA10 — Implement a generic governed primary-source fundamental metric derivation engine and execute a bounded pilot

## Outcome

ME-DATA10 implemented a ticker-independent, deterministic primary-source
fundamental metric derivation engine and executed one bounded real-world pilot.
The engine derived two AAPL FY2026 Q2 metrics through an explicit
checksum-bound approval and the existing DATA07, DATA06, and RUN31 downstream
path:

- `gross_margin = 0.492705785005`;
- `operating_margin = 0.322753273852`.

`debt_to_equity` remained blocked. The approved fact package did not contain a
complete raw-tag-backed set for commercial paper, current term debt,
non-current term debt, and total equity. No missing component was treated as
zero and total liabilities was not used as a substitute.

AAPL remained `partial` and not advice-input-ready. The measured aggregate
baseline remained 6 complete, 39 partial, 907 missing, 6 advice-input-ready, 0
full-advice-ready, and 946 unable-to-advise. No regression occurred.

## Architecture

The implementation uses five explicit boundaries:

1. a versioned primary-source fact package contains canonical facts and raw
   source-tag lineage;
2. a versioned declarative formula catalog defines machine-readable operand
   roles, exact concepts, aggregation, alignment, compatibility, safety, and
   applicability policies;
3. the generic engine validates and derives candidates without approval
   authority;
4. a separate operator decision binds the candidate and all relevant inputs by
   checksum after the runtime emits only a non-authoritative pending candidate;
5. one authoritative approval validator replays the derivation, revalidates the
   DATA09 direct approval, reconstructs the governed package, and only then
   allows DATA07 to parse, snapshot, or execute downstream work.

The runtime performs only two generic operations: a one-numerator ratio and an
explicit component-sum ratio. Formula identity, canonical concepts, selected
fact IDs, applicability, accounting framework, issuer identity, raw source
tags, period, currency, unit, and scale are input data. There is no issuer or
ticker dispatch.

## DATA08 and DATA07 Contract Relationship

ME-DATA10 does not alter or reinterpret DATA08-v1. The
`market-engine-data08-operator-fundamental-metric-input-v1` contract remains a
direct-evidence structural contract and still forbids derived ratios.

DATA07-v1 also remains direct-only. ME-DATA10 adds the explicit
`market-engine-data07-governed-fundamental-metrics-v2` adapter. DATA07 detects
that version before parsing and requires a successful DATA10 derivation
approval. Unknown or missing direct/derived lineage is rejected. Existing
DATA09 direct packages continue through the unchanged v1 approval path.

Derived values are never placed in `raw_source_field` as if the source
published the ratio. Each v2 metric contains `evidence_type: direct` or
`evidence_type: derived`. The normalized DATA07 evidence preserves the same
classification and the complete appropriate lineage object.

## Contract and Schema Versions

- primary facts: `market-engine-data10-primary-source-fact-package-v1`;
- formula catalog: `market-engine-data10-fundamental-metric-formula-catalog-v2`;
- derived metrics: `market-engine-data10-derived-fundamental-metrics-v1`;
- derivation validation: `market-engine-data10-derivation-validation-v1`;
- derivation approval: `market-engine-data10-derivation-approval-decision-v2`;
- approval validation: `market-engine-data10-derivation-approval-validation-v2`;
- DATA07 adapter: `market-engine-data07-governed-fundamental-metrics-v2`;
- compact evidence: `market-engine-data10-compact-derived-metric-pilot-evidence-v1`;
- engine: `market-engine-data10-primary-source-metric-derivation-engine-v2`.

## Formula Catalog

The versioned catalog is
`config/market_engine/data10_fundamental_metric_formula_catalog.json`.

| Formula ID | Version | Canonical expression | Period type |
|---|---:|---|---|
| `gross_margin` | `2.0.0` | `gross_profit / revenue` | duration |
| `operating_margin` | `2.0.0` | `operating_income / revenue` | duration |
| `debt_to_equity` | `2.0.0` | `sum(explicitly_approved_interest_bearing_debt_components) / total_equity` | instant |

Each formula now carries a machine-readable `operand_contract`. Gross margin
requires exactly `gross_profit` as numerator and `revenue` as denominator.
Operating margin requires exactly `operating_income` as numerator and
`revenue` as denominator. Canonical expression text is retained for lineage,
but a correct string cannot mask wrong machine operands and a changed string is
rejected as a lineage mismatch.

The debt formula permits only explicit canonical interest-bearing concepts:
`commercial_paper`, `short_term_borrowings`, `current_term_debt`,
`noncurrent_term_debt`, and `total_interest_bearing_debt`. The request must
declare the complete required component set and matching fact IDs. A reported
`total_interest_bearing_debt` cannot be combined with its components, duplicate
component concepts are rejected, missing components are not zero-filled, and
the engine never substitutes total liabilities.

## Canonical Concepts

The runtime recognizes:

- `revenue`;
- `gross_profit`;
- `operating_income`;
- `commercial_paper`;
- `short_term_borrowings`;
- `current_term_debt`;
- `noncurrent_term_debt`;
- `total_interest_bearing_debt`;
- `total_equity`.

US-GAAP, IFRS, and issuer-specific source tags remain lineage data. Only an
approved canonical mapping permits a source fact to use one of these concepts.

## Generic Runtime Boundary

Production code contains no AAPL, Apple, ASML, issuer allowlist,
issuer-specific XBRL tag, or ticker-specific formula branch. Generic tests use
AAA, BBB, and CCC. The same runtime derives a US-GAAP margin, an IFRS margin,
and the same formula for a second arbitrary ticker without code changes.

Applicability is explicit input evidence. A sector or issuer does not receive
gross-margin treatment automatically. `not_applicable` must carry a reviewed
applicability reference and produces no derived metric.

## Validation and Safety

The engine fails closed on:

- missing, duplicate, or conflicting facts;
- malformed fact or formula packages;
- unsupported accounting framework or canonical concept;
- unknown formula ID, formula version, or operation;
- duration/instant mismatch;
- quarter/year-to-date or period-start mismatch;
- period-end or fiscal-context mismatch;
- currency, unit, or scale mismatch;
- zero or negative revenue/equity;
- negative debt components;
- missing numerator, denominator, or required debt component;
- NaN or Infinity;
- absent applicability review;
- derivation reproduction drift;
- missing, rejected, blocked, or checksum-mismatched approval.

Failed derivation and failed approval tests prove that DATA07 is not invoked.
Consequently no raw snapshot is written and DATA06/RUN31 do not execute. The
derivation and import path performs zero provider and network calls.

## Integrity Review Corrections

The first blocking finding was caused by treating `canonical_expression` as
descriptive lineage while validating only formula identity, metric identity,
operation, and cardinality. Version 2 of the formula catalog now defines exact
numerator and denominator roles and canonical concepts, component aggregation
and overlap rules, period alignment, unit/currency/scale compatibility,
denominator safety, and applicability policy. Runtime enforcement uses stable
codes including `FORMULA_NUMERATOR_CONCEPT_MISMATCH`,
`FORMULA_DENOMINATOR_CONCEPT_MISMATCH`,
`FORMULA_COMPONENT_CONCEPT_NOT_ALLOWED`, `FORMULA_COMPONENT_OVERLAP`,
`FORMULA_OPERAND_ROLE_MISMATCH`, `FORMULA_OPERAND_SET_INVALID`, and
`FORMULA_CANONICAL_EXPRESSION_MISMATCH`.

The second blocking finding was caused by validating bound artifacts and their
file checksums independently. The authoritative v2 approval validator now:

1. validates every bound schema and recomputes every file checksum;
2. revalidates the exact DATA09 package through
   `validate_source_approval_decision` and its original input, report, source
   documents, and approval;
3. validates the fact package and formula catalog and deterministically calls
   `derive_primary_source_metrics` again;
4. compares both the replayed derived package and derivation validation using
   canonical serialization and independently recalculates each calculation
   checksum;
5. reconstructs the DATA07-v2 package from the exact bound direct and derived
   artifacts and compares it with the offered package;
6. reconciles package IDs, approval references, metric identities, ticker,
   instrument, company, fiscal context, evidence type, and checksum sets.

Both `execute_approved_derivation_import` and the ordinary DATA07 operator
import route call this same validator before parsing. Negative tests rechecksum
tampered fact, catalog, direct, derived, governed, validation, and approval
files where applicable and still prove fail-closed reconciliation. They also
cover operand substitution, swapped roles, wrong denominators, total-liability
substitution, debt-total/component overlap, duplicate metrics, identity and
fiscal mismatches, malformed artifacts, and prevention of parser, raw snapshot,
DATA06, and RUN31 execution after failure.

## AAPL Primary Facts

The bounded pilot uses the official SEC Form 10-Q for the quarter ended
2026-03-28. Only the following exact three-month facts were admitted:

| Canonical concept | Raw inline-XBRL tag | Raw value | Unit | Scale | Normalized value |
|---|---|---:|---|---:|---:|
| `revenue` | `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` | 111,184 | USD | 6 | 111,184,000,000 |
| `gross_profit` | `us-gaap:GrossProfit` | 54,781 | USD | 6 | 54,781,000,000 |
| `operating_income` | `us-gaap:OperatingIncomeLoss` | 35,885 | USD | 6 | 35,885,000,000 |

All facts use:

- accounting framework: `us_gaap`;
- duration: 2025-12-28 through 2026-03-28;
- fiscal context: FY2026 Q2;
- currency and unit: USD;
- scale exponent: 6;
- source publication date: 2026-05-01;
- source document checksum:
  `a61f508a797f02384801dadc55b18feef248e28c83ec07281e960dd7d0f4620d`.

The six-month year-to-date values were not selected. Balance-sheet values
visible in the filing were not admitted because the locally checksum-bound
evidence did not provide the complete raw-tag lineage required by this
contract. They are documented as unused rather than assumed.

## Calculations

### Gross margin

```text
54,781,000,000 / 111,184,000,000 = 0.492705785005
```

Calculation checksum:
`02b47e7ea1f21286a8c492bd2363fb2c30cf10d295920c8e37401718d70b934c`.

### Operating margin

```text
35,885,000,000 / 111,184,000,000 = 0.322753273852
```

Calculation checksum:
`c0c0ab90f90546fb1c7a37fb7320490577594a43f0d362c25fcda5950d62cc46`.

### Debt to equity

Status: blocked.

Reason codes:

- `DEBT_COMPONENT_MISSING`;
- `DENOMINATOR_MISSING`;
- `FORMULA_DENOMINATOR_CONCEPT_MISMATCH`;
- `FORMULA_INPUT_CARDINALITY_INVALID`;
- `FORMULA_OPERAND_SET_INVALID`.

No calculation result was emitted.

## Direct and Derived Evidence

The merged package retains DATA09 direct evidence:

- `revenue_growth_yoy = 0.17`, `evidence_type: direct`;
- `eps_growth_yoy = 0.22`, `evidence_type: direct`.

It adds DATA10 evidence:

- `gross_margin = 0.492705785005`, `evidence_type: derived`;
- `operating_margin = 0.322753273852`, `evidence_type: derived`.

The direct lineage retains DATA09 decision
`me-data09-source-approval-20260719T155116Z`, approval checksum
`5fd17e54c948a54a448abb3849b51ce29c87c65c458d9b28de3fa73daf10b684`,
and package checksum
`9507a07605168841163460ef4e7417e7c51a4d881bbea9e15585b02aeff8f41e`.
The original DATA09 package was not mutated.

## Derivation Approval

Runtime creates only a checksum-bound pending approval candidate; it has no
approval authority. For this pilot the normal flow persisted the candidate and
an explicit operator review changed only its decision and review statuses to
approved. No checksum field was manually fabricated or edited. The resulting
decision is `me-data10-derivation-approval-20260721T094346Z`. It covers source
authenticity, primary-source status, permitted use, publication boundary,
identity, accounting framework, canonical and raw-tag mappings, formula
identity, numerator, denominator, debt components, reporting period,
unit/currency/scale, denominator safety, evidence classification, freshness,
and every relevant checksum.

The decision approves only AAPL, gross margin, and operating margin. It records
debt-to-equity as explicitly blocked. DATA07 cannot parse the governed package
until this decision validates.

## Checksums

- fact package: `315ddaf3cc80bc1e6e92aa5786426c60d895eb3829aa972d0828424fa297591a`;
- formula catalog: `7ae5ca07ad9b45eaed54fccd9e77de254616da6f598d42cd44ff78dcd80673d1`;
- derived package: `c6fa2759d9629fc0276083b6e497a132d03c35229803adf91e0fc2f5ee094b01`;
- derivation validation: `6457d8b15ad3f8d7c70ed8e5c30d6cbc546dd4155357685f46a515364d760248`;
- governed DATA07 package: `048405bf32005d1d956f93608e6f97e27374e126b16bb4fcac8ce48c441e3cde`;
- derivation approval: `1ad1376c2019353d48fbb1b62461ef3cf022ed62017859dc12d6da2b35bcad32`.

## Pilot and Downstream Result

Run identities:

- derivation: `me-data10-aapl-primary-facts-fy2026-q2-20260721T094346Z`;
- package validation: `me-data10-package-validation-20260721T094346Z`;
- approval: `me-data10-derivation-approval-20260721T094346Z`;
- DATA07: `me-data10-aapl-governed-operator-pilot-20260721T094346Z`;
- DATA06: `me-data06-after-me-data10-aapl-20260721T094346Z`;
- RUN31: `me-run31-after-me-data10-aapl-20260721T094346Z`.

DATA07 counts:

- selected: 12;
- imported: 1;
- normalized: 1;
- successful: 1;
- blocked: 11;
- failed: 0;
- pending: 0;
- not selected: 940;
- reconciliation: passed.

The raw snapshot status is `validated`; its checksum equals the governed
package checksum. Provider calls and network calls during derivation/import
were both zero.

## Coverage Attribution

Current-sprint comparison:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| fundamental complete | 6 | 6 | 0 |
| fundamental partial | 39 | 39 | 0 |
| fundamental missing | 907 | 907 | 0 |
| canonical advice-input-ready | 6 | 6 | 0 |
| full-advice-ready | 0 | 0 | 0 |
| unable-to-advise | 946 | 946 | 0 |

AAPL remained partial and not advice-input-ready. The current sprint added two
derived metrics and retained two direct metrics. The historical 4/17/931 to
6/39/907 improvement remains marked `attributable_to_current_sprint: false`.
All regression counters are zero.

## Compact Artifact Map

Committed compact evidence:

```text
artifacts/market_engine/run_evidence/me-data10-generic-derived-metric-pilot-20260721T094346Z/
```

It contains the manifest, source-fact summary, formula snapshot, derivation
validation, derived evidence, approval validation, pilot summary, coverage
delta, downstream index, checksum index, and report.

Full local-only evidence:

```text
artifacts/market_engine/fundamental_metric_sourcing_runs/me-data10-aapl-governed-operator-pilot-20260721T094346Z/
artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data10-aapl-20260721T094346Z/
artifacts/market_engine/run_evidence/me-run31-after-me-data10-aapl-20260721T094346Z/
artifacts/market_engine/me_data10_full_advice_readiness_runs/me-run31-after-me-data10-aapl-20260721T094346Z/
data/market_engine/source_snapshots/fundamental_metrics/me-data10-aapl-governed-operator-pilot-20260721T094346Z/
```

The superseded committed `20260719T183522Z` operator input and compact evidence
are removed and replaced by the immutable `20260721T094346Z` pilot. Earlier
local self-review runs remain uncommitted. The replacement includes the
certified DATA09 package file checksum in direct metric lineage and enforces
direct-versus-derived instrument, company, and fiscal identity. No DATA09
historical artifact was changed.

## Governance Boundary

ME-DATA10 changes fundamental evidence classification only. It adds no
recommendation, ranking, conviction, tradeability, urgency, allocation,
position sizing, portfolio mutation, delivery, Telegram, broker, order, or
Decision Engine behavior. No provider or model invocation occurs. The Decision
Engine remains the only allocation authority.

## Validation

The required focused suites passed with 60 derivation tests, 57 source-approval
tests, 29 operator-package tests, 42 DATA07 sourcing tests, 5 DATA09 compact
evidence tests, and 2 DATA10 compact evidence tests. The aggregate commands
passed with 270 data tests, 1,295 Market Engine tests, and 1,962 repository-wide
tests. The matrix covers US-GAAP, IFRS, a second arbitrary ticker, formulas,
debt components, every period/unit/currency/scale safety gate, denominator
safety, malformed packages, deterministic ordering, approval failure, DATA07
non-invocation, and compact artifact publication boundaries.

## Remaining Limitations and Next Sprint

The bounded real pilot did not establish a safe AAPL debt-to-equity value and
did not change aggregate readiness. It proves the universal engine and two real
derived metrics, not broad issuer coverage.

The logical next sprint is **ME-DATA11 — Execute a diversified US-GAAP/IFRS
multi-ticker derivation pilot**. ME-DATA11 must reuse this generic engine and
formula catalog without ticker-specific runtime changes. Each new issuer still
requires explicit fact mappings, applicability evidence, and checksum-bound
approval.
