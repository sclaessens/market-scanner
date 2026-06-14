# Operational Sprint 5 Coverage Audit Utility Developer Specification

## 1. Status and Scope

Status: DEVELOPER SPECIFICATION

This document defines the future implementation contract for a governed coverage audit utility.

This is a documentation-only PM, Functional Analyst, Technical Analyst, Scrum Master, and Governance artifact.

This document does not implement code, tests, CSV files, generated artifacts, reports, provider integration, credentials, runtime orchestration changes, Reporting changes, Telegram changes, Decision Engine changes, or runtime behavior.

This document does not modify:

- `scripts/`
- `tests/`
- `data/`
- `reports/`
- `.github/workflows/`
- Python code
- generated files
- CSV files
- runtime behavior

No sprint is closed or certified complete by this document.

Future implementation is not authorized by this document alone. A separate implementation prompt, branch, review, and validation cycle must approve and bound any code changes.

## 2. Purpose

The purpose of the future coverage audit utility is to create an observational utility that measures metadata and fundamentals coverage across a selected target universe.

The utility must identify:

- missing source data;
- partial source data;
- stale source data;
- invalid source data;
- unmatched source rows;
- ticker/date match failures;
- recurring missing-field patterns.

The utility exists to support data stewardship and future source-data expansion.

The utility prepares the project for later analysis features only after sufficient data coverage exists.

The utility must not create analytical authority, ranking authority, scoring authority, allocation authority, tradeability semantics, urgency semantics, conviction semantics, or Decision Engine bypass.

## 3. Authorized Future Implementation Scope

A separately approved future implementation sprint may implement the following only:

- a coverage audit script outside the runtime Decision Engine path;
- coverage metrics for portfolio metadata;
- coverage metrics for fundamentals;
- explicit target-universe selection;
- missing-field analysis;
- ticker/date match diagnostics;
- optional local CSV coverage report output;
- optional text or markdown coverage summary;
- focused tests for the coverage audit utility.

The utility must remain observational. Its outputs may guide human data stewardship, future source-data expansion, and future governance discussion, but they must not change runtime decision behavior.

## 4. Explicit Forbidden Future Scope

The future implementation must not:

- change Decision Engine logic;
- loosen Decision Engine `REVIEW` behavior;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change validation logic;
- change context logic;
- change timing logic;
- change provider-assisted prefill behavior unless separately authorized;
- call provider APIs;
- add credentials or secrets;
- infer missing metadata;
- infer missing fundamentals;
- create rankings;
- create scores;
- create tradeability labels;
- create urgency labels;
- create conviction labels;
- allocate capital;
- filter rows;
- suppress rows;
- change runtime orchestration;
- write generated processed artifacts unless separately authorized;
- become a Decision Engine input unless separately governed;
- create hidden prioritization or hidden eligibility semantics;
- compensate for missing source data through Reporting or Telegram inference.

## 5. Proposed Future File Boundaries

This developer specification does not modify the following future implementation files. It only defines the proposed boundaries for later approved work.

Suggested future implementation file:

```text
scripts/diagnostics/audit_data_coverage.py
```

Suggested future tests:

```text
tests/diagnostics/test_audit_data_coverage.py
```

Optional future docs or runbook:

```text
docs/active/runbooks/data_coverage_audit.md
```

The utility belongs under `scripts/diagnostics/` or an equivalent non-runtime diagnostics path because it is observational and operator-facing. It must not live under the Decision Engine, Reporting, Telegram, scanner, validation, context, timing, fundamental classification, or portfolio intelligence implementation paths.

The diagnostics boundary makes the utility clearly separate from:

- runtime classification layers;
- allocation and execution authority;
- Reporting and Telegram communication artifacts;
- provider-assisted prefill scripts;
- runtime orchestration.

A diagnostics path prevents the audit utility from being mistaken for a production pipeline stage or a Decision Engine dependency.

## 6. Inputs

A future implementation may read only approved observational inputs.

Permitted candidate inputs:

- a selected target universe artifact, for example scanner output, watchlist, portfolio holdings, or an explicitly provided ticker list;
- `data/portfolio/portfolio_metadata.csv`;
- local/operator-managed `data/raw/fundamentals.csv`;
- optionally `data/processed/fundamental_quality.csv`;
- optionally `data/processed/portfolio_intelligence.csv`;
- optionally `data/processed/final_decisions.csv` for observational comparison only.

Reading Decision Engine outputs for observation does not make the utility a Decision Engine authority.

The utility must not write to Decision Engine input artifacts.

The utility must not write to Decision Engine output artifacts.

The utility must not modify `data/processed/final_decisions.csv`.

The utility must not use Decision Engine output fields to infer missing fundamentals, infer missing metadata, loosen `REVIEW` behavior, or create allocation decisions.

## 7. Target Universe Options

A future implementation must require explicit target-universe selection.

Supported target universe modes:

- portfolio holdings only;
- portfolio plus watchlist;
- scanner output rows;
- A/B-grade scanner rows;
- explicit ticker list;
- full scanner universe.

Target-universe selection is for coverage measurement only.

Target-universe selection must not become:

- hidden filtering;
- hidden ranking;
- hidden scoring;
- tradeability classification;
- urgency classification;
- conviction classification;
- allocation prioritization;
- Decision Engine bypass.

For A/B-grade scanner rows, the grade selection may be used only to narrow the audit universe for data stewardship. It must not imply that A/B rows are more actionable, more tradeable, or more eligible for allocation.

## 8. Portfolio Metadata Coverage Metrics

A future implementation must produce portfolio metadata coverage metrics for the selected target universe.

Required metrics:

- total target tickers;
- metadata complete count;
- metadata partial count;
- metadata missing count;
- metadata invalid count;
- duplicate metadata ticker count;
- missing sector count;
- missing industry count;
- missing asset class count;
- missing currency count;
- missing metadata source count;
- missing metadata last updated count;
- metadata freshness distribution;
- metadata coverage percentage.

Portfolio metadata coverage metrics must remain descriptive and observational.

They must not create sector ranking, industry ranking, asset-class ranking, allocation preference, portfolio weighting, or exposure recommendations.

## 9. Fundamentals Coverage Metrics

A future implementation must produce fundamentals coverage metrics for the selected target universe.

Required metrics:

- total target ticker/date rows;
- fundamentals sufficient count;
- fundamentals partial count;
- fundamentals stale count;
- fundamentals insufficient count;
- fundamentals invalid count;
- source row missing count;
- duplicate ticker/date identity count;
- ticker/date match success count;
- ticker/date match failure count;
- missing `revenue_growth_yoy` count;
- missing `eps_growth_yoy` count;
- missing `gross_margin` count;
- missing `operating_margin` count;
- missing `debt_to_equity` count;
- missing `free_cash_flow_positive` count;
- source freshness distribution;
- fundamentals coverage percentage.

Fundamentals coverage metrics must remain descriptive and observational.

They must not create fundamental ranking, quality scoring, tradeability labels, conviction labels, urgency labels, allocation recommendations, or hidden filtering.

## 10. Date Matching Diagnostics

The future utility must capture the current operational Fundamental Layer behavior.

Current behavior to diagnose:

- fundamentals are matched by `ticker + as_of_date` against the opportunity row `date`;
- date mismatches should be reported explicitly;
- `source_last_updated` later than the opportunity row `date` currently produces invalid metadata behavior and should be detected;
- the utility may report the issue but must not change Fundamental Layer behavior.

Required diagnostics:

- target ticker/date identity;
- source ticker/date identity where present;
- match success status;
- match failure reason;
- source freshness status;
- invalid future-source-update detection;
- missing source row detection;
- duplicate ticker/date detection.

If future governance later changes Fundamental Layer date semantics, this specification must be updated before implementation is changed.

## 11. Output Contract

A future implementation may produce diagnostic and observational outputs only.

Recommended outputs:

- terminal summary JSON or table;
- optional local CSV report, for example:

```text
reports/diagnostics/data_coverage_audit.csv
```

- optional markdown summary, for example:

```text
reports/diagnostics/data_coverage_audit.md
```

Output rules:

- outputs are diagnostic and observational only;
- outputs must not be committed unless separately authorized;
- outputs must not be consumed by the Decision Engine;
- outputs must not be Decision Engine source artifacts;
- outputs must not be Reporting source artifacts;
- outputs must not be Telegram source artifacts;
- outputs must not be written under `data/processed/` unless separately authorized;
- outputs must not alter runtime behavior.

If a future implementation writes local diagnostic artifacts, those artifacts should remain outside the runtime pipeline and should be treated as operator review material.

## 12. Governance and Semantic Boundaries

The future coverage audit utility must preserve the certified project doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- Reporting is communication-only;
- Telegram is communication-only;
- coverage metrics are observational;
- no ranking authority;
- no scoring authority;
- no hidden filtering;
- no tradeability semantics;
- no urgency semantics;
- no conviction semantics;
- no Decision Engine bypass.

The utility may report coverage gaps. It must not decide what to buy, sell, hold, trim, review, prioritize, rank, score, suppress, or allocate.

The utility may support source-data stewardship. It must not become a strategy engine, research engine, allocation layer, Reporting inference layer, Telegram inference layer, or runtime decision dependency.

## 13. Error Handling

A future implementation must use deterministic error handling.

The utility should fail safely or report diagnostics clearly depending on severity.

Required future behavior:

| Condition | Required Behavior |
|---|---|
| Missing required target-universe input | Fail safely with a clear error unless an explicit empty-universe mode is provided. |
| Ignored or unavailable `data/raw/fundamentals.csv` | Report fundamentals source unavailable and continue metadata audit if metadata inputs are available. |
| Missing portfolio metadata artifact | Report metadata source unavailable and continue fundamentals audit if fundamentals inputs are available. |
| Malformed CSV | Fail safely for the affected input with file path and parse context. |
| Missing required columns | Fail safely for required identity columns; report missing optional metric columns as coverage gaps. |
| Duplicate metadata tickers | Report duplicate metadata ticker count and affected tickers. |
| Duplicate ticker/date identities | Report duplicate ticker/date identity count and affected identities. |
| Malformed dates | Report malformed date count and affected rows; fail safely if identity matching cannot proceed. |
| Empty target universe | Fail safely with a clear message unless explicitly allowed for smoke testing. |
| Unknown target universe mode | Fail safely with supported mode list. |
| Future `source_last_updated` relative to opportunity row date | Report invalid source freshness diagnostic without changing Fundamental Layer behavior. |

Error messages must be deterministic, English-only, and operator-readable.

The utility must not silently drop rows.

The utility must not suppress rows to make metrics cleaner.

## 14. Testing Requirements for Future Implementation

If implementation is later approved, Codex must add focused tests for the coverage audit utility.

Required tests:

- portfolio metadata complete coverage;
- portfolio metadata missing coverage;
- portfolio metadata partial coverage;
- fundamentals sufficient coverage;
- fundamentals partial coverage;
- fundamentals missing source rows;
- fundamentals date mismatch;
- invalid source freshness date;
- duplicate ticker handling;
- duplicate ticker/date handling;
- explicit ticker list target universe;
- scanner output target universe;
- no Decision Engine file changes;
- no Reporting file changes;
- no Telegram file changes;
- no generated processed artifacts committed;
- no ranking fields emitted;
- no scoring fields emitted;
- no tradeability fields emitted;
- no allocation fields emitted;
- no urgency fields emitted;
- no conviction fields emitted;
- deterministic malformed input handling;
- deterministic empty target-universe handling.

Tests must remain scoped to diagnostics behavior. They must not require provider/API access, credentials, runtime orchestration changes, Reporting changes, Telegram changes, Decision Engine changes, or generated production artifacts.

## 15. Acceptance Criteria for Future Implementation

Future implementation is acceptable only if:

- the utility is outside the runtime Decision Engine path;
- the utility is observational only;
- the target universe is explicit;
- metadata coverage metrics are produced;
- fundamentals coverage metrics are produced;
- date-match diagnostics are produced;
- missing-field patterns are reported;
- no Decision Engine logic changes are made;
- no Reporting logic changes are made;
- no Telegram logic changes are made;
- no provider/API calls are added;
- no credentials or secrets are added;
- no generated processed artifacts are committed;
- tests pass;
- English-only repository content is preserved;
- no ranking, scoring, tradeability, urgency, conviction, allocation, or hidden filtering fields are emitted;
- generated diagnostics remain outside runtime source artifacts unless separately authorized.

## 16. Implementation Non-Goals

Explicit non-goals:

- no implementation in this document;
- no provider selection;
- no provider/API integration;
- no data fetching;
- no automatic refresh;
- no Decision Engine changes;
- no Reporting changes;
- no Telegram changes;
- no strategy tuning;
- no allocation logic;
- no ranking semantics;
- no scoring semantics;
- no tradeability semantics;
- no urgency semantics;
- no conviction semantics;
- no hidden filtering;
- no analysis features yet;
- no production dashboard;
- no runtime orchestration changes;
- no generated processed artifact commits.

## 17. Backlog Impact Assessment

Existing backlog item `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata` is sufficient for this developer specification.

The coverage audit utility remains a bounded diagnostic candidate within the existing Operational Sprint 5 data coverage and source-data stewardship scope.

No additional backlog item is required because this document does not identify a new independent scope beyond BL-0017.

Backlog impact assessment:
- No new backlog items identified.

## 18. Recommended Next Step

After this developer specification is reviewed and merged, the recommended next step is a separate Codex implementation prompt for the coverage audit utility.

That implementation prompt must remain bounded to diagnostics and coverage measurement only.

It must explicitly preserve:

- diagnostics-only placement;
- no Decision Engine changes;
- no Reporting changes;
- no Telegram changes;
- no provider/API calls;
- no credentials or secrets;
- no generated processed artifact commits;
- no ranking, scoring, tradeability, urgency, conviction, allocation, or hidden filtering semantics.

Implementation should not proceed until the future implementation prompt is reviewed as a separate governed task.