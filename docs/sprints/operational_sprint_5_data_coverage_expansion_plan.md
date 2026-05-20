# Operational Sprint 5 Data Coverage Expansion Plan

## 1. Status and Purpose

Status: PROPOSED SPRINT PLAN

This document defines a governed Data Coverage Expansion Sprint plan.

This is a documentation-only PM, Functional Analyst, Technical Analyst, Scrum Master, and Governance artifact. It does not authorize implementation by itself.

This document does not modify scripts, tests, data files, generated artifacts, reports, workflows, Python code, CSV files, or runtime behavior.

No sprint is closed or certified complete by this document.

## 2. Background

The project has proven that provider-assisted source artifacts can be consumed correctly by the existing pipeline.

The validated source-artifact path is:

```text
provider/operator export
-> prefill script
-> governed source artifact
-> existing classification layers
-> Decision Engine
-> Reporting / Telegram
```

Recently completed work added operator-triggered local prefill scripts outside the runtime Decision Engine path. These scripts support governed local source artifacts without adding live provider/API calls, Decision Engine changes, Reporting changes, Telegram changes, workflow changes, or runtime orchestration changes.

Portfolio metadata has been added and committed for six tickers:

- C
- GM
- GS
- PLD
- TT
- WELL

The committed source artifact is:

```text
data/portfolio/portfolio_metadata.csv
```

Portfolio Intelligence correctly produced:

```text
portfolio_metadata_status = COMPLETE
```

for those six tickers.

The Decision Engine correctly processed those rows as:

```text
final_action = NO_ACTION
allocation_decision = NO_ALLOCATION_ACTION
execution_decision = NO_EXECUTION_ACTION
arbitration_state = NO_CONFLICT
```

The Fundamental Layer has also been locally validated with:

```text
data/raw/fundamentals.csv
```

This file is currently local/operator-managed because `data/raw/` is ignored by `.gitignore`.

Local validation proved that the Fundamental Layer can consume fundamentals when `ticker + as_of_date` matches the upstream opportunity row date.

Observed local test result:

- GM, PLD, TT, WELL -> `SUFFICIENT_DATA`
- C, GS -> `PARTIAL_DATA`

The path works technically, but current coverage remains limited.

## 3. Problem Statement

Six tickers are enough for contract validation but insufficient for broad analysis.

The system needs broader portfolio metadata and fundamental coverage before advanced analysis, historical learning, ranking research, decision-quality evaluation, prediction tracking, or analytical intelligence becomes meaningful.

Data coverage must improve without:

- loosening Decision Engine behavior;
- allowing Reporting to infer missing data;
- allowing Telegram to infer missing data;
- adding live provider/API calls inside runtime pipeline execution;
- introducing ranking, scoring, tradeability, urgency, conviction, allocation, or hidden filtering outside the Decision Engine.

## 4. Sprint Objective

Operational Sprint 5 objective:

Expand source-data coverage for portfolio metadata and fundamentals across a governed target universe, measure coverage quality and missing-data patterns, preserve deterministic pipeline behavior and source provenance, and produce enough source-data coverage to support future analytical work.

This sprint is not an analysis-feature sprint.

This sprint prepares the data foundation needed before deeper analysis features are appropriate.

## 5. Target Universe Decision

The sprint must explicitly select a governed target universe before data expansion begins.

Candidate target universes:

### Option A — Portfolio Holdings Only

Advantages:

- small and manageable;
- directly relevant to current positions;
- easy to validate manually;
- low operational overhead.

Limitations:

- too narrow for opportunity analysis;
- does not improve scanner coverage;
- does not support broad setup comparison.

### Option B — Portfolio + Watchlist

Advantages:

- operationally relevant;
- supports current holdings and near-term monitored opportunities;
- manageable first expansion stage.

Limitations:

- watchlist contract must be clear;
- may still be too narrow for scanner-wide analysis.

### Option C — Current Scanner Output Rows

Advantages:

- directly aligned with current opportunity universe;
- supports Decision Engine input coverage;
- enables broader coverage diagnostics.

Limitations:

- may include many low-priority rows;
- requires stronger source workflow and audit support.

### Option D — A/B-Grade Scanner Rows Only

Advantages:

- focuses on higher-quality technical setups;
- balances usefulness and operational effort;
- avoids immediately filling every low-signal row.

Limitations:

- must not become hidden filtering authority;
- A/B selection must be treated as coverage prioritization only, not allocation priority.

### Option E — Full Scanner Universe

Current scanner-related universe is approximately 291 rows.

Advantages:

- broadest coverage;
- strongest future analytical foundation;
- supports comprehensive missing-data diagnostics.

Limitations:

- operationally heavier;
- likely requires provider evaluation or better sourcing workflow;
- may produce significant partial/stale/missing coverage initially;
- may be too much for a first expansion sprint.

### Option F — Staged Approach

Recommended direction: staged approach.

Stages:

1. portfolio holdings + active watchlist / high-priority tickers;
2. A/B-grade scanner rows;
3. broader scanner universe.

This is preferred because it improves useful coverage quickly while preserving operational control. Trying to fill all 291 rows immediately may be operationally heavy and may require provider evaluation, better sourcing workflow, or additional coverage tooling.

The staged approach must remain coverage-oriented only. It must not create ranking authority, allocation authority, hidden filtering, or tradeability semantics.

## 6. Required Data Coverage Fields

### 6.1 Portfolio Metadata Fields

Required portfolio metadata fields:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

Optional descriptive fields:

- `sector_taxonomy`
- `industry_group`
- `country`
- `region`
- `exchange`
- `notes`

These fields are descriptive source metadata only.

They must not become ranking, scoring, tradeability, urgency, conviction, allocation, or filtering mechanisms.

### 6.2 Fundamental Fields

Required fundamentals fields:

- `ticker`
- `as_of_date`
- `source_name`
- `source_last_updated`
- `report_period`
- `currency`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`
- `free_cash_flow_positive`

These fields support Fundamental Layer descriptive quality classification only.

They do not authorize ranking, tradeability, urgency, conviction, allocation, or hidden filtering.

## 7. Date Matching and Freshness Rules

Observed operational behavior:

- Fundamental Layer matches source rows by ticker and opportunity date.
- `data/raw/fundamentals.csv` uses `as_of_date`.
- `as_of_date` must align with the upstream opportunity row `date`.
- `source_last_updated` must not be later than the opportunity row date under current behavior, otherwise metadata may be considered invalid.

This rule is operationally important because a local validation attempt with non-matching dates produced `row_missing`, and a later source update date than the opportunity date produced invalid metadata.

If this interpretation is too strict for real provider data, a future governance item may be required to define provider reporting-date semantics, as-of matching, and freshness interpretation.

Until then, data coverage expansion must respect the current behavior rather than loosening Fundamental Layer rules ad hoc.

## 8. Coverage Metrics

Operational Sprint 5 must define and produce coverage metrics for the selected target universe.

Required coverage metrics:

- total tickers in target universe;
- metadata complete count;
- metadata partial count;
- metadata missing count;
- fundamentals sufficient count;
- fundamentals partial count;
- fundamentals stale count;
- fundamentals insufficient count;
- most common missing fundamental fields;
- source freshness distribution;
- row match success count;
- row match failure count;
- duplicate detection result;
- invalid row count;
- source artifact coverage rate;
- coverage by source label;
- coverage by target-universe stage.

Coverage metrics are observational only.

They must not become ranking authority, scoring authority, tradeability, allocation authority, urgency, conviction, or hidden filtering.

## 9. Allowed Implementation Candidates for Later Codex Work

This sprint plan may propose future implementation candidates, but it does not implement them.

Candidate future tasks:

- coverage audit script;
- coverage report document;
- source artifact coverage dashboard CSV;
- runbook for provider-assisted data expansion;
- fixture/test coverage for broader source artifacts;
- optional provider evaluation matrix;
- data steward checklist for source artifact updates;
- documentation of date matching and freshness behavior.

These remain future candidates under existing backlog scope until separately approved through developer specification and implementation authorization.

## 10. Explicit Forbidden Scope

Operational Sprint 5 must not allow:

- Decision Engine loosening;
- Reporting inference;
- Telegram inference;
- live provider/API calls inside runtime pipeline;
- live provider/API calls during Decision Engine execution;
- hidden ranking;
- hidden scoring;
- upstream tradeability semantics;
- upstream urgency semantics;
- upstream conviction semantics;
- allocation outside Decision Engine;
- hidden filtering;
- generated processed artifact commits unless separately authorized;
- credentials or secrets;
- workflow changes unless separately approved;
- automatic source refresh during normal runtime execution;
- provider-specific assumptions embedded in Decision Engine logic;
- changes to Reporting or Telegram to compensate for missing source data.

## 11. Governance Boundaries

Operational Sprint 5 must preserve the certified doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- upstream layers classify only;
- Reporting communicates only;
- Telegram communicates only;
- source artifacts are descriptive inputs, not decision outputs;
- coverage metrics are observational;
- no ranking or scoring authority outside the Decision Engine;
- no hidden filtering;
- no Decision Engine bypass.

Coverage expansion improves data availability. It does not authorize allocation decisions.

## 12. Proposed Sprint Phases

### Phase 1 — Universe Selection and Source Inventory

Select the governed target universe for the first coverage expansion stage.

Recommended first-stage target:

```text
portfolio holdings + active watchlist / high-priority tickers
```

Document source files, current coverage, ignored paths, and operator-managed source artifacts.

### Phase 2 — Metadata Coverage Expansion

Expand portfolio metadata coverage for the selected universe using the approved source artifact contract:

```text
data/portfolio/portfolio_metadata.csv
```

Validate sector, industry, asset class, currency, metadata source, metadata freshness, and duplicate handling.

### Phase 3 — Fundamentals Coverage Expansion

Expand fundamentals coverage for the selected universe using the approved local/operator-managed source artifact contract:

```text
data/raw/fundamentals.csv
```

Respect current ticker/date matching behavior and source freshness interpretation.

### Phase 4 — Coverage Audit and Gap Report

Produce coverage metrics for the selected target universe.

Identify missing metadata, missing fundamentals, partial fundamentals, stale data, invalid rows, row match failures, and common missing fields.

### Phase 5 — Pipeline Validation

Run the existing pipeline stages required to confirm that expanded source data is consumed correctly.

Validation must remain observational.

Generated processed artifacts and logs must not be committed unless separately authorized.

### Phase 6 — Backlog Reconciliation and Next-Step Recommendation

Assess new gaps discovered during coverage expansion.

Add backlog items only if they exceed the existing scope of BL-0017.

Recommend whether the next step should be a coverage audit utility, provider evaluation matrix, data steward runbook, or later analytical feature design.

## 13. Acceptance Criteria

Operational Sprint 5 is acceptable only if:

- target universe is explicitly selected;
- source artifacts are expanded or expansion workflow is documented;
- coverage metrics are produced;
- missing data patterns are identified;
- date matching and freshness behavior are documented;
- pipeline remains deterministic;
- Decision Engine authority remains unchanged;
- Reporting authority remains unchanged;
- Telegram authority remains unchanged;
- source artifacts remain descriptive inputs;
- coverage metrics remain observational;
- no generated processed artifacts are committed unless separately authorized;
- no credentials or secrets are added;
- no runtime provider/API dependency is introduced;
- repository content remains English-only;
- backlog impact is reconciled.

## 14. Backlog Impact Assessment

Existing backlog item `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata` is sufficient for this sprint plan.

The following topics remain captured as candidate work within the Operational Sprint 5 plan and do not require separate backlog IDs at this stage:

- operational date semantics for fundamentals source freshness and as-of matching;
- governed data coverage audit and reporting utility;
- provider evaluation matrix for wider coverage.

These topics must be reconsidered during Operational Sprint 5 execution review, implementation audit, or closeout if they exceed the existing scope of BL-0017 or become independently actionable.

Backlog impact assessment:
- No new backlog items identified.

## 15. Recommended Next Step

Recommended next governance-safe step:

Create a developer specification for a coverage audit utility before implementing any new scripts.

Do not proceed directly to analysis features yet.

The coverage audit utility should measure metadata and fundamentals coverage across a selected target universe without ranking, scoring, filtering, or changing Decision Engine behavior.

Only after coverage is sufficient should the project consider deeper analysis, historical learning, prediction tracking, or decision-quality evaluation features.
