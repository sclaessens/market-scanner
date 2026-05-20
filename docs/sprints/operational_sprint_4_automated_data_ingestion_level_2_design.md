# Operational Sprint 4 Automated Data Ingestion Level 2 Design

## 1. Status

Status: LEVEL 2 DESIGN

This document defines a governed Level 2 design for automated or provider-assisted data ingestion covering fundamentals and portfolio metadata.

This is a design-only governance artifact. It does not authorize implementation, provider/API integration, credentials, secrets, runtime orchestration changes, tests, generated files, CSV edits, Reporting changes, Telegram changes, Decision Engine changes, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Future implementation requires a separate developer specification after this design is reviewed and approved.

## 2. Purpose and Scope

The purpose of this Level 2 design is to define a safe future architecture for moving from manually maintained MVP CSV source artifacts toward governed automated or provider-assisted ingestion.

The scope covers two validated source-artifact contracts:

```text
data/raw/fundamentals.csv
data/portfolio/portfolio_metadata.csv
```

The design covers ingestion governance, source artifact principles, provider evaluation criteria, freshness strategy, validation expectations, credential boundaries, and future staging.

It does not choose a provider, implement an API integration, create ingestion scripts, change runtime orchestration, or alter current pipeline behavior.

## 3. Current Validated MVP Contracts

### 3.1 Fundamental Source Artifact

The Fundamental data MVP validated the optional manually maintained source artifact:

```text
data/raw/fundamentals.csv
```

The Fundamental Layer can now produce descriptive quality states such as:

- `SUFFICIENT_DATA`
- `PARTIAL_DATA`
- `STALE_DATA`
- `INSUFFICIENT_DATA`

This proves that the Fundamental Layer can consume explicit local source data and classify quality without gaining allocation, ranking, tradeability, urgency, conviction, or filtering authority.

### 3.2 Portfolio Metadata Source Artifact

The Portfolio Metadata MVP validated the optional manually maintained source artifact:

```text
data/portfolio/portfolio_metadata.csv
```

Portfolio Intelligence can now produce descriptive metadata completeness states such as:

- `portfolio_metadata_status = COMPLETE`
- `portfolio_metadata_status = PARTIAL`

Operational verification showed that when portfolio metadata was present for C, GM, GS, PLD, TT, and WELL, Portfolio Intelligence produced `portfolio_metadata_status = COMPLETE` for those six tickers. The Decision Engine then moved those six rows from `REVIEW` to `NO_ACTION`, while remaining rows stayed `REVIEW` because metadata and/or fundamental coverage was still incomplete.

This proves that the Portfolio Metadata contract can clear source-supported metadata blockers without loosening the Decision Engine.

### 3.3 Stable MVP Contract Interpretation

These two CSV artifacts are stable MVP contracts for now because they are explicit, inspectable, deterministic once written, and compatible with the existing pipeline authority boundaries.

They are source artifacts, not decision artifacts.

They may remain temporary data steward artifacts while future ingestion design proceeds.

## 4. Problem Statement

Manually maintained CSV files are useful for MVP validation but risky as a permanent operating model.

Manual maintenance can lead to stale data, inconsistent source provenance, incomplete coverage, taxonomy drift, manual copy errors, weak audit trails, and pressure to weaken conservative Decision Engine behavior.

The project needs a governed path toward automated or provider-assisted ingestion that preserves:

- deterministic builds;
- explicit source artifacts;
- source provenance;
- freshness metadata;
- schema validation;
- auditability;
- reproducibility;
- classification-only upstream layers;
- Decision Engine as the only allocation authority;
- Reporting and Telegram as communication-only layers.

Provider/API integration must not be added ad hoc.

## 5. Design Goals

This design aims to preserve and extend the current source-artifact model.

Design goals:

- keep builds deterministic;
- keep raw/source artifacts explicit;
- include source provenance for all provider-assisted records;
- include freshness metadata for all source records;
- support schema validation before runtime consumption;
- preserve auditability and reproducibility;
- avoid live provider dependency during Decision Engine execution unless separately approved;
- preserve classification-only upstream layers;
- preserve the Decision Engine as the only allocation authority;
- preserve Reporting and Telegram as communication-only layers;
- ensure missing, partial, stale, or invalid data remains visible instead of inferred;
- support provider-assisted or automated refresh without hidden filtering;
- support future runbooks and validation logs;
- prevent ranking, scoring, tradeability, urgency, conviction, allocation, hidden filtering, or Decision Engine bypass.

## 6. Non-Goals

This Level 2 design does not:

- select a provider, except to define future evaluation criteria;
- implement API calls;
- add credentials;
- add secrets;
- change runtime orchestration;
- add live API calls inside the current pipeline;
- loosen the Decision Engine;
- change Reporting behavior;
- change Telegram behavior;
- allow Reporting or Telegram to infer missing metadata or fundamentals;
- commit generated CSV files;
- change scripts;
- change tests;
- change runtime data files;
- change GitHub workflows;
- add ranking semantics;
- add scoring authority;
- add tradeability semantics;
- add urgency semantics;
- add conviction semantics;
- add allocation authority;
- introduce hidden filtering;
- introduce Decision Engine bypass.

## 7. Ingestion Model Comparison

### 7.1 Option A — Continue Manual CSVs as Temporary Data Steward Artifacts

Description: keep `data/raw/fundamentals.csv` and `data/portfolio/portfolio_metadata.csv` as manually maintained source artifacts.

Advantages:

- preserves deterministic local pipeline reads;
- avoids premature provider/API coupling;
- keeps the MVP contract simple and inspectable;
- supports continued contract validation;
- requires no credentials or runtime provider dependency.

Risks:

- manual data can become stale;
- coverage may remain incomplete;
- manual updates can introduce copy errors;
- source provenance can weaken unless enforced manually;
- manual stewardship may not scale.

Governance fit: safe as a temporary operating model, but not sufficient as the long-term operational strategy.

### 7.2 Option B — Provider-Assisted Prefill Scripts That Write to Governed CSV Artifacts

Description: create future governed prefill scripts that fetch or prepare provider data and write to explicit CSV source artifacts before the pipeline runs.

Advantages:

- preserves deterministic runtime reads;
- keeps provider data auditable through local source artifacts;
- allows schema, freshness, and provenance validation before pipeline consumption;
- avoids live provider calls during Decision Engine execution;
- supports controlled refresh workflows;
- supports reviewable output artifacts.

Risks:

- requires provider selection and provider governance;
- may require credentials or paid access;
- requires validation logs and failure handling;
- scripts must not embed ranking, scoring, or decision semantics.

Governance fit: strong candidate for the next implementation stage after developer specification.

### 7.3 Option C — Fully Automated Provider/API Ingestion Inside the Pipeline

Description: add live provider/API calls directly to the runtime pipeline so fundamentals and metadata are fetched during normal pipeline execution.

Advantages:

- can reduce manual refresh work;
- may improve freshness when providers are reliable;
- can scale to larger ticker universes.

Risks:

- creates live provider dependency during runtime execution;
- can make builds nondeterministic;
- rate limits, outages, provider revisions, or transient failures can alter Decision Engine inputs;
- requires credentials and secret handling;
- complicates reproducibility and auditability;
- increases risk of hidden provider semantics influencing upstream layers.

Governance fit: not recommended as the next stage. This requires separate approval and stronger runtime governance if ever pursued.

### 7.4 Option D — Hybrid Model With External Ingestion Into Raw Artifacts and Deterministic Pipeline Reads

Description: provider/manual source data is collected outside the Decision Engine execution path, written into governed raw/source artifacts, validated, and then read by existing classification layers.

Advantages:

- combines automation with deterministic pipeline execution;
- keeps raw/source artifacts explicit and auditable;
- supports validation before consumption;
- avoids live provider dependency in the Decision Engine path;
- allows controlled refresh cadence;
- preserves existing classification and allocation boundaries.

Risks:

- requires refresh ownership;
- requires failed-refresh behavior;
- requires artifact provenance and validation logs;
- requires careful distinction between ingestion and runtime decision execution.

Governance fit: recommended Level 2 direction.

### 7.5 Option E — Research-Only Enrichment Separate From Decision Engine Inputs

Description: use external or provider data only for research, diagnostics, provider comparison, or exploratory analysis without routing it into Decision Engine inputs.

Advantages:

- supports provider evaluation without changing runtime decisions;
- protects the Decision Engine from unapproved data sources;
- can compare field quality, coverage, taxonomy, and stability;
- useful before selecting an operational provider.

Risks:

- research outputs may be mistaken for operational signals;
- outputs must remain clearly isolated from runtime source artifacts;
- Reporting and Telegram must not present research-only fields as operational decisions.

Governance fit: safe if explicitly labeled research-only and isolated from Decision Engine inputs.

## 8. Recommended Level 2 Direction

Recommended direction: staged hybrid model.

The project should:

1. keep current manual CSV contracts stable for now;
2. design provider-assisted or automated ingestion as a separate governed ingestion layer;
3. write provider data into explicit raw/source artifacts first;
4. validate schema, freshness, provenance, duplicates, and missing data before runtime consumption;
5. keep runtime classification layers reading local governed artifacts;
6. avoid live API calls during Decision Engine execution unless explicitly approved later.

The preferred future architecture is provider-assisted or automated prefill into local source artifacts, not live provider calls in the Decision Engine execution path.

## 9. Proposed Target Architecture

The future safe architecture should follow this pattern:

```text
provider/manual source
-> governed ingestion or prefill layer
-> raw/source artifacts
-> schema/freshness/provenance validation
-> existing classification layers
-> Fundamental Layer / Portfolio Intelligence
-> Decision Engine
-> Reporting / Telegram
```

This is a design proposal only.

The governed ingestion or prefill layer is not authorized by this document.

The key principle is that provider data should become explicit source artifacts before existing runtime layers consume it.

The runtime pipeline should remain deterministic and auditable by reading local artifacts rather than relying on live provider calls during Decision Engine execution.

## 10. Source Artifact Contract Principles

### 10.1 Shared Contract Principles

Both source artifacts should follow shared principles:

- explicit path;
- explicit row identity;
- deterministic duplicate handling;
- required columns;
- source provenance fields;
- freshness metadata fields;
- row-level classification for incomplete data when safe;
- fail-fast behavior for ambiguous schema or duplicate authority;
- no hidden filtering;
- no ranking authority;
- no scoring authority;
- no allocation authority;
- no tradeability semantics;
- no urgency semantics;
- no conviction semantics;
- no Decision Engine bypass.

### 10.2 `data/raw/fundamentals.csv`

Expected contract principles:

- row identity should be based on normalized `ticker` and any approved date or period field required by the Fundamental Layer contract;
- duplicate rows for the same governed row identity should fail fast if they create ambiguous data authority;
- required fields must be defined in developer specification before implementation;
- source provenance should identify provider or manual steward source;
- freshness metadata should identify when the data was last updated and, where applicable, the fiscal period or statement date;
- missing required values should produce descriptive quality states rather than row loss when row-level classification is safe;
- partial data should remain visible as `PARTIAL_DATA` or equivalent descriptive state;
- stale data should remain visible as `STALE_DATA` or equivalent descriptive state;
- invalid schema or duplicate authority should fail deterministically;
- invalid row values should be classified at row level where safe and fail fast where interpretation would be ambiguous.

The Fundamental Layer must remain classification-only.

### 10.3 `data/portfolio/portfolio_metadata.csv`

Expected contract principles:

- row identity should be normalized `ticker`;
- duplicate rows for the same normalized ticker should fail fast;
- required fields should include source-supported descriptive metadata such as sector, industry, asset class, currency, metadata source, and metadata last updated;
- source provenance should identify provider, manual steward source, or approved taxonomy source;
- freshness metadata should identify when the metadata was last updated;
- missing metadata rows should preserve output rows and keep metadata incomplete;
- partial metadata should remain visible through `portfolio_metadata_status = PARTIAL` or equivalent descriptive state;
- stale metadata should not be treated as complete;
- invalid metadata should not be treated as complete;
- metadata-only tickers should not expand the opportunity universe unless a later governed row-universe change is approved.

Portfolio Intelligence must remain descriptive and classification-only.

## 11. Deterministic Failure Versus Row-Level Classification

The future developer specification must distinguish deterministic failure from row-level classification.

Fail fast when:

- required columns are missing from a source artifact;
- duplicate row identity creates ambiguous source authority;
- dates or schema fields cannot be interpreted safely at artifact level;
- provider export format changes in a way that breaks the contract;
- credential or provider configuration is missing for an authorized prefill script and the script cannot proceed safely.

Classify row level when:

- a ticker has no matching source row;
- a row has partial but interpretable source data;
- a row has stale but interpretable source data;
- a row has invalid values that can be isolated without compromising the artifact;
- data is incomplete but row identity can be preserved safely.

No row should be removed because source data is missing, partial, stale, invalid, or unavailable unless a later governance decision explicitly changes the row-universe contract.

## 12. Provider Selection Criteria

This design does not select a provider.

Future provider evaluation should compare candidates against criteria including:

- ticker universe coverage;
- fundamental field coverage;
- portfolio metadata field coverage;
- sector and industry taxonomy clarity;
- asset-class taxonomy clarity;
- field stability;
- update cadence;
- freshness guarantees;
- historical availability;
- API reliability;
- rate limits;
- cost;
- terms of use;
- licensing constraints;
- exportability to CSV or local raw artifacts;
- source provenance support;
- auditability;
- reproducibility;
- error transparency;
- testability;
- compatibility with deterministic pipeline reads.

Provider selection must be handled through a later governance artifact or developer specification before implementation.

## 13. Credential and Secret Boundary

No credentials are authorized by this design.

Future credentials must use approved secret handling.

No secrets may be committed to the repository.

Documentation must not include real credentials, tokens, API keys, account identifiers, or provider secrets.

Any future use of local `.env`, environment variables, keychain storage, or GitHub secrets requires later developer specification.

Credential handling must remain outside CSV source artifacts and outside documentation content.

If provider-assisted ingestion is approved later, the developer specification must define:

- where credentials are read from;
- how missing credentials are handled;
- how credentials are excluded from logs;
- how credentials are excluded from generated artifacts;
- how local and CI usage differ;
- how provider errors are reported without leaking secrets.

## 14. Refresh and Freshness Strategy

Future ingestion design should support controlled refresh without weakening deterministic runtime behavior.

Safe refresh options include:

- manual refresh by a data steward;
- provider-assisted prefill on demand;
- scheduled refresh outside Decision Engine execution;
- research-only refresh for provider evaluation;
- local refresh before a full pipeline run.

Freshness strategy should define:

- freshness thresholds per artifact and field type;
- how `metadata_last_updated` or equivalent fields are written;
- how fiscal or statement period dates are represented;
- how stale rows are classified;
- how failed refreshes are logged;
- whether stale existing artifacts may still be consumed as stale classifications;
- whether refresh failures block runtime or preserve last known source artifacts;
- how operator-visible warnings are recorded.

Failed-refresh behavior should be deterministic.

A failed provider refresh should not silently create complete data. It should either leave prior governed artifacts unchanged with a logged failure or write an explicit failed-refresh artifact only if later approved.

## 15. Validation and Audit Requirements

Future implementation must include validation and audit requirements before provider-assisted data can be trusted.

Expected validation controls:

- schema validation;
- required column validation;
- duplicate ticker or row-key checks;
- normalized ticker checks;
- date parsing checks;
- freshness threshold checks;
- source metadata checks;
- provider export format checks;
- row preservation checks;
- metadata-only ticker handling checks;
- missing data classification checks;
- partial data classification checks;
- stale data classification checks;
- invalid data classification checks;
- no hidden filtering checks;
- no upstream allocation semantics checks;
- no ranking, scoring, tradeability, urgency, or conviction semantics checks;
- no live API dependency in the Decision Engine path unless later approved.

Expected audit outputs or logs should be defined in the future developer specification and may include:

- refresh timestamp;
- provider/source label;
- row counts;
- matched ticker counts;
- missing ticker counts;
- stale row counts;
- invalid row counts;
- duplicate detection results;
- source artifact write status;
- validation status;
- failure reason.

Audit outputs must remain descriptive and must not create decision authority.

## 16. Governance Boundaries

This document does not authorize implementation.

No provider/API integration is authorized.

No credentials or secrets are authorized.

No runtime orchestration changes are authorized.

No sprint is closed or certified complete.

Future work requires developer specification before implementation.

The Decision Engine must not be loosened.

Reporting and Telegram must remain communication-only.

Ingestion must not introduce allocation, ranking, tradeability, urgency, conviction, hidden filtering, or Decision Engine bypass.

Runtime classification layers must remain classification-only.

Provider data must not be interpreted as allocation advice, tradeability, urgency, conviction, ranking, or scoring unless a future Decision Engine governance change explicitly authorizes a new decision input.

## 17. Implementation Staging Proposal

### Stage 1 — Keep Manual CSV Contracts Stable

Continue using the current manual CSV contracts as stable source artifacts while governance design proceeds.

No additional implementation is authorized by this stage.

### Stage 2 — Define Provider Evaluation Matrix

Create a provider evaluation matrix covering fundamentals and portfolio metadata.

The matrix should compare coverage, field quality, source provenance, freshness, reliability, exportability, cost, and terms of use.

### Stage 3 — Create Developer Specification for Provider-Assisted Prefill

After Level 2 design review, create a developer specification for provider-assisted prefill outside the runtime decision path.

The developer specification should define permitted files, forbidden files, validation logic, credential boundaries, tests, and acceptance criteria.

### Stage 4 — Implement Provider-Assisted Prefill Outside Runtime Path

Only after developer specification approval, implement a provider-assisted prefill workflow that writes to governed local source artifacts.

This stage should not add live provider calls during Decision Engine execution.

### Stage 5 — Add Validation and Runbook Documentation

Add runbook documentation for refresh, validation, troubleshooting, stale data handling, and failure recovery.

Runbooks must remain procedural and must not change runtime contracts.

### Stage 6 — Evaluate Whether Deeper Automation Is Justified

After provider-assisted prefill proves reliable, evaluate whether scheduled refresh or deeper automation is justified.

Any move toward live provider dependency inside runtime execution requires separate governance approval.

## 18. Backlog Impact Assessment

BL-0017 already captures the required future work:

```text
BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata
```

This Level 2 design remains within that captured backlog scope.

No additional backlog item is required at this stage.

Backlog impact assessment:
- No new backlog items identified.

## 19. Recommended Next Step

After this Level 2 design is reviewed and approved, create a developer specification for provider-assisted prefill outside the runtime Decision Engine path.

The developer specification should not authorize live API calls during Decision Engine execution unless a separate governance decision explicitly approves that architecture.

The preferred next implementation path is provider-assisted prefill into governed local source artifacts, followed by deterministic reads by existing classification layers.
