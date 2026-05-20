# Operational Sprint 4 Data Source Strategy Follow-up

## 1. Status

Status: INVESTIGATION FOLLOW-UP

This document records the operational finding that manually maintained metadata and fundamental CSV files are useful as MVP contract-validation artifacts, but should not be treated as the permanent long-term operational data strategy.

This is a documentation-only governance follow-up. It does not authorize implementation, runtime code changes, tests, generated artifact changes, CSV edits, provider integration, API calls, credentials, runtime orchestration changes, Reporting changes, Telegram changes, or Decision Engine loosening.

No sprint is closed or certified complete by this document.

Future work requires Level 2 design before implementation.

## 2. Investigation Context

The project has recently implemented and verified two staged MVP data-source contracts.

First, the Fundamental data MVP introduced an optional manually maintained source artifact:

```text
data/raw/fundamentals.csv
```

Second, the Portfolio Metadata MVP introduced an optional manually maintained source artifact:

```text
data/portfolio/portfolio_metadata.csv
```

Operational verification confirmed that both contracts can move source-supported descriptive metadata through the existing pipeline without loosening the Decision Engine or changing Reporting and Telegram authority boundaries.

The verified pipeline remains:

```text
scanner -> validation -> context -> fundamental -> timing -> portfolio state -> portfolio review -> portfolio intelligence -> final decisions -> reporting -> Telegram delivery
```

The finding is two-sided:

1. manually maintained CSV files are useful for validating MVP contracts safely;
2. manually maintained CSV files should not become the assumed permanent operational data strategy.

## 3. What the Fundamental Data MVP Proved

The Fundamental data MVP proved that an optional local raw artifact can support descriptive Fundamental Layer quality classification.

With source data present, the Fundamental Layer can now produce real descriptive quality states such as:

- `SUFFICIENT_DATA`
- `PARTIAL_DATA`
- `STALE_DATA`
- `INSUFFICIENT_DATA`

This confirms that the contract path works from a governed raw source artifact into downstream pipeline inputs.

The Fundamental Layer remains classification-only. It does not rank securities, determine tradeability, create urgency, express conviction, allocate capital, or bypass the Decision Engine.

## 4. What the Portfolio Metadata MVP Proved

The Portfolio Metadata MVP proved that a separate optional local metadata artifact can support descriptive Portfolio Intelligence metadata completeness.

With portfolio metadata present for C, GM, GS, PLD, TT, and WELL, Portfolio Intelligence produced:

```text
portfolio_metadata_status = COMPLETE
```

for those six tickers.

The Decision Engine then moved those six rows from:

```text
final_action = REVIEW
```

to:

```text
final_action = NO_ACTION
```

The remaining 285 rows stayed:

```text
final_action = REVIEW
```

because metadata and/or fundamental coverage was still incomplete.

This confirmed that the contract path works and that the Decision Engine can clear metadata blockers when required descriptive inputs are source-supported.

It also confirmed that the Decision Engine remains correctly conservative when coverage is incomplete.

## 5. Why Manual CSVs Are Acceptable as MVP Contract-Validation Artifacts

Manually maintained CSV files are acceptable for staged MVP contract validation because they are:

- explicit source artifacts;
- inspectable in version control or local working state;
- deterministic once written;
- easy to validate against schema expectations;
- compatible with row-preserving pipeline behavior;
- useful for proving whether downstream contracts react correctly to complete, partial, stale, missing, or invalid source data;
- governance-safe when treated as temporary data steward artifacts.

The MVP purpose is contract validation, not permanent operational ingestion.

The manual artifacts allow the team to prove that the Fundamental Layer, Portfolio Intelligence, Decision Engine, Reporting, and Telegram can preserve authority boundaries while consuming source-supported descriptive data.

## 6. Why Manual CSVs Are Not Ideal as the Long-Term Operational Strategy

Manual CSV maintenance is not ideal as the long-term operational data strategy because it can introduce operational drift.

Long-term risks include:

- stale metadata;
- incomplete provider provenance;
- inconsistent freshness timestamps;
- manual copy/paste errors;
- inconsistent taxonomy values;
- slow updates when ticker universes change;
- hidden assumptions about sectors, industries, fundamentals, or asset classes;
- pressure to loosen the Decision Engine to compensate for missing data;
- Reporting or Telegram pressure to infer missing values for user convenience.

The long-term goal should be governed, auditable, deterministic ingestion into explicit source artifacts.

## 7. Governance Risks of Ad Hoc Provider/API Integration

Provider/API integration must not be added ad hoc.

Ad hoc provider/API integration could create the following governance risks:

- nondeterministic pipeline inputs if live calls occur during runtime decision execution;
- unstable outputs caused by provider availability, rate limits, revisions, or transient failures;
- unclear source provenance;
- unclear freshness metadata;
- credentials or secrets introduced without governance;
- provider terms, field definitions, or taxonomy assumptions embedded in runtime logic without documentation;
- hidden scoring, ranking, tradeability, urgency, conviction, or allocation semantics leaking into upstream layers;
- Decision Engine bypass pressure if provider data is interpreted as direct action guidance;
- insufficient testability and reproducibility.

No API calls, credentials, ingestion logic, provider selection, or automation are authorized by this document.

## 8. Governance Risks of Permanent Manual Maintenance

Permanent manual maintenance is also risky.

If manual CSVs become the de facto long-term operational strategy without governance controls, the project can accumulate silent data quality problems.

Risks include:

- manual data drift;
- unverified freshness;
- inconsistent coverage across tickers;
- missing audit trail for source updates;
- unclear responsibility for data stewardship;
- stale metadata that still appears structurally valid;
- inconsistent sector or industry taxonomy;
- accidental editing of source artifacts to force desired decisions;
- pressure to weaken conservative `REVIEW` behavior.

Manual files should remain stable MVP source artifacts for now, but future operational quality requires a governed ingestion strategy.

## 9. Possible Safe Future Directions

### 9.1 Option A — Continue Manually Maintained CSVs as Temporary Data Steward Artifacts

Description: keep `data/raw/fundamentals.csv` and `data/portfolio/portfolio_metadata.csv` as manually maintained source artifacts while the team designs a governed ingestion strategy.

Advantages:

- preserves deterministic local pipeline behavior;
- avoids premature provider/API coupling;
- keeps source artifacts explicit and inspectable;
- supports continued MVP contract validation;
- minimizes runtime risk.

Risks:

- manual updates can become stale;
- coverage may remain incomplete;
- long-term source provenance may remain weak if not governed;
- data steward responsibility must be explicit.

Governance fit: safe as a temporary operating mode, not as a permanent strategy.

### 9.2 Option B — Provider-Assisted Prefill Scripts That Write to Governed CSV Artifacts

Description: create governed scripts outside the runtime decision path that prefill or refresh local source artifacts from an approved provider.

Advantages:

- preserves local deterministic pipeline reads;
- provider data becomes auditable through explicit artifacts;
- refreshes can be controlled and reviewed;
- source provenance and freshness metadata can be written into the CSV contract;
- easier to test than live runtime API calls.

Risks:

- still requires provider selection and governance;
- requires credential handling if a paid provider is used;
- requires deterministic error handling and refresh logs;
- prefill scripts must not introduce hidden ranking, scoring, or decision semantics.

Governance fit: strong candidate for the next governed stage after Level 2 design.

### 9.3 Option C — Fully Automated Provider/API Ingestion Inside the Pipeline

Description: call provider APIs directly during pipeline execution to fetch fundamentals and/or portfolio metadata.

Advantages:

- can reduce manual maintenance;
- may improve freshness if provider reliability is high;
- can support scalable ticker coverage.

Risks:

- runtime nondeterminism;
- provider outages and rate limits can affect Decision Engine inputs;
- harder reproducibility;
- credential and secret management required;
- freshness and provenance must be enforced at runtime;
- live provider semantics may pressure upstream layers to act as ranking or scoring layers.

Governance fit: not recommended as the next step. This requires stronger governance and explicit approval before runtime use.

### 9.4 Option D — Hybrid Model With External Ingestion Into Raw Artifacts and Deterministic Pipeline Reads

Description: fetch or prepare provider data outside the Decision Engine execution path, write it into governed raw/source artifacts, then run the deterministic pipeline against those local artifacts.

Advantages:

- combines automation with deterministic builds;
- preserves source artifacts as auditable contracts;
- supports controlled refresh cycles;
- keeps live provider calls away from Decision Engine execution;
- allows validation, freshness checks, and provenance checks before pipeline consumption.

Risks:

- requires clear refresh ownership;
- requires stale artifact handling;
- requires provider failure policy;
- requires artifact versioning or logging rules.

Governance fit: preferred long-term direction, provided it is designed through Level 2 governance first.

### 9.5 Option E — Research-Only Enrichment Separate From Decision Engine Inputs

Description: use provider or external metadata for research, diagnostics, or exploratory analysis without routing it into Decision Engine inputs.

Advantages:

- supports learning and data evaluation;
- avoids changing runtime decisions;
- can compare providers before approving operational ingestion;
- preserves current pipeline authority boundaries.

Risks:

- research outputs could be misread as operational signals;
- must remain clearly separate from Decision Engine inputs;
- must not influence Reporting or Telegram as if source-approved.

Governance fit: safe if explicitly labeled research-only and isolated from runtime decision artifacts.

## 10. Recommended Direction

Recommended direction: use a staged approach.

For now, keep the current manually maintained CSV contracts as stable source artifacts:

```text
data/raw/fundamentals.csv
data/portfolio/portfolio_metadata.csv
```

Then design provider-assisted or automated ingestion as a separate governed layer.

The preferred future architecture should preserve deterministic pipeline behavior by writing provider data into explicit raw/source artifacts first.

Runtime classification layers should continue reading local governed artifacts rather than relying on live API calls during Decision Engine execution, unless a later governance decision explicitly approves live runtime provider usage.

The next governance step should create a Level 2 design for automated or provider-assisted ingestion covering both fundamentals and portfolio metadata.

The Level 2 design should define at least:

1. source selection criteria;
2. source provenance fields;
3. freshness metadata;
4. refresh cadence;
5. failure behavior;
6. credential and secret boundaries;
7. raw artifact paths;
8. schema validation;
9. stale data handling;
10. partial coverage handling;
11. duplicate handling;
12. row identity;
13. logging and audit trail;
14. provider-assisted prefill versus runtime ingestion boundaries;
15. testing requirements;
16. forbidden semantics checks.

## 11. Implementation Authorization Boundary

This document does not authorize implementation.

No provider/API integration is authorized.

No credentials or secrets are authorized.

No runtime orchestration changes are authorized.

No sprint is closed or certified complete.

Future work requires Level 2 design before implementation.

The Decision Engine must not be loosened.

Reporting and Telegram must remain communication-only.

Ingestion must not introduce allocation, ranking, tradeability, urgency, conviction, hidden filtering, or Decision Engine bypass.

No generated artifact, processed CSV, runtime file, test file, script, workflow, or report change is authorized by this document.

## 12. Backlog Impact Assessment

New backlog item identified and added to `docs/sprints/project_backlog.md`:

- BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata

Backlog impact assessment:
- New backlog items identified and added to project_backlog.md

## 13. Recommended Next Step

Create a governed Level 2 design for automated or provider-assisted data ingestion covering fundamentals and portfolio metadata.

The Level 2 design should prefer the hybrid model: external/provider-assisted ingestion writes into governed raw/source artifacts first, and deterministic runtime layers read those local artifacts.

Implementation must not begin until the Level 2 design is reviewed and approved.
