# Operational Sprint Planning — Portfolio Metadata Coverage MVP

## 1. Status and Scope

This document is a documentation-only PM / Functional Analyst / Technical Analyst / Governance planning artifact.

It prepares the Portfolio Metadata Coverage Repair / Expansion workstream.

This document does not implement:

- code;
- tests;
- CSV files;
- generated artifacts;
- provider integration;
- API calls;
- scraping;
- credentials;
- runtime behavior;
- Decision Engine logic;
- Reporting logic;
- Telegram logic;
- scanner logic;
- Fundamental Layer logic;
- portfolio source CSV edits;
- portfolio metadata CSV edits.

No sprint is closed or certified complete by this document.

No metadata source is approved by this document.

No CSV edit is authorized by this document.

## 2. Background

The Approved Fundamental Data Source MVP has been implemented and merged.

The Portfolio Source Contract Repair has also been implemented and merged.

Post-merge operational validation showed:

- full test suite: `252 passed`;
- direct Fundamental builder succeeds;
- full pipeline succeeds end-to-end:
  `scanner -> validation -> context -> fundamentals -> timing -> portfolio -> portfolio intelligence -> final decisions -> reporting -> Telegram`;
- fundamentals are source-supported for 6 rows;
- final decisions are no longer all `REVIEW`;
- 285 rows remain `REVIEW`;
- the dominant blocker is portfolio metadata coverage;
- `portfolio_metadata_status = PARTIAL` for 285 rows;
- `arbitration_state = MISSING_METADATA` for 285 rows.

Relevant backlog item:

- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`.

Related backlog items:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`;
- `BL-0011 — Define and repair authoritative active portfolio source`.

This planning document treats the post-Fundamental-MVP validation result as operational validation input.

## 3. Governance Boundaries

Portfolio metadata must remain descriptive.

Portfolio metadata must not create or imply:

- buy/sell advice;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- ranking authority;
- scoring authority;
- eligibility;
- hidden filtering;
- Decision Engine bypass;
- Reporting recommendation semantics;
- Telegram recommendation semantics.

Any portfolio action remains Decision Engine-controlled.

Reporting and Telegram may communicate source-supported metadata, but they must not infer missing metadata or turn metadata into recommendations.

## 4. Problem Statement

The pipeline now has source-supported fundamentals for some rows, but most final decisions remain `REVIEW` because portfolio metadata is incomplete for most scanner-universe rows.

The current metadata coverage appears sufficient only for the 6 locally source-supported rows and missing or partial for 285 rows.

The system needs a governed metadata source contract that can describe the scanner universe consistently enough for Portfolio Intelligence and Decision Engine inputs.

Key design question:

Should portfolio metadata coverage apply only to current portfolio holdings, or to the full scanner/opportunity universe?

Recommended answer:

For Decision Engine input quality, metadata required by Portfolio Intelligence must cover the full upstream opportunity universe, not only current holdings, while clearly distinguishing:

- current portfolio position metadata;
- security reference metadata;
- sector/industry/asset-class metadata;
- valuation or price enrichment;
- decision authority.

## 5. Recommended Contract Decision

Portfolio metadata should be split conceptually into five categories.

### 5.1 Current Portfolio Position Metadata

Current portfolio position metadata describes owned holdings only.

It may include:

- position status;
- quantity;
- cost basis;
- transaction-derived information.

It must not create action authority.

### 5.2 Security Reference Metadata

Security reference metadata describes any ticker in the scanner/opportunity universe.

It should include:

- sector;
- industry;
- asset class;
- country or region if needed.

It should be available for both holdings and non-holdings.

### 5.3 Portfolio Exposure Metadata

Portfolio exposure metadata describes how current holdings map to sector, industry, and asset-class exposure.

It should be source-supported and descriptive.

### 5.4 Market / Valuation Enrichment

Market or valuation enrichment may include:

- last price;
- market value;
- unrealized profit/loss.

It must not be treated as transaction-derived source-of-truth unless separately governed.

### 5.5 Decision Outputs

Decision outputs include:

- actions;
- allocation decisions;
- trim/add/sell decisions.

They must remain Decision Engine-only.

Current blocker conclusion:

The current blocker is most likely security reference metadata coverage, not current position repair.

## 6. Candidate Metadata Source Artifact Contract

Proposed manual governed MVP artifact path:

```text
data/portfolio/portfolio_metadata.csv
```

If the repository already uses this path, the future developer specification must inspect and reconcile the existing contract before implementation.

This planning document does not edit the file.

Proposed required columns for security reference metadata:

- `ticker`;
- `metadata_source`;
- `metadata_freshness_date`;
- `asset_class`;
- `sector`;
- `industry`.

Optional columns:

- `country`;
- `region`;
- `exchange`;
- `currency`;
- `portfolio_relevance`;
- `metadata_notes`.

If used, `portfolio_relevance` must be descriptive only.

It must not imply:

- eligibility;
- priority;
- tradeability;
- allocation.

Missing metadata must be classified explicitly, not inferred.

## 7. Coverage Requirements

Minimum expected coverage:

- every upstream ticker/date row entering Portfolio Intelligence should be able to join to security reference metadata by ticker;
- no upstream rows should be dropped if metadata is missing;
- missing metadata should produce explicit descriptive states;
- metadata coverage should be reported as counts and distributions;
- current portfolio holdings should be separately identifiable from non-holdings;
- holdings metadata and security reference metadata must not be conflated.

Candidate coverage states:

- `COMPLETE`;
- `PARTIAL`;
- `MISSING`;
- `STALE`;
- `UNSUPPORTED`;
- `SOURCE_ERROR`.

The future developer specification must reconcile these with existing implementation states.

## 8. Freshness and Provenance Requirements

Required source provenance and freshness controls:

- metadata source name or reference;
- metadata freshness date;
- deterministic freshness rule;
- stale classification;
- no future freshness dates;
- no silent inference;
- no hidden fallback to unknown sector or industry unless explicitly classified.

`metadata_freshness_date` must be required for `COMPLETE` status.

Missing freshness must not be treated as fresh.

## 9. Portfolio Intelligence Output Expectations

After future implementation, `data/processed/portfolio_intelligence.csv` should:

- remain row-preserving relative to the upstream timing/opportunity universe;
- preserve current portfolio annotations where applicable;
- add or preserve descriptive security metadata;
- explicitly classify missing, partial, and stale metadata;
- not introduce allocation or decision semantics;
- not drop rows;
- not create duplicate ticker/date rows;
- not convert missing metadata into hidden filtering.

Expected inspected fields may include:

- `portfolio_metadata_status`;
- sector/industry metadata fields;
- asset class metadata fields;
- metadata freshness/status fields;
- portfolio position descriptors;
- source/provenance descriptors.

The future developer specification must inspect the current schema before approving changes.

## 10. Decision Engine Boundary

The Decision Engine may consume descriptive portfolio metadata only if already governed.

Portfolio metadata must not decide actions upstream.

Any action outcomes remain in:

- `data/processed/final_decisions.csv`;
- Decision Engine-owned fields only.

Metadata-related blockers may result in `REVIEW`, but the upstream metadata layer must not create buy/sell/hold/trim/add decisions.

## 11. Candidate Repair Options

### Option A — Manual Governed Metadata CSV MVP

Use or expand `data/portfolio/portfolio_metadata.csv` as a source-supported metadata artifact for the scanner universe.

Pros:

- auditable;
- simple;
- no credentials;
- fast to unblock;
- deterministic testing.

Cons:

- manual upkeep;
- freshness discipline needed;
- scalability limitations.

### Option B — Provider-Assisted Metadata Export

Use a provider/platform export, transformed into governed CSV.

Pros:

- broader coverage;
- more scalable than manual entry.

Cons:

- terms/export rights;
- source provenance;
- repeatability;
- possible credentials.

### Option C — Direct API Metadata Integration

Use an automated metadata source.

Pros:

- scalable;
- refreshable.

Cons:

- credentials;
- rate limits;
- provider failure handling;
- nondeterminism;
- governance overhead.

Recommended planning direction:

Start with Option A, a manual governed metadata CSV MVP, while keeping provider-assisted ingestion as future work under `BL-0017`.

## 12. Recommended Sprint Shape

Recommended sprint name:

```text
Operational Sprint — Portfolio Metadata Coverage MVP
```

Recommended scope:

1. inspect current portfolio metadata source contract;
2. define required security reference metadata fields;
3. define coverage states;
4. define freshness/provenance rules;
5. define join behavior into Portfolio Intelligence;
6. preserve row identity and row count;
7. classify missing metadata explicitly;
8. keep portfolio metadata descriptive only;
9. prevent Decision Engine, Reporting, and Telegram authority leakage;
10. prepare Codex developer specification.

## 13. Files Codex Should Inspect Later

Future developer-spec and implementation steps should inspect:

- `scripts/core/build_portfolio_intelligence.py`;
- tests related to portfolio intelligence and portfolio metadata;
- `data/portfolio/portfolio_metadata.csv` if present;
- `data/portfolio/portfolio_positions.csv`;
- `data/portfolio/portfolio_transactions.csv`;
- `data/processed/portfolio_intelligence.csv`;
- `data/processed/final_decisions.csv`;
- `docs/sprints/project_backlog.md`;
- `docs/sprints/operational_sprint_portfolio_metadata_coverage_planning.md`;
- `.gitignore`.

## 14. Future Implementation Scope

Likely allowed in a future implementation step only:

- `scripts/core/build_portfolio_intelligence.py`;
- focused tests under `tests/core/` or `tests/portfolio/`;
- possibly a small test fixture;
- contract documentation if needed.

Not allowed unless explicitly approved:

- Decision Engine logic;
- Reporting logic;
- Telegram logic;
- scanner logic;
- Fundamental Layer logic;
- generated processed CSV commits;
- portfolio source CSV edits without approval;
- provider/API integration;
- credentials/secrets;
- workflows.

## 15. Acceptance Criteria for Future Implementation

Future implementation may be accepted only if:

- scanner/opportunity universe metadata coverage is defined;
- security reference metadata can classify at least provided source rows;
- missing metadata remains explicit;
- no row loss occurs;
- no duplicate ticker/date rows occur;
- `portfolio_metadata_status` distribution improves when valid metadata is provided;
- rows without metadata remain `REVIEW` or equivalent only through Decision Engine-controlled logic;
- no upstream allocation/tradeability semantics are introduced;
- focused tests pass;
- full test suite passes;
- generated artifacts are not improperly committed.

## 16. Backlog Impact Assessment

Existing backlog items remain sufficient.

`BL-0016` remains sufficient for the portfolio metadata and sector exposure contract.

`BL-0017` remains sufficient for future provider-assisted or automated ingestion.

`BL-0011` remains related for authoritative active portfolio source repair, but this planning document does not reopen active position repair.

Backlog impact assessment:
- No new backlog items identified.

## 17. Recommended Next Step

Create a Codex developer-spec preparation prompt for:

```text
Operational Sprint — Portfolio Metadata Coverage MVP
```

Do not implement yet.

After developer specification review and merge, launch a separate implementation sprint if approved.
