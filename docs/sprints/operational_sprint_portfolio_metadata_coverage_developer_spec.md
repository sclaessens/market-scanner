# Operational Sprint Developer Specification - Portfolio Metadata Coverage MVP

## 1. Status and Scope

This document is a documentation-only developer specification for the Operational Sprint - Portfolio Metadata Coverage MVP.

This document does not implement:

- code
- tests
- CSV files
- generated artifacts
- provider integration
- API calls
- scraping
- credentials
- runtime behavior
- Decision Engine logic
- Reporting logic
- Telegram logic
- scanner logic
- Fundamental Layer logic
- portfolio source CSV edits
- portfolio metadata CSV edits

No sprint is closed or certified complete by this document.

No metadata source is approved by this document beyond what is already approved in planning.

Implementation requires a later explicit Codex implementation prompt after this developer specification is reviewed and approved.

## 2. Background

This developer specification follows `docs/sprints/operational_sprint_portfolio_metadata_coverage_planning.md`.

Relevant backlog items:

- `BL-0016 - Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0017 - Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0011 - Define and repair authoritative active portfolio source`

The Approved Fundamental Data Source MVP has been implemented and merged.

The Portfolio Source Contract Repair has been implemented and merged.

Post-merge validation showed:

- full test suite passed with `252 passed`;
- direct Fundamental builder succeeds;
- full pipeline runs end to end:
  `scanner -> validation -> context -> fundamentals -> timing -> portfolio -> portfolio intelligence -> final decisions -> reporting -> Telegram`;
- final decisions are no longer all `REVIEW`;
- `data/processed/fundamental_quality.csv` contains 291 rows with 4 `SUFFICIENT_DATA`, 2 `PARTIAL_DATA`, and 285 `INSUFFICIENT_DATA` rows;
- `data/processed/portfolio_intelligence.csv` contains 291 rows with 6 `portfolio_metadata_status = COMPLETE` rows and 285 `portfolio_metadata_status = PARTIAL` rows;
- `data/processed/final_decisions.csv` contains 6 `NO_ACTION` rows and 285 `REVIEW` rows;
- 285 rows are blocked by `portfolio_metadata_status = PARTIAL` and `arbitration_state = MISSING_METADATA`.

This sprint is about metadata coverage for the scanner and opportunity universe, not current position repair.

The current blocker is security reference metadata coverage for non-holding opportunity rows, not transaction-derived portfolio source repair.

## 3. Governance Boundaries

Portfolio metadata must remain descriptive.

Portfolio metadata must not create or imply:

- buy advice
- sell advice
- allocation authority
- tradeability
- urgency
- conviction
- ranking authority
- scoring authority
- eligibility
- hidden filtering
- Decision Engine bypass
- Reporting recommendation semantics
- Telegram recommendation semantics

Any portfolio action remains controlled by `scripts/core/decision_engine.py`.

Reporting and Telegram may communicate source-supported metadata only. They must not infer missing metadata or turn metadata into recommendations.

## 4. Contract Decision To Implement Later

Portfolio metadata should be split conceptually into five categories.

### 4.1 Current Portfolio Position Metadata

Current portfolio position metadata describes owned holdings only.

It may include:

- position status;
- quantity;
- cost basis;
- transaction-derived information.

It must not create action authority.

### 4.2 Security Reference Metadata

Security reference metadata describes any ticker in the scanner and opportunity universe.

It includes:

- sector;
- industry;
- asset class;
- optionally country;
- optionally region;
- optionally exchange;
- optionally currency.

It should be available for both holdings and non-holdings.

### 4.3 Portfolio Exposure Metadata

Portfolio exposure metadata describes how current holdings map to sector, industry, and asset-class exposure.

It must remain descriptive and source-supported.

### 4.4 Market / Valuation Enrichment

Market or valuation enrichment may include:

- last price;
- market value;
- unrealized profit or loss.

It must not be treated as transaction-derived source-of-truth unless separately governed.

### 4.5 Decision Outputs

Decision outputs include:

- actions;
- allocation decisions;
- trim decisions;
- add decisions;
- sell decisions.

They must remain Decision Engine-only.

The current blocker is security reference metadata coverage for the full scanner and opportunity universe, not current position repair.

## 5. Proposed Metadata Source Artifact Contract

Proposed MVP artifact path:

```text
data/portfolio/portfolio_metadata.csv
```

Read-only inspection for this specification found that `data/portfolio/portfolio_metadata.csv` already exists and is tracked by Git.

Read-only inspection also found that `data/portfolio/portfolio_metadata.csv` is currently used by `scripts/core/build_portfolio_intelligence.py` through `PORTFOLIO_METADATA_PATH`.

The current implementation requires:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

The planning document proposes these required columns:

- `ticker`
- `metadata_source`
- `metadata_freshness_date`
- `asset_class`
- `sector`
- `industry`

Future implementation must reconcile `metadata_freshness_date` with the currently implemented `metadata_last_updated` field before changing code or source data. Codex must not introduce unreviewed schema drift between planning, implementation, tests, and the tracked metadata artifact.

Proposed optional columns:

- `country`
- `region`
- `exchange`
- `currency`
- `portfolio_relevance`
- `metadata_notes`

The existing tracked artifact currently includes additional descriptive optional fields, including `sector_taxonomy`, `industry_group`, `country`, `region`, `exchange`, and `notes`.

The future implementation should read `data/portfolio/portfolio_metadata.csv` as a governed manual metadata source artifact unless a later prompt explicitly approves a different source path or provider-assisted workflow.

Because `data/portfolio/` is not ignored by `.gitignore`, future implementation must treat edits to `data/portfolio/portfolio_metadata.csv` as source-controlled data changes requiring explicit approval. Codex must inspect `.gitignore` before deciding what can be committed.

`portfolio_relevance`, if used, must remain descriptive only. It must not imply eligibility, priority, tradeability, urgency, conviction, allocation, ranking, or scoring.

Missing metadata must be classified explicitly, not inferred.

## 6. Portfolio Intelligence Output Contract

The expected output remains:

```text
data/processed/portfolio_intelligence.csv
```

The output must:

- remain row-preserving relative to the upstream timing and opportunity universe;
- preserve ticker/date row identity;
- preserve current portfolio annotations where applicable;
- join security reference metadata by ticker;
- explicitly classify metadata as complete, partial, missing, stale, unsupported, or source error;
- not drop rows when metadata is missing;
- not multiply rows when metadata has duplicate ticker entries;
- not introduce allocation or decision semantics;
- not convert missing metadata into hidden filtering.

Expected output concepts to inspect and reconcile:

- `portfolio_metadata_status`
- sector metadata fields
- industry metadata fields
- asset class metadata fields
- metadata source/provenance fields
- metadata freshness/status fields
- current portfolio position descriptors
- source/provenance descriptors

Read-only inspection found that the current `PORTFOLIO_COLUMNS` appended by `scripts/core/build_portfolio_intelligence.py` include `portfolio_metadata_status`, `portfolio_metadata_reason`, `portfolio_source_provenance`, and `portfolio_classification_rationale`, but do not currently append raw sector, industry, asset class, source, or freshness columns to `portfolio_intelligence.csv`.

Future implementation must inspect the current schema before changing output fields. Any schema change must be explicitly documented and covered by focused tests.

## 7. Coverage States

Candidate coverage states:

- `COMPLETE`
- `PARTIAL`
- `MISSING`
- `STALE`
- `UNSUPPORTED`
- `SOURCE_ERROR`

Future implementation must reconcile these candidate states with the current implementation before changing code.

Read-only inspection found current Portfolio Intelligence states and classifications including:

- `COMPLETE`
- `PARTIAL`
- `MISSING`
- internal metadata classification states such as `INVALID` and `STALE`

Suggested semantics:

- `COMPLETE`: required metadata fields are present, fresh, and valid.
- `PARTIAL`: metadata row exists but one or more required descriptive fields are missing or incomplete.
- `MISSING`: no metadata row exists for the ticker.
- `STALE`: metadata exists but freshness rules fail.
- `UNSUPPORTED`: ticker or security type is not supported by the current metadata source.
- `SOURCE_ERROR`: metadata source artifact exists but fails structural validation.

If a candidate state is not currently represented in the output contract, implementation must avoid adding it unless the contract impact is explicitly approved and tested.

## 8. Classification Rules

Future implementation must use deterministic classification rules.

### 8.1 Metadata File Missing

If `data/portfolio/portfolio_metadata.csv` is missing:

- preserve row count;
- classify rows as missing or partial according to the existing contract;
- do not fail if missing-source fallback is already established and intended.

### 8.2 Required Columns Missing

If required metadata columns are missing:

- fail fast with a clear English-only error;
- do not produce misleading metadata.

### 8.3 Duplicate Ticker Metadata Rows

If duplicate ticker metadata rows exist:

- fail fast unless the contract explicitly supports multiple dated metadata rows;
- do not silently pick one row.

### 8.4 Ticker Missing From Metadata Source

If an upstream ticker is missing from the metadata source:

- preserve the upstream row;
- classify as `MISSING` or the existing equivalent;
- do not drop or filter the row.

### 8.5 Required Metadata Fields Missing

If a metadata row exists but required metadata fields are missing:

- classify as `PARTIAL`;
- do not infer sector, industry, asset class, source, or freshness values.

### 8.6 Metadata Freshness Missing

If metadata freshness is missing:

- do not treat it as fresh;
- classify as partial, stale, missing, invalid, or source error according to deterministic rules.

### 8.7 Metadata Stale

If metadata is stale:

- classify as `STALE` or the existing equivalent;
- do not treat stale metadata as complete.

### 8.8 Future Freshness Date

If metadata freshness date is in the future:

- fail fast or classify as source error according to the approved contract;
- do not accept future metadata freshness as valid.

### 8.9 Valid Metadata Row

If a metadata row is valid:

- classify as `COMPLETE`;
- preserve source/provenance evidence;
- do not create action semantics.

## 9. Freshness and Provenance Rules

Minimum freshness and provenance requirements:

- `metadata_source` is required for `COMPLETE`.
- Metadata freshness date is required for `COMPLETE`.
- A deterministic stale threshold must be configurable or defined as a named constant.
- Missing freshness must not be treated as fresh.
- Invalid dates must fail or classify explicitly.
- Future dates must fail or classify explicitly.
- Stale metadata must not be classified as complete.
- Source/provenance values must not contain credentials or secrets.

Read-only inspection found a current `METADATA_STALE_THRESHOLD_DAYS = 365` constant in `scripts/core/build_portfolio_intelligence.py`.

Future implementation must reconcile the planning term `metadata_freshness_date` with current `metadata_last_updated` behavior before changing source artifacts or code.

## 10. Row Identity and Merge Rules

Future implementation must guarantee:

- one output row per upstream ticker/date row;
- no row loss;
- no row multiplication;
- no hidden filtering;
- no hidden priority sorting;
- no allocation semantics;
- no tradeability semantics;
- upstream order preserved unless an existing implementation defines deterministic ordering;
- fail fast on duplicate metadata ticker rows if the current-source contract expects one row per ticker;
- holdings and non-holdings distinction preserved.

The metadata artifact is not an opportunity source and must not expand the scanner or opportunity universe.

## 11. Files Codex Is Expected To Inspect Before Implementation

Before implementing, Codex should inspect:

- `scripts/core/build_portfolio_intelligence.py`
- tests related to portfolio intelligence and portfolio metadata under `tests/core/` and `tests/portfolio/`
- `data/portfolio/portfolio_metadata.csv` if present
- `data/portfolio/portfolio_positions.csv`
- `data/portfolio/portfolio_transactions.csv`
- `data/processed/portfolio_intelligence.csv`
- `data/processed/final_decisions.csv`
- `docs/sprints/project_backlog.md`
- `docs/sprints/operational_sprint_portfolio_metadata_coverage_planning.md`
- `.gitignore`

This developer specification describes expected touchpoints only. It does not authorize runtime file changes.

## 12. Future Implementation Scope

Likely allowed in a later implementation prompt:

- `scripts/core/build_portfolio_intelligence.py`
- focused tests under `tests/core/` or `tests/portfolio/`
- possibly a small test fixture
- targeted contract documentation if implementation clarifies the contract

Likely not allowed unless explicitly approved:

- Decision Engine logic
- Reporting logic
- Telegram logic
- scanner logic
- Fundamental Layer logic
- generated processed CSV commits
- portfolio source CSV edits without approval
- provider/API integration
- credentials/secrets
- workflows
- analyst expectations logic

If implementation pressure suggests editing `data/portfolio/portfolio_metadata.csv`, Codex must confirm that explicit source-data edit authorization exists before proceeding.

## 13. Required Tests For Future Implementation

Future implementation must add or update tests for:

- metadata source file missing;
- required metadata columns missing;
- valid metadata-supported row;
- ticker missing from metadata source;
- required metadata field missing;
- stale metadata freshness date;
- future metadata freshness date;
- duplicate ticker metadata rows;
- row preservation;
- holdings/non-holdings distinction preserved;
- metadata enrichment does not introduce action semantics;
- forbidden semantics not introduced.

Tests must verify no fields or values introduce:

- buy advice
- sell advice
- allocation
- tradeability
- urgency
- conviction
- eligibility
- ranking authority
- scoring authority
- hidden filtering
- Decision Engine bypass
- Reporting recommendation semantics
- Telegram recommendation semantics

## 14. Validation Plan For Future Implementation

Future implementation should run:

```bash
git diff --check
PYTHONPATH=. .venv/bin/python -m pytest tests/core -k "portfolio"
PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio
PYTHONPATH=. .venv/bin/python -m pytest
git status --short
```

If safe and in scope, future implementation may also run:

```bash
PYTHONPATH=. .venv/bin/python scripts/core/build_portfolio_intelligence.py
```

Full pipeline validation should run only if the implementation scope approves generated artifact handling:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_full_pipeline.py
```

Generated artifacts must not be committed unless explicitly approved.

## 15. Acceptance Criteria

Future implementation is acceptable only if:

- approved metadata source contract is documented;
- `portfolio_metadata.csv` contract is reconciled with existing implementation;
- valid metadata rows can improve `portfolio_metadata_status`;
- missing metadata remains explicit;
- stale metadata is handled explicitly;
- partial metadata is handled explicitly;
- malformed metadata is handled explicitly;
- output remains row-preserving;
- no duplicate ticker/date rows are introduced;
- no hidden filtering is introduced;
- no upstream allocation semantics are introduced;
- no upstream tradeability semantics are introduced;
- no Decision Engine authority leakage is introduced;
- no Reporting semantic drift is introduced;
- no Telegram semantic drift is introduced;
- focused tests pass;
- full test suite passes;
- runtime artifacts are not improperly committed.

## 16. Implementation Prompt Skeleton

Future use only. Do not execute this prompt until this developer specification is reviewed and approved.

```text
You are operating inside the institutional `market-scanner` repository.

Task:
Implement the Operational Sprint - Portfolio Metadata Coverage MVP.

Follow the approved developer specification:

- `docs/sprints/operational_sprint_portfolio_metadata_coverage_developer_spec.md`

This is a governed implementation task.

Do NOT modify:
- Decision Engine logic;
- Reporting logic;
- Telegram logic;
- scanner logic;
- Fundamental Layer logic;
- provider/API integration;
- credentials or secrets;
- generated processed CSV files;
- workflows;
- portfolio source CSV values unless explicitly approved.

Goal:
Improve source-supported Portfolio Intelligence metadata coverage for the scanner/opportunity universe while keeping portfolio metadata descriptive only.

Required behavior:
- preserve one output row per upstream ticker/date row;
- join security reference metadata by ticker only;
- classify complete, partial, missing, stale, unsupported, or source-error metadata deterministically;
- do not infer missing metadata;
- do not add allocation, tradeability, urgency, conviction, ranking, scoring, eligibility, hidden filtering, or Decision Engine bypass semantics;
- reconcile `metadata_freshness_date` planning terminology with current `metadata_last_updated` implementation before changing code or data.

Before editing:
- run `git status`;
- inspect `scripts/core/build_portfolio_intelligence.py`;
- inspect existing portfolio metadata tests;
- inspect `data/portfolio/portfolio_metadata.csv`;
- inspect `.gitignore`;
- inspect the current `portfolio_intelligence.csv` and `final_decisions.csv` schemas if present.

After implementation:
- run `git diff --check`;
- run focused portfolio metadata / Portfolio Intelligence tests;
- run `PYTHONPATH=. .venv/bin/python -m pytest tests/portfolio`;
- run `PYTHONPATH=. .venv/bin/python -m pytest`;
- run `git status --short`;
- do not commit generated artifacts unless explicitly approved.
```

## 17. Backlog Impact Assessment

Existing backlog items remain sufficient.

`BL-0016` remains sufficient for the portfolio metadata and sector exposure contract.

`BL-0017` remains sufficient for future provider-assisted or automated ingestion.

`BL-0011` remains related for authoritative active portfolio source repair, but this developer specification does not reopen active position repair.

Backlog impact assessment:
- No new backlog items identified.

## 18. Recommended Next Step

Review and merge this developer specification.

After review approval, launch a separate Codex implementation sprint for the Portfolio Metadata Coverage MVP.
