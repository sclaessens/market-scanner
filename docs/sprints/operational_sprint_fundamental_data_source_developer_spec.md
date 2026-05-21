# Operational Sprint Developer Specification - Approved Fundamental Data Source MVP

## 1. Status and Scope

This document is a documentation-only developer specification for the Operational Sprint - Approved Fundamental Data Source MVP.

This document does not implement:

- code changes
- tests
- data changes
- CSV files
- generated artifacts
- provider integration
- API calls
- scraping
- credentials or secrets
- runtime behavior changes
- Decision Engine logic
- Reporting logic
- Telegram logic

No sprint is closed or certified complete by this document.

No fundamental data source is approved by this document unless that source has already been approved in existing governance documentation.

Implementation requires a later explicit Codex implementation prompt after this developer specification is reviewed and approved.

## 2. Background

This developer specification follows `docs/sprints/operational_sprint_fundamental_data_source_planning.md`.

Relevant backlog items:

- `BL-0015 - Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 - Define governed automated data ingestion strategy for fundamentals and portfolio metadata`

The pipeline can run technically end to end, but final decisions have remained constrained because the Fundamental Layer historically lacked an approved real fundamental data source.

Previous investigation found:

- `data/processed/fundamental_quality.csv` contained `291` rows.
- all rows had `quality_state = INSUFFICIENT_DATA`.
- all rows had `quality_metadata_status = source_missing`.
- all rows had `source_data_status = source_missing`.
- `missing_fundamentals_count = 291`.
- `partial_data_count = 0`.
- `stale_data_count = 0`.

That state showed that the Fundamental Layer could preserve rows and classify source absence, but could not yet provide source-supported fundamental quality classifications.

Real fundamental data source support is now higher priority than analyst expectations provider work because the current blocking issue is source-data coverage for required fundamentals, not analyst target or expectations research. Analyst expectations work remains observational research and should not distract from the approved source-data foundation required for Fundamental Layer classification.

## 3. Governance Boundaries

The Fundamental Layer remains descriptive and classification-only.

It must not produce or imply:

- buy advice
- sell advice
- allocation
- tradeability
- urgency
- conviction
- hidden filtering
- Decision Engine bypass
- Reporting decision semantics
- Telegram decision semantics

Any downstream action use remains controlled by `scripts/core/decision_engine.py`.

The Decision Engine remains the only allocation authority.

Reporting and Telegram remain communication-only.

## 4. Recommended MVP Source Path

The recommended MVP implementation path is:

1. governed manual raw fundamentals CSV artifact first;
2. provider/API integration deferred;
3. hybrid provider-assisted path allowed later only after explicit source governance.

The recommended raw artifact path is:

```text
data/raw/fundamentals.csv
```

This path is safer than starting with provider/API integration because it:

- requires no credentials;
- is easier to audit locally;
- supports deterministic local testing;
- validates the raw source contract before automation;
- reduces provider terms and licensing risk;
- avoids live network or rate-limit failure modes;
- unblocks Fundamental Layer classification faster;
- preserves a clean boundary between source data and Decision Engine authority.

Provider/API integration should remain deferred until source terms, access, credentials, rate limits, caching, auditability, and failure handling are governed separately.

## 5. Proposed Raw Source Artifact Contract

Proposed raw input artifact path:

```text
data/raw/fundamentals.csv
```

Read-only inspection for this specification found that a local `data/raw/fundamentals.csv` file exists in this workspace. This developer specification does not authorize editing, committing, or treating that local file as an approved source artifact.

Read-only inspection also found that `.gitignore` ignores:

```text
data/raw/
```

Future Codex implementation must inspect `.gitignore` before deciding whether any raw source artifact, fixture, or template may be committed. The default expectation is that live `data/raw/fundamentals.csv` remains local/operator-managed and should not be committed unless a future prompt explicitly authorizes it.

If the file is absent during implementation, the builder must preserve the existing safe fallback behavior.

If the file is created during implementation or data-preparation work, that creation must be separately authorized.

### 5.1 Required Identity and Provenance Columns

Minimum proposed required identity and provenance columns:

- `ticker`
- `as_of_date`
- `source_name`
- `source_reference`
- `source_freshness_date`

These columns identify the source row, the date represented by the source data, and the provenance/freshness evidence required for auditability.

Future implementation must reconcile these proposed names with the current implementation before modifying code. Read-only inspection found current Fundamental Layer code references `source_last_updated` rather than `source_freshness_date`, and current tests exercise `source_last_updated`. Codex must not introduce an unreviewed schema drift between this specification, existing tests, and runtime code.

### 5.2 Optional Metric Columns

Proposed optional metric columns:

- `currency`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `free_cash_flow_margin`
- `debt_to_equity`
- `current_ratio`
- `return_on_equity`
- `return_on_invested_capital`
- `fundamental_notes`

Optional metric columns are descriptive source fields only. They must not create allocation, ranking, tradeability, urgency, conviction, eligibility, or hidden filtering semantics.

Future implementation must define which metric columns are required for `SUFFICIENT_DATA` and which remain optional descriptive enrichment. The current implementation may already require a narrower metric set; Codex must reconcile that existing behavior before changing any contract.

## 6. Fundamental Quality Output Contract

After implementation, `data/processed/fundamental_quality.csv` must remain row-preserving relative to the approved upstream universe.

Expected output concepts include:

- `quality_state`
- `quality_reason`
- `quality_metadata_status`
- `source_data_status`
- `source_freshness_status`
- `missing_fundamentals_count`
- `partial_data_count`
- `stale_data_count`

Candidate states:

- `SUFFICIENT_DATA`
- `PARTIAL_DATA`
- `STALE_DATA`
- `INSUFFICIENT_DATA`
- `SOURCE_UNSUPPORTED`
- `SOURCE_ERROR`

Codex must reconcile these candidate names with the current implementation before modifying code. Read-only inspection found existing output fields and tests for `quality_state`, `quality_reason`, `quality_metadata_status`, `source_data_status`, `source_freshness_days`, `missing_required_fields`, `partial_data_reason`, `stale_data_reason`, and `invalid_data_reason`.

If a candidate state or field is not currently supported, implementation must either:

- avoid adding it unless explicitly approved; or
- document the contract impact before adding it.

## 7. Classification Rules

Classification rules must be deterministic and auditable.

### 7.1 Source File Missing

If `data/raw/fundamentals.csv` is missing, preserve every upstream row and classify absence safely.

Expected behavior:

- `quality_state = INSUFFICIENT_DATA`
- `quality_metadata_status = source_missing`
- `source_data_status = source_missing`

### 7.2 Required Columns Missing

If required identity or provenance columns are missing, fail fast with a deterministic English-only error.

The builder must not silently continue with ambiguous source data.

### 7.3 Ticker Missing From Source

If an upstream ticker/date row has no matching approved raw source row, preserve the upstream row and classify the source row as missing or insufficient.

The builder must not drop the row.

### 7.4 Provenance Metadata Missing

If source provenance metadata is missing, the source row must not be treated as fully supported.

Depending on final contract reconciliation, this should classify as `INSUFFICIENT_DATA`, `PARTIAL_DATA`, `SOURCE_ERROR`, or an existing equivalent invalid/partial state.

### 7.5 Source Freshness Missing

If source freshness metadata is missing, the row must not be treated as fresh.

Missing freshness must classify explicitly and must not be silently accepted.

### 7.6 Metric Fields Partially Missing

If some metric fields are present and valid but one or more required metrics are missing, classify as `PARTIAL_DATA` or the reconciled equivalent.

Missing metrics must be represented in a deterministic missing-field field such as `missing_required_fields`.

### 7.7 Source Data Stale

If source data exists but violates the approved freshness threshold, classify as `STALE_DATA` or the reconciled equivalent.

Stale rows must not be silently accepted as sufficient.

### 7.8 Source Row Malformed

Malformed dates, malformed numeric values, invalid boolean values, or unreadable rows must classify explicitly or fail fast according to the final approved contract.

Malformed source rows must not be silently coerced into complete data.

### 7.9 Duplicate Ticker Keys

If the contract expects one current source row per ticker, duplicate ticker rows must fail fast.

If the contract uses ticker plus `as_of_date`, duplicate raw source rows for the same ticker and `as_of_date` must fail fast.

The implementation must make the row identity explicit before changing code.

### 7.10 Valid Source-supported Row

If an upstream row has a matching source row, required identity/provenance fields are present, freshness is valid, required metrics are present and valid, and no duplicate ambiguity exists, classify as `SUFFICIENT_DATA` or the reconciled equivalent.

## 8. Freshness Rules

`source_freshness_date` or its reconciled current equivalent is required for `SUFFICIENT_DATA`.

The stale threshold should be configurable or clearly documented. Read-only inspection found the current Fundamental Layer references a `120` day stale threshold. Future implementation must confirm whether that remains approved before changing it.

Freshness rules:

- stale rows should become `STALE_DATA`, not silently accepted;
- missing freshness should not be treated as fresh;
- malformed freshness dates should fail validation or classify explicitly;
- future freshness dates should fail validation or classify explicitly;
- freshness calculations must use deterministic calendar-day logic;
- source freshness must remain descriptive and must not imply allocation readiness.

## 9. Row Identity and Merge Rules

Future implementation must require:

- one output row per upstream ticker/date row;
- no row loss;
- no row multiplication;
- no hidden filtering;
- deterministic handling of missing source rows;
- fail-fast duplicate handling for the approved raw source identity;
- preservation of upstream order unless the existing implementation defines deterministic ordering.

The current upstream input is expected to remain `data/processed/context_strength.csv` unless a later prompt explicitly changes that contract.

Raw-only tickers must not create new Fundamental Layer output rows.

## 10. Files Codex Is Expected To Inspect Before Implementation

Before implementation, Codex must inspect:

- `scripts/core/build_fundamental_layer.py`
- relevant tests under `tests/core/`, especially `tests/core/test_build_fundamental_layer.py`
- `data/processed/fundamental_quality.csv`
- `data/processed/context_strength.csv`
- `data/processed/timing_state_layer.csv`, if downstream timing contract impact is relevant
- `docs/sprints/project_backlog.md`
- `docs/sprints/operational_sprint_fundamental_data_source_planning.md`
- `.gitignore`
- any existing raw fundamentals file, if present

Expected touchpoints from read-only inspection:

- `scripts/core/build_fundamental_layer.py` already references `data/raw/fundamentals.csv`.
- `tests/core/test_build_fundamental_layer.py` already contains focused Fundamental Layer tests.
- `.gitignore` ignores `data/raw/`.
- A local `data/raw/fundamentals.csv` may exist but should not be committed without explicit approval.

This developer specification does not modify those files.

## 11. Future Implementation Scope

Likely allowed during a later implementation prompt only:

- `scripts/core/build_fundamental_layer.py`
- focused tests under `tests/core/`
- possibly a small fixture file under `tests/`, if needed
- documentation update if implementation changes the final contract

Likely not allowed unless explicitly approved:

- Decision Engine logic
- Reporting logic
- Telegram logic
- scanner logic
- validation, context, timing, or portfolio intelligence runtime logic
- generated processed CSV files
- portfolio files
- watchlist files
- GitHub Actions workflows
- provider/API credentials
- live API integrations
- committed live `data/raw/fundamentals.csv`

## 12. Required Tests For Future Implementation

Future implementation must add or update focused tests for:

- raw source file missing;
- required columns missing;
- source metadata missing;
- valid source-supported row;
- partially missing metrics;
- stale freshness date;
- future freshness date;
- duplicate ticker rows or duplicate ticker/date rows, depending on approved identity;
- unsupported ticker;
- row preservation;
- raw-only tickers not creating output rows;
- forbidden semantics not introduced.

Tests must verify that no upstream tradeability, allocation, urgency, conviction, buy/sell action, eligibility, or hidden filtering fields are introduced.

Tests must verify that the Fundamental Layer does not import or call Decision Engine, Reporting, or Telegram modules.

## 13. Validation Plan For Future Implementation

The following commands are for a future implementation step only. They should not be run as part of this documentation-only specification task.

Recommended future validation:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/core/test_build_fundamental_layer.py
PYTHONPATH=. .venv/bin/python -m pytest
PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py
git diff --check
git status --short
```

If the implementation prompt explicitly approves pipeline validation, also run the full pipeline command used by current operations and inspect generated artifacts.

Generated artifact inspection must not result in committing generated outputs unless those outputs are already tracked and explicitly approved for the implementation.

Codex must review the diff and confirm no unauthorized files changed.

## 14. Acceptance Criteria

Future implementation is acceptable only if:

- approved raw source contract is documented;
- builder consumes the approved source path or classifies absence deterministically;
- valid source rows can produce non-`INSUFFICIENT_DATA` states;
- missing source still produces deterministic safe classification;
- stale data is handled explicitly;
- partial data is handled explicitly;
- malformed data is handled explicitly;
- output remains row-preserving;
- raw-only tickers do not create output rows;
- no hidden filtering is introduced;
- no Decision Engine authority leakage is introduced;
- no Reporting or Telegram semantic drift is introduced;
- no provider/API integration is introduced unless separately approved;
- no credentials or secrets are added;
- tests pass;
- runtime artifacts are not improperly committed.

## 15. Implementation Prompt Skeleton

The following prompt skeleton is for future use only.

Implementation must start only after this developer specification is reviewed and approved.

```text
You are operating inside the institutional market-scanner repository.

Task:
Implement the approved Fundamental Data Source MVP according to:
docs/sprints/operational_sprint_fundamental_data_source_developer_spec.md

This is an implementation task only after human approval of the developer specification.

Before editing:
- run git status;
- confirm the working tree is clean;
- create an implementation branch;
- read AGENTS.md;
- read docs/sprints/operational_sprint_fundamental_data_source_developer_spec.md;
- inspect scripts/core/build_fundamental_layer.py;
- inspect tests/core/test_build_fundamental_layer.py;
- inspect .gitignore;
- inspect whether data/raw/fundamentals.csv exists locally.

Scope:
- keep the Fundamental Layer descriptive/classification-only;
- preserve row count and upstream row identity;
- preserve missing-source fallback behavior;
- reconcile proposed source columns with current implementation before changing code;
- add or update focused Fundamental Layer tests.

Allowed files:
- scripts/core/build_fundamental_layer.py;
- tests/core/test_build_fundamental_layer.py;
- test fixtures under tests/ if needed;
- documentation only if final contract differs from the approved spec.

Forbidden:
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- scanner changes;
- portfolio or watchlist changes;
- provider/API calls;
- credentials or secrets;
- generated processed CSV commits;
- live data/raw/fundamentals.csv commits unless explicitly approved;
- ranking, scoring, tradeability, urgency, conviction, allocation, eligibility, or hidden filtering semantics.

Validation:
- run focused Fundamental Layer tests;
- run full pytest if feasible;
- run git diff --check;
- inspect generated artifacts without committing unauthorized outputs.

Final report:
- files changed;
- tests run;
- confirmation that output is row-preserving;
- confirmation that no Decision Engine, Reporting, Telegram, provider/API, credentials, or generated artifact changes were made;
- any unresolved contract questions.
```

## 16. Backlog Impact Assessment

`BL-0015` and `BL-0017` remain sufficient.

`BL-0015` covers the approved Fundamental data source and quality classification contract.

`BL-0017` covers governed automated or provider-assisted ingestion strategy for fundamentals and portfolio metadata.

This developer specification prepares the implementation handoff for those existing backlog items and does not identify additional deferred work.

Backlog impact assessment:
- No new backlog items identified.

## 17. Recommended Next Step

Review and merge this developer specification.

After review and approval, launch a separate Codex implementation sprint if approved.

The implementation sprint must preserve Fundamental Layer classification-only authority, Decision Engine allocation authority, row preservation, deterministic behavior, source provenance, and English-only repository content.
