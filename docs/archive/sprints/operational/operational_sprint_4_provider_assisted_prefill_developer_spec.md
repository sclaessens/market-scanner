# Operational Sprint 4 Provider-Assisted Prefill Developer Specification

## 1. Status and Scope

Status: DEVELOPER SPECIFICATION

This document defines the future implementation contract for provider-assisted prefill outside the runtime Decision Engine path.

This is a documentation-only developer specification. It does not implement code, tests, CSV files, provider integration, credentials, runtime orchestration changes, Reporting changes, Telegram changes, Decision Engine changes, generated artifacts, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Future implementation requires a separate implementation prompt after this developer specification is reviewed and approved.

## 2. Context

The project has completed the governed Level 2 design for automated or provider-assisted data ingestion covering fundamentals and portfolio metadata.

Relevant documents:

- `docs/sprints/operational_sprint_4_data_source_strategy_followup.md`
- `docs/sprints/operational_sprint_4_automated_data_ingestion_level_2_design.md`
- `docs/sprints/project_backlog.md`

Relevant backlog item:

```text
BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata
```

The validated MVP source-artifact contracts are:

```text
data/raw/fundamentals.csv
data/portfolio/portfolio_metadata.csv
```

The Level 2 design recommends a staged hybrid model:

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

This developer specification preserves that direction.

## 3. Authorized Future Implementation Scope

A future implementation sprint may implement provider-assisted prefill only after this developer specification is reviewed and approved.

Authorized future implementation scope may include:

- a provider-assisted prefill workflow outside the runtime Decision Engine path;
- writing provider-assisted output into governed local source artifacts;
- schema validation before writing or before pipeline consumption;
- freshness metadata validation;
- source provenance metadata;
- deterministic error handling;
- audit/log output for the prefill workflow;
- focused tests for the prefill workflow if and only if implementation is later authorized;
- documentation or runbook updates if implementation findings require operational clarification.

Provider-assisted prefill means an operator or separately approved refresh process prepares source artifacts before the normal runtime pipeline consumes them.

It does not mean live provider calls during Decision Engine execution.

## 4. Explicitly Forbidden Future Scope

The future implementation must not:

- call provider APIs during Decision Engine execution;
- loosen Decision Engine review behavior;
- change Decision Engine allocation logic;
- change Decision Engine arbitration logic;
- change Reporting logic;
- change Telegram logic;
- infer missing fundamentals in Reporting;
- infer missing portfolio metadata in Reporting;
- infer missing fundamentals in Telegram;
- infer missing portfolio metadata in Telegram;
- introduce allocation semantics outside the Decision Engine;
- introduce ranking authority outside the Decision Engine;
- introduce scoring authority outside the Decision Engine;
- introduce tradeability semantics upstream;
- introduce urgency semantics upstream;
- introduce conviction semantics upstream;
- introduce hidden filtering;
- introduce Decision Engine bypass;
- commit generated processed artifacts;
- commit credentials or secrets;
- change GitHub workflows unless separately authorized;
- change scanner behavior;
- change validation behavior;
- change context behavior;
- change timing behavior;
- change Reporting behavior;
- change Telegram behavior;
- change Decision Engine behavior;
- make live provider data a direct runtime dependency unless separately approved.

## 5. Proposed Future Files That May Be Changed

This document does not modify implementation files.

A future approved implementation may add or modify only files required for provider-assisted prefill and its focused tests.

Preferred future implementation structure:

```text
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

A shared support module may be acceptable if needed, for example:

```text
scripts/data_sources/common.py
```

Focused tests may be added under `tests/` using an appropriate future test path.

Documentation or runbook updates may be added if needed.

### 5.1 Separate Scripts Preferred

Separate prefill scripts are preferred for:

- fundamentals;
- portfolio metadata.

Rationale:

- `data/raw/fundamentals.csv` and `data/portfolio/portfolio_metadata.csv` have different schemas;
- the artifacts have different freshness rules;
- the artifacts have different validation criteria;
- provider field mapping may differ;
- failure handling may differ;
- separate scripts reduce accidental cross-contract coupling.

A single combined prefill script is not recommended unless a later implementation review proves that shared orchestration improves safety without blending the contracts.

## 6. Files and Areas That Must Not Be Changed Without Separate Authorization

The future implementation must not modify the following without separate governance authorization:

- `scripts/core/decision_engine.py`
- Reporting scripts
- Telegram scripts
- scanner scripts
- validation logic
- context logic
- timing logic
- fundamental classification behavior beyond source-artifact consumption already approved by its MVP contract
- portfolio intelligence behavior beyond source-artifact consumption already approved by its MVP contract
- `.github/workflows/`
- generated `data/processed/*`
- generated `reports/*`
- committed CSV data files unless explicitly approved for fixture or test purposes
- runtime orchestration scripts
- runtime scan execution scripts
- production reporting or Telegram delivery flows

If implementation requires any of these files, the implementation must stop and request separate governance review.

## 7. Source Artifact Targets

The future prefill workflow may write only to governed local source artifacts after implementation approval:

```text
data/raw/fundamentals.csv
data/portfolio/portfolio_metadata.csv
```

These artifacts are source artifacts.

They are not Decision Engine outputs.

They are not Reporting outputs.

They are not Telegram outputs.

They must remain auditable and deterministic once written.

They must contain source provenance and freshness metadata.

They must not contain allocation, ranking, scoring, tradeability, urgency, conviction, hidden filtering, or action semantics.

## 8. Fundamentals Prefill Contract

### 8.1 Contract Purpose

The future fundamentals prefill workflow may prepare `data/raw/fundamentals.csv` from provider-assisted or operator-approved source data.

The artifact supports descriptive Fundamental Layer quality classification only.

### 8.2 Row Identity

The future developer implementation must preserve the row identity approved by the existing Fundamental data MVP contract.

If the implementation needs to refine row identity, it must define it explicitly before code changes.

Expected row identity principles:

- normalized `ticker` must be part of row identity;
- any approved reporting period, fiscal date, source date, or as-of date must be documented if used;
- duplicate row identities must fail fast if they create ambiguous source authority.

### 8.3 Ticker Normalization

Ticker matching must be deterministic.

Future implementation should normalize tickers by:

- trimming whitespace;
- converting to uppercase;
- rejecting blank tickers;
- detecting duplicate normalized tickers or duplicate normalized row identities.

### 8.4 Required Fields

The future implementation must not invent unsupported fields silently.

Required fields must be derived from existing repository evidence and the approved Fundamental data MVP contract. If additional fields are needed, they must be marked as proposed future fields in the implementation plan and reviewed before implementation.

At minimum, the future implementation must preserve the ability of the Fundamental Layer to classify:

- `SUFFICIENT_DATA`
- `PARTIAL_DATA`
- `STALE_DATA`
- `INSUFFICIENT_DATA`

### 8.5 Source Provenance Fields

Future provider-assisted fundamentals prefill must include provenance metadata.

Expected provenance principles:

- source/provider label must be recorded;
- source extraction or refresh date must be recorded;
- provider field mapping assumptions must be documented;
- manual override or stewarded input must be identified if used;
- provenance must not include credentials, tokens, account identifiers, or secrets.

### 8.6 Freshness Fields

Future fundamentals prefill must include freshness metadata.

Freshness metadata may include fields such as source update date, statement period date, fiscal period date, or metadata refresh date if approved by the existing contract or future implementation review.

Freshness fields must be parseable and deterministic.

Stale data must remain visible as a descriptive state and must not be treated as complete data.

### 8.7 Missing, Partial, Stale, and Invalid Data

Missing source data should preserve row identity where safe and allow downstream classification as insufficient or missing.

Partial source data should remain visible as partial quality.

Stale source data should remain visible as stale quality.

Invalid source data should not silently become sufficient data.

The future implementation must distinguish:

- safe row-level classification;
- unsafe artifact-level ambiguity requiring deterministic failure.

### 8.8 Deterministic Failure Conditions

The future fundamentals prefill workflow must fail deterministically when:

- required artifact columns are missing;
- duplicate row identity creates ambiguous data authority;
- provider response format cannot be mapped safely;
- required date fields are malformed at artifact level;
- write operation fails;
- credentials required for an approved provider flow are missing;
- provider configuration is invalid.

### 8.9 Forbidden Semantics

Fundamentals prefill must not create:

- ranking;
- scoring authority;
- tradeability;
- urgency;
- conviction;
- allocation;
- hidden filtering;
- buy/sell/hold/trim semantics;
- Decision Engine bypass.

## 9. Portfolio Metadata Prefill Contract

### 9.1 Contract Purpose

The future portfolio metadata prefill workflow may prepare `data/portfolio/portfolio_metadata.csv` from provider-assisted or operator-approved source data.

The artifact supports descriptive Portfolio Intelligence metadata completeness only.

### 9.2 Row Identity

Required row identity:

```text
ticker
```

Ticker identity must be normalized by trimming whitespace and converting to uppercase.

Each normalized ticker may appear at most once.

Duplicate normalized tickers must fail fast.

### 9.3 Required Descriptive Fields

The future implementation should preserve the Portfolio Metadata MVP contract fields, including:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

Optional descriptive fields may include:

- `sector_taxonomy`
- `industry_group`
- `country`
- `region`
- `exchange`
- `notes`

Optional fields must remain descriptive only.

### 9.4 Source Provenance

Portfolio metadata prefill must include source provenance.

Expected provenance principles:

- `metadata_source` must identify provider, manual source, or approved taxonomy source;
- `metadata_last_updated` must indicate the date the metadata was last updated or refreshed;
- optional taxonomy notes may document GICS, ICB, manual, or provider-specific taxonomy sources;
- provenance must not include credentials or secrets.

### 9.5 Freshness Validation

Future implementation must validate `metadata_last_updated` or equivalent approved freshness fields.

Stale metadata must not be treated as complete.

Malformed freshness dates must be classified as invalid or fail deterministically according to the implementation contract.

### 9.6 Missing, Partial, Stale, and Invalid Metadata

Missing metadata rows must not remove output rows from downstream artifacts.

Partial metadata must remain incomplete.

Stale metadata must remain incomplete.

Invalid metadata must remain incomplete.

Metadata-only tickers must not expand the opportunity universe unless a separate row-universe governance decision approves it.

### 9.7 Forbidden Semantics

Portfolio metadata prefill must not create:

- opportunity rows;
- allocation authority;
- ranking authority;
- scoring authority;
- tradeability;
- urgency;
- conviction;
- hidden filtering;
- Reporting decision logic;
- Telegram decision logic;
- Decision Engine bypass.

## 10. Provider Abstraction Boundary

Future implementation must isolate provider-specific logic from core classification layers.

Required principles:

- provider-specific parsing must remain in provider-assisted prefill code;
- provider-specific field names must be mapped into governed artifact fields;
- provider errors must not silently create complete data;
- provider availability must not affect Decision Engine execution unless the operator explicitly runs the prefill process before the pipeline;
- provider assumptions must be documented;
- provider raw responses should not be committed unless explicitly approved;
- provider-specific taxonomy values must be normalized or recorded as provenance before runtime consumption;
- core classification layers must continue reading governed local source artifacts.

Provider-specific code must not be imported into Decision Engine execution unless separately authorized.

## 11. Provider Selection Status

Provider selection remains unresolved.

This developer specification does not select a provider.

Future implementation may use a placeholder provider interface or manual/provider-assisted import boundary only if approved later.

Any future provider choice must be evaluated against the criteria defined in the Level 2 design, including:

- coverage;
- field quality;
- stability;
- rate limits;
- cost;
- terms of use;
- taxonomy clarity;
- historical availability;
- freshness guarantees;
- API reliability;
- exportability to CSV or local raw artifacts;
- auditability;
- reproducibility;
- deterministic pipeline compatibility.

## 12. Credential and Secret Handling

No credentials are authorized by this document.

No secrets may be committed.

Documentation must not contain real credentials, tokens, API keys, account identifiers, or provider secrets.

Future implementation may read credentials only from approved locations if later authorized, such as:

- environment variables;
- local `.env` files excluded from version control;
- local keychain or secure credential store;
- GitHub secrets if workflow usage is separately approved.

Credentials must never be written to:

- logs;
- CSV artifacts;
- reports;
- generated files;
- documentation;
- error messages;
- audit outputs.

Missing credentials must fail deterministically in the prefill workflow.

Missing credentials must not fail inside Decision Engine execution because provider calls must not occur there.

## 13. Prefill Execution Model

Future provider-assisted prefill must be explicitly operator-triggered or separately scheduled outside the runtime Decision Engine path.

Permitted future execution models, subject to implementation approval:

- manual/on-demand execution by an operator;
- local prefill before a full pipeline run;
- provider-assisted import from a downloaded or exported provider file;
- scheduled refresh only after separate governance approval.

Forbidden execution models without separate approval:

- automatic provider calls from the Decision Engine;
- automatic provider calls from Reporting;
- automatic provider calls from Telegram;
- automatic provider calls from core classification layers;
- automatic provider calls from runtime orchestration;
- silent refresh during normal scan execution.

The prefill workflow must produce governed local source artifacts first.

Only after artifacts exist and pass validation may existing classification layers consume them.

## 14. Validation Rules

Future implementation must include deterministic validation rules.

Expected validation checks:

- schema validation;
- required column validation;
- duplicate ticker checks;
- duplicate row identity checks;
- normalized ticker checks;
- blank ticker checks;
- date parsing;
- freshness thresholds;
- source metadata checks;
- provider export format checks;
- missing data handling;
- partial data handling;
- stale data handling;
- invalid data handling;
- artifact write path validation;
- no hidden filtering checks;
- no upstream allocation semantics checks;
- no ranking, scoring, tradeability, urgency, or conviction semantics checks;
- no live API dependency in the Decision Engine path.

Validation must run before writing final source artifacts or before marking a prefill run successful.

If partial artifacts are written for diagnostics, they must not replace governed source artifacts unless explicitly approved.

## 15. Audit and Logging Requirements

Future provider-assisted prefill must produce credential-safe descriptive audit output.

Expected audit/log fields:

- run timestamp;
- provider/source label;
- requested ticker count;
- matched ticker count;
- missing ticker count;
- written row count;
- stale row count;
- invalid row count;
- partial row count;
- duplicate detection result;
- artifact write path;
- validation status;
- failure reason;
- refresh mode;
- source artifact target;
- credential-safe status only.

Audit logs must remain descriptive.

Audit logs must not create decision authority.

Audit logs must not include credentials or secrets.

Audit logs must not contain allocation, ranking, scoring, tradeability, urgency, conviction, hidden filtering, or action semantics.

## 16. Error Handling

Future implementation must prefer safe deterministic failure over silent data corruption.

### 16.1 Provider Unavailable

If the provider is unavailable, the prefill workflow must fail safely and preserve existing governed source artifacts unless explicit overwrite behavior is approved.

### 16.2 Rate Limit Reached

If a rate limit is reached, the prefill workflow must fail or pause deterministically according to the approved implementation contract.

It must not write incomplete data as complete data.

### 16.3 Invalid Provider Response

If the provider response cannot be mapped safely, the prefill workflow must fail deterministically.

### 16.4 Missing Credentials

If credentials are required and missing, the prefill workflow must fail deterministically before provider access.

### 16.5 Malformed Dates

Malformed dates must either fail artifact-level validation or classify affected rows as invalid according to the approved contract.

They must not be treated as fresh data.

### 16.6 Duplicate Tickers

Duplicate normalized tickers or duplicate row identities must fail fast where they create ambiguous source authority.

### 16.7 Missing Required Fields

Missing required artifact columns must fail fast.

Missing row values may be row-level partial or invalid states where safe.

### 16.8 Stale Records

Stale records must not be treated as complete.

They should remain visible as stale or incomplete according to the relevant artifact contract.

### 16.9 Partial Records

Partial records must remain visible as partial and must not be inferred into complete data.

### 16.10 Write Failure

Write failures must fail deterministically and must not leave partially written governed source artifacts as successful outputs.

### 16.11 Validation Failure

Validation failure must prevent successful prefill status.

Provider failure, validation failure, or write failure must not force the Decision Engine to loosen decisions.

## 17. Testing Requirements for Future Implementation

If implementation is approved later, Codex must add focused tests for the prefill workflow.

Required future test coverage should include:

- successful fundamentals prefill;
- successful portfolio metadata prefill;
- missing credentials;
- provider unavailable;
- rate limit handling if provider behavior is implemented;
- malformed provider response;
- duplicate ticker handling;
- duplicate row identity handling;
- missing required artifact columns;
- missing required row values;
- stale records;
- partial records;
- invalid records;
- output artifact schema;
- source provenance fields;
- freshness metadata fields;
- credential-safe logging;
- no Decision Engine file changes;
- no Reporting file changes;
- no Telegram file changes;
- no generated processed artifacts committed;
- no forbidden semantics introduced.

Tests must remain focused on provider-assisted prefill and must not rewrite existing pipeline authority boundaries.

## 18. Acceptance Criteria for Future Implementation

Future implementation is acceptable only if all conditions below are met:

- provider-assisted prefill runs outside the Decision Engine runtime path;
- provider data is written into governed local source artifacts;
- source artifacts include provenance metadata;
- source artifacts include freshness metadata;
- validation is deterministic;
- failures are deterministic and safe;
- existing governed source artifacts are not silently corrupted;
- no Decision Engine changes are made;
- no Reporting changes are made;
- no Telegram changes are made;
- no live provider call is added to runtime Decision Engine execution;
- no credentials are committed;
- no secrets are logged;
- no generated processed artifacts are committed;
- focused tests pass;
- full suite passes if required by implementation scope;
- English-only repository content is preserved;
- no allocation, ranking, scoring, tradeability, urgency, conviction, hidden filtering, or Decision Engine bypass is introduced.

## 19. Implementation Non-Goals

This developer specification does not authorize:

- provider choice;
- API implementation;
- credentials;
- secrets;
- runtime pipeline automation;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- strategy tuning;
- allocation logic;
- ranking logic;
- scoring logic;
- tradeability semantics;
- urgency semantics;
- conviction semantics;
- hidden filtering;
- provider calls during normal runtime execution;
- generated artifact commits;
- workflow changes.

## 20. Backlog Impact Assessment

BL-0017 already captures the required future work:

```text
BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata
```

This developer specification remains within that captured backlog scope.

No additional backlog item is required at this stage.

Backlog impact assessment:
- No new backlog items identified.

## 21. Recommended Next Step

After this developer specification is reviewed and merged, create a separate implementation prompt for Codex.

The implementation prompt must remain bounded to provider-assisted prefill outside the runtime Decision Engine path.

It must preserve:

- local governed source artifacts as the provider data boundary;
- classification-only upstream layers;
- Decision Engine as only allocation authority;
- Reporting and Telegram as communication-only layers;
- no live provider calls during Decision Engine execution unless separately approved.
