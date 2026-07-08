# ME-CI06 - Advisory Artifact Schema Validation and Contract Enforcement Audit

## Objective

ME-CI06 implements deterministic, fail-closed contract validation for:

```text
schema_version: market-engine-chatgpt-ready-advisory-artifact-v1
artifact_type: market-engine-chatgpt-ready-advisory-artifact
```

The sprint enforces the ME-CI05 advisory artifact contract. It does not create a
new assembler, investment analysis engine, prompt runner, ChatGPT runtime,
provider integration, delivery channel, broker layer, portfolio mutation,
watchlist mutation, or Decision Engine semantic change.

## Inspected architecture

The implementation started from fresh `main` after ME-CI05 was merged.

Source main SHA:

```text
ab23d14bd4d5e52d9e2e73aa3c8e867616827760
```

Inspected sources included:

* ME-CI01 Structured Decision Output contract;
* ME-CI02 ChatGPT Advisory Context contract;
* ME-CI03 Portfolio Intelligence Context contract;
* ME-CI04 Explainability / Change-Rationale Context contract;
* ME-CI05 assembler, CLI, tests, audit, backlog entry, and roadmap entry;
* central Market Engine backlog and roadmap;
* relevant JSON examples under `docs/market_engine/contracts/examples`;
* existing local artifact persistence conventions.

## Current CI05 artifact composition

ME-CI05 emits these top-level fields:

```text
contract_identity
artifact_identity
run_identity
instrument_identity
generated_at
source_artifact_references
composition_status
advisory_eligibility
structured_decision_context
portfolio_intelligence_context
explainability_change_rationale_context
governor_context
dispatch_context
provenance_context
freshness_context
uncertainty_context
blockers
missing_context
validation_summary
```

The required context is Structured Decision Output. Optional supported context
families are Portfolio Intelligence Context, Explainability / Change-Rationale
Context, Governor context, and Dispatch context. ME-CI05 also uses the optional
ME-CI02 Advisory Context as an input source for advisory eligibility, source
references, freshness, and uncertainty, but it does not emit a separate
top-level embedded ME-CI02 payload.

Context families may be:

* embedded;
* referenced;
* absent when the CI05 contract permits absence.

## Validation architecture

ME-CI06 adds:

```text
src/market_engine/advisory/advisory_artifact_validation.py
```

The public validator is:

```text
validate_chatgpt_ready_advisory_artifact
```

The validator separates:

1. JSON/document shape validity;
2. top-level advisory artifact contract validity;
3. embedded/referenced advisory context compatibility;
4. cross-context identity and consistency validity.

The persistence path now follows:

```text
load inputs
-> assemble deterministic artifact in memory
-> validate complete artifact
-> persist only when valid
-> write validation evidence into artifact and manifest
```

## Schema strategy

ME-CI06 uses explicit standard-library validation rather than introducing a new
schema dependency. This matches the current repository style and keeps the
scope bounded to the existing ME-CI05 artifact.

The validator enforces:

* required top-level fields;
* primitive types;
* object/list/null semantics;
* enum values;
* contract identity;
* artifact type;
* schema version;
* timestamp shape;
* embedded, referenced, and absent context rules;
* validation metadata shape;
* provenance, freshness, uncertainty, blockers, and missingness shape;
* contextual forbidden authority fields.

## Supported contract families

ME-CI06 validates only families used by ME-CI05:

| Family | Validation mode |
| --- | --- |
| Structured Decision Output | required embedded canonical context |
| Portfolio Intelligence Context | optional embedded, referenced, or absent |
| Explainability / Change-Rationale Context | optional embedded, referenced, or absent |
| Governor context | optional embedded, referenced, or absent |
| Dispatch context | optional embedded, referenced, or absent |

ME-CI02 Advisory Context is validated at the CI05 input/assembly boundary and
through the emitted advisory eligibility/freshness/uncertainty fields. ME-CI06
does not invent a new top-level ME-CI02 embedded payload that CI05 does not
emit.

## Validation result model

ME-CI06 adds typed result objects:

```text
AdvisoryArtifactValidationResult
AdvisoryArtifactValidationIssue
```

Each issue contains:

```text
code
path
message
severity
contract_family
expected
actual
```

`contract_family`, `expected`, and `actual` are included only when useful.

Validation result payloads include:

```text
status
validator_version
validated_schema_version
issue_count
issues
```

## Error taxonomy

The machine-readable issue codes include:

```text
missing_required_field
invalid_field_type
invalid_enum_value
unsupported_schema_version
unsupported_contract_version
artifact_type_mismatch
ticker_identity_mismatch
instrument_identity_mismatch
run_identity_mismatch
timestamp_invalid
context_identity_mismatch
embedded_reference_conflict
missing_required_context
malformed_context
cross_context_conflict
invalid_provenance_shape
invalid_freshness_shape
validation_metadata_invalid
forbidden_field_present
```

The validator does not expose free text as the only failure interface.

## Cross-context consistency rules

ME-CI06 enforces:

* parent ticker equals Structured Decision Output ticker;
* parent instrument ticker equals nested instrument ticker;
* parent run id equals Structured Decision Output run id;
* parent `source_structured_decision_run_id` equals Structured Decision Output
  run id;
* ME-CI02 advisory context run id, when recorded in `context_run_ids`, equals
  the parent run id;
* embedded Portfolio, Governor, Dispatch, and Explainability tickers do not
  conflict with the parent ticker;
* referenced context tickers do not conflict with the parent ticker when a
  reference ticker is supplied;
* Explainability `current_run_identity.run_id` equals the parent run id.

Missing identity evidence is not treated as proof of compatibility.

## Embedded/reference semantics

For embedded contexts, the embedded payload must be an object and must validate
against the supported family shape.

For referenced contexts, the reference must include:

```text
artifact_ref
schema_version
artifact_type
```

Referenced context validation confirms reference shape and lineage only. It
does not claim to fully validate external content that is not embedded.

For absent contexts, absence is valid only for optional families and must keep
`payload` as `null` with an explicit missing reason.

## Missingness enforcement

ME-CI06 preserves ME-CI05 missingness semantics:

* absent portfolio context is not interpreted as an empty portfolio;
* unavailable cash is not converted to zero;
* absent explainability context is not interpreted as unchanged;
* unknown freshness remains unknown and is not converted to fresh;
* generated-at timestamps do not refresh upstream evidence.

## Forbidden authority enforcement

The validator rejects forbidden authority fields only at relevant advisory
artifact control locations, including composition status, advisory eligibility,
validation summary, Governor payload, and Dispatch payload.

The validator does not perform a naive global string search because some
upstream contract fields, such as Structured Decision Output `target_weight` or
Portfolio Intelligence allocation-context placeholders, are legitimate when
present inside their approved upstream contract boundaries.

## Persistence boundary

`persist_chatgpt_ready_advisory_artifact` now validates the full artifact before
writing `chatgpt_ready_advisory.json` or `manifest.json`.

Invalid artifacts raise a controlled `ChatGPTReadyAdvisoryArtifactError` and no
valid advisory artifact is persisted.

Valid persisted artifacts include validation evidence in:

* `validation_summary.contract_validation`;
* manifest fields:
  * `validation_status`;
  * `validator_version`;
  * `validated_schema_version`;
  * `validation_timestamp`;
  * `validation_issue_count`.

Overwrite protection remains unchanged.

## CLI behavior

The existing ME-CI05 CLI remains the entrypoint. It now inherits the persistence
gate:

```text
valid input set
-> artifact assembled
-> validation passes
-> persistence succeeds
-> exit code 0
```

```text
invalid artifact/input relationship
-> validation fails
-> no valid advisory artifact is written
-> exit code 2
```

## Test evidence

ME-CI06 adds a dedicated validation test file covering:

* minimal valid artifact;
* full valid artifact;
* optional absent context;
* referenced context;
* deterministic validation;
* missing required top-level field;
* wrong primitive type;
* unknown enum value;
* unsupported schema version;
* artifact type mismatch;
* malformed nested context;
* ticker and run identity mismatches;
* Structured Decision Output source identity conflict;
* Governor, Dispatch, Portfolio, and Explainability identity conflicts;
* missingness boundaries;
* unknown freshness;
* invalid referenced context shape;
* embedded/reference conflict;
* unauthorized semantic override;
* forbidden authority field;
* invalid artifact not persisted;
* valid artifact persisted with validation evidence;
* overwrite protection;
* CLI validation failure;
* deterministic CLI output.

## Governance boundary

ME-CI06 adds no:

* OpenAI or ChatGPT API call;
* prompt execution;
* LLM output;
* internet call;
* provider call;
* SEC, EDGAR, or yfinance access;
* live price retrieval;
* Telegram, Messenger, Signal, or email send;
* broker integration;
* order;
* portfolio write;
* watchlist write;
* production write;
* scheduler behavior;
* UI;
* new recommendation, Governor, Portfolio Review, or Decision Engine semantics;
* allocation, target weight, position sizing, execution advice, conviction,
  urgency, tradeability, causality, or materiality inference.

## Residual gaps

ME-CI06 validates the deterministic ChatGPT-ready advisory artifact, but it
does not provide:

* prompt contract;
* LLM runtime;
* OpenAI API integration;
* answer parser;
* response-grounding validator;
* notification delivery;
* broker integration;
* autonomous decision making.

ME-CI06 also does not introduce a generic schema framework for every historical
Market Engine contract family. It validates the currently supported CI05
artifact boundary.

## Recommended next sprint

The next logical sprint remains prompt-boundary work after validation:

```text
ME-CI07 - Define ChatGPT advisory prompt and response-grounding contract
```

That sprint should remain contract-first and must not introduce live ChatGPT or
notification delivery until prompt inputs, allowed wording, response grounding,
and refusal behavior are defined.
