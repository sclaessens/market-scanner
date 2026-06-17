# ME-RUN09 — Cached-source end-to-end local execution contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN09

## Purpose

ME-RUN09 defines the first cached-source end-to-end local execution contract for Market Engine.

The contract exists to bridge the current local dry-run fixture path and a future bounded cached-source execution path without approving live provider calls, broker calls, Telegram/email delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, production report generation, or Decision Engine action/allocation authority.

ME-RUN09 is documentation-only. It does not add Python code, tests, fixtures, commands, runtime behavior, provider access, data refresh behavior, artifact writing behavior, or delivery behavior.

## Current approved baseline

The approved local execution baseline before ME-RUN09 is:

```text
ME-RUN05 -> optional local dry-run artifact persistence
ME-RUN06 -> local_snapshot_fixture input wrapper
ME-RUN07 -> realistic non-production local fixture artifact execution
ME-RUN08 -> deterministic local fixture matrix coverage
```

The approved dry-run and local artifact contracts remain:

```text
market-engine-end-to-end-dry-run-v1
market-engine-local-dry-run-input-fixture-v1
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

ME-RUN09 does not replace those contracts. It defines the next input boundary that a future implementation may add.

## New future contract family

A future implementation may introduce this local execution input contract:

```text
market-engine-cached-source-local-execution-input-v1
```

This contract represents local, deterministic, review-only execution from already-existing cached source snapshots.

It is not a source refresh contract.

It is not a provider adapter contract.

It is not a live-data execution contract.

It is not a production reporting contract.

It is not a trading, allocation, scoring, ranking, conviction, urgency, or tradeability contract.

## Approved input source

A future cached-source local execution may consume only cached source snapshots that already exist on disk before the command starts.

Approved input source category:

```text
data/market_engine/source_snapshots/
```

A future implementation may support an override path for local development only if all path-containment and non-production checks from ME-RUN05/ME-RUN06 are preserved.

The cached-source input must include enough metadata to prove:

* source snapshot format version;
* source family;
* ticker or issuer identifier;
* CIK when applicable;
* snapshot creation timestamp;
* provider/source provenance;
* retrieval timestamp if present;
* cache age;
* whether the snapshot is stale for review;
* whether required source fields are missing;
* non-production local execution marker when the command is run in local mode.

## Approved command shape

A future implementation may add a new local input mode to the existing dry-run command family:

```text
--input-mode cached_source_snapshot
```

The future command may accept:

```text
--source-snapshot-json <path>
--dry-run-id <id>
--generated-at <timestamp>
--write-local-artifact
--artifact-output-root <path>
--artifact-created-at <timestamp>
```

The existing behavior must remain unchanged for:

```text
--input-mode synthetic_contract_fixture
--input-mode local_snapshot_fixture
--input-mode explicit_in_memory_payload
```

Artifact writing must remain opt-in only through the existing explicit artifact flag.

## Required end-to-end flow boundary

A future cached-source local execution may convert cached source snapshot data into the approved Market Engine downstream chain only through existing or explicitly approved job contracts:

```text
cached source snapshot
-> Source Context
-> Fundamental Observations
-> Derived Observations
-> Setup Detection
-> Analysis Review
-> Recommendation Review
-> Portfolio Review
-> Decision Engine handoff
-> Delivery / Reporting
-> End-to-end dry-run summary
-> optional local dry-run artifact
```

Each downstream stage must preserve its existing non-actionable and missing-data-aware behavior.

A future implementation must fail closed when the cached source cannot produce a valid approved downstream contract.

## Required output

The final local execution output must still be:

```text
market-engine-end-to-end-dry-run-v1
```

If artifact writing is explicitly requested, the persisted artifact must still use:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

The dry-run payload must record:

* input mode: `cached_source_snapshot`;
* dry-run id;
* generated timestamp;
* source snapshot path or logical reference;
* source snapshot identity;
* source snapshot staleness status;
* missing-data markers;
* stale-data markers;
* blocked stages and blocked reasons;
* numeric-zero evidence;
* provenance across every stage that could be constructed;
* forbidden-side-effect confirmation;
* authority-boundary confirmation.

## Missing, stale, blocked, and numeric-zero behavior

A future implementation must preserve these rules:

* Missing fields stay explicit missing-data markers.
* Stale cached snapshots stay explicit stale-data markers.
* Numeric zero values remain valid evidence and may not be converted to missing values.
* Unsupported or incomplete source snapshots fail closed or produce a blocked dry-run state.
* Blocked states must record the blocked stage and blocked reasons.
* Local execution may not silently repair, enrich, normalize, or backfill data with live provider calls.

## Approved stale-data behavior

The cached-source contract must make staleness visible, not hidden.

A future implementation must expose staleness at source and dry-run level through fields equivalent to:

```text
source_snapshot_age_days
source_snapshot_created_at
source_snapshot_stale_for_review
stale_data_markers
```

The implementation may continue execution with limitations only when existing downstream contracts can safely represent the limitations.

It must block when required downstream evidence cannot be constructed safely.

## Approved provenance behavior

A future implementation must preserve provenance from cached source to final dry-run artifact.

Required provenance categories:

* cached source path or logical reference;
* source snapshot format version;
* source provider or source family;
* retrieval timestamp if present;
* source snapshot timestamp;
* transformation stage identifiers;
* downstream run identifiers;
* omitted-stage reasons when a stage cannot be constructed;
* artifact reference when local artifact writing is explicitly requested.

## Fail-closed requirements

A future cached-source local execution must fail closed when:

* the source snapshot file does not exist;
* the source snapshot path escapes the approved local input boundary;
* the source snapshot is not valid JSON;
* the source snapshot format version is unsupported;
* required source identity metadata is missing;
* the source snapshot is ambiguous across tickers/CIKs;
* numeric values cannot be parsed without unsafe coercion;
* required downstream contract identity cannot be produced;
* provider access would be required;
* broker, Telegram, email, scheduler, UI, portfolio, or watchlist access would be required;
* artifact writing would overwrite existing local artifacts without explicit permission;
* any action/allocation/ranking/scoring/tradeability semantics would be introduced outside approved guardrail text.

## Explicit non-goals

ME-RUN09 does not approve:

* SEC/EDGAR provider calls;
* yfinance or live market data calls;
* external API calls;
* source refresh jobs;
* broker integration;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* all-ticker production runs;
* automatic cache refresh;
* automatic cache cleanup;
* Decision Engine decisions;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

## Future implementation requirements

The logical next implementation sprint is:

```text
ME-RUN10 — Implement cached-source end-to-end local execution path
```

ME-RUN10 must remain bounded to local cached snapshots and deterministic local tests.

Minimum implementation requirements:

* add `cached_source_snapshot` input mode without breaking existing input modes;
* load exactly one approved cached source snapshot from disk;
* validate source identity, format version, path containment, and JSON shape;
* transform only through approved Market Engine contract builders;
* preserve missing-data, stale-data, blocked-state, numeric-zero, and provenance evidence;
* keep artifact writing opt-in only;
* add local tests for success, missing file, unsafe path, malformed JSON, unsupported version, stale snapshot, missing required fields, numeric-zero preservation, blocked downstream construction, and artifact persistence;
* add audit documentation;
* synchronize backlog and roadmap.

## Acceptance criteria for ME-RUN09

ME-RUN09 is complete when:

* the cached-source local execution contract is documented;
* approved input, command, output, failure, provenance, stale-data, missing-data, and numeric-zero boundaries are explicit;
* forbidden side effects and forbidden action-authority semantics are explicit;
* ME-RUN10 implementation requirements are defined;
* backlog and roadmap entries are synchronized;
* no runtime code, tests, provider calls, data writes, delivery behavior, portfolio behavior, watchlist behavior, scheduler behavior, UI behavior, or Decision Engine behavior is introduced.

## Conclusion

ME-RUN09 moves Market Engine closer to real local analysis by defining how already-cached source data may feed the existing end-to-end dry-run chain. It deliberately stops before implementation, provider refresh, production reporting, channel delivery, portfolio mutation, and action/allocation authority.
