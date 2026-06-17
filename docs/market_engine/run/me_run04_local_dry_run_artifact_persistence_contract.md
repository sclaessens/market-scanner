# ME-RUN04 - Local dry-run artifact persistence contract

## Status

COMPLETED BY ME-RUN04

## Sprint

ME-RUN04 - Define local dry-run artifact persistence contract

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN04 defines the contract for optional local, non-production persistence of `market-engine-end-to-end-dry-run-v1` payloads emitted by the ME-RUN03 dry-run command.

This sprint exists because ME-RUN03 intentionally prints inspectable JSON to stdout only. Before any future implementation may write dry-run output to disk, the repository needs an explicit persistence boundary, approved local paths, metadata requirements, retention expectations, and fail-closed side-effect rules.

ME-RUN04 is documentation-only. It does not implement artifact writes.

## Architectural position

Approved chain after ME-RUN04:

```text
Source Refresh / raw snapshots
-> Source Context
-> Fundamental Observations
-> Derived Observations
-> Setup Detection
-> Analysis Review
-> Recommendation Review
-> Portfolio Review
-> Decision Engine handoff / action authority
-> Delivery / Reporting
-> End-to-end dry-run summary
-> Local dry-run command output
-> Optional local non-production dry-run artifact persistence
```

The persistence boundary sits after the dry-run summary has already been built. It must not become a new analysis layer, Decision Engine substitute, reporting channel, scheduler, or production delivery mechanism.

Decision Engine remains the only future action/allocation authority. Delivery / Reporting remains non-actionable. Dry-run artifacts remain local integration-review evidence only.

## Approved input

A future ME-RUN05 implementation may persist only an already-built payload with:

```text
market-engine-end-to-end-dry-run-v1
```

The persistence layer must not independently construct, modify, enrich, normalize, repair, or re-interpret upstream stage payloads.

Approved input modes remain the ME-RUN01 modes already represented in the dry-run payload:

* `synthetic_contract_fixture`
* `local_snapshot_fixture`
* `explicit_in_memory_payload`

Live provider fetches, broker exports, production portfolio files, watchlist state, Telegram/email queues, old generated reports, legacy `scripts` runtime output, and archived reference material are not approved persistence inputs.

## Approved artifact output

A future implementation may write only local, non-production JSON artifacts representing the exact dry-run payload.

The persisted artifact must contain:

* the complete `market-engine-end-to-end-dry-run-v1` payload;
* dry-run format identity and version;
* dry-run identifier;
* generated timestamp;
* input mode;
* ticker/entity identifiers where available;
* run state;
* stage statuses;
* blocked reasons when present;
* missing-data summary;
* stale-data summary;
* numeric-zero evidence summary;
* provenance summary;
* delivery report reference when present;
* forbidden-side-effect confirmation;
* authority-boundary confirmation;
* artifact metadata.

The artifact metadata must include, at minimum:

* artifact format version: `market-engine-local-dry-run-artifact-v1`;
* artifact creation timestamp;
* artifact persistence mode;
* artifact path category;
* source dry-run identifier;
* source dry-run generated timestamp;
* source input mode;
* explicit non-production marker.

## Approved local path category

A future implementation may write only under a dedicated local non-production Market Engine dry-run artifact folder.

Approved path category:

```text
artifacts/market_engine/dry_runs/
```

The contract approves the category, not an implementation. A future implementation must still validate the resolved path before writing.

The persistence layer must not write to:

* production data folders;
* broker-connected folders;
* user-facing report folders;
* Telegram queues;
* email queues;
* portfolio state folders;
* watchlist state folders;
* scheduler state;
* UI state;
* archived legacy runtime folders;
* old generated report folders.

## Filename requirements

A future implementation must use deterministic, inspectable filenames derived from safe metadata.

Required filename components:

* dry-run identifier or a sanitized equivalent;
* generated date or artifact creation date;
* `.json` extension.

Filenames must not include broker account identifiers, personal portfolio identifiers, raw provider payload names, or unsafe path traversal components.

## Serialization requirements

A future implementation must serialize JSON deterministically enough for operator review and tests.

Required behavior:

* UTF-8 text;
* JSON object at the top level;
* stable key ordering where practical;
* human-readable pretty output by default;
* no lossy conversion of numeric zero values;
* no conversion of missing values into zero;
* no removal of stale-data or limitation markers;
* no removal of blocked states or blocked reasons;
* no addition of trading/allocation semantics.

Compact output may be supported only as an explicit option.

## Retention expectations

Local dry-run artifacts are review evidence, not production records.

A future implementation must document its retention behavior before enabling writes. Until retention is implemented explicitly, the persistence layer may only write artifacts and may not delete, rotate, upload, sync, or archive artifacts automatically.

Automatic cleanup, cloud sync, publication, or report bundling requires a separate approved sprint.

## Fail-closed behavior

A future implementation must fail closed and avoid writing an artifact when:

* the payload is not `market-engine-end-to-end-dry-run-v1`;
* the payload is not a JSON object / mapping;
* required dry-run identity fields are missing;
* the target path resolves outside the approved local dry-run artifact folder;
* the target folder cannot be created safely;
* the target file already exists and overwrite was not explicitly approved;
* serialization fails;
* the payload contains prohibited action/allocation/delivery semantics outside guardrail/audit contexts;
* the artifact would require provider, broker, Telegram, email, scheduler, UI, portfolio, or watchlist access.

Failed persistence must not mutate the dry-run payload.

## Side-effect boundary

ME-RUN04 approves only a future optional local JSON write under the approved non-production path category.

ME-RUN04 does not approve:

* provider calls;
* SEC/EDGAR calls;
* live market data calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* uploads;
* cloud sync;
* automatic retention cleanup;
* Decision Engine decisions;
* new financial analysis logic;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target weights;
* target prices;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

Forbidden concepts may appear in contract guardrails, tests, and audit documents only as prohibited behavior.

## Future implementation requirements

A future ME-RUN05 implementation must:

* persist only already-built `market-engine-end-to-end-dry-run-v1` payloads;
* add a small local persistence module or command option without moving orchestration authority out of the dry-run harness;
* validate the artifact path category;
* create local directories only under the approved path category;
* refuse unsafe paths;
* refuse accidental overwrite unless explicitly allowed;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero values;
* preserve blocked states and blocked reasons;
* preserve provenance;
* add local synthetic tests only;
* avoid live provider, broker, Telegram, email, portfolio, watchlist, scheduler, UI, and production report side effects.

## Outcome

ME-RUN04 defines the safe boundary for future local non-production persistence of dry-run JSON artifacts. It keeps the Market Engine dry-run as an integration-review mechanism only and prevents artifact writing from becoming a hidden production report, delivery channel, scheduler, portfolio mutation path, or Decision Engine substitute.
