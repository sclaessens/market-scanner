# ME-RUN02 - End-to-end dry-run harness implementation

## Status

COMPLETED BY ME-RUN02

## Sprint

ME-RUN02 - Implement end-to-end dry-run harness

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN02 implements the deterministic local dry-run harness defined by ME-RUN01.

The harness validates and summarizes approved Market Engine contract payloads in sequence. It emits `market-engine-end-to-end-dry-run-v1` and remains an integration-review artifact only.

It does not fetch providers, call live market data, call brokers, deliver messages, write production reports, mutate portfolio/watchlist state, schedule work, create UI behavior, or add trading/allocation authority.

## Implemented runtime

```text
src/market_engine/run/end_to_end_dry_run.py
src/market_engine/run/__init__.py
```

## Implemented tests

```text
tests/market_engine/run/test_end_to_end_dry_run.py
```

## Output contract

The harness emits:

```text
market-engine-end-to-end-dry-run-v1
```

The payload includes dry-run format identity, dry-run id, generated timestamp, input mode, ticker/entity identifiers where available, ordered stage results, observed contract versions, stage statuses, blocked stage, blocked reasons, missing-data summary, stale-data summary, numeric-zero evidence summary, provenance summary, delivery report reference, forbidden-side-effect confirmation, authority-boundary confirmation, and audit metadata.

## Approved input modes

The implementation accepts only:

* `synthetic_contract_fixture`;
* `local_snapshot_fixture`;
* `explicit_in_memory_payload`.

Unsupported input modes fail closed as `dry_run_unsupported_input` before any stage starts.

## Required stage coverage

The harness inspects the required ME-RUN01 stages in order:

1. Source Context
2. Fundamental Observations
3. Derived Observations
4. Setup Detection
5. Analysis Review
6. Recommendation Review
7. Portfolio Review
8. Decision Engine handoff
9. Delivery / Reporting
10. Dry-run summary

If an upstream stage blocks, downstream stages are marked `not_started`.

## Stage status behavior

The implementation supports:

* `completed`;
* `completed_with_limitations`;
* `blocked`;
* `unsupported_input`;
* `contract_violation`;
* `not_started`.

Missing or stale markers produce `completed_with_limitations` when the stage contract itself is otherwise supported and not blocked.

Unsupported contract versions produce `unsupported_input` and fail closed.

Malformed stage payloads and prohibited semantic fields produce `contract_violation` and fail closed.

## Run state behavior

The implementation emits:

* `dry_run_completed` when all required stages complete without missing/stale limitations;
* `dry_run_completed_with_limitations` when stages complete while preserving missing or stale evidence;
* `dry_run_blocked` when a stage is missing or preserves a blocked upstream state;
* `dry_run_unsupported_input` when input mode or stage contract version is unsupported;
* `dry_run_contract_violation` when payload structure or prohibited semantics violate the contract.

## Preserved evidence

ME-RUN02 preserves:

* stage-level contract identity;
* blocked state and blocked reasons;
* missing-data markers;
* stale-data markers;
* valid numeric-zero values;
* provenance and references available in each payload;
* delivery report references.

## Forbidden behavior preserved

ME-RUN02 does not introduce provider calls, SEC/EDGAR calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, new financial analysis logic, BUY / SELL / HOLD semantics, allocation advice, target weights, target prices, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability.

## Tests added

Synthetic tests cover completed, completed-with-limitations, blocked, unsupported-input, malformed-input, missing-stage, stale-data, missing-data, numeric-zero, provenance, delivery report reference, contract-violation, serialization, and side-effect import guard cases.

## Outcome

ME-RUN02 implements the first end-to-end Market Engine dry-run harness. The sprint proves the approved contract chain can be inspected deterministically through Delivery / Reporting and summarized as `market-engine-end-to-end-dry-run-v1` without crossing provider, broker, delivery, scheduler, portfolio, watchlist, UI, or action/allocation boundaries.
