# ME-RUN01 - End-to-end dry-run contract

## Status

COMPLETED BY ME-RUN01

## Sprint

ME-RUN01 - Define end-to-end dry-run contract

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN01 defines the canonical contract for a future Market Engine end-to-end dry-run.

The dry-run is the first integration boundary after the completed review and delivery/reporting chain. It is meant to prove that approved Market Engine contracts can be connected in sequence and inspected as one run, without introducing live provider calls, broker integration, Telegram/email delivery, portfolio mutation, watchlist mutation, scheduling, UI behavior, or execution authority.

The dry-run is not a production run. It is a controlled, local, deterministic integration harness for human and test review.

## Architectural position

Approved chain covered by the dry-run contract:

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
```

The dry-run sits outside the individual job families. It may orchestrate approved Market Engine contracts, but it must not become a new analysis layer, a hidden Decision Engine, a delivery channel, or a production scheduler.

Decision Engine remains the only future action and allocation authority. Delivery / Reporting remains non-actionable. The dry-run only verifies and summarizes contract flow.

## Approved upstream contracts

A future ME-RUN02 implementation may consume only approved Market Engine contract payloads or deterministic synthetic fixtures that exactly model those payloads.

Approved payload families for the first dry-run contract are:

```text
sec-companyfacts-source-context-v1
sec-companyfacts-fundamental-observations-v1
sec-companyfacts-derived-cash-generation-observations-v1
sec-companyfacts-setup-detection-v1
sec-companyfacts-analysis-review-v1
sec-companyfacts-recommendation-review-v1
sec-companyfacts-portfolio-review-v1
market-engine-portfolio-context-v1
market-engine-decision-engine-handoff-v1
market-engine-delivery-report-v1
```

The dry-run must not bypass contract boundaries by reading raw provider responses, old scanner output, old generated reports, watchlists, broker exports, legacy `scripts` runtime output, or archived reference material as active runtime input.

## Approved input modes

ME-RUN01 approves only these future dry-run input modes:

| Mode | Meaning | Provider access | Production writes |
| --- | --- | --- | --- |
| `synthetic_contract_fixture` | Fully synthetic payloads used to prove contract flow. | Forbidden | Forbidden |
| `local_snapshot_fixture` | Local non-production snapshots previously approved by source/data governance. | Forbidden during dry-run | Forbidden |
| `explicit_in_memory_payload` | Caller-supplied in-memory contract payloads. | Forbidden | Forbidden |

Live provider fetches are not part of ME-RUN01. A future real-data run requires a separate, explicitly approved sprint.

## Output contract

A future ME-RUN02 implementation should emit:

```text
market-engine-end-to-end-dry-run-v1
```

The payload must include:

* dry-run format identity and version;
* dry-run identifier;
* generated timestamp;
* input mode;
* ticker/entity identifiers where available;
* ordered stage results;
* contract version observed at every stage;
* stage status for every stage;
* blocked stage, if any;
* blocked reasons, if any;
* missing-data summary;
* stale-data summary;
* numeric-zero evidence summary;
* provenance summary;
* delivery/report reference when available;
* forbidden-side-effect confirmation;
* authority-boundary confirmation;
* audit metadata.

The output is an integration-review artifact only. It must not be treated as advice, a recommendation, a trade plan, a channel-delivery message, or a production report.

## Stage status model

Approved dry-run stage statuses:

| Status | Meaning |
| --- | --- |
| `not_started` | Stage was not executed or not inspected because an earlier stage blocked the run. |
| `completed` | Stage produced or validated the expected approved contract payload. |
| `completed_with_limitations` | Stage completed while preserving missing, stale, partial, or limitation evidence. |
| `blocked` | Stage was blocked by contract validation, upstream state, missing evidence, stale evidence, or explicit governance rule. |
| `unsupported_input` | Stage input contract or version is unsupported. |
| `contract_violation` | Stage input/output violates the approved contract or contains prohibited semantics. |

Blocked upstream states must remain blocked downstream. The dry-run must not continue as if a blocked stage were valid.

## Run-level states

Approved dry-run run states:

| State | Meaning |
| --- | --- |
| `dry_run_completed` | All required stages completed and the final delivery report is available for human review. |
| `dry_run_completed_with_limitations` | Required stages completed but missing-data, stale-data, or limitation evidence remains visible. |
| `dry_run_blocked` | One or more stages blocked the run. |
| `dry_run_unsupported_input` | At least one required payload uses an unsupported contract/version. |
| `dry_run_contract_violation` | A payload or stage introduced prohibited behavior or violated required contract boundaries. |

Run-level states are review states only. They must not become trading or allocation states.

## Required stage coverage

The first implementation sprint must cover, at minimum:

* Source Context;
* Fundamental Observations;
* Derived Observations;
* Setup Detection;
* Analysis Review;
* Recommendation Review;
* Portfolio Review;
* Decision Engine handoff;
* Delivery / Reporting;
* dry-run summary.

If a stage is represented by a fixture rather than a runtime builder, the dry-run summary must mark it as fixture-backed and preserve the fixture provenance.

## Required lineage preservation

The dry-run must preserve lineage from every available upstream stage.

At minimum, the dry-run summary must preserve references for:

* source context;
* source facts / provider family when present upstream;
* fundamental observations;
* derived observations;
* setup detection;
* analysis review;
* recommendation review;
* portfolio review;
* portfolio context;
* Decision Engine handoff;
* delivery report.

The dry-run must not invent lineage. Missing lineage must be reported as missing.

## Missing-data handling

Missing data must remain explicit across the dry-run.

The dry-run must not:

* fill missing facts from old reports;
* infer missing values from prior periods;
* treat missing numeric values as zero;
* hide missing input behind a clean run summary;
* convert missing portfolio context into a neutral portfolio state.

## Stale-data handling

Stale data must remain explicit across the dry-run.

The dry-run must not describe stale upstream evidence as current. If stale markers are present at any stage, the dry-run state must be either `dry_run_completed_with_limitations` or a blocked/unsupported/violation state, depending on the stage contract.

## Numeric-zero handling

Valid numeric zero values must be preserved.

Examples include zero quantity, zero market value, zero exposure, zero cash, zero weight, zero capex, zero free-cash-flow component, or any other explicit upstream zero.

Zero must not be treated as missing.

## Forbidden dry-run behavior

The dry-run must not emit, create, infer, prepare, or trigger:

* buy instruction;
* sell instruction;
* hold instruction;
* allocation advice;
* target weights;
* target price;
* position sizing;
* order generation;
* execution instruction;
* broker-ready payload;
* trade ticket;
* urgency label;
* conviction label or score;
* ranking;
* best-pick language;
* watchlist mutation;
* portfolio mutation;
* Telegram/email/user notification;
* scheduler behavior;
* production report write;
* live provider fetch;
* live market data fetch.

Forbidden concepts may appear in contract guardrails, tests, and audit documents only as prohibited behavior.

## Side-effect boundary

ME-RUN01 approves no side effects except future local test execution and optional local non-production dry-run artifacts if a future implementation sprint explicitly defines their path and retention policy.

The dry-run must not write to production data folders, broker-connected folders, user-facing report folders, Telegram queues, email queues, portfolio state, watchlist state, or scheduler state.

## Failure model

The dry-run must fail closed when:

* required input is missing;
* input is malformed;
* a contract version is unsupported;
* ticker/entity identity is inconsistent;
* provenance is missing where required;
* a stage emits prohibited language or fields;
* a blocked upstream state is softened or ignored;
* missing data is converted to zero;
* stale data is treated as current;
* any side-effect channel is requested.

Fail-closed output must preserve the reason and the stage where the failure occurred.

## Future implementation requirements

Recommended next sprint:

```text
ME-RUN02 - Implement end-to-end dry-run harness
```

ME-RUN02 must:

* consume only approved contract payloads or deterministic fixtures;
* emit `market-engine-end-to-end-dry-run-v1`;
* preserve stage-by-stage contract identity;
* preserve blocked states;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* preserve provenance;
* include local synthetic tests only;
* test completed, completed-with-limitations, blocked, unsupported-input, malformed-input, stale-data, missing-data, numeric-zero, and contract-violation cases;
* avoid provider calls;
* avoid live market data calls;
* avoid broker calls;
* avoid Telegram/email delivery;
* avoid portfolio/watchlist writes;
* avoid scheduler behavior;
* avoid user-facing production report generation;
* avoid ranking, conviction, urgency, target-price, buy/sell/hold, allocation, or execution semantics.

## Explicit non-goals

ME-RUN01 does not implement:

* Python runtime code;
* tests;
* a CLI command;
* Streamlit/UI behavior;
* scheduler behavior;
* live provider fetches;
* SEC/EDGAR calls;
* market data calls;
* broker integration;
* Telegram/email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* Decision Engine decisions;
* new financial analysis logic;
* real-data all-ticker execution;
* generated dry-run artifacts.

## Acceptance criteria for ME-RUN02

ME-RUN02 is complete only when:

* the dry-run harness validates every approved stage contract it consumes;
* the dry-run output contract is `market-engine-end-to-end-dry-run-v1`;
* blocked upstream state remains blocked;
* stage-level and run-level states are deterministic;
* required provenance is preserved;
* missing/stale markers are preserved;
* numeric zero is preserved;
* forbidden action/allocation/delivery language is not emitted;
* tests are synthetic and provider-free;
* no live provider, broker, Telegram, email, portfolio, watchlist, scheduler, production report, or execution side effects are introduced.

## Next sprint

```text
ME-RUN02 - Implement end-to-end dry-run harness
```
