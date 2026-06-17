# ME-RUN01 - End-to-end dry-run contract audit

## Status

COMPLETED BY ME-RUN01

## Sprint audited

ME-RUN01 - Define end-to-end dry-run contract

## Audit scope

This audit verifies that ME-RUN01 defines a dry-run contract only and does not introduce runtime behavior, provider access, broker integration, delivery channels, portfolio/watchlist mutation, scheduling, production writes, or trading/allocation authority.

Files introduced by ME-RUN01:

* `docs/market_engine/run/me_run01_end_to_end_dry_run_contract.md`
* `docs/market_engine/audits/me_run01_end_to_end_dry_run_contract_audit.md`
* `docs/market_engine/backlog/me_run01_end_to_end_dry_run_backlog_entry.md`
* `docs/market_engine/roadmap/me_run01_roadmap_update.md`

## Repository context inspected

ME-RUN01 was defined after the completed chain through ME-DL02.

Relevant inspected context:

* `docs/market_engine/roadmap/market_engine_roadmap.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/delivery_reporting/me_dl01_delivery_reporting_contract.md`
* `src/market_engine/delivery_reporting/sec_companyfacts_delivery_report.py`

The current chain ends with `market-engine-delivery-report-v1`, emitted by Delivery / Reporting from `market-engine-decision-engine-handoff-v1` input.

## Contract outcome

ME-RUN01 defines the future dry-run output contract:

```text
market-engine-end-to-end-dry-run-v1
```

The contract defines:

* the dry-run architectural position after Delivery / Reporting;
* approved upstream contract families;
* approved non-live input modes;
* dry-run output payload requirements;
* stage-level statuses;
* run-level states;
* required stage coverage;
* lineage preservation;
* missing-data handling;
* stale-data handling;
* numeric-zero preservation;
* fail-closed behavior;
* forbidden action/allocation/delivery semantics;
* side-effect boundaries;
* ME-RUN02 implementation requirements.

## Authority boundary audit

ME-RUN01 preserves existing authority boundaries.

The dry-run contract does not authorize:

* buy instruction;
* sell instruction;
* hold instruction;
* allocation advice;
* target weights;
* target price;
* position sizing;
* order generation;
* execution instruction;
* broker-ready output;
* trade ticket generation;
* urgency label;
* conviction label or score;
* ranking;
* best-pick language.

Decision Engine remains the only future action and allocation authority.

## Side-effect audit

ME-RUN01 introduces no side effects.

Not introduced:

* Python runtime code;
* tests;
* CLI behavior;
* Streamlit/UI behavior;
* scheduler behavior;
* live provider fetches;
* SEC/EDGAR calls;
* market data calls;
* broker integration;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* generated dry-run artifacts.

ME-RUN01 permits only future local, deterministic, non-production dry-run behavior if ME-RUN02 implements it under the defined contract.

## Data-handling audit

The contract preserves required Market Engine data rules:

* missing data remains explicit;
* stale data remains explicit;
* numeric zero remains valid evidence;
* missing numeric values are not converted to zero;
* stale evidence is not presented as current;
* blocked upstream states remain blocked;
* missing lineage remains missing rather than invented.

## Contract-boundary audit

The dry-run may orchestrate approved payloads, but it may not bypass any job-family contract.

The contract explicitly prevents the future dry-run from reading these as active canonical inputs:

* raw provider responses;
* old scanner output;
* old generated reports;
* watchlists;
* broker exports;
* legacy `scripts` runtime output;
* archived reference material.

## Backlog and roadmap audit

ME-RUN01 records the next logical implementation sprint:

```text
ME-RUN02 - Implement end-to-end dry-run harness
```

Because the existing consolidated backlog file is large and currently ends after ME-DL02, ME-RUN01 adds a dedicated backlog entry file under `docs/market_engine/backlog/` and a dedicated roadmap update file under `docs/market_engine/roadmap/`. This preserves the identified sprint sequence without rewriting unrelated backlog history.

## Acceptance result

ME-RUN01 is accepted as a documentation-only contract sprint.

It is complete because it:

* defines `market-engine-end-to-end-dry-run-v1`;
* defines approved input modes;
* defines stage and run states;
* defines fail-closed behavior;
* defines provenance, missing-data, stale-data, and numeric-zero requirements;
* defines explicit forbidden behavior;
* defines ME-RUN02 requirements;
* introduces no runtime code, tests, providers, brokers, delivery channels, portfolio/watchlist writes, scheduler behavior, generated reports, or execution authority.
