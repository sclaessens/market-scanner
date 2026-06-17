# ME-RUN01 / ME-RUN02 / ME-RUN03 / ME-RUN04 - Roadmap update

## Status

ACTIVE ROADMAP UPDATE AFTER ME-RUN04

## Purpose

This roadmap update preserves the Market Engine sprint sequence after ME-DL02 and records the Run / orchestration job family introduced by ME-RUN01, implemented by ME-RUN02, wired into a local command by ME-RUN03, and extended by ME-RUN04 with a local non-production dry-run artifact persistence contract.

The consolidated roadmap remains the historical chain reference. This file records the ME-RUN additions without rewriting unrelated roadmap history.

## Chain update

ME-DL02 implemented the controlled Delivery / Reporting output contract and emitted `market-engine-delivery-report-v1` as a non-actionable presentation payload.

ME-RUN01 defined the next integration boundary: an end-to-end dry-run contract that can connect the approved Market Engine chain in a deterministic local run and emit a dry-run summary.

ME-RUN02 implemented the first deterministic local dry-run harness for that boundary.

ME-RUN03 wired that harness into a runnable local terminal command that emits inspectable JSON.

ME-RUN04 defined the safe boundary for optional local non-production persistence of emitted dry-run JSON artifacts before any future artifact-writing implementation.

## Updated architectural chain

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

Decision Engine remains the only action/allocation authority.

Delivery / Reporting remains non-actionable.

End-to-end dry-run summary remains a local integration-review artifact only.

Local dry-run command output remains stdout JSON. ME-RUN04 approves only a future optional local non-production artifact persistence boundary and does not implement artifact writes.

## Completed chain addition

| Sprint | Job family | Status |
| --- | --- | --- |
| ME-RUN01 | Run / orchestration | Completed |
| ME-RUN02 | Run / orchestration | Completed |
| ME-RUN03 | Run / orchestration | Completed |
| ME-RUN04 | Run / orchestration | Completed |

ME-RUN01 defined:

* output contract: `market-engine-end-to-end-dry-run-v1`;
* approved upstream contract families through `market-engine-delivery-report-v1`;
* approved input modes: `synthetic_contract_fixture`, `local_snapshot_fixture`, and `explicit_in_memory_payload`;
* stage-level statuses;
* run-level states;
* required stage coverage;
* provenance preservation;
* missing-data preservation;
* stale-data preservation;
* numeric-zero preservation;
* fail-closed behavior;
* forbidden side effects and action/allocation/delivery semantics;
* contract document: `docs/market_engine/run/me_run01_end_to_end_dry_run_contract.md`;
* audit: `docs/market_engine/audits/me_run01_end_to_end_dry_run_contract_audit.md`;
* implementation sprint: `ME-RUN02 - Implement end-to-end dry-run harness`.

ME-RUN02 implemented:

* output contract: `market-engine-end-to-end-dry-run-v1`;
* module: `src/market_engine/run/end_to_end_dry_run.py`;
* package exports: `src/market_engine/run/__init__.py`;
* tests: `tests/market_engine/run/test_end_to_end_dry_run.py`;
* implementation documentation: `docs/market_engine/run/me_run02_end_to_end_dry_run_implementation.md`;
* audit: `docs/market_engine/audits/me_run02_end_to_end_dry_run_implementation_audit.md`.

ME-RUN02 validates stage-by-stage contract identity, preserves blocked states, missing-data markers, stale-data markers, numeric-zero values, provenance, and delivery report references, and derives deterministic dry-run states.

ME-RUN02 did not introduce provider calls, live market data calls, SEC/EDGAR calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, order generation, or execution advice.

ME-RUN03 implemented:

* local command module: `src/market_engine/run/end_to_end_dry_run_command.py`;
* console script: `market-engine-dry-run`;
* runnable module command: `python -m market_engine.run.end_to_end_dry_run_command`;
* embedded deterministic synthetic fixture mode;
* explicit local JSON stage-payload input mode;
* stdout JSON output for `market-engine-end-to-end-dry-run-v1`;
* tests: `tests/market_engine/run/test_end_to_end_dry_run_command.py`;
* implementation documentation: `docs/market_engine/run/me_run03_dry_run_local_command_implementation.md`;
* audit: `docs/market_engine/audits/me_run03_dry_run_local_command_implementation_audit.md`.

ME-RUN03 made the dry-run locally runnable without introducing provider calls, live market data calls, SEC/EDGAR calls, broker calls, Telegram/email delivery, production report generation, dry-run artifact persistence, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, order generation, or execution advice.

ME-RUN04 defined:

* approved upstream input: already-built `market-engine-end-to-end-dry-run-v1` payloads only;
* future local artifact metadata contract: `market-engine-local-dry-run-artifact-v1`;
* approved non-production path category: `artifacts/market_engine/dry_runs/`;
* filename requirements;
* deterministic JSON serialization requirements;
* retention expectations;
* fail-closed persistence behavior;
* numeric-zero preservation;
* missing-data preservation;
* stale-data preservation;
* blocked-state preservation;
* provenance preservation;
* contract document: `docs/market_engine/run/me_run04_local_dry_run_artifact_persistence_contract.md`;
* audit: `docs/market_engine/audits/me_run04_local_dry_run_artifact_persistence_contract_audit.md`;
* implementation candidate: `ME-RUN05 - Implement local dry-run artifact persistence`.

ME-RUN04 did not introduce Python code, tests, runtime behavior, provider calls, live market data calls, SEC/EDGAR calls, broker calls, Telegram/email delivery, production report generation, artifact writes, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, order generation, or execution advice.

## Future roadmap note

No implementation sprint is executed by ME-RUN04.

The logical next candidate is `ME-RUN05 - Implement local dry-run artifact persistence`.

That future implementation must remain narrowly scoped to optional local JSON artifact persistence under `artifacts/market_engine/dry_runs/` and must preserve all ME-RUN04 side-effect boundaries.

Future work beyond local artifacts, such as real-data intake governance, a safe all-ticker dry-run contract, or an operator review/reporting workflow, still requires a separate approved roadmap/backlog sprint.
