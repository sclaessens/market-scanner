# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-RUN07

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-RUN07.

ME-RUN07 proved that the Market Engine can run end-to-end locally through `local_snapshot_fixture` using realistic non-production fixture data and can persist a deterministic local dry-run review artifact through the existing RUN05 artifact flag.

The next approved sprint is preserved as `ME-RUN08 - Expand local fixture matrix coverage for multiple dry-run states`.

## Completed Chain

Completed job-scoped chain:

| Sprint   | Job family               | Status    |
| -------- | ------------------------ | --------- |
| ME-SR01  | Source Refresh           | Completed |
| ME-SC01  | Source Context           | Completed |
| ME-SC02  | Source Context           | Completed |
| ME-FO01  | Fundamental Observations | Completed |
| ME-FO02  | Fundamental Observations | Completed |
| ME-DO01  | Derived Observations     | Completed |
| ME-AR01  | Analysis Review          | Completed |
| ME-AR02  | Analysis Review          | Completed |
| ME-RM01  | Roadmap / Governance     | Completed |
| ME-SD01  | Setup Detection          | Completed |
| ME-SD02  | Setup Detection          | Completed |
| ME-AR03  | Analysis Review          | Completed |
| ME-AR04  | Analysis Review          | Completed |
| ME-RR03  | Recommendation Review    | Completed |
| ME-RR04  | Recommendation Review    | Completed |
| ME-PR01  | Portfolio Review         | Completed |
| ME-PR02  | Portfolio Review         | Completed |
| ME-DE01  | Decision Engine handoff  | Completed |
| ME-DE02  | Decision Engine handoff  | Completed |
| ME-DL01  | Delivery / Reporting     | Completed |
| ME-DL02  | Delivery / Reporting     | Completed |
| ME-RUN05 | Run / orchestration      | Completed |
| ME-RUN06 | Run / orchestration      | Completed |
| ME-RUN07 | Run / orchestration      | Completed |

## Recent RUN chain

ME-RUN05 implemented local dry-run artifact persistence with:

* input contract: `market-engine-end-to-end-dry-run-v1`;
* artifact contract: `market-engine-local-dry-run-artifact-v1`;
* manifest contract: `market-engine-local-dry-run-artifact-manifest-v1`;
* approved path category: `artifacts/market_engine/dry_runs/`;
* module: `src/market_engine/run/local_dry_run_artifacts.py`;
* command integration: `src/market_engine/run/end_to_end_dry_run_command.py`;
* tests: `tests/market_engine/run/test_local_dry_run_artifacts.py`;
* implementation documentation: `docs/market_engine/run/me_run05_local_dry_run_artifact_persistence_implementation.md`;
* audit: `docs/market_engine/audits/me_run05_local_dry_run_artifact_persistence_audit.md`.

ME-RUN05 preserved stdout-only dry-run behavior by default and requires explicit `--write-local-artifact` invocation before local artifact writing.

ME-RUN06 implemented local dry-run fixture/data input with:

* input fixture contract: `market-engine-local-dry-run-input-fixture-v1`;
* approved local command mode: `--input-mode local_snapshot_fixture` with `--stage-payloads-json`;
* runtime module: `src/market_engine/run/local_dry_run_inputs.py`;
* command integration: `src/market_engine/run/end_to_end_dry_run_command.py`;
* tests: `tests/market_engine/run/test_local_dry_run_inputs.py` and `tests/market_engine/run/test_end_to_end_dry_run_command.py`;
* implementation documentation: `docs/market_engine/run/me_run06_local_dry_run_fixture_data_input_implementation.md`;
* audit: `docs/market_engine/audits/me_run06_local_dry_run_fixture_data_input_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run06_local_dry_run_fixture_data_input_backlog_entry.md`.

ME-RUN06 preserved embedded synthetic dry-run behavior by default, preserved raw `explicit_in_memory_payload` compatibility, and required a non-production wrapper for `local_snapshot_fixture` data input.

ME-RUN07 executed a realistic local fixture dry-run and persisted a review artifact with:

* fixture: `tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json`;
* tests: `tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py`;
* implementation / execution documentation: `docs/market_engine/run/me_run07_realistic_local_fixture_dry_run_artifact_execution.md`;
* audit: `docs/market_engine/audits/me_run07_realistic_local_fixture_dry_run_artifact_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run07_realistic_local_fixture_dry_run_artifact_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run07_realistic_local_fixture_dry_run_artifact_roadmap_entry.md`.

ME-RUN07 local validation confirmed:

```text
18 passed in 0.04s
```

ME-RUN07 proved:

* `local_snapshot_fixture` can drive the end-to-end dry-run command path;
* local artifact persistence remains opt-in only through `--write-local-artifact`;
* persisted artifacts are deterministic, inspectable, local, and non-production;
* numeric-zero evidence remains present;
* missing-data markers remain present;
* stale-data markers remain present;
* blocked stage and blocked reasons remain present;
* provenance remains present across all dry-run stages;
* overwrite protection prevents accidental replacement of an existing local artifact directory.

## Next Approved Sprint

### ME-RUN08 - Expand local fixture matrix coverage for multiple dry-run states

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: NEXT APPROVED SPRINT AFTER ME-RUN07

Goal: expand the local non-production dry-run fixture coverage from one realistic fixture into a deterministic fixture matrix for completed, limited, blocked, stale-data, missing-data, numeric-zero, unsupported-input, and provenance-heavy states.

Rationale: ME-RUN07 proves one realistic local blocked-review fixture. Before approving broader live execution, cached-source-to-review orchestration, channel adapters, user-facing reports, or production-style workflows, the local dry-run path should prove deterministic behavior across representative state families.

Scope: local fixtures, local tests, command documentation, and audit documentation only.

Non-goals: no provider calls, live data, broker calls, message delivery, portfolio or watchlist writes, production artifacts, new financial logic, action semantics, allocation behavior, order generation, ranking, scoring, urgency, conviction, or tradeability authority.

Acceptance criteria:

* fixture matrix is non-production;
* existing Market Engine payload contracts are reused;
* `local_snapshot_fixture` remains the local fixture input mode;
* existing embedded synthetic and explicit in-memory modes remain compatible;
* artifact writing remains opt-in only through the RUN05 flag;
* tests cover completed, limited, blocked, stale-data, missing-data, numeric-zero, unsupported-input, and provenance behavior;
* documentation records exact local commands, expected output, side-effect boundaries, and audit conclusion;
* roadmap and backlog remain synchronized after completion.

## Candidate Follow-Ups After ME-RUN08

These are candidates only and are not inserted ahead of ME-RUN08 unless ME-RUN08 discovers a blocker or governance gap:

* `ME-RUN09 - Define cached-source end-to-end local execution contract`;
* `ME-SR02 - Build bounded SEC CompanyFacts source refresh job runner`;
* `ME-QA01 - Add cross-job dry-run contract regression suite`.
