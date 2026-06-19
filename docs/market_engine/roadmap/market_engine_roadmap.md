# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-RUN16

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-RUN16.

ME-RUN07 proved that the Market Engine can run end-to-end locally through `local_snapshot_fixture` using realistic non-production fixture data and can persist a deterministic local dry-run review artifact through the existing RUN05 artifact flag.

ME-RUN08 expanded local non-production dry-run coverage into a deterministic fixture matrix for completed, completed-with-limitations, blocked, stale-data, missing-data, numeric-zero, unsupported-input, and provenance-heavy states.

ME-RUN09 defined the cached-source end-to-end local execution contract as the next safe boundary toward real local analysis from already-existing cached source snapshots.

ME-RUN10 implemented that cached-source local execution path from already-existing cached SEC CompanyFacts snapshots into the approved Market Engine dry-run chain.

ME-RUN11 validated that same cached-source path against a small deterministic ticker bundle using the existing per-ticker command path.

ME-RUN12 defined the safe future contract for broader all-ticker cached-source batch dry-runs without approving implementation, provider refresh, production execution, delivery, portfolio/watchlist mutation, scheduler behavior, UI behavior, or action/allocation authority.

ME-RUN13 implemented the safe cached-source batch dry-run runtime behavior defined by ME-RUN12.

ME-UNI01 defined the canonical ticker universe contract required before broader canonical-universe RUN and Telegram sequencing can proceed.

ME-UNI02 implemented the canonical ticker universe loader and validation layer required by ME-UNI01. ME-RUN16 remains the downstream RUN sprint that may consume the validated canonical universe.

ME-RUN16 consumed the canonical ticker universe in the cached-source batch dry-run command and executed the first canonical-universe batch. The execution selected 13 active `cached_source_only` tickers, excluded SMCI as `manual_review_only`, and failed closed for every selected ticker because no local cached SEC CompanyFacts source snapshots were present.

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
| ME-RUN08 | Run / orchestration      | Completed |
| ME-RUN09 | Run / orchestration      | Completed |
| ME-RUN10 | Run / orchestration      | Completed |
| ME-RUN11 | Run / orchestration      | Completed |
| ME-RUN12 | Run / orchestration      | Completed |
| ME-RUN13 | Run / orchestration      | Completed |
| ME-UNI01 | Ticker Universe          | Completed |
| ME-UNI02 | Ticker Universe          | Completed |
| ME-UNI03 | Ticker Universe          | Completed |
| ME-RUN16 | Run / orchestration      | Completed with blocked ticker outcome |

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

ME-RUN08 expanded local fixture matrix coverage with:

* deterministic cases for completed, completed-with-limitations, blocked, stale-data, missing-data, numeric-zero, unsupported-input, and provenance-heavy states;
* test: `tests/market_engine/run/test_me_run08_local_fixture_matrix_coverage.py`;
* implementation documentation: `docs/market_engine/run/me_run08_local_fixture_matrix_coverage_implementation.md`;
* audit: `docs/market_engine/audits/me_run08_local_fixture_matrix_coverage_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run08_local_fixture_matrix_coverage_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run08_local_fixture_matrix_coverage_roadmap_entry.md`.

ME-RUN08 validation confirmed:

```text
9 passed in 0.03s
49 passed in 0.08s
```

ME-RUN08 preserved all local-only, non-production, opt-in artifact, no-provider, no-delivery, no-portfolio-write, no-watchlist-write, no-scheduler, no-UI, and non-actionable boundaries.

ME-RUN09 defined the cached-source end-to-end local execution contract with:

* future input mode: `cached_source_snapshot`;
* future input contract family: `market-engine-cached-source-local-execution-input-v1`;
* approved cached-source path category: `data/market_engine/source_snapshots/`;
* final output contract preserved as `market-engine-end-to-end-dry-run-v1`;
* local artifact contracts preserved as `market-engine-local-dry-run-artifact-v1` and `market-engine-local-dry-run-artifact-manifest-v1`;
* contract document: `docs/market_engine/run/me_run09_cached_source_end_to_end_local_execution_contract.md`;
* audit: `docs/market_engine/audits/me_run09_cached_source_end_to_end_local_execution_contract_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run09_cached_source_end_to_end_local_execution_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run09_cached_source_end_to_end_local_execution_roadmap_entry.md`.

ME-RUN09 is documentation-only and does not introduce Python code, tests, runtime behavior, provider calls, source refresh jobs, live market data calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production runs, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

ME-RUN10 implemented cached-source local execution with:

* input mode: `cached_source_snapshot`;
* wrapper input contract: `market-engine-cached-source-local-execution-input-v1`;
* final output contract preserved as `market-engine-end-to-end-dry-run-v1`;
* local artifact contracts preserved as `market-engine-local-dry-run-artifact-v1` and `market-engine-local-dry-run-artifact-manifest-v1`;
* runtime module: `src/market_engine/run/cached_source_execution.py`;
* command integration: `src/market_engine/run/end_to_end_dry_run_command.py`;
* tests: `tests/market_engine/run/test_me_run10_cached_source_local_execution.py`;
* implementation documentation: `docs/market_engine/run/me_run10_cached_source_local_execution_implementation.md`;
* audit: `docs/market_engine/audits/me_run10_cached_source_local_execution_implementation_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run10_cached_source_local_execution_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run10_cached_source_local_execution_roadmap_entry.md`.

ME-RUN10 remains local, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

ME-RUN10 does not introduce source refresh jobs, provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, all-ticker production runs, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

ME-RUN11 validated cached-source local execution with:

* deterministic ticker bundle: `NVDA`, `MSFT`, `AMD`;
* per-ticker output contract preserved as `market-engine-end-to-end-dry-run-v1`;
* input mode preserved as `cached_source_snapshot`;
* test: `tests/market_engine/run/test_me_run11_cached_source_ticker_bundle_execution.py`;
* implementation documentation: `docs/market_engine/run/me_run11_cached_source_ticker_bundle_execution.md`;
* audit: `docs/market_engine/audits/me_run11_cached_source_ticker_bundle_execution_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run11_cached_source_ticker_bundle_execution_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run11_cached_source_ticker_bundle_execution_roadmap_entry.md`.

ME-RUN11 remains local, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

ME-RUN11 does not introduce a batch runner, source refresh jobs, provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, all-ticker production execution, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

ME-RUN12 defined safe all-ticker cached-source batch dry-run behavior with:

* future batch-level contract family: `market-engine-cached-source-batch-dry-run-v1`;
* approved cached-source root boundary: `data/market_engine/source_snapshots/`;
* explicit future input mode direction: `cached_source_batch`;
* approved local ticker universe sources;
* forbidden live/provider/broker/watchlist ticker universe sources;
* deterministic cached-source discovery rules;
* ambiguity handling rules;
* per-ticker execution and failure isolation;
* final per-ticker output preserved as `market-engine-end-to-end-dry-run-v1`;
* batch summary and count expectations;
* opt-in local artifact expectations;
* operator visibility requirements;
* missing-data, stale-data, blocked-state, numeric-zero, and provenance preservation;
* fail-closed batch and ticker-level behavior;
* contract document: `docs/market_engine/run/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract.md`;
* audit: `docs/market_engine/audits/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_backlog_entry.md`;
* roadmap entry: `docs/market_engine/roadmap/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_roadmap_entry.md`.

ME-RUN12 is documentation-only and does not introduce Python code, tests, fixtures, provider calls, source refresh jobs, SEC/EDGAR live calls, yfinance calls, live market data calls, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production execution, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

ME-RUN13 implemented safe all-ticker cached-source batch dry-run behavior with:

* batch contract: `market-engine-cached-source-batch-dry-run-v1`;
* per-ticker output contract preserved as `market-engine-end-to-end-dry-run-v1`;
* runtime module: `src/market_engine/run/cached_source_batch_execution.py`;
* deterministic cached-source discovery under an explicit local root;
* explicit requested ticker support;
* deterministic cached-ticker discovery mode;
* per-ticker failure isolation;
* missing, invalid, unsupported, ambiguous, downstream, and unexpected local error states;
* local batch artifact writing only when explicitly requested;
* test: `tests/market_engine/run/test_me_run13_cached_source_batch_dry_run.py`;
* implementation documentation: `docs/market_engine/run/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation.md`;
* audit: `docs/market_engine/audits/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation_audit.md`.

ME-RUN13 remains cached-source/local-only, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

ME-RUN13 does not introduce live provider calls, SEC/EDGAR fetches, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Next Implementation Candidate

### ME-RUN14 - Add cached-source batch dry-run command interface

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-RUN13

Goal: add a narrow operator-facing command interface for the ME-RUN13 cached-source batch dry-run runtime behavior.

Rationale: ME-RUN13 implements the safe batch behavior as a runtime function and artifact writer. A separate sprint should add any operator-facing command interface so command arguments, terminal output, artifact flags, and failure messages remain explicit and reviewable.

Scope: command interface, local argument parsing, terminal JSON output, opt-in artifact wiring, deterministic local tests, documentation, and audit only unless explicitly re-scoped.

Non-goals: no provider refresh, live market data, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, new financial logic, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

Acceptance criteria:

* command consumes only local cached-source inputs;
* command emits `market-engine-cached-source-batch-dry-run-v1`;
* command preserves per-ticker `market-engine-end-to-end-dry-run-v1`;
* artifact writing remains opt-in only;
* command has deterministic argument validation and fail-closed errors;
* tests cover command success, malformed arguments, missing root, artifact default-off, opt-in artifacts, and forbidden side effects;
* roadmap and backlog remain synchronized after completion.

## Candidate Follow-Ups After ME-RUN13

These are candidates only and are not inserted ahead of ME-RUN14 unless ME-RUN14 discovers a blocker or governance gap:

* `ME-SR02 - Build bounded SEC CompanyFacts source refresh job runner`;
* `ME-QA01 - Add cross-job dry-run contract regression suite`.

## Completed Sprint

### ME-UNI02 - Implement canonical ticker universe loading and validation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI02

Goal: implement the canonical ticker universe loader and validation layer defined by ME-UNI01.

ME-UNI02 implemented:

* contract version: `market-engine-canonical-ticker-universe-v1`;
* default canonical CSV path: `data/market_engine/ticker_universe/ticker_universe.csv`;
* explicit path override support for tests and future command integration;
* required-column validation;
* required-value validation;
* allowed-value validation;
* duplicate ticker and market rejection;
* active cached-source default selection;
* explicit inactive, blocked and manual-review-only inclusion when requested;
* deterministic ordering by priority, ticker and market;
* normalized typed entries and result metadata.

Implemented runtime:

```text
src/market_engine/ticker_universe/
```

Implemented tests:

```text
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni02_canonical_ticker_universe_loader_implementation.md
docs/market_engine/audits/me_uni02_canonical_ticker_universe_loader_audit.md
```

Outcome: Market Engine can now load and validate the canonical ticker universe without provider calls, live data, source refresh, Telegram behavior, portfolio writes, watchlist writes, Decision Engine behavior or action authority.

## Next Canonical-Universe RUN Candidate

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-UNI02

Goal: consume the ME-UNI02 canonical ticker universe loader in the cached-source batch dry-run path and execute the first real cached-source batch dry-run using the canonical universe.

Scope: cached-source/local-only RUN integration, canonical universe visibility, fail-closed invalid-universe behavior, local tests, documentation and audit only unless explicitly re-scoped.

Non-goals: no provider refresh, live market data, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, new financial logic, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH BLOCKED TICKER OUTCOME BY ME-RUN16

Goal: execute the first cached-source batch dry-run selected from the canonical ticker universe.

Outcome:

* canonical universe loaded from `data/market_engine/ticker_universe/ticker_universe.csv`;
* 14 canonical rows loaded;
* 13 active `cached_source_only` tickers selected;
* SMCI excluded because `source_policy=manual_review_only`;
* all 13 selected tickers returned `blocked_missing_cached_source`;
* no provider or live data fallback occurred;
* generated local batch manifest under `artifacts/market_engine/...`, not committed.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
```

Implemented tests:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run16_first_canonical_universe_cached_source_batch_dry_run_execution.md
docs/market_engine/audits/me_run16_first_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run16_first_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run16_first_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

## Next Source Refresh Candidate

### ME-SR02 - Produce bounded canonical-universe SEC CompanyFacts cached source snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN16

Goal: produce or validate bounded local SEC CompanyFacts cached source snapshots for the canonical universe so a later RUN sprint can execute downstream dry-runs from real cached source evidence.

Rationale: ME-RUN16 proved canonical-universe RUN selection and fail-closed behavior. It also showed that this checkout has no cached source snapshots under `data/market_engine/source_snapshots`, so every selected ticker blocks before downstream dry-run execution.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.
