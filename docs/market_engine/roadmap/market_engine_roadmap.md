# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-UNI09

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-UNI09.

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

ME-RUN17 fixed RUN cached-source discovery for the ME-SR02 source-refresh snapshot layout. It discovered 12 canonical-universe snapshots, generated 12 local dry-run artifacts, kept HO blocked as missing cached source, and preserved downstream blocked states without provider calls or action authority.

Since ME-RUN17, the project completed the supported-universe execution and readable-output chain through ME-RUN19, ME-SR05, ME-RUN20, ME-RUN21, ME-RUN22, ME-OUT01, ME-OUT02, ME-CANDIDATE01, and ME-CANDIDATE02.

ME-UNI09 then implemented controlled Professional Swing Universe expansion from non-actionable candidate-classification output. It preserves existing universe entries, includes only eligible candidates with valid proposed universe rows, excludes unsafe or ineligible candidates with explicit reasons, and remains non-actionable universe maintenance only.

The active planning direction is now expanded-universe execution. The project should scale from the current supported subset toward a larger Professional Swing Universe / target analysis universe before prioritizing additional polish, QA, delivery preview, portfolio-context persistence, Decision Engine handoff review hardening, or extra governance layers.

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
| ME-SR02  | Source Refresh           | Completed |
| ME-RUN17 | Run / orchestration      | Completed with downstream blocked outcome |
| ME-RUN19 | Run / orchestration      | Completed |
| ME-SR05  | Source Refresh / Source Coverage | Completed |
| ME-RUN20 | Run / orchestration      | Completed |
| ME-RUN21 | Run / orchestration      | Completed |
| ME-RUN22 | Run / orchestration      | Completed |
| ME-OUT01 | Output / Operator Reporting | Completed |
| ME-OUT02 | Output / Operator Reporting | Completed |
| ME-CANDIDATE01 | Candidate Classification | Completed |
| ME-CANDIDATE02 | Candidate Classification | Completed |
| ME-UNI09 | Ticker Universe | Completed |

## Active Next Direction

### ME-SR06 - Classify source support for expanded Professional Swing Universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: NEXT ACTIVE CANDIDATE AFTER ME-UNI09

Goal: classify cached-source support for the expanded Professional Swing Universe produced or proposed by ME-UNI09.

Scope: source-support classification only. Use existing approved local source artifacts and metadata. No provider calls, live data, source refresh, analysis changes, report changes, candidate-classification changes, delivery, portfolio/watchlist writes, or trading authority.

Rationale: the system now has a local dry-run, readable operator output, and non-actionable candidate classification path. The priority is to scale ticker coverage before adding more polish or governance layers.

### Planned Next Candidate After ME-SR06

```text
ME-RUN23 - Execute expanded supported-universe cached-source run and produce readable/candidate outputs
```

ME-RUN23 should execute the expanded supported-universe cached-source run and produce readable operator output plus non-actionable candidate classification over the larger supported universe after ME-SR06 classifies source support.

## Deferred Follow-up Candidates

These sprints are not rejected and not blocked. They are intentionally deferred below expanded-universe execution to avoid refinement loops before broad ticker execution is proven:

* ME-CANDIDATE03 - Candidate classification QA/review contract.
* ME-OUT03 - Operator report readability/polish improvements.
* ME-DL03 - Non-production delivery preview.
* ME-PR03 - Approved portfolio context source/persistence contract.
* ME-DE03 - Decision Engine handoff review hardening.
* ME-QAxx / ME-GOVxx - Additional QA/governance only from concrete run evidence.

## Scale-first Planning Rule

After ME-CANDIDATE02 and ME-UNI09, do not insert additional QA, polish, delivery, portfolio, governance, or candidate-classification refinement sprints ahead of ME-SR06 / ME-RUN23 unless a concrete blocker is discovered in local execution, source support, report generation, or candidate-classification output over the expanded universe.

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

## Historical Implementation Candidate

### ME-RUN14 - Add cached-source batch dry-run command interface

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: HISTORICAL CANDIDATE SUPERSEDED BY COMPLETED RUN COMMAND WORK

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

## Completed Sprint

### ME-RUN24 - Non-production portfolio-context fixture for expanded scans

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN24

Goal: unlock the next ME-RUN23 Portfolio Review blocker by allowing expanded cached-source scans to opt in to an explicit non-production portfolio-context fixture.

Outcome:

* ME-RUN23 source-support and cached-source scan selection remain unchanged;
* expanded scans can pass fixture-backed `portfolio_contexts_by_ticker` into cached-source batch execution only when explicitly requested;
* fixture provenance records absent versus non-production fixture context;
* fixture validation fails closed for missing, malformed, or unsupported input;
* all behavior remains local, non-production, non-actionable, and mutation-free.

Next: rerun the expanded cached-source scan locally with the non-production fixture enabled and inspect the next downstream state before planning follow-up work.

## Completed Sprint

### ME-SR07 - Cached-source snapshot acquisition plan for missing expanded universe entries

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR07

Roadmap position:

```text
ME-UNI09 -> ME-SR06 -> ME-RUN23 -> ME-RUN24 -> ME-SR07
```

ME-SR07 documents the current expanded-universe cached-source coverage baseline and plans the policy for future snapshot acquisition or staging. It does not acquire snapshots or add runtime provider behavior.

Next logical sprint:

```text
ME-SR08 - Define cached-source snapshot acquisition manifest contract
```

ME-SR08 should formalize acquisition metadata, checksum, stale-data, validation, and real/synthetic/derived classification fields before any future staging or acquisition implementation begins.

Future source-governance candidate:

```text
ME-SR12 - Define non-US ticker source-family and source-mapping governance contract
```

ME-SR12 is future work only. It should define how non-US tickers, ADRs, foreign listings, dual listings, and `needs_source_mapping` entries can be admitted into cached-source coverage through explicit source-family rules and source identity mapping. It must cover entries such as ASML, NVO, RHM, RR, ADYEN, and similar future entries without acquiring snapshots, implementing provider access, or marking non-US tickers supported merely because a current classifier can load a snapshot.

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR03

Goal: resolve or precisely document the source-coverage blockers exposed by ME-RUN19.

Outcome:

* ASML resolved through annual `20-F` `us-gaap` `EUR` source mapping;
* TSM resolved through annual `20-F` `ifrs-full` `USD` source mapping;
* HO remains blocked because no approved cached source snapshot exists locally;
* canonical rerun reached 12 completed tickers and 1 blocked ticker.

Implemented documentation:

```text
docs/market_engine/source_data/me_sr03_canonical_universe_cached_source_coverage_blockers.md
docs/market_engine/audits/me_sr03_canonical_universe_cached_source_coverage_blockers_audit.md
docs/market_engine/backlog/me_sr03_canonical_universe_cached_source_coverage_blockers_backlog_entry.md
docs/market_engine/roadmap/me_sr03_canonical_universe_cached_source_coverage_blockers_roadmap_entry.md
```

## Next Source Refresh Candidate

### ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR04

Goal: decide whether HO should receive an approved source identity/backfill path or be moved out of default cached-source execution until a valid source exists.

Rationale: ASML and TSM no longer block after source mapping remediation. HO remains the only canonical cached-source blocker.

Outcome:

* HO remains in the canonical universe as Thales on Euronext;
* HO source policy changed to `manual_review_only`;
* default canonical SEC CompanyFacts cached-source execution excludes HO and SMCI;
* canonical cached-source rerun selected 12 supported tickers and completed 12 with zero blocked tickers.

Scope: Source Refresh / source identity only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Universe Governance Sprint

### ME-UNI08 - Add first-class Professional Swing Universe CLI flag

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

Goal: expose the approved editable Professional Swing Universe as a first-class local cached-source batch runtime CLI choice.

Outcome:

* added `--professional-swing-universe` to the cached-source batch dry-run command;
* routed the flag through the existing ME-UNI07 runtime-input builder;
* preserved custom `--canonical-ticker-universe <path>` behavior;
* added mutual-exclusion failure behavior for Professional Swing and custom universe inputs.

Scope: ME-UNI08 did not introduce provider calls, source refresh, output/reporting behavior, delivery behavior, scheduler behavior, portfolio/watchlist writes, Decision Engine action semantics, or trading authority.

## Next Source Support Candidate

### ME-SR05 - Classify source support for Professional Swing Universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR05

Goal: classify actual cached-source support for Professional Swing Universe rows before broad supported-universe cached-source scanning.

Rationale: ME-UNI08 makes the editable Professional Swing Universe easy to select at runtime, but source-support classification remains a separate Source Refresh responsibility.

Outcome:

* implemented deterministic Professional Swing Universe source-support classification;
* consumed the validated editable Professional Swing Universe loader;
* classified local SEC CompanyFacts support from approved cached snapshots and provider error records;
* emitted explicit `supported_cached`, `missing_snapshot`, `unsupported_sec_companyfacts`, `missing_required_source_field`, `malformed_or_unreadable_source_artifact`, `ambiguous_identity`, `manual_review_only`, and `excluded` statuses;
* preserved source artifact references, provider error references, missing-field evidence, universe row references, and numeric-zero evidence;
* preserved source-support-only boundaries.

Planned sequence after ME-SR05:

```text
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-RUN21 - Inspect and summarize supported-universe cached-source scan outputs
ME-RUN22 - Produce first human-readable Market Engine interpretation report from cached-source supported-universe outputs
ME-OUT01 - Define readable operator report contract from dry-run artifacts
ME-OUT02 - Implement readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output
```

### ME-RUN20 - Execute clean supported-universe cached-source scan

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN20

Goal: execute a local cached-source scan against the currently supported active subset of the editable Professional Swing Universe and produce inspectable local artifacts.

Scope: ME-RUN20 should consume ME-SR05 source-support classification results and must keep unsupported, missing, malformed, ambiguous, manual-review-only, and excluded rows explicit instead of silently treating them as clean supported cached-source rows.

Outcome:

* executed the 12 ME-SR05-supported cached-source tickers through the existing local cached-source batch dry-run path;
* requested 12, discovered 12 cached snapshots, executed 12, completed 12;
* observed 0 blocked, 0 failed, 0 skipped, 0 missing, 0 ambiguous, 0 unsupported, and 0 stale source results inside the supported subset;
* wrote local non-production artifacts under `artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/`;
* did not commit generated artifacts by default;
* preserved cached-source/local-only and non-actionable boundaries.

### ME-RUN21 - Inspect and summarize supported-universe cached-source scan outputs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN21

Goal: inspect the ME-RUN20 supported-universe cached-source scan artifacts and summarize whether the outputs are complete, consistent, and usable as the basis for first human-readable Market Engine interpretation.

Outcome:

* inspected the ME-RUN20 artifact root under `artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/`;
* confirmed 12 ticker directories contain valid `dry_run.json` and `manifest.json`;
* confirmed all 12 ticker payloads use `market-engine-end-to-end-dry-run-v1`;
* confirmed all 12 ticker payloads completed all expected dry-run stages;
* observed no missing-data markers, stale-data markers, blocked stages, malformed JSON, or structural inconsistency in the supported subset;
* documented readiness for the next non-actionable interpretation/reporting sprint.

### ME-RUN22 - Produce first human-readable Market Engine interpretation report from cached-source supported-universe outputs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN22

Goal: produce the first human-readable, non-actionable Market Engine interpretation report from the ME-RUN20 cached-source supported-universe artifacts.

Scope: ME-RUN22 must preserve the non-actionable boundary. It may summarize and explain generated artifacts, but it must not introduce BUY / SELL / HOLD advice, allocation, ranking, scoring, target prices, urgency, conviction, tradeability, position sizing, execution instructions, broker-ready output, Telegram delivery, or production writes.

Outcome:

* implemented `market-engine-interpretation-report-v1`;
* added a deterministic local report generator for cached-source dry-run artifacts;
* generated Markdown and companion JSON summary outputs;
* preserved per-ticker artifact paths, stage states, missing/stale/blocked markers, and provenance references;
* handled missing and malformed ticker artifacts with explicit skipped reasons;
* added focused tests for happy path, missing files, malformed JSON, deterministic ordering, guardrail metadata, and CLI output;
* generated a local sample report under `artifacts/market_engine/me-run22-human-readable-report-me-run20-supported-universe-20260623T120000Z/`;
* did not commit generated local report artifacts by default;
* preserved the non-actionable, provider-free, local-only boundary.

### ME-OUT01 - Define readable operator report contract from dry-run artifacts

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

Status: COMPLETED BY ME-OUT01

Goal: define a readable, non-actionable operator report contract from generated dry-run artifacts without introducing delivery, trading authority, ranking, scoring, allocation, or execution behavior.

Outcome:

* defined `market-engine-readable-operator-report-v1`;
* formalized approved local dry-run artifact inputs;
* defined required Markdown report sections;
* defined required machine-readable companion summary fields;
* documented artifact integrity, stage completion, per-ticker summary, missing-data, stale-data, blocked-state, provenance, numeric-zero, fail-closed, and advisory-language guardrail requirements.

### ME-OUT02 - Implement readable operator report from dry-run artifacts

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

Status: COMPLETED BY ME-OUT02

Goal: implement `market-engine-readable-operator-report-v1` as a deterministic local operator report generator from existing dry-run artifacts.

Outcome:

* implemented a local report builder and CLI command module;
* emitted `operator_report.md` and `operator_report_summary.json`;
* preserved local artifact integrity, stage-state, missing-data, stale-data, blocked-state, provenance, and numeric-zero evidence;
* added explicit per-ticker skip reasons for incomplete, malformed, or unsupported artifacts;
* added focused tests and implementation/audit documentation;
* preserved local-only, provider-free, non-production, and non-actionable boundaries.

### ME-CANDIDATE01 - Define non-actionable candidate classification contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

Status: COMPLETED BY ME-CANDIDATE01

Goal: define a non-actionable candidate classification contract after readable operator reporting exists.

Outcome:

* defined `market-engine-candidate-classification-v1`;
* defined approved local readable operator and dry-run artifact inputs;
* defined fixed non-actionable candidate buckets;
* documented required evidence, provenance, missing-data, stale-data, blocked-state, malformed-artifact, unsupported-input, numeric-zero, deterministic-output, fail-closed, and advisory-language guardrail behavior.

### ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

Status: COMPLETED BY ME-CANDIDATE02

Goal: implement the ME-CANDIDATE01 candidate classification contract from readable operator output.

Outcome:

* implemented `market-engine-candidate-classification-v1`;
* added local candidate classification builder and CLI module;
* emitted `candidate_classification_report.md` and `candidate_classification_summary.json`;
* preserved evidence references, blocking reasons, safety flags, missing-data markers, stale-data markers, blocked-state markers, provenance presence, and numeric-zero evidence presence;
* added focused tests and implementation/audit documentation;
* preserved local-only, provider-free, non-production, non-actionable boundaries.

Next planning note: ME-CANDIDATE02 does not insert an immediate blocking follow-up. Candidate-classification QA/review, output readability polish, delivery-preview work, portfolio-context persistence, stronger Decision Engine handoff review, and additional governance remain valid deferred follow-up candidates. They should be picked up only after expanded-universe execution produces concrete inspection, QA, governance, or delivery evidence that justifies them, or if such a concrete blocker is discovered earlier. The active next direction is to scale from the current supported subset toward a larger Professional Swing Universe / target analysis universe and then execute readable/candidate outputs over that larger universe.

## Completed Sprint

### ME-RUN17 - Canonical-universe cached-source batch dry-run with ME-SR02 snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH DOWNSTREAM BLOCKED OUTCOME BY ME-RUN17

Goal: execute and fix canonical-universe cached-source batch dry-run behavior using ME-SR02 snapshots.

Outcome:

* fixed RUN discovery for `sec_companyfacts/<snapshot_id>/raw/*.json`;
* preserved older `*/raw/*.json` discovery;
* selected 13 canonical active `cached_source_only` tickers;
* excluded SMCI as `manual_review_only`;
* discovered 12 ME-SR02 raw snapshots;
* executed 12 local end-to-end dry-run payloads;
* kept HO blocked as missing cached source;
* generated 12 local per-ticker artifacts plus a batch manifest;
* preserved provider-free, local-only, non-actionable boundaries.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_execution.py
```

Implemented test change:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run17_canonical_universe_cached_source_batch_dry_run_with_me_sr02_snapshots.md
docs/market_engine/audits/me_run17_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run17_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run17_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

## Historical RUN Candidate

### ME-RUN18 - Provide portfolio context for canonical-universe cached-source dry-runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: HISTORICAL CANDIDATE SUPERSEDED BY COMPLETED ME-RUN18 / ME-RUN19 PATH

Goal: provide approved local portfolio context to canonical-universe cached-source dry-runs so downstream review stages can progress without production portfolio writes.

Rationale: ME-RUN17 now discovers ME-SR02 snapshots and executes 12 dry-run payloads, but the chain remains blocked downstream because required local portfolio context is not provided.

Scope: local cached-source RUN behavior only. No provider refresh, live market data, portfolio writes, watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-RUN19 - Portfolio-context-aware canonical cached-source dry-run

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN19

Goal: execute the canonical-universe cached-source batch dry-run with ME-SR02 snapshots and approved local non-production portfolio context.

Outcome:

* the existing ME-RUN18 command path ran successfully without runtime code changes;
* the run selected 13 active `cached_source_only` tickers and excluded SMCI as `manual_review_only`;
* 12 cached snapshots were discovered and executed;
* 10 tickers completed through Portfolio Review, Decision Engine handoff, Delivery / Reporting, and dry-run summary;
* ASML and TSM blocked at Recommendation Review after preserving missing-field evidence;
* HO blocked because no cached source snapshot exists locally;
* generated artifacts remain local uncommitted evidence.

Implemented documentation:

```text
docs/market_engine/run/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_execution.md
docs/market_engine/audits/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_audit.md
docs/market_engine/backlog/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_roadmap_entry.md
```

## Next Source Refresh Candidate

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN19

Goal: resolve the remaining cached-source coverage blockers exposed by ME-RUN19 before broader canonical-universe validation or Telegram preview work.

Rationale: local portfolio context is no longer the default blocker. Remaining blockers are cached-source coverage and canonical source-field completeness for HO, ASML, and TSM.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.
