# Python Runtime Cleanup Developer Specification

Status: ACTIVE DEVELOPER SPECIFICATION

## 1. Purpose

This document is a developer-ready specification for a future Python runtime cleanup sprint.

This document does not authorize implementation by itself. Cleanup implementation requires explicit later approval.

Sprint C.3 changed no code, tests, CSV files, generated outputs, raw data, provider calls, scraping behavior, pipeline runs, or runtime behavior.

## 2. Architecture and Governance Baseline

The active runtime architecture remains:

scanner -> validation -> context -> fundamentals -> timing state -> portfolio intelligence -> Decision Engine -> reporting

The certified governance boundary remains unchanged:

- upstream layers classify;
- allocation happens downstream;
- `scripts/core/decision_engine.py` is the only allocation authority;
- Reporting communicates final Decision Engine outputs and must not create decision logic;
- Telegram delivery and command processing must not create allocation logic;
- no upstream layer may create tradeability, urgency, conviction, allocation, ranking, scoring, eligibility, buy/sell, or hidden filtering semantics;
- future cleanup must preserve deterministic outputs, row preservation, source traceability, and auditability.

Sprint D selected "Option A - Compatibility Wrapper First" for fundamentals history implementation. Therefore future fundamentals work must keep `scripts/core/build_fundamental_layer.py` as the pipeline-facing compatibility surface until a later governed replacement is approved.

## 3. Cleanup Objective

The Python runtime cleanup should reduce the number of active Python surfaces without breaking behavior.

The cleanup aims to:

- reduce duplicate or obsolete Python entrypoints;
- preserve active entrypoints used by workflows, tests, or manual operations;
- move useful logic from incorrect locations into coherent modules;
- merge duplicate diagnostic, reporting, portfolio, and legacy helper logic only after references are verified;
- keep compatibility wrappers where external or manual callers may still depend on old paths;
- remove only code proven obsolete, reference-free, and without unique required logic;
- avoid breaking scheduled workflows, manual operator workflows, tests, Reporting, Telegram, scanner, portfolio, Decision Engine, and future Sprint E fundamentals work.

## 4. Non-Scope

Sprint C.3 performs no implementation.

This sprint does not authorize:

- code edits;
- Python file moves;
- Python file deletions;
- import changes;
- refactors;
- test edits;
- test execution;
- CSV edits;
- generated artifact updates;
- raw data changes;
- provider/API calls;
- scraping;
- pipeline runs;
- Decision Engine changes;
- Reporting runtime changes;
- Telegram runtime changes;
- fundamentals runtime changes;
- portfolio runtime changes.

## 5. Cleanup Batch Strategy

Future cleanup should be batched so each step has a small blast radius and clear rollback.

### Batch 1 - Safe Documentation and Reference Alignment

Purpose:
Update active documentation, runbooks, and references after cleanup decisions are approved.

Allowed later only after approval:

- update docs/runbooks that point at removed or wrapped scripts;
- document compatibility entrypoints;
- do not move or delete code in this batch.

### Batch 2 - Safe Wrapper and Entrypoint Consolidation

Purpose:
Handle wrapper scripts and duplicate entrypoints after references are verified.

Candidates:

- `scripts/reporting/build_telegram_summary.py`;
- selected duplicate top-level validation entrypoints if confirmed unused;
- wrapper-like scripts with no unique logic.

Allowed future action:

- convert to explicit compatibility wrappers;
- or delete after references are removed and validation passes.

### Batch 3 - Diagnostics Consolidation

Purpose:
Move reusable diagnostic and validation-analysis logic into a coherent diagnostics module.

Candidates:

- `scripts/analyze_validation.py`;
- `scripts/validate_scans.py`;
- `scripts/core/analyze_validation.py`;
- `scripts/core/validate_scans.py`;
- `scripts/core/build_context_backfill.py`;
- `scripts/core/build_entry_quality_backfill.py`;
- `scripts/core/validator.py`;
- `scripts/core/log_scans.py`;
- `scripts/diagnostics/audit_data_coverage.py`.

Allowed future action:

- move reusable diagnostic logic under `scripts/diagnostics/`;
- keep CLI wrappers only where manual workflows still use them;
- delete old entrypoints only after references are removed and tests pass.

### Batch 4 - Portfolio Legacy Cleanup

Purpose:
Clean old portfolio scripts while preserving the active portfolio source contract.

Candidates:

- `scripts/portfolio/parse_trade_commands.py`;
- `scripts/portfolio/portfolio_manager.py`;
- `scripts/portfolio/test_portfolio.py`;
- `scripts/portfolio/reporter.py` if it is introduced or rediscovered later.

Allowed future action:

- preserve `scripts/portfolio/build_portfolio.py`;
- preserve `scripts/portfolio/evaluate_positions.py`;
- preserve Telegram command compatibility until inbound command workflows are governed;
- move obsolete portfolio logic to legacy or delete only after tests and reference checks.

### Batch 5 - Reporting and Telegram Compatibility Cleanup

Purpose:
Remove old report and Telegram surfaces only after the Reporting contract is protected.

Candidates:

- `scripts/reporting/reporter.py`;
- `scripts/reporting/build_telegram_summary.py`;
- Telegram wrappers only if duplicate and safe.

Allowed future action:

- keep `scripts/reporting/build_reporting_layer.py`;
- keep `scripts/reporting/send_telegram.py`;
- preserve the communication-only Reporting boundary;
- delete old scripts only after no active docs, workflows, tests, or manual operations reference them.

### Batch 6 - Watchlist Legacy Cleanup

Purpose:
Handle `scripts/watchlist/*.py` as legacy timing/watchlist surfaces.

Allowed future action:

- move to `legacy/` or delete only after confirming active pipeline independence;
- preserve useful logic only if it does not reintroduce execution, tradeability, urgency, ranking, scoring, or allocation semantics outside the Decision Engine.

### Batch 7 - Source-Data Helper Rationalization

Purpose:
Rationalize provider/source-data helper scripts after fundamentals and source-data scope is approved.

Candidates:

- `scripts/data_sources/common.py`;
- `scripts/data_sources/prefill_fundamentals.py`;
- `scripts/data_sources/prefill_portfolio_metadata.py`;
- provider-assisted or source-data intake helpers.

Allowed future action:

- do not delete before Sprint E/F source-data scope is clear;
- keep source-data helpers as `REQUIRES_REVIEW` or future source-data scope until raw-history and source-artifact contracts are approved.

## 6. File-Level Cleanup Specification

| File | Current classification | Future action | Batch | Logic to preserve | Target destination | Preconditions | Required tests/validation | Risk level | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `scripts/run_scan.py` | ACTIVE_ENTRYPOINT / PIPELINE_ORCHESTRATION | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Full daily scan orchestration | Current file | Explicit orchestration scope | Operator visibility tests, workflow check, focused smoke check if approved | High | Referenced by GitHub Actions and `scripts/run_full_pipeline.py`. |
| `scripts/run_full_pipeline.py` | ACTIVE_ENTRYPOINT / PIPELINE_ORCHESTRATION | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Full ordered pipeline execution | Current file | Explicit orchestration scope | Operator visibility tests, compile check, focused CLI smoke if approved | High | Also functions as a manual compatibility entrypoint. |
| `scripts/core/build_validation_layer.py` | ACTIVE_LIBRARY | KEEP_AS_IS | none | Validation classification builder | Current file | None for cleanup | Existing validation tests | Medium | Upstream classification layer. |
| `scripts/core/build_context_layer.py` | ACTIVE_LIBRARY | KEEP_AS_IS | none | Context classification builder | Current file | None for cleanup | Existing context tests | Medium | Upstream classification layer. |
| `scripts/core/build_fundamental_layer.py` | ACTIVE_LIBRARY / COMPATIBILITY_WRAPPER_CANDIDATE | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | Sprint E scope | Pipeline-facing fundamental quality compatibility | Current file first, future helper modules behind it | Explicit Sprint E approval | Fundamental Layer tests, compatibility tests, downstream tests | High | Sprint D requires compatibility wrapper first. |
| `scripts/core/build_timing_state_layer.py` | ACTIVE_LIBRARY / DOWNSTREAM_CONSUMER | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Timing-state classification | Current file | Explicit timing scope | Timing-state tests, Decision Engine boundary checks | High | Must not absorb watchlist legacy semantics. |
| `scripts/core/build_portfolio_intelligence.py` | ACTIVE_LIBRARY / DOWNSTREAM_CONSUMER | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Portfolio intelligence classification | Current file | Explicit portfolio intelligence scope | Portfolio intelligence tests, Decision Engine boundary checks | High | Must not determine allocation. |
| `scripts/core/decision_engine.py` | ACTIVE_LIBRARY / ONLY_ALLOCATION_AUTHORITY | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Final allocation decision logic | Current file | Explicit Decision Engine scope | Full Decision Engine contract tests | Critical | No cleanup should weaken this boundary. |
| `scripts/core/build_stability_layer.py` | ACTIVE_LIBRARY | KEEP_AS_IS | none | Stability metadata | Current file | None for cleanup | Stability tests | Medium | Reporting may communicate stability metadata only. |
| `scripts/core/data_fetcher.py` | ACTIVE_LIBRARY / DATA_FETCHER | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Market data fetching | Current file | Explicit data-fetching scope | Scanner/data-fetch tests and smoke checks if approved | High | Do not alter provider behavior during cleanup. |
| `scripts/core/scanner.py` | ACTIVE_LIBRARY / SCANNER | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Discovery logic | Current file | Explicit scanner scope | Scanner tests or smoke checks if approved | High | Scanner is upstream discovery only. |
| `scripts/core/indicators.py` | ACTIVE_LIBRARY | KEEP_AS_IS | none | Indicator calculations | Current file | None for cleanup | Existing or future indicator tests | Medium | Shared by scanner/data paths. |
| `scripts/core/regime.py` | ACTIVE_LIBRARY | KEEP_AS_IS | none | Market regime helper | Current file | None for cleanup | Existing or future regime tests | Medium | Shared classification helper. |
| `scripts/core/build_context_backfill.py` | DIAGNOSTIC_TOOL / BACKFILL_HELPER | MOVE_LOGIC_TO_NEW_MODULE | 3 | Input validation and backfill transformations | `scripts/diagnostics/` module | Confirm no active orchestration dependency | Existing backfill tests, compile check | Medium | Keep CLI wrapper if manually useful. |
| `scripts/core/build_entry_quality_backfill.py` | DIAGNOSTIC_TOOL / BACKFILL_HELPER | MOVE_LOGIC_TO_NEW_MODULE | 3 | Entry-quality backfill transformations | `scripts/diagnostics/` module | Confirm no active orchestration dependency | Existing backfill tests, compile check | Medium | Keep as diagnostic, not strategy logic. |
| `scripts/core/analyze_validation.py` | DIAGNOSTIC_TOOL | MOVE_LOGIC_TO_NEW_MODULE | 3 | Validation summary aggregation | `scripts/diagnostics/validation_analysis.py` | Confirm top-level duplicate relationship | Focused diagnostic tests, compile check | Medium | More current than top-level `scripts/analyze_validation.py`. |
| `scripts/core/validate_scans.py` | DIAGNOSTIC_TOOL | MOVE_LOGIC_TO_NEW_MODULE | 3 | Local processed-price validation logic | `scripts/diagnostics/scan_validation.py` | Confirm no active orchestration dependency | Focused diagnostic tests, compile check | Medium | Avoid provider calls in cleanup. |
| `scripts/core/log_scans.py` | DIAGNOSTIC_TOOL / UNKNOWN | REQUIRES_REVIEW | 3 | Scan logging behavior if still used | TBD | Reference verification | Compile check and any related tests | Medium | Do not delete without manual workflow review. |
| `scripts/core/validator.py` | LEGACY_TOOL | MOVE_TO_LEGACY | 3 | Any unique validation helpers | `legacy/` or `scripts/diagnostics/` if useful | Confirm no imports and no active docs | Compile check, grep reference check | Medium | Candidate only; no deletion approved here. |
| `scripts/analyze_validation.py` | LEGACY_TOOL / TOP_LEVEL_DIAGNOSTIC | DELETE_AFTER_MIGRATION | 3 | Group summary ideas if still useful | `scripts/diagnostics/validation_analysis.py` | Migrate useful logic or prove superseded | Focused diagnostic tests, grep reference check | Medium | Contains legacy operator output and xlsx writing. |
| `scripts/validate_scans.py` | LEGACY_TOOL / PROVIDER_DIAGNOSTIC | DELETE_AFTER_MIGRATION | 3 | Any still-needed historical validation concepts | `scripts/diagnostics/scan_validation.py` | Prove no active manual dependency; avoid provider calls | Focused diagnostic tests, grep reference check | High | Uses `yfinance`; cleanup must not call providers. |
| `scripts/diagnostics/audit_data_coverage.py` | ACTIVE_DIAGNOSTIC_TOOL | KEEP_BUT_REFACTOR_LATER | 3/7 | Coverage audit logic and target-universe support | Current file or future diagnostics package | Future diagnostics scope | Existing diagnostics tests | Medium | Source-data related; keep until source-data scope is clear. |
| `scripts/ops/capture_historical_evidence.py` | ACTIVE_OPS_TOOL | KEEP_AS_IS | none | Historical evidence capture | Current file | None for cleanup | Existing ops tests | Low | Not part of runtime pipeline. |
| `scripts/data_sources/common.py` | ACTIVE_LIBRARY / SOURCE_DATA_HELPER | KEEP_AS_IS | 7 | Shared source-data validation helpers | Current file | Future source-data scope | Existing source-data tests | Medium | Likely reusable for raw-history intake. |
| `scripts/data_sources/prefill_fundamentals.py` | DATA_SOURCE_TOOL / REPLACEMENT_CANDIDATE | KEEP_BUT_REFACTOR_LATER | 7 | Governed prefill validation patterns | Future raw-history intake helpers | Sprint E/F source-data approval | Existing prefill tests plus raw-history tests | High | Current schema is metric-like and not target raw history. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | ACTIVE_DATA_SOURCE_TOOL | KEEP_AS_IS | 7 | Portfolio metadata prefill workflow | Current file | Future source-data approval for changes | Existing portfolio metadata prefill tests | Medium | Separate from fundamentals migration. |
| `scripts/portfolio/build_portfolio.py` | ACTIVE_PORTFOLIO_TOOL | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Active portfolio source contract | Current file | Explicit portfolio scope | Portfolio source contract tests | High | Preserve active portfolio source behavior. |
| `scripts/portfolio/evaluate_positions.py` | ACTIVE_PORTFOLIO_TOOL | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Position evaluation support | Current file | Explicit portfolio scope | Portfolio tests or smoke checks if approved | High | Must not determine final allocation. |
| `scripts/portfolio/parse_trade_commands.py` | PORTFOLIO_TOOL / TELEGRAM_DEPENDENCY | MOVE_LOGIC_TO_NEW_MODULE | 4 | User transaction-command parsing and call to `log_trade` | Future governed portfolio command module | Telegram command compatibility plan | Command parser tests, Telegram command tests, grep reference check | High | Referenced by `scripts/telegram/process_telegram_commands.py`. |
| `scripts/portfolio/portfolio_manager.py` | LEGACY_PORTFOLIO_TOOL | DELETE_AFTER_MIGRATION | 4 | Transaction append and position rebuild logic if still needed | Future governed portfolio command module or legacy | Migrate useful logic; preserve manual workflow if active | Portfolio tests, command tests, grep reference check | High | Referenced by parser and legacy `scripts/portfolio/test_portfolio.py`. |
| `scripts/portfolio/test_portfolio.py` | TEST_ONLY_OR_DEV_HELPER | DELETE_AFTER_REFERENCE_REMOVAL | 4 | None if superseded by formal tests | None | Confirm no active test runner relies on it | Formal portfolio tests pass | Low | This is a script-level test helper, not a pytest file under `tests/`. |
| `scripts/portfolio/reporter.py` | NOT_PRESENT | REQUIRES_REVIEW | 4 | Not applicable | Not applicable | If reintroduced, classify before cleanup | Not applicable | Low | Exact file not present during Sprint C.3 inspection. |
| `scripts/reporting/build_reporting_layer.py` | ACTIVE_REPORTING_TOOL | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Authoritative Reporting Layer builder | Current file | Explicit Reporting scope | Reporting tests and no-hidden-filtering checks | High | Reporting communicates only. |
| `scripts/reporting/build_telegram_summary.py` | COMPATIBILITY_WRAPPER | CONVERT_TO_COMPATIBILITY_WRAPPER | 2/5 | Delegation to authoritative Reporting builder | Current file as explicit wrapper | Confirm wrapper has no independent semantics | Reporting and Telegram summary tests | Medium | Already delegates to `build_reporting_layer.py`; may remain as stable entrypoint. |
| `scripts/reporting/send_telegram.py` | ACTIVE_TELEGRAM_DELIVERY | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Delivery-only Telegram summary send | Current file | Explicit Telegram delivery scope | Delivery smoke tests with no network unless approved | High | Imported by `scripts/run_scan.py`; do not alter in cleanup. |
| `scripts/reporting/reporter.py` | LEGACY_REPORTING_TOOL | DELETE_AFTER_REFERENCE_REMOVAL | 5 | None unless unique markdown reporting is still needed | None or `legacy/` | Confirm no active docs/actions/manual flows use it | Reporting tests, grep reference check | Medium | Historical Sprint 8 evidence says this was quarantined as legacy. |
| `scripts/telegram/process_telegram_commands.py` | ACTIVE_TELEGRAM_COMMAND_TOOL | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE | none | Inbound Telegram command handling | Current file | Explicit Telegram command scope | Command tests, compile check | High | Depends on portfolio command parser. |
| `scripts/utils/utils.py` | LEGACY_UTILITY / UNREFERENCED_BY_ACTIVE_CODE | DELETE_AFTER_REFERENCE_REMOVAL | 3 | Any helper still used manually | Existing module destinations if needed | Confirm no imports in active code | Compile check, grep reference check | Low | Grep found no active project imports. |
| `scripts/watchlist/auto_watchlist_from_scan.py` | LEGACY_WATCHLIST_TOOL | MOVE_TO_LEGACY | 6 | None unless timing-state-safe helper exists | `legacy/` only after approval | Confirm no active pipeline dependency | Compile check, grep reference check, no forbidden semantics check | High | Must not recreate allocation-like watchlist behavior. |
| `scripts/watchlist/build_watchlist.py` | LEGACY_WATCHLIST_TOOL | MOVE_TO_LEGACY | 6 | Active watchlist file build logic if still manually needed | `legacy/` or governed timing input module | Confirm manual workflow needs | Compile check, grep reference check | High | Treat as legacy until watchlist policy is refreshed. |
| `scripts/watchlist/evaluate_watchlist.py` | LEGACY_WATCHLIST_TOOL | MOVE_TO_LEGACY | 6 | Timing-context ideas only if governance-safe | `legacy/` or governed timing module | Confirm no execution semantics leak | Compile check, no forbidden semantics check | High | Existing comments/output include legacy language concerns. |
| `scripts/watchlist/parse_watchlist_commands.py` | LEGACY_WATCHLIST_TOOL | MOVE_TO_LEGACY | 6 | Manual watchlist command parsing if still needed | `legacy/` or governed timing command module | Confirm operator workflow | Command tests, no forbidden semantics check | High | Must not emit tradeability or allocation semantics. |
| `scripts/watchlist/update_watchlist_actions.py` | LEGACY_WATCHLIST_TOOL | MOVE_TO_LEGACY | 6 | None unless timing-only state updates are still needed | `legacy/` | Confirm not active in pipeline | Compile check, no forbidden semantics check | High | `run_full_pipeline.py` says legacy watchlist action updater is excluded. |
| `scripts/parse_trade_commands.py` | NOT_PRESENT | REQUIRES_REVIEW | none | Not applicable | Not applicable | If reintroduced, classify before cleanup | Not applicable | Low | Actual file is `scripts/portfolio/parse_trade_commands.py`. |
| `scripts/portfolio_manager.py` | NOT_PRESENT | REQUIRES_REVIEW | none | Not applicable | Not applicable | If reintroduced, classify before cleanup | Not applicable | Low | Actual file is `scripts/portfolio/portfolio_manager.py`. |

## 7. Logic Relocation Specifications

### `scripts/analyze_validation.py`

- Current role: legacy top-level validation summary script.
- Problem: overlaps conceptually with `scripts/core/analyze_validation.py` and writes a legacy xlsx summary outside the current processed artifact pattern.
- Useful logic to preserve: group-by summary concepts only if still useful.
- Recommended target: `scripts/diagnostics/validation_analysis.py`.
- Future implementation steps: compare outputs with `scripts/core/analyze_validation.py`, keep one canonical diagnostic implementation, add a small wrapper only if manual workflows need the old path.
- References to update: active docs/runbooks and any manual command notes; archived docs do not block cleanup.
- Required tests: focused diagnostic summary tests and `python -m compileall scripts`.
- Deletion or legacy preconditions: no active references, useful logic migrated or proven superseded, rollback by restoring wrapper.
- Rollback plan: keep a compatibility wrapper for one sprint if manual usage is uncertain.
- Risk level: Medium.

### `scripts/validate_scans.py`

- Current role: legacy top-level scan validation script using provider-backed future price download.
- Problem: provider calls and legacy scan validation behavior do not belong in a cleanup sprint and overlap with local processed-price validation.
- Useful logic to preserve: historical validation concepts only if still needed.
- Recommended target: `scripts/diagnostics/scan_validation.py`, without live provider calls unless separately approved.
- Future implementation steps: isolate provider-dependent logic, prefer local deterministic validation, and keep provider calls out of routine cleanup.
- References to update: active docs/runbooks if they mention this top-level script.
- Required tests: focused scan validation tests, provider-free fixtures, compile check.
- Deletion or legacy preconditions: no active references, no manual dependency, no unique required provider workflow.
- Rollback plan: move to legacy first instead of deleting if manual use is uncertain.
- Risk level: High.

### `scripts/core/analyze_validation.py`

- Current role: current processed validation summary builder.
- Problem: diagnostic code lives under `scripts/core/`, which should primarily host active runtime layers.
- Useful logic to preserve: validation result reading, grouping, summary output schema.
- Recommended target: `scripts/diagnostics/validation_analysis.py`.
- Future implementation steps: move implementation to diagnostics, leave a compatibility wrapper if imports or manual commands use the old path.
- References to update: tests and docs that call the core path.
- Required tests: existing validation summary tests or new diagnostics tests, compile check.
- Deletion or legacy preconditions: callers updated or wrapper retained.
- Rollback plan: restore wrapper to old path.
- Risk level: Medium.

### `scripts/core/validate_scans.py`

- Current role: current local processed-price scan validation builder.
- Problem: diagnostic validation logic lives under `scripts/core/` and is not part of the active classification pipeline.
- Useful logic to preserve: local price-data validation, output columns, no-provider deterministic behavior.
- Recommended target: `scripts/diagnostics/scan_validation.py`.
- Future implementation steps: move logic behind a diagnostics module and keep old CLI wrapper if needed.
- References to update: docs, tests, and manual commands.
- Required tests: focused scan validation tests, compile check.
- Deletion or legacy preconditions: callers updated or wrapper retained.
- Rollback plan: restore old path wrapper.
- Risk level: Medium.

### `scripts/core/build_context_backfill.py`

- Current role: context backfill helper with tested input validation.
- Problem: backfill and repair utilities are diagnostics/ops concerns, not core runtime classification entrypoints.
- Useful logic to preserve: `validate_scans_input` and deterministic backfill transformations.
- Recommended target: `scripts/diagnostics/` or `scripts/ops/` depending on approved future ownership.
- Future implementation steps: move implementation, preserve wrapper if manual workflows depend on old path.
- References to update: `tests/core/test_build_context_backfill.py` and documentation references.
- Required tests: existing backfill tests.
- Deletion or legacy preconditions: none until tests are moved and callers are updated.
- Rollback plan: keep old file as wrapper.
- Risk level: Medium.

### `scripts/core/build_entry_quality_backfill.py`

- Current role: entry-quality backfill helper.
- Problem: backfill code is not a core runtime layer.
- Useful logic to preserve: deterministic backfill transformations and schema handling.
- Recommended target: `scripts/diagnostics/` or `scripts/ops/`.
- Future implementation steps: move implementation behind a stable diagnostics/ops module and update tests.
- References to update: `tests/core/test_build_entry_quality_backfill.py` and any docs.
- Required tests: existing entry-quality backfill tests.
- Deletion or legacy preconditions: old path wrapper retained until callers are updated.
- Rollback plan: restore wrapper.
- Risk level: Medium.

### `scripts/portfolio/parse_trade_commands.py`

- Current role: parses user transaction commands and calls `portfolio_manager.log_trade`.
- Problem: transaction-command parsing is coupled to legacy portfolio manager internals and is used by Telegram command processing.
- Useful logic to preserve: command syntax parsing and validation for explicit user commands.
- Recommended target: a governed portfolio command module, or keep current path until Telegram command scope is approved.
- Future implementation steps: add tests around command parsing, introduce target module, keep compatibility import path, then migrate Telegram command processing.
- References to update: `scripts/telegram/process_telegram_commands.py` and any command docs.
- Required tests: parser tests, Telegram command tests, no forbidden semantics checks.
- Deletion or legacy preconditions: no active imports to the old module or wrapper retained.
- Rollback plan: keep old module as wrapper to the new implementation.
- Risk level: High.

### `scripts/portfolio/portfolio_manager.py`

- Current role: legacy portfolio transaction and position file manager.
- Problem: overlaps with active portfolio source-contract builders and is directly used by legacy parser/tests.
- Useful logic to preserve: any still-approved transaction append or position rebuild behavior.
- Recommended target: future governed portfolio command module or legacy quarantine.
- Future implementation steps: define whether manual portfolio commands remain active, migrate only approved transaction logic, update parser and tests.
- References to update: `scripts/portfolio/parse_trade_commands.py`, `scripts/portfolio/test_portfolio.py`, docs/runbooks if active.
- Required tests: portfolio source contract tests, parser tests, compile check.
- Deletion or legacy preconditions: useful logic migrated or declared obsolete, no active imports, rollback path clear.
- Rollback plan: move to legacy first if manual workflows are uncertain.
- Risk level: High.

### `scripts/reporting/build_telegram_summary.py`

- Current role: compatibility wrapper around `scripts/reporting/build_reporting_layer.py`.
- Problem: old name can look like an independent Reporting path even though it delegates to the authoritative builder.
- Useful logic to preserve: stable compatibility function and CLI behavior if manually used.
- Recommended target: keep as explicit wrapper, or delete only after reference removal.
- Future implementation steps: make wrapper intent explicit in code comments/docs if code edit is approved, then remove only if tests and docs no longer need it.
- References to update: reporting tests and any runbooks.
- Required tests: `tests/reporting/test_build_telegram_summary.py`, `tests/reporting/test_build_reporting_layer.py`.
- Deletion or legacy preconditions: no active references and no manual dependency.
- Rollback plan: restore wrapper.
- Risk level: Medium.

### `scripts/reporting/reporter.py`

- Current role: quarantined legacy markdown reporter.
- Problem: historical Reporting surface predates the authoritative Reporting Layer and can confuse the active communication boundary.
- Useful logic to preserve: normally none; preserve only if a later review finds non-semantic formatting utility.
- Recommended target: delete after reference removal or move to legacy if audit trail wants source preservation.
- Future implementation steps: confirm no active workflow uses `market_scan_*.md` legacy output as active reporting.
- References to update: active docs/runbooks if any.
- Required tests: Reporting tests and grep reference checks.
- Deletion or legacy preconditions: no active references, no unique required logic.
- Rollback plan: restore from git or keep in legacy first.
- Risk level: Medium.

### `scripts/utils/utils.py`

- Current role: legacy utility module with no active project imports found during C.3 inspection.
- Problem: generic utility modules can become hidden dependency sinks and obscure ownership.
- Useful logic to preserve: only helpers proven useful by future reference inspection.
- Recommended target: move individual helpers to owning modules, otherwise delete after reference removal.
- Future implementation steps: inspect contents, map each helper to an owner, avoid creating a new generic dumping ground.
- References to update: any discovered imports.
- Required tests: compile check and affected module tests.
- Deletion or legacy preconditions: no imports, no docs/runbooks, no unique manual use.
- Rollback plan: restore file if missed dependency appears.
- Risk level: Low.

### `scripts/watchlist/*.py`

- Current role: legacy watchlist command, build, evaluate, and action-update surfaces.
- Problem: legacy watchlist scripts can imply action, status, or timing readiness semantics that must not leak into upstream allocation authority.
- Useful logic to preserve: only timing-state-safe data handling if still manually needed.
- Recommended target: `legacy/` first, or a governed timing/watchlist module only after approval.
- Future implementation steps: verify active pipeline independence, inspect forbidden semantics, then either quarantine or migrate safe helpers.
- References to update: docs/runbooks and any workflow mentions.
- Required tests: timing-state tests, Decision Engine boundary tests, no forbidden semantics grep checks, compile check.
- Deletion or legacy preconditions: active pipeline and manual workflows do not depend on them.
- Rollback plan: move to legacy before deleting.
- Risk level: High.

### `scripts/data_sources/prefill_fundamentals.py`

- Current role: provider/operator-assisted prefill for the current metric-like fundamentals MVP.
- Problem: future architecture targets raw financial statement history, not current metric-like MVP rows.
- Useful logic to preserve: source-data validation patterns, forbidden-column checks, dry-run/overwrite safety, audit shape.
- Recommended target: future raw-history intake helpers after Sprint E/F approval.
- Future implementation steps: do not refactor before raw-history schema and migration scope are approved.
- References to update: source-data tests and active fundamentals docs after approved implementation.
- Required tests: current prefill tests plus future raw-history intake tests.
- Deletion or legacy preconditions: replacement raw-history intake exists and current MVP compatibility no longer needed.
- Rollback plan: keep current utility until migration is complete.
- Risk level: High.

## 8. Files Not To Touch Before Approved Scope

| File | Reason not to touch | Earliest allowed cleanup point | Notes |
|---|---|---|---|
| `scripts/run_scan.py` | Active daily workflow and orchestration entrypoint | Explicit orchestration cleanup scope | Referenced by `.github/workflows/daily-market-scan.yml`. |
| `scripts/run_full_pipeline.py` | Active full-pipeline/manual entrypoint | Explicit orchestration cleanup scope | Must preserve deterministic pipeline order. |
| `scripts/core/decision_engine.py` | Only allocation authority | Explicit Decision Engine scope | Not part of cleanup. |
| `scripts/core/build_fundamental_layer.py` | Sprint D compatibility surface for Sprint E | Approved Sprint E implementation scope | Keep pipeline-facing compatibility first. |
| `scripts/core/build_timing_state_layer.py` | Sensitive downstream classification layer | Explicit timing-state scope | Do not absorb legacy watchlist semantics. |
| `scripts/core/build_portfolio_intelligence.py` | Sensitive downstream portfolio classification layer | Explicit portfolio intelligence scope | Must not determine allocation. |
| `scripts/reporting/build_reporting_layer.py` | Authoritative Reporting Layer | Explicit Reporting scope | Reporting remains communication-only. |
| `scripts/reporting/send_telegram.py` | Active Telegram delivery dependency | Explicit Telegram delivery scope | Imported by `scripts/run_scan.py`. |
| `scripts/telegram/process_telegram_commands.py` | Active inbound command surface | Explicit Telegram command scope | Depends on portfolio command parsing. |
| `scripts/core/data_fetcher.py` | Active data-fetching surface | Explicit data-fetching scope | Do not change provider behavior in cleanup. |
| `scripts/core/scanner.py` | Active scanner discovery surface | Explicit scanner scope | Discovery only. |
| `scripts/portfolio/build_portfolio.py` | Active portfolio source contract builder | Explicit portfolio scope | Preserve source contract behavior. |
| `scripts/portfolio/evaluate_positions.py` | Active portfolio support tool | Explicit portfolio scope | Must remain descriptive/supportive. |
| `scripts/data_sources/prefill_fundamentals.py` | Future raw-history migration dependency | Sprint E/F source-data scope | Keep until replacement strategy is approved. |

## 9. Reference Cleanup Requirements

Before any future deletion or move, a cleanup sprint must check:

- grep references;
- imports;
- GitHub Actions;
- active docs and runbooks;
- tests;
- manual entrypoint use;
- archived docs only as historical evidence.

Required commands for a future cleanup sprint:

```bash
grep -R "<filename-or-symbol>" -n scripts tests docs .github || true
python -m compileall scripts
pytest
```

These commands were not run as implementation validation in Sprint C.3, except for inspection-only grep/listing commands needed to draft this specification.

## 10. Required Future Tests and Validation

Before actual cleanup, a future implementation sprint must run:

- focused tests for every affected module;
- full `pytest` if imports, shared modules, or entrypoints change;
- `python -m compileall scripts`;
- import smoke checks for preserved public entrypoints;
- CLI smoke checks for preserved entrypoints where approved;
- Reporting output compatibility tests if Reporting surfaces change;
- Telegram command/delivery tests if Telegram surfaces change;
- Decision Engine contract tests if any upstream/downstream boundary is touched;
- no forbidden semantics grep checks for watchlist, portfolio, Reporting, Telegram, and upstream layers;
- no generated output commits unless explicitly approved;
- no pipeline run unless explicitly approved.

## 11. Deletion Policy

A Python file may be deleted only when all of these are true:

- it has no unique required logic, or useful logic has already been migrated;
- no active imports reference it;
- no active docs/runbooks call it;
- no GitHub Actions call it;
- tests pass after removal;
- rollback is simple;
- deletion is explicitly approved.

Otherwise prefer one of these safer outcomes:

- keep as a compatibility wrapper;
- move to `legacy/`;
- mark `REQUIRES_REVIEW`;
- defer until a narrower implementation scope exists.

No deletion is approved by Sprint C.3.

## 12. Recommended Future Execution Plan

Recommended sequence:

1. Sprint C.4 - Reference and dependency verification, documentation-only or validation-only.
2. Sprint C.5 - Safe wrapper cleanup and diagnostics consolidation implementation.
3. Sprint C.6 - Portfolio, Reporting, Telegram, and watchlist legacy cleanup implementation.
4. Sprint E1 - Raw history implementation, only after cleanup risk is acceptable.
5. Sprint E2+ - Metrics, quality compatibility, and analysis implementation.

Safer alternative:

1. Sprint C.4 verifies references and manual entrypoints.
2. Sprint E1 implements fundamentals compatibility helpers without broad cleanup.
3. Later C-series cleanup handles diagnostics and legacy scripts after Sprint E compatibility is stable.

The safer alternative is preferred if reference verification shows active manual use of legacy scripts.

## 13. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog and active specs already cover the observed governance and technical cleanup needs.

## 14. Validation

Sprint C.3 validation commands:

```bash
git checkout main
git pull origin main
git status
git checkout -b docs/sprint-c3-python-runtime-cleanup-spec
find scripts -type f -name "*.py" | sort
find tests -type f -name "*.py" | sort
find .github -type f | sort
grep -R "run_scan.py\|run_scan" -n . || true
grep -R "run_full_pipeline.py\|run_full_pipeline" -n . || true
grep -R "build_fundamental_layer.py\|build_fundamental_layer" -n . || true
grep -R "build_reporting_layer.py\|build_reporting_layer" -n . || true
grep -R "build_telegram_summary.py\|build_telegram_summary" -n . || true
grep -R "send_telegram.py\|send_telegram" -n . || true
grep -R "analyze_validation.py\|analyze_validation" -n . || true
grep -R "validate_scans.py\|validate_scans" -n . || true
grep -R "parse_trade_commands.py\|parse_trade_commands" -n . || true
grep -R "portfolio_manager.py\|portfolio_manager" -n . || true
grep -R "reporter.py\|reporter" -n . || true
grep -R "utils.py\|from scripts.utils\|import scripts.utils" -n . || true
grep -R "watchlist" -n scripts tests docs .github || true
grep -R "prefill" -n scripts tests docs .github || true
git status
git diff --stat main...HEAD
git diff --name-status main...HEAD
git diff --check main...HEAD
```

Validation confirmation:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run.
