# Python Runtime Surface Rationalization Plan

Status: ACTIVE INVENTORY / RATIONALIZATION PLAN

## 1. Purpose

This document is a documentation-only rationalization plan for the Python runtime surface before any future code cleanup or Sprint E fundamentals implementation.

This sprint changed no code, moved no files, deleted no files, changed no imports, changed no runtime behavior, and authorized no implementation.

The purpose is to identify active entrypoints, active libraries, wrappers, diagnostics, source-data tools, legacy surfaces, duplication, misplaced logic, and safe future cleanup paths before implementation work starts.

## 2. Scope and Non-Scope

Scope:

- inspect the Python runtime surface under `scripts/`;
- classify Python files by current role and reference status;
- identify duplication;
- identify misplaced logic;
- identify legacy candidates;
- identify future safe cleanup paths;
- preserve current architecture and Decision Engine authority.

Non-scope:

- code edits;
- test edits;
- CSV edits;
- generated artifact updates;
- file moves;
- file deletion;
- import changes;
- runtime changes;
- provider/API calls;
- scraping;
- pipeline runs;
- Sprint E implementation.

## 3. Active Architecture Baseline

The active certified pipeline is:

```text
scanner
-> validation_layer
-> context_layer
-> fundamental_layer
-> timing_state_layer
-> portfolio_intelligence_layer
-> decision_engine
-> reporting
```

Upstream layers classify only. Portfolio Intelligence is descriptive only. The Decision Engine is the only allocation, execution, arbitration, and final-action authority. Reporting communicates Decision Engine outputs only.

Sprint D selected the future fundamentals implementation approach:

```text
Option A - Compatibility Wrapper First
```

That means `scripts/core/build_fundamental_layer.py` should remain the pipeline-facing compatibility surface until a governed Sprint E scope introduces raw-history, metrics, quality mapping, and analysis helpers safely behind or alongside it.

## 4. Repository Python Surface Overview

| Folder | Python file count | Main responsibility | Notes |
|---|---:|---|---|
| `scripts/` | 4 | Top-level orchestration and legacy validation entrypoints | `run_scan.py` and `run_full_pipeline.py` are active; top-level validation scripts need review. |
| `scripts/core/` | 17 | Core scanner, pipeline layers, decision engine, validation/backfill tools | Contains active pipeline core plus older validation/backfill helpers. |
| `scripts/data_sources/` | 3 | Governed prefill helpers and shared source-data validation | Useful, but fundamentals prefill is tied to the current metric-like MVP. |
| `scripts/diagnostics/` | 1 | Read-only coverage diagnostics | Active diagnostic utility; needs future alignment with raw-history architecture. |
| `scripts/ops/` | 1 | Historical evidence capture | Active operational evidence utility. |
| `scripts/portfolio/` | 5 | Portfolio source, state, review, and command helpers | Active pipeline uses `build_portfolio.py` and `evaluate_positions.py`; command helpers contain legacy duplication. |
| `scripts/reporting/` | 4 | Reporting output, Telegram summary wrapper, Telegram delivery, legacy report builder | Reporting layer is active; legacy markdown reporter should remain quarantined. |
| `scripts/telegram/` | 1 | Telegram polling command entrypoint | Referenced by GitHub Actions and has network/API behavior. |
| `scripts/utils/` | 1 | Generic file helpers | Currently unreferenced. |
| `scripts/watchlist/` | 5 | Legacy/supporting watchlist utilities | Not part of active pipeline; should be reviewed before future automation. |

Total tracked Python files under `scripts/`: 42.

## 5. Runtime Entry Point Inventory

| File | Entry point type | Called by | Current role | Keep status | Notes |
|---|---|---|---|---|---|
| `scripts/run_scan.py` | ACTIVE_ENTRYPOINT / PIPELINE_ORCHESTRATION | GitHub Actions; `scripts/run_full_pipeline.py`; tests | Main deterministic end-to-end scan pipeline | KEEP_AS_IS | Do not touch before approved orchestration scope. |
| `scripts/run_full_pipeline.py` | ACTIVE_ENTRYPOINT / compatibility wrapper | Operator usage; tests; historical docs | Thin wrapper around `scripts/run_scan.py` | KEEP_AS_IS | Useful stable operator entrypoint, but not the GitHub Actions entrypoint. |
| `scripts/telegram/process_telegram_commands.py` | ACTIVE_ENTRYPOINT / TELEGRAM_TOOL | GitHub Actions | Polls Telegram commands before scan | KEEP_BUT_REFACTOR_LATER | Network/API behavior; not safe for broad cleanup without explicit scope. |
| `scripts/reporting/send_telegram.py` | ACTIVE_ENTRYPOINT / TELEGRAM_TOOL | `scripts/run_scan.py`; manual invocation | Sends generated Telegram message | KEEP_AS_IS | Runtime network sender; keep isolated from reporting semantics. |
| `scripts/reporting/build_reporting_layer.py` | ACTIVE_ENTRYPOINT / REPORTING_TOOL | `scripts/run_scan.py`; tests; manual invocation | Authoritative Reporting Layer builder | KEEP_AS_IS | Sensitive active downstream surface. |
| `scripts/reporting/build_telegram_summary.py` | ACTIVE_ENTRYPOINT / REPORTING_TOOL | Tests; historical docs; manual invocation | Compatibility wrapper that delegates to Reporting Layer | KEEP_AS_IS | Already behaves as wrapper; may stay as compatibility surface. |
| `scripts/core/decision_engine.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests; historical evidence utility | Sole allocation authority | KEEP_AS_IS | Do not touch before approved Decision Engine scope. |
| `scripts/core/build_validation_layer.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests | Validation and entry-quality layer builder | KEEP_AS_IS | Active pipeline layer. |
| `scripts/core/build_context_layer.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests | Context layer builder | KEEP_AS_IS | Active pipeline layer. |
| `scripts/core/build_fundamental_layer.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests | Current Fundamental Layer compatibility surface | KEEP_BUT_REFACTOR_LATER | Sprint E should preserve it as pipeline-facing wrapper first. |
| `scripts/core/build_timing_state_layer.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests | Timing State builder | KEEP_AS_IS | Do not touch before approved downstream compatibility scope. |
| `scripts/core/build_portfolio_intelligence.py` | ACTIVE_ENTRYPOINT / ACTIVE_LIBRARY | `scripts/run_scan.py`; tests | Portfolio Intelligence builder | KEEP_AS_IS | Sensitive downstream consumer of fundamentals/timing. |
| `scripts/portfolio/build_portfolio.py` | ACTIVE_ENTRYPOINT / PORTFOLIO_TOOL | `scripts/run_scan.py`; tests | Rebuilds portfolio positions from transaction source | KEEP_AS_IS | Active portfolio source contract surface. |
| `scripts/portfolio/evaluate_positions.py` | ACTIVE_ENTRYPOINT / PORTFOLIO_TOOL | `scripts/run_scan.py`; tests | Builds portfolio review artifact | KEEP_AS_IS | Active pipeline support artifact. |
| `scripts/diagnostics/audit_data_coverage.py` | DIAGNOSTIC_TOOL | Tests; docs; manual invocation | Source coverage diagnostic | KEEP_BUT_REFACTOR_LATER | Needs future raw-history alignment. |
| `scripts/data_sources/prefill_fundamentals.py` | DATA_SOURCE_TOOL | Tests; docs; manual invocation | Current fundamentals MVP prefill | KEEP_BUT_REFACTOR_LATER | Replace or wrap after raw-history contract implementation. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | DATA_SOURCE_TOOL | Tests; docs; manual invocation | Portfolio metadata prefill | KEEP_AS_IS | Separate from fundamentals migration. |
| `scripts/ops/capture_historical_evidence.py` | DIAGNOSTIC_TOOL / ops entrypoint | Tests; docs; manual invocation | Captures historical pipeline evidence | KEEP_AS_IS | Active operational evidence utility. |
| `scripts/core/build_context_backfill.py` | DIAGNOSTIC_TOOL / backfill entrypoint | Tests; docs; manual invocation | Historical context backfill | KEEP_BUT_REFACTOR_LATER | Uses external price download; not active pipeline. |
| `scripts/core/build_entry_quality_backfill.py` | DIAGNOSTIC_TOOL / backfill entrypoint | Tests; docs; manual invocation | Historical entry-quality backfill | KEEP_BUT_REFACTOR_LATER | Uses external price download; not active pipeline. |
| `scripts/core/build_stability_layer.py` | ACTIVE_LIBRARY / optional downstream entrypoint | Tests; Reporting optionally consumes output | Stability metadata layer | KEEP_AS_IS | Not currently in `run_scan.py`; still active architecture surface. |
| `scripts/analyze_validation.py` | LEGACY_TOOL | No active tests or docs except duplicate-name reference | Legacy validation summary script | MOVE_TO_LEGACY | Duplicates newer `scripts/core/analyze_validation.py` intent. |
| `scripts/validate_scans.py` | LEGACY_TOOL | Docs/research references; no active tests | Legacy yfinance validation script | MOVE_TO_LEGACY | Duplicates newer local-data validation concept in `scripts/core/validate_scans.py`. |

## 6. Full Python File Rationalization Table

| File | Current role | Reference status | Unique logic | Proposed action | Target destination if logic moves | Deletion/move preconditions | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/analyze_validation.py` | Legacy validation summary | Low active reference; no active tests | Excel-style summary from `data/scans_validation.csv` | MOVE_TO_LEGACY | `legacy/scripts/analyze_validation.py` or merge useful summary into `scripts/core/analyze_validation.py` | Confirm no manual workflow still uses `data/scans_validation.csv`; preserve any required summary output | Medium | Contains non-ASCII operator output; not active doctrine-aligned. |
| `scripts/run_full_pipeline.py` | Full-pipeline wrapper | Tested and historically referenced | Stable wrapper around `scripts/run_scan.py` | KEEP_AS_IS | Not applicable | Confirm external/manual users before any future rename | Low | Useful compatibility entrypoint. |
| `scripts/run_scan.py` | Main pipeline orchestration | GitHub Actions, wrapper, tests | End-to-end sequencing and operator visibility | KEEP_AS_IS | Not applicable | Any change needs governed orchestration scope and tests | High | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE. |
| `scripts/validate_scans.py` | Legacy validation/backtest script | Low active reference; no active tests | Downloads future price data and validates entry/stop/target outcomes | MOVE_TO_LEGACY | A future diagnostics/backtesting module, if still needed | Confirm no manual workflow depends on it; avoid provider calls in normal validation | High | Uses yfinance and legacy paths under `data/`. |
| `scripts/core/analyze_validation.py` | Current validation summary helper | No active tests found | Summarizes `validation_results.csv` into `validation_summary.csv` | REQUIRES_REVIEW | Possibly `scripts/diagnostics/analyze_validation.py` | Add tests or document manual-only status before moving | Medium | More aligned than top-level legacy analyzer but still not part of active pipeline. |
| `scripts/core/build_context_backfill.py` | Context backfill utility | Tested; historical docs | Downloads historical OHLCV and builds historical context | MOVE_LOGIC_TO_NEW_MODULE | Future `scripts/diagnostics/backfills/build_context_backfill.py` | Preserve tests; isolate provider calls; update docs and imports | Medium | Useful logic lives in `core/` although it is not an active pipeline layer. |
| `scripts/core/build_context_layer.py` | Active Context Layer | `run_scan.py`, tests | Current context layer builder | KEEP_AS_IS | Not applicable | Future changes need layer contract review | High | Active pipeline layer. |
| `scripts/core/build_entry_quality_backfill.py` | Entry-quality backfill utility | Tested; historical docs | Historical indicator and entry-quality backfill | MOVE_LOGIC_TO_NEW_MODULE | Future `scripts/diagnostics/backfills/build_entry_quality_backfill.py` | Preserve tests; isolate provider calls; update docs/imports | Medium | Useful but misplaced in `core/` if considered non-runtime backfill. |
| `scripts/core/build_fundamental_layer.py` | Current Fundamental Layer | `run_scan.py`, tests, active specs | Builds `fundamental_quality.csv` from context and MVP raw fundamentals | KEEP_BUT_REFACTOR_LATER | Remain as compatibility wrapper during Sprint E | Sprint E approval; compatibility tests; raw-history helpers implemented | High | DO_NOT_TOUCH before approved Sprint E scope. |
| `scripts/core/build_portfolio_intelligence.py` | Active Portfolio Intelligence | `run_scan.py`, tests | Descriptive portfolio metadata layer | KEEP_AS_IS | Not applicable | Future changes need downstream contract review | High | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE. |
| `scripts/core/build_stability_layer.py` | Stability metadata layer | Tests; Reporting optionally reads output | Persistence/stability classification from decisions | KEEP_AS_IS | Not applicable | Confirm orchestration status before adding to `run_scan.py` | Medium | Active architecture surface but not current pipeline step. |
| `scripts/core/build_timing_state_layer.py` | Active Timing State | `run_scan.py`, tests | Timing metadata layer | KEEP_AS_IS | Not applicable | Future changes need downstream compatibility tests | High | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE. |
| `scripts/core/build_validation_layer.py` | Active Validation Layer | `run_scan.py`, tests | Validation and entry-quality output | KEEP_AS_IS | Not applicable | Future schema changes require governance review | High | Active pipeline layer. |
| `scripts/core/data_fetcher.py` | Active data fetch library | `run_scan.py` | Ticker loading and OHLCV fetch | KEEP_AS_IS | Not applicable | Provider behavior changes need explicit approval | High | Active network/provider surface in scanner pipeline. |
| `scripts/core/decision_engine.py` | Sole allocation authority | `run_scan.py`, tests, ops utility | Final action and allocation decisions | KEEP_AS_IS | Not applicable | Only change under explicit Decision Engine scope | Critical | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE. |
| `scripts/core/indicators.py` | Active indicator helper | `run_scan.py`, portfolio helpers, backfills | Indicator calculations | KEEP_AS_IS | Not applicable | Tests needed before shared indicator changes | Medium | Shared active library. |
| `scripts/core/log_scans.py` | Scanner log helper | No active tests or direct callers found | Appends scanner-ranked rows to scan log | REQUIRES_REVIEW | Possibly fold into `run_scan.py` or ops evidence capture | Confirm whether scan logs still needed; add tests before migration | Medium | Potentially stale because current pipeline writes processed artifacts directly. |
| `scripts/core/regime.py` | Active regime classifier | `run_scan.py`, scanner, legacy tools | Market regime classification | KEEP_AS_IS | Not applicable | Add tests before changing semantics | Medium | Small shared library. |
| `scripts/core/scanner.py` | Active scanner library | `run_scan.py`, validation layer, docs | Setup discovery and ranking support | KEEP_AS_IS | Not applicable | Any scanner behavior change requires architecture review | High | Active discovery layer; not part of cleanup before Sprint E. |
| `scripts/core/validate_scans.py` | Local validation/backtest helper | Referenced by context backfill/tests indirectly | Local processed-data scan validation | KEEP_BUT_REFACTOR_LATER | Possibly `scripts/diagnostics/validate_scans.py` | Preserve tests; distinguish from top-level provider-backed validator | Medium | Better aligned than top-level `scripts/validate_scans.py`. |
| `scripts/core/validator.py` | Legacy setup validator | No active references found | Ensures base directories and legacy files exist | MOVE_TO_LEGACY | `legacy/scripts/core/validator.py` | Confirm no manual bootstrap workflow depends on it | Low | Likely superseded by current pipeline setup. |
| `scripts/data_sources/common.py` | Shared source-data helper library | Imported by prefill tools; tested | Validation, path guards, atomic CSV writes | KEEP_AS_IS | Not applicable | Future raw-history intake may reuse it | Medium | Good shared utility. |
| `scripts/data_sources/prefill_fundamentals.py` | Current fundamentals MVP prefill | Tests and docs | Prepares metric-like `data/raw/fundamentals.csv` | KEEP_BUT_REFACTOR_LATER | Future raw-history intake helper or wrapper | Sprint E raw-history spec; migration tests | Medium | Must not be deleted before raw-history intake path exists. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | Portfolio metadata prefill | Tests and docs | Prepares governed portfolio metadata artifact | KEEP_AS_IS | Not applicable | Separate approval for portfolio metadata changes | Medium | Keep outside fundamentals cleanup. |
| `scripts/diagnostics/audit_data_coverage.py` | Coverage diagnostic | Tests and docs | Audits portfolio metadata and current fundamentals coverage | KEEP_BUT_REFACTOR_LATER | Same file, extended for raw-history architecture | Raw-history artifacts and tests must exist first | Medium | Future work should add raw-history/metrics/analysis coverage modes. |
| `scripts/ops/capture_historical_evidence.py` | Ops evidence capture | Tests and docs | Captures run, artifact, decision/reporting evidence | KEEP_AS_IS | Not applicable | Update only with evidence contract changes | Medium | Active ops utility. |
| `scripts/portfolio/build_portfolio.py` | Active portfolio builder | `run_scan.py`, tests | Builds positions from transaction ledger | KEEP_AS_IS | Not applicable | Portfolio source changes require contract review | High | Active portfolio source contract surface. |
| `scripts/portfolio/evaluate_positions.py` | Active portfolio review builder | `run_scan.py`, tests | Descriptive review artifact for open positions | KEEP_AS_IS | Not applicable | Confirm role before future schema changes | Medium | Active support artifact. |
| `scripts/portfolio/parse_trade_commands.py` | Telegram/manual transaction parser | Telegram command processor | Parses transaction commands and logs trades through legacy manager | MOVE_LOGIC_TO_NEW_MODULE | Future `scripts/portfolio/transaction_commands.py` using current portfolio source contract | Keep command behavior tested; update Telegram caller; preserve transaction ledger semantics | High | Contains network-facing command flow through Telegram. |
| `scripts/portfolio/portfolio_manager.py` | Legacy portfolio transaction/positions manager | Used by `parse_trade_commands.py` and manual test helper | Transaction logging and duplicate position builder | DELETE_AFTER_MIGRATION | Migrate transaction logging into a current contract-aligned helper | Update `parse_trade_commands.py`; preserve transaction append behavior; add tests | High | Duplicates `build_portfolio.py` position logic. |
| `scripts/portfolio/test_portfolio.py` | Manual dev helper | No active test runner dependency found | Temporary smoke/demo around `portfolio_manager.py` | DELETE_AFTER_REFERENCE_REMOVAL | Not applicable | Replace with real tests before deletion; confirm no manual workflow depends on it | Low | Should not live under runtime `scripts/portfolio/` long-term. |
| `scripts/reporting/build_reporting_layer.py` | Authoritative Reporting Layer | `run_scan.py`, tests | Dashboard, log, and Telegram message generation | KEEP_AS_IS | Not applicable | Reporting changes require contract tests | High | DO_NOT_TOUCH_BEFORE_APPROVED_SCOPE. |
| `scripts/reporting/build_telegram_summary.py` | Compatibility wrapper | Tests and docs | Delegates to authoritative Reporting Layer | KEEP_AS_IS | Not applicable | Keep until direct callers are known; tests should remain | Low | Already a wrapper; no independent reporting semantics observed. |
| `scripts/reporting/reporter.py` | Legacy markdown reporter | Historical docs; status tracker | Legacy markdown market scan report builder | MOVE_TO_LEGACY | `legacy/scripts/reporting/reporter.py` | Confirm no active caller; keep archive trail | Medium | Historical docs already describe it as legacy. |
| `scripts/reporting/send_telegram.py` | Active Telegram sender | `run_scan.py`, manual invocation | Sends generated message to Telegram API | KEEP_AS_IS | Not applicable | Network changes require explicit approval | High | Delivery only; should not create reporting semantics. |
| `scripts/telegram/process_telegram_commands.py` | Active Telegram command poller | GitHub Actions | Polls Telegram API and processes transaction commands | KEEP_BUT_REFACTOR_LATER | Keep entrypoint; eventually delegate to contract-aligned command helper | Update only with Telegram/portfolio command scope and tests | High | DO_NOT_TOUCH before approved command-processing scope. |
| `scripts/utils/utils.py` | Generic utility helper | No active references found | Save text and CSV helpers | DELETE_AFTER_REFERENCE_REMOVAL | Not applicable | Confirm no hidden manual imports; remove only after reference sweep | Low | Candidate only; no deletion approved. |
| `scripts/watchlist/auto_watchlist_from_scan.py` | Legacy/supporting watchlist automation | Historical docs | Adds A-grade scan rows to watchlist transactions | MOVE_TO_LEGACY | Future watchlist package only if watchlist is reactivated | Governance review before automation; tests needed | Medium | Not part of active pipeline; may imply automated watchlist actions. |
| `scripts/watchlist/build_watchlist.py` | Legacy/supporting watchlist builder | Historical docs | Builds active watchlist from transactions | MOVE_TO_LEGACY | Future watchlist package if reactivated | Confirm current manual workflows; add tests before migration | Medium | Supporting input only, not allocation authority. |
| `scripts/watchlist/evaluate_watchlist.py` | Legacy/supporting watchlist evaluator | Historical docs | Evaluates watchlist timing/status | MOVE_TO_LEGACY | Future watchlist package after governance review | Review semantics before reuse; tests needed | High | Historical audits warn against reusing as Sprint 4 schema pattern. |
| `scripts/watchlist/parse_watchlist_commands.py` | Legacy/supporting command parser | Historical docs | Manual watch/unwatch command parser | MOVE_TO_LEGACY | Future watchlist command module if reactivated | Confirm command workflows; tests needed | Medium | Not in active pipeline. |
| `scripts/watchlist/update_watchlist_actions.py` | Legacy/supporting watchlist action updater | Historical docs | Appends watchlist action rows from status | MOVE_TO_LEGACY | Future watchlist package if reactivated | Governance review before automation | Medium | Excluded from full-pipeline wrapper by design. |

## 7. Logic Relocation Candidates

### `scripts/analyze_validation.py`

- Current location problem: top-level legacy validation analyzer duplicates the newer validation summary concept under `scripts/core/analyze_validation.py`.
- Useful logic to preserve: group-level validation summaries and any operator-friendly summary format still needed.
- Recommended target: merge required summary behavior into `scripts/core/analyze_validation.py` or a future `scripts/diagnostics/analyze_validation.py`.
- Required future code changes: update any manual documentation or scripts that still call the top-level file.
- Required tests: focused summary tests for any preserved output.
- Deletion or wrapper preconditions: confirm no manual workflow depends on `data/scans_validation.csv`.
- Risk level: Medium.

### `scripts/validate_scans.py`

- Current location problem: top-level legacy validation script performs provider-backed validation and uses old `data/scans_log.csv` and `data/scans_validation.csv` paths.
- Useful logic to preserve: lookahead outcome validation, same-day ambiguity handling, and summary metrics if still analytically useful.
- Recommended target: future diagnostics/backtesting module, separate from active pipeline validation.
- Required future code changes: decide whether provider-backed validation is still allowed; update paths to current contracts or quarantine it.
- Required tests: provider-free fixture tests for lookahead validation.
- Deletion or wrapper preconditions: confirm no manual workflow depends on the old top-level command.
- Risk level: High because it can call yfinance.

### `scripts/core/build_context_backfill.py`

- Current location problem: backfill tool lives under `core/`, but it is not the active Context Layer builder.
- Useful logic to preserve: historical context calculation, price alignment, percentile classification, and logging.
- Recommended target: future diagnostics/backfills namespace.
- Required future code changes: update imports, test imports, and docs after a governed move.
- Required tests: existing `tests/core/test_build_context_backfill.py` should move or be updated.
- Deletion or wrapper preconditions: no deletion; keep as a wrapper if manual calls exist.
- Risk level: Medium.

### `scripts/core/build_entry_quality_backfill.py`

- Current location problem: backfill tool lives under `core/`, but it performs historical/provider-backed support work rather than active runtime layer work.
- Useful logic to preserve: point-in-time indicator calculation, scan/date alignment, entry-quality helper output.
- Recommended target: future diagnostics/backfills namespace.
- Required future code changes: update tests, docs, and any manual command references.
- Required tests: existing entry-quality backfill tests plus provider-free fixture coverage.
- Deletion or wrapper preconditions: no deletion until backfill status is decided.
- Risk level: Medium.

### `scripts/core/analyze_validation.py`

- Current location problem: validation analysis helper is not an active core pipeline layer.
- Useful logic to preserve: grouped validation summary generation from `validation_results.csv`.
- Recommended target: `scripts/diagnostics/analyze_validation.py` if validation analysis remains operational.
- Required future code changes: update any manual command references.
- Required tests: add tests before moving.
- Deletion or wrapper preconditions: keep wrapper if operators call the current path.
- Risk level: Medium.

### `scripts/core/validate_scans.py`

- Current location problem: validation/backtest helper lives under `core/` but is not active pipeline validation.
- Useful logic to preserve: local processed-data validation without provider calls.
- Recommended target: `scripts/diagnostics/validate_scans.py`.
- Required future code changes: update context backfill references if any remain.
- Required tests: preserve current context backfill-related expectations and add direct validation tests.
- Deletion or wrapper preconditions: keep a wrapper or update all references first.
- Risk level: Medium.

### `scripts/portfolio/parse_trade_commands.py`

- Current location problem: transaction command parsing delegates to `portfolio_manager.py`, whose position-building logic duplicates `build_portfolio.py`.
- Useful logic to preserve: command parsing, decimal comma handling, validation of side/quantity/price, transaction append semantics.
- Recommended target: a current contract-aligned portfolio transaction command helper.
- Required future code changes: update `scripts/telegram/process_telegram_commands.py` to call the new helper.
- Required tests: command parser tests, transaction ledger append tests, Telegram command integration tests.
- Deletion or wrapper preconditions: preserve behavior and external command syntax first.
- Risk level: High because it is connected to Telegram command handling.

### `scripts/portfolio/portfolio_manager.py`

- Current location problem: duplicates portfolio position-building behavior that now belongs to `scripts/portfolio/build_portfolio.py`.
- Useful logic to preserve: transaction append helper currently used by Telegram trade commands.
- Recommended target: move transaction append behavior into a focused helper that does not rebuild positions.
- Required future code changes: migrate `parse_trade_commands.py` away from `portfolio_manager.py`.
- Required tests: transaction append tests and portfolio source contract tests.
- Deletion or wrapper preconditions: remove all imports and preserve transaction command behavior.
- Risk level: High.

### `scripts/reporting/reporter.py`

- Current location problem: legacy markdown reporter is not the active Reporting Layer.
- Useful logic to preserve: none identified for active reporting unless manually required.
- Recommended target: legacy archive path, not active runtime.
- Required future code changes: confirm no imports or manual workflows use it.
- Required tests: none if moved to legacy; reporting tests must continue to cover `build_reporting_layer.py`.
- Deletion or wrapper preconditions: no direct deletion until references are cleared.
- Risk level: Medium.

### `scripts/utils/utils.py`

- Current location problem: generic utility file has no active references.
- Useful logic to preserve: simple save-text and save-CSV helpers only if future code imports them.
- Recommended target: delete after reference removal, or keep only if a future module needs shared file-write helpers.
- Required future code changes: none unless references are discovered.
- Required tests: import/reference sweep.
- Deletion or wrapper preconditions: confirm no external/manual usage.
- Risk level: Low.

## 8. Merge / Wrapper Candidates

| File | Candidate type | Rationale | Future direction |
|---|---|---|---|
| `scripts/reporting/build_telegram_summary.py` | Compatibility wrapper | Already delegates to `build_reporting_layer.py` and avoids independent reporting semantics. | Keep wrapper until all direct callers are known; do not merge before reporting scope exists. |
| `scripts/run_full_pipeline.py` | Compatibility wrapper | Thin operator wrapper around `run_scan.py`. | Keep as stable manual entrypoint. |
| `scripts/core/build_fundamental_layer.py` | Future compatibility wrapper | Sprint D requires this as the pipeline-facing surface for Option A. | Refactor only in approved Sprint E scope. |
| `scripts/analyze_validation.py` and `scripts/core/analyze_validation.py` | Merge candidate | Duplicate validation-summary concepts. | Preserve one diagnostics-oriented analyzer if still needed. |
| `scripts/validate_scans.py` and `scripts/core/validate_scans.py` | Merge/quarantine candidate | Duplicate validation concepts with different data/provider assumptions. | Prefer provider-free diagnostics path; quarantine provider-backed legacy script. |
| `scripts/portfolio/portfolio_manager.py` and `scripts/portfolio/build_portfolio.py` | Merge/delete-after-migration candidate | Both can build positions; active contract uses `build_portfolio.py`. | Preserve transaction append helper separately, then delete legacy manager after migration. |
| `scripts/data_sources/prefill_fundamentals.py` | Future source-data wrapper candidate | Tied to metric-like MVP `fundamentals.csv`. | Replace or wrap after raw-history intake exists. |

## 9. Legacy and Delete Candidates

| File | Candidate type | Why candidate | Must preserve first | References to remove first | Safer alternative | Final deletion allowed? |
|---|---|---|---|---|---|---|
| `scripts/analyze_validation.py` | MOVE_TO_LEGACY | Top-level legacy duplicate of validation analysis | Any required summary behavior | Manual docs/workflows | Move to legacy | NO - future approval required |
| `scripts/validate_scans.py` | MOVE_TO_LEGACY | Legacy provider-backed validator with old paths | Lookahead validation logic if needed | Manual docs/workflows | Move to legacy | NO - future approval required |
| `scripts/core/validator.py` | MOVE_TO_LEGACY | Legacy setup/bootstrap helper with no active references found | Any bootstrap checks still required | Manual docs/workflows | Move to legacy | NO - future approval required |
| `scripts/core/log_scans.py` | REQUIRES_REVIEW | No active caller found; may be stale scan-log helper | Scan logging semantics if still needed | Manual docs/workflows | Keep until logging policy reviewed | NO - future approval required |
| `scripts/portfolio/portfolio_manager.py` | DELETE_AFTER_MIGRATION | Duplicates position-building behavior and is used by command parser | Transaction append helper | `parse_trade_commands.py`; manual helper | Convert to wrapper first | NO - future approval required |
| `scripts/portfolio/test_portfolio.py` | DELETE_AFTER_REFERENCE_REMOVAL | Manual dev helper inside runtime tree | None beyond smoke behavior | Manual docs/workflows | Move to tests or legacy | NO - future approval required |
| `scripts/reporting/reporter.py` | MOVE_TO_LEGACY | Historical legacy markdown reporter | None identified for active reporting | Status/doc references only | Move to legacy | NO - future approval required |
| `scripts/utils/utils.py` | DELETE_AFTER_REFERENCE_REMOVAL | Unreferenced generic helper | Simple save helper behavior only if needed | External/manual imports | Move to legacy first | NO - future approval required |
| `scripts/watchlist/auto_watchlist_from_scan.py` | MOVE_TO_LEGACY | Legacy watchlist automation outside active pipeline | Any future watchlist intake behavior | Manual workflows | Move to legacy | NO - future approval required |
| `scripts/watchlist/build_watchlist.py` | MOVE_TO_LEGACY | Legacy watchlist support outside active pipeline | Active watchlist reconstruction if still used | Manual workflows | Move to legacy | NO - future approval required |
| `scripts/watchlist/evaluate_watchlist.py` | MOVE_TO_LEGACY | Legacy evaluator with historical governance warnings | Any timing-state logic only after review | Manual workflows | Move to legacy | NO - future approval required |
| `scripts/watchlist/parse_watchlist_commands.py` | MOVE_TO_LEGACY | Legacy command parser outside active pipeline | Manual watch/unwatch syntax if still used | Manual workflows | Move to legacy | NO - future approval required |
| `scripts/watchlist/update_watchlist_actions.py` | MOVE_TO_LEGACY | Legacy action updater intentionally excluded from wrapper | None unless watchlist automation is reactivated | Manual workflows | Move to legacy | NO - future approval required |

## 10. Files Not To Touch Before Approved Scope

| File | Reason |
|---|---|
| `scripts/run_scan.py` | Main pipeline orchestration; changes affect sequencing, freshness, generated outputs, Telegram delivery, and downstream contracts. |
| `scripts/run_full_pipeline.py` | Stable operator wrapper; changes affect manual execution expectations. |
| `scripts/core/decision_engine.py` | Sole allocation authority. |
| `scripts/core/build_fundamental_layer.py` | Sprint E compatibility surface under Option A. |
| `scripts/core/build_timing_state_layer.py` | Downstream consumer of `fundamental_quality.csv`; compatibility-sensitive. |
| `scripts/core/build_portfolio_intelligence.py` | Downstream consumer of timing/fundamental metadata; Decision Engine input surface. |
| `scripts/reporting/build_reporting_layer.py` | Authoritative Reporting Layer; must remain communication-only. |
| `scripts/reporting/send_telegram.py` | Runtime Telegram delivery surface with network/API behavior. |
| `scripts/telegram/process_telegram_commands.py` | GitHub Actions entrypoint with Telegram polling and transaction-command side effects. |
| `scripts/core/data_fetcher.py` | Active provider/data-fetch surface for scanner pipeline. |
| `scripts/core/scanner.py` | Active discovery layer; changes may affect opportunity distribution. |
| `scripts/portfolio/build_portfolio.py` | Active portfolio source contract implementation. |
| `scripts/portfolio/evaluate_positions.py` | Active portfolio review artifact builder used by current pipeline. |

## 11. Test and Validation Needs for Future Cleanup

Future cleanup should use focused validation before any move, merge, wrapper conversion, or deletion:

- import/reference checks for every changed path;
- CLI smoke tests for active entrypoints and wrappers;
- GitHub Actions reference review;
- existing pytest suite;
- focused tests for any moved logic;
- pipeline dry validation only if explicitly approved;
- reporting output compatibility tests;
- Decision Engine contract tests;
- no forbidden semantics checks;
- provider/API isolation checks for backfill and validation tools;
- rollback plan for every move or deletion.

Do not run these in Sprint C.2.

## 12. Recommended Future Cleanup Sequence

1. Remove or wrap clearly obsolete top-level duplicate entrypoints.
2. Consolidate diagnostics helpers.
3. Consolidate reporting and Telegram compatibility wrappers only after active reporting references are confirmed.
4. Rationalize source-data helpers after fundamentals implementation scope is approved.
5. Preserve Decision Engine and downstream pipeline files until governed scope exists.
6. Move watchlist utilities to legacy only after manual workflow review.
7. Migrate portfolio transaction command logic away from duplicate position builders.
8. Only delete files after migration, reference cleanup, tests, and explicit approval.

## 13. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

The current backlog and active specs already cover fundamentals simplification, source-data strategy, raw-history architecture, and downstream compatibility. Python runtime surface cleanup should be considered a future implementation/specification follow-up after this rationalization plan is reviewed.

## 14. Validation

Commands run for Sprint C.2:

- `gh --version`
- `gh auth status`
- `git checkout main`
- `git pull origin main`
- `git status`
- `git checkout -b docs/sprint-c2-python-runtime-rationalization-plan`
- `sed -n '1,220p' docs/active/architecture_current_state.md`
- `sed -n '1,220p' docs/active/governance_v2.md`
- `sed -n '1,220p' docs/active/contracts/pipeline_contracts.md`
- `sed -n '1,220p' docs/active/contracts/fundamentals_platform_contract.md`
- `sed -n '1,220p' docs/active/contracts/fundamental_calculations_technical_spec.md`
- `sed -n '1,220p' docs/active/analysis/financial_analysis_contract.md`
- `sed -n '1,220p' docs/active/analysis/functional_analysis_contract.md`
- `sed -n '1,220p' docs/active/analysis/technical_analysis_contract.md`
- `sed -n '1,220p' docs/active/roles_and_responsibilities.md`
- `sed -n '1,260p' docs/active/inventory/fundamentals_code_inventory.md`
- `sed -n '1,260p' docs/active/specs/fundamentals_history_implementation_spec.md`
- `sed -n '1,260p' docs/sprints/fundamentals_simplification_sprint_plan.md`
- `find scripts -type f -name "*.py" | sort`
- `find .github -type f | sort`
- `find tests -type f -name "*.py" | sort`
- `rg -n "scripts/core/build_fundamental_layer.py|build_fundamental_layer" .`
- `rg -n "run_full_pipeline" .`
- `rg -n "run_scan" .`
- `rg -n "build_telegram_summary" .`
- `rg -n "validate_scans" .`
- `rg -n "analyze_validation" .`
- `rg -n "prefill_fundamentals" .`
- `rg -n "prefill_portfolio_metadata" .`
- `rg -n "python" .github docs scripts tests | head -300`
- `sed -n '1,220p' .github/workflows/daily-market-scan.yml`
- `find scripts -maxdepth 2 -type f -name "*.py" | sed 's#^scripts/##' | awk 'BEGIN{FS="/"} {if (NF==1) c["scripts/"]++; else c["scripts/"$1"/"]++} END{for (k in c) print k, c[k]}' | sort`
- `rg -n "^(from scripts|import scripts|def |class |if __name__ == .__main__.)" scripts -g "*.py"`
- `rg -n "Path\\(|to_csv|read_csv|open\\(|write_text|mkdir|requests\\.|yfinance|yf\\.|TELEGRAM|telegram|reports/|data/|\\.csv|\\.txt" scripts -g "*.py"`
- `rg -n "from scripts|import scripts" tests -g "*.py"`
- `git ls-files 'scripts/**/*.py' 'scripts/*.py' | sort`
- `git status --short --untracked-files=all`
- `rg -n "scripts/(analyze_validation|validate_scans|core/analyze_validation|core/validate_scans|core/log_scans|core/validator|utils/utils|portfolio/test_portfolio|reporting/reporter|portfolio/portfolio_manager|watchlist/)" docs tests scripts .github`
- selected read-only `sed` inspections of runtime entrypoints, validation tools, portfolio tools, reporting tools, Telegram tools, and watchlist tools.

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
