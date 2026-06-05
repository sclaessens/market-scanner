# v2 Report Artifact Runtime Boundary Migration

## Status

Completed by RESET-10L-BL34.

## Reset stage

RESET-10L-BL34 — Migrate Report Artifact Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts report artifact runtime migration away from legacy report/Telegram scripts and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical report artifact boundary under:

```text
src/market_scanner/reporting/
```

The new boundary is deterministic and dry-run/planning-only. It defines report artifact ownership and safety policy, but writes no report file, creates no `reports/daily/telegram_message.txt`, sends no Telegram message, runs no provider, writes no production data, modifies no portfolio/watchlist file, executes no production pipeline, and produces no investment recommendation.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`

Policy result:

- New report artifact Python files were created only because BL28 approved `src/market_scanner/reporting/` as the canonical report artifact ownership boundary.
- A new reporting unit test file was created because no existing test file owned canonical report artifact boundary behavior.
- Existing legacy runner, report, Telegram, message, delivery, portfolio, and watchlist files were inspected but not edited.
- No one-off report helper, quick report runner, Telegram report file, write-report helper, delivery bridge, or parallel runtime shortcut was created.

## Files inspected

Governance and backlog:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`
- `docs/active/backlog.md`

Canonical app, scanner, analysis, decision, messaging, reporting, and tests:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`
- `tests/unit/test_v2_canonical_messaging.py`

Legacy runtime, report, message, and Telegram files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `src/market_scanner/reporting/reporting_input_adapter.py`
- `src/market_scanner/reporting/reporting_input_contracts.py`
- `src/market_scanner/reporting/reporting_engine.py`
- `src/market_scanner/reporting/report_records.py`
- `src/market_scanner/reporting/telegram_contracts.py`
- `src/market_scanner/reporting/telegram_renderer.py`

Repository structure and references were inspected with static file inventory and grep searches for report, artifact, daily, summary, `telegram_message`, write behavior, Telegram, send, delivery, notification, `run_scan`, `run_full_pipeline`, and forbidden investment semantics.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/reporting/__init__.py`
- `src/market_scanner/reporting/report_contracts.py`
- `src/market_scanner/reporting/report_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime, report writer, Telegram, delivery, message, portfolio, or watchlist files were changed.

## Canonical report artifact boundary result

The canonical report artifact boundary is now established under:

```text
src/market_scanner/reporting/
```

Public functions:

- `build_report_artifact_plan`
- `build_report_artifact_policy`
- `build_review_report_plan`

Core records:

- `ReportArtifactPlan`
- `ReportArtifactPolicy`
- `ReportArtifactStage`

The report artifact plan describes:

- report artifact stage name;
- upstream message/review data category;
- allowed artifact-planning types;
- blocked report write behavior;
- blocked delivery behavior;
- blocked final state codes;
- blocked behavior codes;
- whether provider calls are allowed;
- whether production data writes are allowed;
- whether report file writes are allowed;
- whether `reports/daily/telegram_message.txt` writes are allowed;
- whether Telegram delivery is allowed;
- whether portfolio/watchlist mutation is allowed;
- whether final output behavior is allowed;
- whether production pipeline execution is allowed;
- legacy migration status.

The initial report artifact boundary has two planning stages:

```text
review_report_artifact_planning
report_write_policy_block
```

All stages are side-effect-free by default.

Allowed artifact-planning types:

```text
review_report_artifact
limited_analysis_report_artifact
evidence_gap_report_artifact
dry_run_report_artifact
operator_review_artifact
```

## App integration result

`src/market_scanner/app.py` now imports the canonical report artifact boundary and includes the deterministic report artifact plan in the canonical app dry-run runtime plan.

The canonical app report artifact stage now references:

```text
src/market_scanner/reporting/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, legacy report builders, legacy Telegram delivery files, legacy message files, or production report writers.

## Legacy report/Telegram status

Legacy report, message, and Telegram responsibilities remain migration targets and were not expanded.

Current legacy and reporting-adjacent report authority remains concentrated in:

- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `src/market_scanner/reporting/reporting_engine.py`
- `src/market_scanner/reporting/reporting_input_adapter.py`
- `src/market_scanner/reporting/telegram_renderer.py`

`scripts/reporting/build_reporting_layer.py` still owns script-era dashboard/log output and Telegram message artifact writing. `scripts/reporting/build_telegram_summary.py` still delegates to that reporting layer. `scripts/reporting/send_telegram.py` still owns legacy Telegram delivery behavior. Existing `src/market_scanner/reporting/` modules remain reporting/Telegram scaffolds and migration references; the new `report_boundary.py` is the canonical report artifact planning owner for BL34.

These legacy files were not edited, moved, deleted, wrapped, or given new canonical runtime authority.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, Telegram delivery, and legacy Decision Engine execution.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_reporting.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- report artifact plan is deterministic;
- report artifact plan is side-effect-free;
- report artifact plan forbids provider calls by default;
- report artifact plan forbids production data writes by default;
- report artifact plan forbids report file writes by default;
- report artifact plan forbids `reports/daily/telegram_message.txt` writes by default;
- report artifact plan forbids Telegram delivery by default;
- report artifact plan forbids portfolio/watchlist mutation by default;
- report artifact plan forbids final BUY/SELL/HOLD by default;
- report artifact plan forbids allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior;
- report artifact plan exposes artifact-planning types only;
- report artifact plan explicitly lists blocked write and delivery behavior;
- canonical app uses the canonical report artifact boundary;
- canonical app does not import or invoke legacy runner scripts;
- canonical report artifact boundary does not import or invoke legacy report/Telegram execution;
- legacy runner, report, Telegram, and message files were not expanded to import canonical reporting/app modules.

## Side-effect guarantees

The canonical report artifact plan records these guarantees for every report artifact stage:

```text
provider_calls_allowed = False
production_data_writes_allowed = False
report_file_writes_allowed = False
daily_message_file_writes_allowed = False
telegram_delivery_allowed = False
portfolio_watchlist_mutation_allowed = False
final_outcomes_allowed = False
delivery_outputs_allowed = False
production_pipeline_allowed = False
```

The report artifact boundary has no import-time side effects and does not create files during plan construction.

## Python file creation justification

The canonical reporting files were created because BL28 approved a dedicated report artifact ownership boundary. They replace scattered or legacy report/reporting/Telegram artifact responsibility over time rather than adding competing temporary report helpers. Existing relevant modules were inspected first, including `src/market_scanner/app.py`, `src/market_scanner/messaging/`, `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `scripts/reporting/send_telegram.py`, and existing `src/market_scanner/reporting/` modules.

`tests/unit/test_v2_canonical_reporting.py` was created because no existing test file owned canonical report artifact boundary behavior. Existing tests cover legacy reporting/Telegram behavior, reporting input scaffolds, the canonical app boundary, or other canonical runtime boundaries.

No one-off runtime helper files, temporary migration files, quick report files, Telegram report files, report writer files, delivery bridges, or parallel shortcut runners were created.

## Blocked report write behavior

The canonical report artifact boundary explicitly blocks:

```text
write_report_file
write_daily_report
write_daily_message_file
write_telegram_message_file
write_reports_daily_telegram_message_txt
production_report_artifact
```

Report artifact planning and production report file writing remain separate.

## Blocked delivery behavior

The canonical report artifact boundary explicitly blocks:

```text
telegram_send
telegram_delivery
email_send
production_notification
```

Report artifact planning and delivery remain separate.

## Blocked investment semantics

The canonical report artifact boundary explicitly blocks final state and behavior semantics including:

```text
buy
sell
hold
allocate
increase_position
reduce_position
target_price
tradeable
not_tradeable
recommendation
allocation
conviction
urgency
scoring
target-price
tradeability
```

No report files, `reports/daily/telegram_message.txt`, Telegram delivery, production notification, final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Remaining migration work

Recommended next migration work:

1. Establish the canonical delivery/Telegram boundary separately from message composition and report artifact planning.
2. Keep production report generation disconnected until a separate report generation sprint explicitly approves file writes.
3. Later, migrate useful communication-only report artifact logic from legacy reporting modules into canonical owners with tests.
4. Keep production data writes, Telegram delivery, portfolio/watchlist updates, provider calls, and Decision Engine behavior disconnected until separately approved.
5. Only after canonical callers exist and tests pass, review whether legacy report/Telegram files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

## Guardrails confirmation

- No credentials committed.
- No raw live payloads committed.
- No production data writes.
- No reports generated.
- No report files written.
- No `reports/daily/telegram_message.txt` created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runners were not expanded.
- Legacy report/message/Telegram files were not expanded.
- Report artifact planning and delivery remain separate.

## Next recommended step

RESET-10L-BL35 — Migrate Delivery and Telegram Runtime Logic to Canonical V2 Boundary.
