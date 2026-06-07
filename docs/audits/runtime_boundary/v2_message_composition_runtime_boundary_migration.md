# v2 Message Composition Runtime Boundary Migration

## Status

Completed by RESET-10L-BL33.

## Reset stage

RESET-10L-BL33 — Migrate Message Composition Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts message composition runtime migration away from legacy report/Telegram scripts and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical message composition boundary under:

```text
src/market_scanner/messaging/
```

The new boundary is deterministic and dry-run/planning-only. It composes no real message payload, writes no report file, sends no Telegram message, runs no provider, writes no production data, modifies no portfolio/watchlist file, executes no production pipeline, and produces no investment recommendation.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`

Policy result:

- New messaging Python files were created only because BL28 approved `src/market_scanner/messaging/` as the canonical message composition ownership boundary.
- A new messaging unit test file was created because no existing test file owned canonical message composition boundary behavior.
- Existing legacy runner, report, Telegram, delivery, portfolio, and watchlist files were inspected but not edited.
- No one-off message helper, Telegram helper, delivery bridge, quick message runner, report builder, or parallel runtime shortcut was created.

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
- `docs/active/backlog.md`

Canonical app, scanner, analysis, decision, and tests:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`

Legacy runtime, message, report, and Telegram files:

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

Repository structure and references were inspected with static file inventory and grep searches for Telegram, message, notification, send, report, daily, summary, compose, format, `run_scan`, `run_full_pipeline`, and forbidden investment semantics.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/messaging/__init__.py`
- `src/market_scanner/messaging/message_contracts.py`
- `src/market_scanner/messaging/message_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime, report, Telegram, delivery, portfolio, or watchlist files were changed.

## Canonical message composition boundary result

The canonical message composition boundary is now established under:

```text
src/market_scanner/messaging/
```

Public functions:

- `build_message_composition_plan`
- `build_message_composition_policy`
- `build_review_message_plan`

Core records:

- `MessageCompositionPlan`
- `MessageCompositionPolicy`
- `MessageCompositionStage`

The message composition plan describes:

- message composition stage name;
- upstream decision/review data category;
- allowed composition-oriented message types;
- blocked delivery behavior;
- blocked final state codes;
- blocked behavior codes;
- whether provider calls are allowed;
- whether data writes are allowed;
- whether report files are allowed;
- whether Telegram delivery is allowed;
- whether portfolio/watchlist mutation is allowed;
- whether final output behavior is allowed;
- whether delivery outputs are allowed;
- legacy migration status.

The initial message composition boundary has two planning stages:

```text
review_message_composition
delivery_separation_review
```

All stages are side-effect-free by default.

Allowed composition-oriented message types:

```text
review_summary
limited_analysis_summary
evidence_gap_summary
dry_run_summary
operator_review_message
```

## App integration result

`src/market_scanner/app.py` now imports the canonical message composition boundary and includes the deterministic message composition plan in the canonical app dry-run runtime plan.

The canonical app message composition stage now references:

```text
src/market_scanner/messaging/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, legacy report builders, legacy Telegram delivery files, or legacy message execution files.

## Legacy message/report/Telegram status

Legacy message, report, and Telegram responsibilities remain migration targets and were not expanded.

Current legacy and reporting-adjacent message authority remains concentrated in:

- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `src/market_scanner/reporting/reporting_input_adapter.py`
- `src/market_scanner/reporting/telegram_renderer.py`

`scripts/reporting/build_reporting_layer.py` still owns script-era reporting output and Telegram message text construction. `scripts/reporting/build_telegram_summary.py` still delegates to that reporting layer. `scripts/reporting/send_telegram.py` still owns legacy Telegram delivery behavior. The newer `src/market_scanner/reporting/` modules remain reporting and Telegram scaffolds, not canonical message composition authorities in this sprint.

These files were not edited, moved, deleted, wrapped, or given new canonical authority.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, Telegram delivery, and legacy Decision Engine execution.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_messaging.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- message composition plan is deterministic;
- message composition plan is side-effect-free;
- message composition plan forbids provider calls by default;
- message composition plan forbids production data writes by default;
- message composition plan forbids report file writes by default;
- message composition plan forbids Telegram delivery by default;
- message composition plan forbids portfolio/watchlist mutation by default;
- message composition plan forbids final BUY/SELL/HOLD by default;
- message composition plan forbids allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior;
- message composition plan exposes composition-oriented message types only;
- message composition plan explicitly lists blocked delivery behavior;
- canonical app uses the canonical message composition boundary;
- canonical app does not import or invoke legacy runner scripts;
- canonical message composition boundary does not import or invoke legacy report/Telegram execution;
- legacy runner, report, Telegram, and message files were not expanded to import canonical messaging/app modules.

## Side-effect guarantees

The canonical message composition plan records these guarantees for every message composition stage:

```text
provider_calls_allowed = False
data_writes_allowed = False
report_files_allowed = False
telegram_delivery_allowed = False
portfolio_watchlist_mutation_allowed = False
final_outcomes_allowed = False
delivery_outputs_allowed = False
```

The message composition boundary has no import-time side effects and does not create files during plan construction.

## Python file creation justification

The canonical messaging files were created because BL28 approved a dedicated message composition ownership boundary. They replace scattered or legacy message/report/Telegram composition responsibility over time rather than adding competing temporary message helpers. Existing relevant modules were inspected first, including `src/market_scanner/app.py`, `src/market_scanner/decision/`, `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `scripts/reporting/send_telegram.py`, and `src/market_scanner/reporting/`.

`tests/unit/test_v2_canonical_messaging.py` was created because no existing test file owned canonical message composition boundary behavior. Existing tests cover legacy reporting/Telegram behavior, the canonical app boundary, or other canonical runtime boundaries.

No one-off runtime helper files, temporary migration files, quick message files, Telegram helper files, report-builder files, delivery bridges, or parallel shortcut runners were created.

## Blocked delivery behavior

The canonical message composition boundary explicitly blocks:

```text
telegram_send
telegram_delivery
email_send
write_report_file
write_daily_message_file
production_notification
```

Message composition and delivery remain separate.

## Blocked investment semantics

The canonical message composition boundary explicitly blocks final state and behavior semantics including:

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

No Telegram delivery, report generation, production notification, final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Remaining migration work

Recommended next migration work:

1. Establish the canonical report artifact boundary separately from message composition and delivery.
2. Keep Telegram delivery disconnected until a separate delivery boundary sprint approves it.
3. Later, migrate useful communication-only formatting logic from legacy reporting/Telegram modules into canonical owners with tests.
4. Keep production data writes, report file generation, Telegram delivery, portfolio/watchlist updates, provider calls, and Decision Engine behavior disconnected until separately approved.
5. Only after canonical callers exist and tests pass, review whether legacy message/report/Telegram files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

## Guardrails confirmation

- No credentials committed.
- No raw live payloads committed.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runners were not expanded.
- Legacy message/report/Telegram files were not expanded.
- Message composition and delivery remain separate.

## Next recommended step

RESET-10L-BL34 — Migrate Report Artifact Runtime Logic to Canonical V2 Boundary.
