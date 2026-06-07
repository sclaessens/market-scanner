# v2 Delivery Runtime Boundary Migration

## Status

Completed by RESET-10L-BL35.

## Reset stage

RESET-10L-BL35 — Migrate Delivery and Telegram Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts delivery runtime migration away from legacy Telegram and delivery scripts and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical delivery boundary under:

```text
src/market_scanner/delivery/
```

The new boundary is deterministic and dry-run/planning-only. It defines delivery ownership and safety policy, but sends no Telegram message, calls no Telegram API, reads no credentials, performs no network call, writes no production data, writes no report file, creates no `reports/daily/telegram_message.txt`, modifies no portfolio/watchlist file, executes no production pipeline, and produces no investment recommendation.

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
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`

Policy result:

- New delivery Python files were created only because BL28 approved `src/market_scanner/delivery/` as the canonical delivery/Telegram ownership boundary.
- A new delivery unit test file was created because no existing test file owned canonical delivery boundary behavior.
- Existing legacy runner, delivery, Telegram, report, message, portfolio, and watchlist files were inspected but not edited.
- No one-off delivery helper, Telegram sender, quick delivery runner, legacy delivery bridge, credential helper, or parallel runtime shortcut was created.

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
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`
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
- `tests/unit/test_v2_canonical_reporting.py`

Legacy runtime, delivery, Telegram, report, and message files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `src/market_scanner/reporting/reporting_input_adapter.py`
- `src/market_scanner/reporting/reporting_engine.py`
- `src/market_scanner/reporting/telegram_renderer.py`

Repository structure and references were inspected with static file inventory and grep searches for Telegram, send, delivery, notification, bot, chat/token, request/network/API behavior, report/daily/write behavior, `run_scan`, `run_full_pipeline`, and forbidden investment semantics.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/delivery/__init__.py`
- `src/market_scanner/delivery/delivery_contracts.py`
- `src/market_scanner/delivery/delivery_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_delivery.py`
- `docs/active/v2_delivery_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime, delivery, Telegram, report writer, message, portfolio, or watchlist files were changed.

## Canonical delivery boundary result

The canonical delivery boundary is now established under:

```text
src/market_scanner/delivery/
```

Public functions:

- `build_delivery_plan`
- `build_delivery_policy`
- `build_telegram_delivery_plan`

Core records:

- `DeliveryPlan`
- `DeliveryPolicy`
- `DeliveryStage`

The delivery plan describes:

- delivery stage name;
- upstream message/report artifact category;
- allowed delivery-planning channels;
- blocked execution behavior;
- blocked final state codes;
- blocked behavior codes;
- whether provider calls are allowed;
- whether network calls are allowed;
- whether credential reads are allowed;
- whether production data writes are allowed;
- whether report file writes are allowed;
- whether `reports/daily/telegram_message.txt` writes are allowed;
- whether Telegram sending is allowed;
- whether portfolio/watchlist mutation is allowed;
- whether final output behavior is allowed;
- whether message composition or report generation is allowed;
- whether production pipeline execution is allowed;
- legacy migration status.

The initial delivery boundary has two planning stages:

```text
telegram_delivery_planning
delivery_execution_policy_block
```

All stages are side-effect-free by default.

Allowed delivery-planning channels:

```text
telegram_planned
operator_review_delivery
dry_run_delivery
manual_delivery_review
```

## App integration result

`src/market_scanner/app.py` now imports the canonical delivery boundary and includes the deterministic delivery plan in the canonical app dry-run runtime plan.

The canonical app delivery stage now references:

```text
src/market_scanner/delivery/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, legacy Telegram senders, legacy report builders, legacy message files, delivery files, credential loaders, or network clients.

## Legacy delivery/Telegram status

Legacy delivery and Telegram responsibilities remain migration targets and were not expanded.

Current legacy delivery authority remains concentrated in:

- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

`scripts/reporting/send_telegram.py` still owns script-era Telegram send behavior, including credential reads and Telegram API calls. `scripts/telegram/process_telegram_commands.py` still owns script-era Telegram command polling and trade command parsing. These files were inspected but not edited, moved, deleted, wrapped, or given new canonical runtime authority.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, Telegram delivery, and legacy Decision Engine execution.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_delivery.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- delivery plan is deterministic;
- delivery plan is side-effect-free;
- delivery plan forbids provider calls by default;
- delivery plan forbids network calls by default;
- delivery plan forbids credential access by default;
- delivery plan forbids production data writes by default;
- delivery plan forbids report file writes by default;
- delivery plan forbids `reports/daily/telegram_message.txt` writes by default;
- delivery plan forbids Telegram sending by default;
- delivery plan forbids portfolio/watchlist mutation by default;
- delivery plan forbids final investment outcomes and investment behavior;
- delivery plan exposes delivery-planning channels only;
- delivery plan explicitly lists blocked execution behavior;
- canonical app uses the canonical delivery boundary;
- canonical app does not import legacy runner scripts, legacy Telegram/report execution, network clients, or credential loaders;
- legacy runner, report, message, Telegram, and delivery files were not expanded.

## Side-effect guarantees

The canonical delivery boundary does not:

- call providers;
- perform network calls;
- read credentials;
- write production data;
- write report files;
- write `reports/daily/telegram_message.txt`;
- create Telegram artifacts;
- send Telegram messages;
- call Telegram APIs;
- modify portfolio/watchlist files;
- execute the production pipeline;
- invoke legacy runners;
- produce final investment outcomes or investment behavior.

## Python file creation justification

The canonical delivery files were created because BL28 approved a dedicated delivery/Telegram ownership boundary. They replace scattered or legacy Telegram/delivery responsibility over time rather than adding competing temporary delivery helpers.

`tests/unit/test_v2_canonical_delivery.py` was created because no existing test file owned canonical delivery-boundary behavior.

No new one-off runtime helper, temporary migration file, Telegram sender, network helper, credential helper, or legacy bridge was created.

## Blocked network/credential behavior

The boundary explicitly blocks:

- `network_post`
- `network_get`
- `credential_read`
- `telegram_api_call`
- `telegram_bot_post`

The delivery plan sets:

```text
network_calls_allowed = False
credentials_allowed = False
```

## Blocked Telegram behavior

The boundary explicitly blocks:

- `telegram_send`
- `telegram_api_call`
- `telegram_bot_post`

The delivery plan sets:

```text
telegram_sending_allowed = False
```

## Blocked report write behavior

The boundary explicitly blocks:

- `write_delivery_artifact`
- `write_reports_daily_telegram_message_txt`

The delivery plan sets:

```text
report_file_writes_allowed = False
daily_message_file_writes_allowed = False
report_generation_allowed = False
```

## Blocked investment semantics

The boundary explicitly blocks:

- `buy`
- `sell`
- `hold`
- `allocate`
- `increase_position`
- `reduce_position`
- `target_price`
- `tradeable`
- `not_tradeable`
- `recommendation`
- `allocation`
- `conviction`
- `urgency`
- `scoring`
- `target-price`
- `tradeability`

No final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Remaining migration work

- Decide when and how legacy Telegram send behavior can be migrated, if future governance approves actual delivery execution.
- Keep message composition, report artifact planning, and delivery planning separate.
- Review whether `scripts/reporting/send_telegram.py` and `scripts/telegram/process_telegram_commands.py` can be archived after a future controlled delivery migration or explicit retirement decision.
- Continue toward legacy runtime script archive readiness review before moving or deleting script-era files.

## Guardrails confirmation

- No credentials committed.
- No credentials read.
- No raw live payloads committed.
- No network calls performed.
- No production data writes.
- No reports generated.
- No report files written.
- No `reports/daily/telegram_message.txt` created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No Telegram API calls made.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runners were not expanded.
- Legacy report/message/Telegram/delivery files were not expanded.
- Delivery, message composition, and reporting remain separate.

No Telegram messages, Telegram API calls, credential reads, network calls, report files, `reports/daily/telegram_message.txt`, production notification, final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Next recommended step

RESET-10L-BL36 — Legacy Runtime Script Archive Readiness Review.
