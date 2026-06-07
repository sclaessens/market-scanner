# High-Risk Script-Era Test Execution Cleanup

## Status

Completed by RESET-10L-BL45.

## Reset stage

RESET-10L-BL45 - Remove High-Risk Script-Era Test Execution.

## Purpose

Remove active pytest execution and import-time dependency from high-risk script-era Python modules under `scripts/` while preserving the old behavior tests as explicit migration blockers.

This sprint did not migrate, archive, delete, or modify script-era production files. It moved active validation toward canonical v2 boundary tests and static blocker checks.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_archived_script_execution_test_cleanup.md`
- `docs/active/backlog.md`
- Canonical boundary records from RESET-10L-BL29 through RESET-10L-BL35.
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Files inspected

Governance and migration records:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_archived_script_execution_test_cleanup.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`
- `docs/active/v2_delivery_runtime_boundary_migration.md`
- `docs/active/backlog.md`

Repository surfaces:

- `tests/`
- `scripts/core/`
- `scripts/fundamentals/`
- `scripts/portfolio/`
- `scripts/reporting/`
- `scripts/telegram/`
- `scripts/watchlist/`
- `scripts/data_sources/`
- `src/market_scanner/`
- `.github/workflows/daily-market-scan.yml`

## Files changed

- `tests/conftest.py`
- `tests/test_operator_visibility.py`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/backlog.md`

No script-era production files, archived scripts, workflows, data files, report files, portfolio/watchlist files, or canonical production source files were changed.

## High-risk test execution before BL45

Static inspection found many active tests importing script-era modules through `from scripts...` imports. These included tests for:

- legacy Decision Engine final/allocation semantics;
- SEC CompanyFacts bulk intake and transformation support;
- script-era fundamentals, metrics, quality, history, and analysis builders;
- script-era context, validation, timing, stability, and portfolio intelligence builders;
- report layer and Telegram summary builders;
- portfolio/data-source/diagnostic/operator utilities.

These tests created an active pytest dependency on high-risk script-era modules even when individual tests used temporary paths or monkeypatches to reduce side effects.

## High-risk test execution after BL45

`tests/conftest.py` now marks the high-risk script-era behavior test modules as collection blockers through `collect_ignore`.

As a result, active pytest collection no longer imports or executes those high-risk legacy behavior suites by default. The old test files remain in the repository as migration evidence and are explicitly classified as inactive blockers, not approved long-term runtime authorities.

Static source references to `from scripts...` remain inside ignored test files. They are retained only as migration evidence and are documented as blockers for future canonical migration work.

## Test dependency classifications

| classification | files |
|---|---|
| `MIGRATE_TO_CANONICAL_BOUNDARY_TEST` | Canonical boundary tests under `tests/unit/test_v2_canonical_*.py` remain the active runtime-boundary validation path. |
| `REWRITE_AS_STATIC_POLICY_TEST` | `tests/test_operator_visibility.py` now statically verifies the inactive high-risk blocker list. Existing canonical tests statically inspect legacy paths as policy evidence without importing or executing them. |
| `KEEP_AS_TEMPORARY_LEGACY_TEST_BLOCKER` | The script-era behavior tests listed in `tests/conftest.py` remain present but are not collected or executed. |
| `REMOVE_ONLY_IF_REDUNDANT` | None. No test files were deleted. |

## Tests rewritten

- `tests/test_operator_visibility.py` now verifies that each retained high-risk script-era behavior test remains present and is listed as an inactive migration blocker.

## Tests retained as temporary blockers

The following active-collection blockers remain for future canonical migration or retirement review:

- `tests/core/test_build_context_backfill.py`
- `tests/core/test_build_context_layer.py`
- `tests/core/test_build_entry_quality_backfill.py`
- `tests/core/test_build_fundamental_analysis.py`
- `tests/core/test_build_fundamental_layer.py`
- `tests/core/test_build_fundamental_metrics.py`
- `tests/core/test_build_fundamentals_history_intake.py`
- `tests/core/test_build_portfolio_intelligence.py`
- `tests/core/test_build_stability_layer.py`
- `tests/core/test_build_timing_state_layer.py`
- `tests/core/test_build_validation_layer.py`
- `tests/core/test_decision_engine.py`
- `tests/core/test_entry_quality.py`
- `tests/core/test_fundamentals_operational_validation.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/data_sources/test_prefill_common.py`
- `tests/data_sources/test_prefill_fundamentals.py`
- `tests/data_sources/test_prefill_portfolio_metadata.py`
- `tests/diagnostics/test_audit_data_coverage.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_sec_ticker_cik_index.py`
- `tests/ops/test_capture_historical_evidence.py`
- `tests/portfolio/test_portfolio_source_contract.py`
- `tests/reporting/test_build_reporting_layer.py`
- `tests/reporting/test_build_telegram_summary.py`

## Coverage preservation

Coverage is preserved through three safer channels:

- canonical v2 boundary tests continue to validate the active app, scanner, analysis, decision/review, messaging, reporting, and delivery planning boundaries;
- static policy tests continue to verify that archived runtime scripts and high-risk script-era test suites are non-canonical migration targets;
- legacy behavior tests remain available as source evidence for future migration, but they are no longer active execution targets.

This is intentionally a fail-closed cleanup step. It avoids executing high-risk script-era modules until each domain is migrated, replaced, or retired under a dedicated controlled sprint.

## Canonical boundary coverage used

Active canonical coverage remains under:

- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `tests/unit/test_v2_canonical_delivery.py`
- fundamentals provider and persistence contract tests that do not import script-era runtime modules.

## Remaining high-risk test dependencies

Static source references remain in ignored legacy behavior test files. Those files are no longer active pytest execution/import dependencies, but they remain migration blockers because their old assertions have not yet been moved into canonical owners.

Explicit status for requested high-risk files:

| high-risk file | active tests still import or execute it after BL45? | status |
|---|---:|---|
| `scripts/core/decision_engine.py` | no | Ignored `tests/core/test_decision_engine.py` remains a temporary migration blocker; canonical decision boundary tests only statically inspect legacy authority. |
| `scripts/reporting/send_telegram.py` | no | Static canonical delivery/reporting/messaging policy reads only. |
| `scripts/telegram/process_telegram_commands.py` | no | Static canonical delivery policy reads only. |
| `scripts/portfolio/portfolio_manager.py` | no | No active pytest import/execution found. |
| `scripts/portfolio/parse_trade_commands.py` | no | No active pytest import/execution found. |
| `scripts/watchlist/parse_watchlist_commands.py` | no | No active pytest import/execution found. |
| `scripts/core/data_fetcher.py` | no | Static canonical scanner metadata only. |
| `scripts/core/scanner.py` | no | Static canonical scanner metadata only. |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | no | Ignored `tests/fundamentals/test_sec_companyfacts_bulk_intake.py` remains a temporary migration blocker. |
| `scripts/reporting/build_reporting_layer.py` | no | Ignored reporting test remains a temporary migration blocker; canonical reporting tests use static policy reads. |
| `scripts/reporting/build_telegram_summary.py` | no | Ignored reporting test remains a temporary migration blocker; canonical messaging/reporting tests use static policy reads. |

No archived scripts are executed by active tests.

## Side-effect guarantees

- No high-risk script-era files were executed as part of the implemented test changes.
- No archived scripts were executed.
- No live providers were called.
- No SEC, EDGAR, broker, Telegram, or other network calls were made.
- No credentials were read.
- No production data was written.
- No report files were generated or written.
- No `reports/daily/telegram_message.txt` file was created or modified.
- No Telegram messages or Telegram artifacts were created.
- No portfolio or watchlist files were mutated.
- No production pipeline was executed.
- No final BUY/SELL/HOLD recommendation behavior was added.
- No allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Guardrails confirmation

- No script-era production files changed.
- No archived scripts modified.
- No archived scripts executed.
- No high-risk script-era files executed.
- No workflows changed.
- No credentials committed.
- No credentials read.
- No raw live payloads committed.
- No network calls performed.
- No production data writes.
- No reports generated.
- No report files written.
- No reports/daily/telegram_message.txt created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No Telegram API calls made.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No one-off temporary helper files created.
- No replacement runtime scripts created.
- Archived scripts remain historical references only.

## Known limitations

- The legacy behavior test files remain in the repository as inactive blockers. Their assertions still need domain-by-domain migration or retirement decisions.
- `tests/conftest.py` is a test-collection guard, not a canonical replacement for the legacy behavior covered by the ignored tests.
- This sprint does not decide whether script-era fundamentals, portfolio, watchlist, reporting, diagnostics, or Decision Engine logic should be migrated, archived, or deleted.

## Next recommended step

Proceed to `RESET-10L-BL46 - Fundamentals Script-Era Side-Effect Migration Review`.

The fundamentals/provider scripts are the safest next domain because they contain important evidence/provider logic, active historical tests, and side-effect risk that should be reviewed before any further archive/delete work.
