# v2 Python Architecture Cleanup and Legacy Decoupling Review

## Status

Completed by RESET-10L-BL27.

## Reset stage

RESET-10L-BL27 — Python Architecture Cleanup and Legacy Decoupling Review.

## Purpose

This review inventories the committed Python architecture before additional real-analysis features are added. It identifies canonical v2 modules, legacy dependencies that are still used, duplicate responsibilities, unclear entrypoints, old runners, scanner duplication, analysis duplication, reporting and Telegram coupling, and cleanup work that should happen before implementation resumes.

This sprint is review-only. No runtime code, tests, generated data, reports, workflows, portfolio data, or watchlist data were changed.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/backlog.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

Policy application:

- No Python files were created.
- No Python files were changed.
- No Python files were moved.
- No Python files were deleted.
- A legacy Python file that is still used is treated as a temporary dependency, not automatically approved for long-term retention.
- Still-used legacy files are classified for migration, decoupling, archive-after-migration, deletion-after-confirmation, or do-not-touch-yet treatment.

## Inspection method

Static inspection only. No provider calls, pipeline execution, report generation, Telegram delivery, portfolio/watchlist changes, or production data writes were performed.

Commands and inspection patterns used:

```bash
find . -name "*.py" \
  -not -path "./.venv/*" \
  -not -path "./venv/*" \
  -not -path "./__pycache__/*" \
  -not -path "*/__pycache__/*"

grep -R "if __name__ == .__main__." -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__

grep -R "telegram" -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__

grep -R "report" -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__

grep -R "scanner\|scan\|universe" -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__

grep -R "analysis\|analy" -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__

grep -R "decision" -n . --include="*.py" \
  --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__
```

Representative runtime files were sampled with direct reads to distinguish compatibility wrappers from responsibility owners.

## Python file inventory summary

Committed Python files outside virtualenv and cache paths:

| Area | Count | Review summary |
|---|---:|---|
| `src/market_scanner/` | 34 | Compact v2 package with contracts, records, synthetic scaffolds, provider/persistence boundary, reporting scaffold, and minimal orchestration. |
| `scripts/` | 54 | Legacy runtime-heavy area with full pipeline, scanner, layer builders, fundamentals analysis, reporting, Telegram, portfolio, watchlist, diagnostics, and compatibility wrappers. |
| `legacy/` | 4 | Explicit legacy Telegram/watchlist command files. |
| `config/` | 1 | Legacy shared settings for data/report paths and scanner constants. |
| `tests/` | 62 | Test support for v2 contracts and legacy script behavior. |
| Total | 155 | The repository has a small canonical v2 package and a much larger script-era runtime surface. |

## Entrypoints and runners found

Entrypoints were found in:

- `scripts/run_full_pipeline.py`
- `scripts/run_scan.py`
- `scripts/analyze_validation.py`
- `scripts/validate_scans.py`
- `scripts/core/analyze_validation.py`
- `scripts/core/build_context_backfill.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/core/build_stability_layer.py`
- `scripts/core/build_timing_state_layer.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/decision_engine.py`
- `scripts/core/log_scans.py`
- `scripts/core/validate_scans.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/ops/capture_historical_evidence.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/portfolio/evaluate_positions.py`
- `scripts/portfolio/parse_trade_commands.py`
- `scripts/portfolio/portfolio_manager.py`
- `scripts/portfolio/test_portfolio.py`
- `scripts/watchlist/auto_watchlist_from_scan.py`
- `scripts/watchlist/build_watchlist.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/watchlist/parse_watchlist_commands.py`
- `scripts/watchlist/update_watchlist_actions.py`
- `legacy/telegram/add_to_watchlist.py`
- `legacy/watchlist/builder.py`
- `legacy/watchlist/evaluator.py`
- `legacy/watchlist/parser.py`

Primary finding: the repository has no single documented canonical v2 runtime entrypoint. `scripts/run_scan.py` remains the broad legacy orchestrator and `scripts/run_full_pipeline.py` wraps it.

## Scanner/universe files found

- `scripts/run_scan.py`
- `scripts/core/scanner.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/indicators.py`
- `scripts/core/regime.py`
- `scripts/core/validator.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/log_scans.py`
- `scripts/core/validate_scans.py`
- `scripts/validate_scans.py`
- `scripts/analyze_validation.py`
- `scripts/core/analyze_validation.py`
- `scripts/watchlist/auto_watchlist_from_scan.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `src/market_scanner/discovery/__init__.py`
- `src/market_scanner/validation/validation_contracts.py`

Primary finding: scanner execution and universe selection remain in legacy scripts, while v2 has validation contracts but no canonical scanner runtime owner.

## Analysis files found

- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_normalization_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `src/market_scanner/fundamentals/source_data_readiness.py`
- `src/market_scanner/fundamentals/source_data_records.py`
- `src/market_scanner/fundamentals/fundamental_contracts.py`

Primary finding: v2 source-data capture, normalization, readiness, and persistence are canonical under `src/market_scanner/fundamentals/`, but legacy metrics, quality, and analysis remain under `scripts/fundamentals/`. `scripts/core/build_fundamental_*` files are compatibility wrappers over `scripts/fundamentals/*`.

## Decision/review files found

- `scripts/core/decision_engine.py`
- `src/market_scanner/decisions/decision_engine.py`
- `src/market_scanner/decisions/decision_records.py`
- `src/market_scanner/reporting/reporting_engine.py`
- `src/market_scanner/reporting/report_records.py`
- `src/market_scanner/reporting/reporting_input_contracts.py`
- `scripts/core/build_stability_layer.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/portfolio/evaluate_positions.py`

Primary finding: the certified allocation authority still lives in legacy `scripts/core/decision_engine.py` for current full-pipeline behavior. The v2 `src/market_scanner/decisions/decision_engine.py` is a review scaffold, not a replacement for allocation authority. This split must be explicitly governed before migration.

## Report/message/Telegram files found

- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/reporting/reporter.py`
- `scripts/telegram/process_telegram_commands.py`
- `legacy/telegram/add_to_watchlist.py`
- `src/market_scanner/reporting/reporting_engine.py`
- `src/market_scanner/reporting/report_records.py`
- `src/market_scanner/reporting/reporting_input_adapter.py`
- `src/market_scanner/reporting/reporting_input_contracts.py`
- `src/market_scanner/reporting/telegram_contracts.py`
- `src/market_scanner/reporting/telegram_renderer.py`

Primary finding: v2 reporting and Telegram rendering are pure in-memory communication scaffolds under `src/market_scanner/reporting/`. Legacy reporting scripts still write `data/processed` and `reports/daily/telegram_message.txt`; Telegram delivery scripts import `requests`, load credentials, and call Telegram APIs. These are separate responsibilities and must be decoupled from analysis review paths.

## Provider/fundamentals files found

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/data_sources/common.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/data_sources/prefill_portfolio_metadata.py`

Primary finding: the v2 provider-to-persistence path is controlled and narrow. SEC bulk intake, ticker-CIK indexing, transformation review, and data-source prefill remain script-era utilities and should not be treated as canonical without migration review.

## Duplicate responsibility groups

| Group | Files involved | Overlap | Likely canonical owner | Legacy files to decouple | Migration risk | Recommended next action |
|---|---|---|---|---|---|---|
| Application entrypoints / runners | `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, many `scripts/*/__main__` files, `src/market_scanner/orchestration/pipeline_core.py` | Multiple runnable surfaces, with `run_scan.py` owning broad side effects and `pipeline_core.py` owning synthetic v2 scaffold only. | No canonical production owner identified yet; `src/market_scanner/orchestration/pipeline_core.py` is canonical for v2 synthetic scaffold only. | `scripts/run_scan.py`, `scripts/run_full_pipeline.py` | High: full pipeline writes data/reports and calls Telegram delivery. | BL28 should define the canonical v2 runtime architecture before migration. |
| Scanner / universe selection | `scripts/core/data_fetcher.py`, `scripts/core/scanner.py`, `scripts/core/indicators.py`, `scripts/core/regime.py`, `scripts/run_scan.py`, `src/market_scanner/discovery/__init__.py` | Legacy scripts own live scanner execution; v2 discovery has no runtime owner. | No canonical owner identified yet. | `scripts/core/*scanner/data_fetcher/indicators/regime*` as needed | High: provider/network/data-write behavior may be embedded. | Define scanner boundary and dry-run requirements before moving logic. |
| Validation layer | `scripts/core/build_validation_layer.py`, `scripts/core/validator.py`, `scripts/core/validate_scans.py`, `scripts/validate_scans.py`, `src/market_scanner/validation/validation_contracts.py` | Script validation builders and v2 contracts coexist. | `src/market_scanner/validation/validation_contracts.py` for contracts; no canonical runtime owner yet. | Script validation builders and wrappers | Medium: existing tests cover legacy behavior. | Map required runtime behavior to v2 contract boundary. |
| Fundamentals provider / normalization / analysis | `src/market_scanner/fundamentals/*`, `scripts/fundamentals/*`, `scripts/core/build_fundamental_*` | v2 owns provider/persistence evidence; scripts own metrics, quality, and analysis outputs. | `src/market_scanner/fundamentals/` for source evidence; no canonical owner yet for real-analysis metrics/quality. | `scripts/fundamentals/build_*`, `scripts/core/build_fundamental_*` wrappers | High: recent BL20-BL26 work depends on script analysis tests. | Define whether analysis logic moves into existing `src/market_scanner/fundamentals/` or a future approved module. |
| Decision / review boundary | `scripts/core/decision_engine.py`, `src/market_scanner/decisions/decision_engine.py`, `src/market_scanner/decisions/decision_records.py` | Legacy Decision Engine owns allocation behavior; v2 decision scaffold is review-only. | `scripts/core/decision_engine.py` remains allocation authority until a governed migration is approved. | None for deletion yet. | Very high: certified doctrine depends on this boundary. | Do not migrate until a dedicated Decision Engine migration plan exists. |
| Portfolio review / intelligence | `scripts/portfolio/*`, `scripts/core/build_portfolio_intelligence.py`, `src/market_scanner/portfolio/*` | Legacy portfolio scripts mutate/read portfolio artifacts; v2 portfolio files are contracts. | `src/market_scanner/portfolio/` for contracts only; no canonical runtime owner yet. | `scripts/portfolio/*`, `scripts/core/build_portfolio_intelligence.py` | High: portfolio state and trade-command parsing are sensitive. | Separate review-only portfolio classification from command-processing behavior. |
| Reporting / message composition | `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `src/market_scanner/reporting/*` | Legacy reporting writes artifacts; v2 reporting is pure communication scaffold. | `src/market_scanner/reporting/` for pure contracts/rendering. | `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py` | Medium-high: tests still validate legacy report artifact behavior. | Decouple rendering from artifact writes and delivery. |
| Telegram delivery | `scripts/reporting/send_telegram.py`, `scripts/telegram/process_telegram_commands.py`, `legacy/telegram/add_to_watchlist.py`, `legacy/watchlist/parser.py` | Outbound and inbound Telegram logic overlaps with report/message composition and command parsing. | No canonical delivery owner identified yet. | Telegram scripts and legacy Telegram files | High: credentials, network, and portfolio/watchlist command effects. | Define delivery boundary separately from renderer before migration. |
| Configuration loading | `config/settings.py`, many hard-coded `Path("data/...")` constants in scripts | Central settings coexist with many local path constants. | No canonical v2 config owner identified yet. | `config/settings.py` and script constants need review. | Medium: path constants determine production writes. | Define v2 config/path policy before runtime migration. |
| Shared utilities | `scripts/utils/utils.py`, `scripts/data_sources/common.py`, `src/market_scanner/shared/*` | v2 shared records/contracts coexist with legacy helper utilities. | `src/market_scanner/shared/` for v2 contracts and records. | `scripts/utils/utils.py`, `scripts/data_sources/common.py` | Low-medium. | Migrate only useful pure helpers after ownership is defined. |

## File classification table

Primary statuses use exactly one of: `CANONICAL_V2`, `LEGACY_DEPENDENCY`, `MIGRATE_LOGIC`, `ARCHIVE_AFTER_MIGRATION`, `DELETE_AFTER_CONFIRMATION`, `DO_NOT_TOUCH_YET`, `TEST_SUPPORT`.

### Canonical v2 package

| File path | Primary status | Responsibility category | Reason | Recommended next action |
|---|---|---|---|---|
| `src/market_scanner/__init__.py` | CANONICAL_V2 | shared_utility | Declares v2 package surface. | Keep; update only through package governance. |
| `src/market_scanner/context/__init__.py` | CANONICAL_V2 | unknown | Placeholder for context layer package. | Keep as placeholder until canonical context runtime is defined. |
| `src/market_scanner/decisions/__init__.py` | CANONICAL_V2 | decision_or_review_boundary | Package marker for v2 decision scaffold. | Keep. |
| `src/market_scanner/decisions/decision_engine.py` | CANONICAL_V2 | decision_or_review_boundary | Pure v2 review scaffold over synthetic pipeline records; not production allocation authority. | Keep; document distinction from legacy allocation engine. |
| `src/market_scanner/decisions/decision_records.py` | CANONICAL_V2 | decision_or_review_boundary | v2 decision/review records. | Keep. |
| `src/market_scanner/discovery/__init__.py` | CANONICAL_V2 | scanner_or_universe_selection | Placeholder for discovery layer package. | Define runtime owner in BL28. |
| `src/market_scanner/fundamentals/__init__.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | Package marker for v2 fundamentals. | Keep. |
| `src/market_scanner/fundamentals/fundamental_contracts.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | v2 fundamentals/source-data contract metadata. | Keep. |
| `src/market_scanner/fundamentals/fundamentals_normalization_adapter.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | Synthetic normalization adapter and no-side-effect boundary. | Keep; reconcile with provider adapter ownership if duplication grows. |
| `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | Normalization/source-readiness contract metadata. | Keep. |
| `src/market_scanner/fundamentals/fundamentals_persistence.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | Controlled synthetic persistence boundary with temp-only writes and forbidden path rejection. | Keep; do not connect to production writes without approval. |
| `src/market_scanner/fundamentals/fundamentals_provider_adapter.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | v2 provider-boundary mapping, derived FreeCashFlow, and growth evidence logic. | Keep; likely owner for source-to-evidence mapping. |
| `src/market_scanner/fundamentals/fundamentals_provider_contracts.py` | CANONICAL_V2 | provider_or_source_access | v2 provider/source contracts and data records. | Keep. |
| `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py` | CANONICAL_V2 | provider_or_source_access | Controlled one-ticker smoke harness boundary. | Keep narrow; no automatic live ingestion. |
| `src/market_scanner/fundamentals/source_data_readiness.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | v2 source-data readiness scaffold. | Keep. |
| `src/market_scanner/fundamentals/source_data_records.py` | CANONICAL_V2 | fundamentals_normalization_or_evidence | v2 source-data records. | Keep. |
| `src/market_scanner/orchestration/__init__.py` | CANONICAL_V2 | runner_or_orchestrator | Package marker for v2 orchestration. | Keep. |
| `src/market_scanner/orchestration/pipeline_core.py` | CANONICAL_V2 | runner_or_orchestrator | Minimal deterministic synthetic v2 pipeline scaffold. | Keep; not a production runner yet. |
| `src/market_scanner/portfolio/__init__.py` | CANONICAL_V2 | configuration | Package marker for v2 portfolio contracts. | Keep. |
| `src/market_scanner/portfolio/portfolio_contracts.py` | CANONICAL_V2 | decision_or_review_boundary | v2 portfolio contract checks without decisions or file effects. | Keep. |
| `src/market_scanner/portfolio/portfolio_source_contracts.py` | CANONICAL_V2 | decision_or_review_boundary | Portfolio source-of-truth and reporting-display contracts. | Keep. |
| `src/market_scanner/reporting/__init__.py` | CANONICAL_V2 | report_generation | Package marker for v2 reporting. | Keep. |
| `src/market_scanner/reporting/report_records.py` | CANONICAL_V2 | report_generation | Communication-only report records. | Keep. |
| `src/market_scanner/reporting/reporting_engine.py` | CANONICAL_V2 | report_generation | In-memory reporting scaffold preserving row identity. | Keep; no artifact writes. |
| `src/market_scanner/reporting/reporting_input_adapter.py` | CANONICAL_V2 | message_composition | Synthetic reporting input adapter for Telegram renderer. | Keep; avoid delivery coupling. |
| `src/market_scanner/reporting/reporting_input_contracts.py` | CANONICAL_V2 | message_composition | Reporting input aggregation contracts. | Keep. |
| `src/market_scanner/reporting/telegram_contracts.py` | CANONICAL_V2 | message_composition | Telegram UX contract metadata. | Keep; no delivery imports. |
| `src/market_scanner/reporting/telegram_renderer.py` | CANONICAL_V2 | message_composition | Pure in-memory Telegram renderer. | Keep; do not add network delivery. |
| `src/market_scanner/shared/__init__.py` | CANONICAL_V2 | shared_utility | Package marker for shared v2 utilities. | Keep. |
| `src/market_scanner/shared/data_contracts.py` | CANONICAL_V2 | shared_utility | Shared fixture/data contract metadata. | Keep. |
| `src/market_scanner/shared/records.py` | CANONICAL_V2 | shared_utility | Shared pipeline records/results. | Keep. |
| `src/market_scanner/timing/__init__.py` | CANONICAL_V2 | unknown | Placeholder for timing package. | Define runtime owner in BL28. |
| `src/market_scanner/validation/__init__.py` | CANONICAL_V2 | configuration | Package marker for validation contracts. | Keep. |
| `src/market_scanner/validation/validation_contracts.py` | CANONICAL_V2 | scanner_or_universe_selection | v2 validation contract checks without filtering or decisions. | Keep. |

### Config, legacy, and scripts

| File path | Primary status | Responsibility category | Reason | Recommended next action |
|---|---|---|---|---|
| `config/settings.py` | DO_NOT_TOUCH_YET | configuration | Central legacy settings for production data/report paths and scanner constants. | Define canonical v2 config/path policy before migration. |
| `legacy/telegram/add_to_watchlist.py` | ARCHIVE_AFTER_MIGRATION | delivery_or_telegram | Legacy Telegram-to-watchlist behavior; not aligned with current review-only delivery guardrails. | Confirm no current imports, then archive or delete after migration confirmation. |
| `legacy/watchlist/builder.py` | ARCHIVE_AFTER_MIGRATION | legacy_utility | Legacy watchlist builder. | Confirm no current runtime dependency before archival. |
| `legacy/watchlist/evaluator.py` | ARCHIVE_AFTER_MIGRATION | legacy_utility | Legacy watchlist evaluator. | Confirm no current runtime dependency before archival. |
| `legacy/watchlist/parser.py` | ARCHIVE_AFTER_MIGRATION | delivery_or_telegram | Legacy Telegram command log parser. | Confirm no current dependency and archive after command-boundary review. |
| `scripts/analyze_validation.py` | DELETE_AFTER_CONFIRMATION | legacy_utility | Root-level validation analysis duplicate separate from `scripts/core/analyze_validation.py`. | Confirm no imports/tests require root file, then remove or archive. |
| `scripts/core/analyze_validation.py` | LEGACY_DEPENDENCY | analysis | Current validation analysis script with data/log path behavior. | Decouple into validation review boundary or archive after replacement. |
| `scripts/core/build_context_backfill.py` | MIGRATE_LOGIC | analysis | Useful context backfill logic but script-era owner. | Migrate useful pure classification logic after context owner is defined. |
| `scripts/core/build_context_layer.py` | LEGACY_DEPENDENCY | analysis | Builds context layer from scanner/validation/sector inputs. | Decouple into canonical context runtime when defined. |
| `scripts/core/build_entry_quality_backfill.py` | MIGRATE_LOGIC | analysis | Useful historical entry-quality backfill logic. | Migrate only if still needed; otherwise archive after confirmation. |
| `scripts/core/build_fundamental_analysis.py` | ARCHIVE_AFTER_MIGRATION | analysis | Compatibility wrapper over `scripts.fundamentals.build_analysis`. | Remove after imports/tests are migrated to canonical owner. |
| `scripts/core/build_fundamental_layer.py` | ARCHIVE_AFTER_MIGRATION | fundamentals_normalization_or_evidence | Compatibility wrapper over `scripts.fundamentals.build_quality`. | Remove after runtime imports migrate. |
| `scripts/core/build_fundamental_metrics.py` | ARCHIVE_AFTER_MIGRATION | analysis | Compatibility wrapper over `scripts.fundamentals.build_metrics`. | Remove after runtime imports migrate. |
| `scripts/core/build_fundamentals_history_intake.py` | ARCHIVE_AFTER_MIGRATION | provider_or_source_access | Compatibility wrapper over `scripts.fundamentals.build_history_intake`. | Remove after imports migrate. |
| `scripts/core/build_portfolio_intelligence.py` | LEGACY_DEPENDENCY | decision_or_review_boundary | Produces portfolio intelligence consumed before Decision Engine. | Separate portfolio state modelling from allocation authority before migration. |
| `scripts/core/build_stability_layer.py` | LEGACY_DEPENDENCY | decision_or_review_boundary | Reads final decisions and builds stability state. | Keep until Decision Engine/reporting migration plan is approved. |
| `scripts/core/build_timing_state_layer.py` | LEGACY_DEPENDENCY | scanner_or_universe_selection | Timing-state classification script. | Define canonical timing owner before migration. |
| `scripts/core/build_validation_layer.py` | LEGACY_DEPENDENCY | scanner_or_universe_selection | Runtime validation layer builder. | Migrate into validation runtime after v2 architecture definition. |
| `scripts/core/data_fetcher.py` | LEGACY_DEPENDENCY | provider_or_source_access | Scanner data source access and ticker loading. | Decouple provider/network access from scanner classification. |
| `scripts/core/decision_engine.py` | LEGACY_DEPENDENCY | decision_or_review_boundary | Current allocation authority for full pipeline; still tested and doctrine-sensitive. | Do not alter until dedicated Decision Engine migration plan exists. |
| `scripts/core/indicators.py` | MIGRATE_LOGIC | scanner_or_universe_selection | Scanner technical indicator logic. | Migrate only after scanner boundary is defined. |
| `scripts/core/log_scans.py` | LEGACY_DEPENDENCY | runner_or_orchestrator | Writes scan logs. | Decouple logging from scanner classification. |
| `scripts/core/regime.py` | MIGRATE_LOGIC | scanner_or_universe_selection | Market regime classification helper. | Decide whether regime belongs in context/timing before migration. |
| `scripts/core/scanner.py` | LEGACY_DEPENDENCY | scanner_or_universe_selection | Core script-era scanner logic. | Migrate into canonical discovery/scanner owner after BL28. |
| `scripts/core/validate_scans.py` | LEGACY_DEPENDENCY | scanner_or_universe_selection | Runtime scan validation. | Reconcile with root `scripts/validate_scans.py` and v2 validation contracts. |
| `scripts/core/validator.py` | MIGRATE_LOGIC | scanner_or_universe_selection | Validation helper logic. | Migrate pure checks into v2 validation runtime if needed. |
| `scripts/data_sources/common.py` | MIGRATE_LOGIC | shared_utility | Shared prefill/data-source helper logic with forbidden-field checks. | Migrate pure helpers to `src/market_scanner/shared/` only if required. |
| `scripts/data_sources/prefill_fundamentals.py` | DO_NOT_TOUCH_YET | provider_or_source_access | Prefill utility may mutate data-source artifacts. | Review separately before any migration or deletion. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | DO_NOT_TOUCH_YET | provider_or_source_access | Prefill utility may mutate portfolio metadata. | Review separately with portfolio/watchlist guardrails. |
| `scripts/diagnostics/audit_data_coverage.py` | DO_NOT_TOUCH_YET | legacy_utility | Diagnostic utility over production-like artifacts and target universes. | Keep until audit ownership is defined. |
| `scripts/fundamentals/__init__.py` | MIGRATE_LOGIC | fundamentals_normalization_or_evidence | Script fundamentals package marker. | Retire after fundamentals runtime migration. |
| `scripts/fundamentals/build_analysis.py` | MIGRATE_LOGIC | analysis | Current real-analysis metrics review owner used by BL20-BL26 tests. | Define canonical analysis owner and migrate carefully. |
| `scripts/fundamentals/build_history_intake.py` | MIGRATE_LOGIC | provider_or_source_access | Fundamentals history validation/intake logic. | Migrate into v2 source-data boundary if still needed. |
| `scripts/fundamentals/build_metrics.py` | MIGRATE_LOGIC | analysis | Current governed metrics/growth computation owner. | Migrate into canonical fundamentals analysis module after BL28. |
| `scripts/fundamentals/build_quality.py` | MIGRATE_LOGIC | fundamentals_normalization_or_evidence | Fundamental quality layer builder. | Separate source-data quality from analysis review before migration. |
| `scripts/fundamentals/run_sec_transformation_review.py` | DO_NOT_TOUCH_YET | provider_or_source_access | SEC transformation review utility. | Keep until SEC source architecture is defined. |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | DO_NOT_TOUCH_YET | provider_or_source_access | Bulk SEC intake utility with external-source behavior. | Do not connect to runtime without explicit approval. |
| `scripts/fundamentals/sec_companyfacts_transform.py` | MIGRATE_LOGIC | provider_or_source_access | SEC companyfacts transform logic tested separately. | Migrate safe transformation logic into provider boundary after source policy. |
| `scripts/fundamentals/sec_ticker_cik_index.py` | MIGRATE_LOGIC | provider_or_source_access | SEC ticker/CIK index and coverage helper. | Migrate only under provider/source architecture review. |
| `scripts/ops/capture_historical_evidence.py` | DO_NOT_TOUCH_YET | legacy_utility | Historical evidence capture over data/reporting artifacts. | Keep isolated until audit artifact policy is defined. |
| `scripts/portfolio/build_portfolio.py` | LEGACY_DEPENDENCY | decision_or_review_boundary | Builds portfolio artifact. | Decouple from pipeline and portfolio mutation behavior. |
| `scripts/portfolio/evaluate_positions.py` | LEGACY_DEPENDENCY | decision_or_review_boundary | Evaluates portfolio positions. | Separate risk-state modelling from Decision Engine authority. |
| `scripts/portfolio/parse_trade_commands.py` | DO_NOT_TOUCH_YET | delivery_or_telegram | Parses trade commands and can affect portfolio workflows. | Review under strict Decision Engine and command-boundary policy. |
| `scripts/portfolio/portfolio_manager.py` | DO_NOT_TOUCH_YET | legacy_utility | Portfolio management utility likely mutates portfolio state. | Do not migrate without dedicated portfolio governance. |
| `scripts/portfolio/test_portfolio.py` | DELETE_AFTER_CONFIRMATION | legacy_utility | Test-like script inside runtime scripts area. | Confirm redundancy with tests, then delete or archive. |
| `scripts/reporting/build_reporting_layer.py` | LEGACY_DEPENDENCY | report_generation | Builds dashboard/log and writes Telegram artifact. | Decouple report generation from message rendering and delivery. |
| `scripts/reporting/build_telegram_summary.py` | ARCHIVE_AFTER_MIGRATION | message_composition | Wrapper around legacy reporting layer. | Retire after v2 renderer/input adapter replaces usage. |
| `scripts/reporting/reporter.py` | DO_NOT_TOUCH_YET | report_generation | Unclear report helper in legacy scripts. | Inspect in a dedicated reporting cleanup sprint. |
| `scripts/reporting/send_telegram.py` | LEGACY_DEPENDENCY | delivery_or_telegram | Loads credentials and calls Telegram API. | Isolate as delivery-only; never import into analysis/report composition. |
| `scripts/run_full_pipeline.py` | LEGACY_DEPENDENCY | application_entrypoint | Full pipeline wrapper around `scripts/run_scan.py`. | Retain until canonical entrypoint is defined; do not broaden. |
| `scripts/run_scan.py` | LEGACY_DEPENDENCY | runner_or_orchestrator | Broad end-to-end runtime with data writes, scanner, layers, Decision Engine, reporting, and Telegram delivery. | Highest-priority decoupling target after BL28 architecture definition. |
| `scripts/telegram/process_telegram_commands.py` | DO_NOT_TOUCH_YET | delivery_or_telegram | Inbound Telegram command processor with network and offset writes. | Keep isolated; review under delivery/command governance. |
| `scripts/utils/utils.py` | MIGRATE_LOGIC | shared_utility | Generic utility module. | Migrate only used pure helpers; otherwise archive after confirmation. |
| `scripts/validate_scans.py` | DELETE_AFTER_CONFIRMATION | legacy_utility | Root-level duplicate validation script separate from `scripts/core/validate_scans.py`. | Confirm no runtime dependency, then remove or archive. |
| `scripts/watchlist/auto_watchlist_from_scan.py` | DO_NOT_TOUCH_YET | scanner_or_universe_selection | Watchlist generation from scan logs. | Review under watchlist timing-state constraints. |
| `scripts/watchlist/build_watchlist.py` | DO_NOT_TOUCH_YET | legacy_utility | Watchlist builder. | Keep until watchlist runtime boundary is defined. |
| `scripts/watchlist/evaluate_watchlist.py` | DO_NOT_TOUCH_YET | legacy_utility | Watchlist evaluator; contains legacy non-English developer comment. | Review separately; do not migrate hidden assumptions. |
| `scripts/watchlist/parse_watchlist_commands.py` | DO_NOT_TOUCH_YET | delivery_or_telegram | Watchlist command parser. | Review with Telegram/command boundaries. |
| `scripts/watchlist/update_watchlist_actions.py` | DO_NOT_TOUCH_YET | legacy_utility | Watchlist action updater. | Keep isolated; action semantics require governance review. |

### Test support

All 62 committed Python files under `tests/` are classified as `TEST_SUPPORT`. They should be reviewed separately from runtime architecture. Current test support covers:

- v2 contracts and unit tests under `tests/contract/` and `tests/unit/`;
- legacy script behavior under `tests/core/`, `tests/reporting/`, `tests/portfolio/`, `tests/fundamentals/`, `tests/data_sources/`, `tests/diagnostics/`, and `tests/ops/`;
- integration scaffolds under `tests/integration/`;
- operator visibility behavior under `tests/test_operator_visibility.py`.

## Legacy dependencies still used

Still-used legacy files are not automatically approved for long-term retention. Current evidence indicates these remain active through tests or runtime imports:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/core/decision_engine.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_timing_state_layer.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/core/build_stability_layer.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`
- `scripts/core/log_scans.py`
- `scripts/core/validate_scans.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/send_telegram.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/portfolio/evaluate_positions.py`

These files should be treated as decoupling targets, not permanent v2 owners.

## Migrate-logic candidates

- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/core/build_context_backfill.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/indicators.py`
- `scripts/core/regime.py`
- `scripts/core/validator.py`
- `scripts/data_sources/common.py`
- `scripts/utils/utils.py`

Migration should happen only after canonical owners are defined.

## Archive-after-migration candidates

- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/reporting/build_telegram_summary.py`
- `legacy/telegram/add_to_watchlist.py`
- `legacy/watchlist/builder.py`
- `legacy/watchlist/evaluator.py`
- `legacy/watchlist/parser.py`

These look like wrappers or explicitly legacy files, but archive requires import/test confirmation first.

## Delete-after-confirmation candidates

- `scripts/analyze_validation.py`
- `scripts/validate_scans.py`
- `scripts/portfolio/test_portfolio.py`

Deletion requires search confirmation, test confirmation, and a dedicated cleanup sprint.

## Do-not-touch-yet files

- `config/settings.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/data_sources/prefill_portfolio_metadata.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/ops/capture_historical_evidence.py`
- `scripts/portfolio/parse_trade_commands.py`
- `scripts/portfolio/portfolio_manager.py`
- `scripts/reporting/reporter.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/watchlist/auto_watchlist_from_scan.py`
- `scripts/watchlist/build_watchlist.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/watchlist/parse_watchlist_commands.py`
- `scripts/watchlist/update_watchlist_actions.py`

These files touch sensitive boundaries, have unclear ownership, or may mutate runtime artifacts. They should not be moved, deleted, or migrated until their boundary is reviewed.

## Canonical v2 architecture proposal

This proposal defines desired ownership only. It does not create, rename, or move files.

| Responsibility | Proposed canonical owner |
|---|---|
| Application entrypoint | No canonical owner identified yet. BL28 should define one. |
| Runner/orchestrator | `src/market_scanner/orchestration/pipeline_core.py` for synthetic scaffold only; no production owner identified yet. |
| Scanner/universe selection | No canonical owner identified yet. `src/market_scanner/discovery/` is the likely package namespace. |
| Provider/source access | `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`, `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`, and `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py` for controlled v2 fundamentals source behavior. |
| Fundamentals normalization/evidence | `src/market_scanner/fundamentals/`, especially provider adapter, normalization contracts, readiness records, and persistence boundary. |
| Fundamental analysis | No canonical owner identified yet. Current owner is `scripts/fundamentals/build_analysis.py`; target ownership should be defined before migration. |
| Decision/review boundary | `scripts/core/decision_engine.py` remains the current allocation authority until a dedicated migration is approved; `src/market_scanner/decisions/` remains review scaffold only. |
| Portfolio state modelling | `src/market_scanner/portfolio/` for contracts only; no runtime owner identified yet. |
| Timing state tracking | No canonical owner identified yet. `src/market_scanner/timing/` is the likely package namespace. |
| Message composition | `src/market_scanner/reporting/reporting_input_adapter.py`, `src/market_scanner/reporting/reporting_input_contracts.py`, and `src/market_scanner/reporting/telegram_renderer.py` for pure in-memory composition. |
| Report generation | `src/market_scanner/reporting/reporting_engine.py` for v2 communication scaffold only; no artifact-writing owner approved yet. |
| Delivery/Telegram | No canonical owner identified yet. Delivery should remain isolated from renderer and analysis. |
| Configuration | No canonical owner identified yet. BL28 should define path/config ownership before runtime migration. |
| Shared utilities | `src/market_scanner/shared/` for v2 records and contracts. |

## Recommended cleanup sequence

1. Define canonical v2 runtime architecture before moving or deleting files.
2. Decide official ownership for application entrypoint, scanner/universe selection, analysis, reporting, delivery, configuration, and portfolio/timing runtime behavior.
3. Freeze `scripts/run_scan.py` as a legacy dependency and stop adding responsibilities to it.
4. Separate message rendering, report artifact writing, and Telegram delivery into explicitly governed boundaries.
5. Migrate fundamentals analysis logic from `scripts/fundamentals/` only after the canonical analysis owner is approved.
6. Preserve `scripts/core/decision_engine.py` as allocation authority until a dedicated Decision Engine migration is designed and approved.
7. Remove compatibility wrappers only after import references and tests are updated.
8. Archive or delete confirmed obsolete root scripts and legacy watchlist/Telegram files in a separate cleanup sprint.

## Risks

- `scripts/run_scan.py` couples scanner execution, production data writes, fundamentals, Decision Engine, reporting, and Telegram delivery.
- Legacy Telegram delivery and command-processing files import `requests`, load credentials, and can write offsets or send messages.
- Legacy portfolio/watchlist files may mutate source-of-truth artifacts or encode action semantics outside a clearly governed boundary.
- The current real-analysis work depends on `scripts/fundamentals/*`, which is useful but not a clean long-term v2 owner.
- `scripts/core/decision_engine.py` is still the certified allocation authority and must not be casually moved, rewritten, or replaced.
- Some legacy files contain developer-facing non-English comments, which should be addressed in a dedicated cleanup only if the file is retained or migrated.

## Guardrails confirmation

- No code changed.
- No tests changed.
- No runtime behavior changed.
- No files moved or deleted.
- No provider calls made.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No portfolio/watchlist updates.
- No Decision Engine investment behavior changed.
- No BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior added.

## Conclusion

The Python codebase has a clear v2 package scaffold under `src/market_scanner/`, but current executable behavior still depends heavily on script-era runtime files. The highest-risk coupling is `scripts/run_scan.py`, which composes scanner execution, data writes, fundamentals, portfolio, Decision Engine, reporting, and Telegram delivery. The fundamentals real-analysis path has advanced, but its analysis and metrics owners still live in `scripts/fundamentals/`.

Before adding more real-analysis features, the project should define the canonical v2 runtime architecture and migration boundaries. Cleanup implementation should not start until the official owners for entrypoint, scanner, analysis, reporting, delivery, configuration, portfolio, timing, and Decision Engine migration policy are documented.

## Next recommended step

RESET-10L-BL28 — Define Canonical V2 Runtime Architecture.

BL28 should remain design/review-only unless separately approved. It should define the single official app entrypoint, scanner flow, analysis flow, message composition flow, report artifact boundary, delivery boundary, configuration owner, and Decision Engine migration boundary before any Python files are moved, deleted, or rewritten.
