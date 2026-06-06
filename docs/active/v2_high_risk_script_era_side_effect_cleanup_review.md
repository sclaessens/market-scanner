# High-Risk Script-Era Side-Effect Cleanup Review

## Status

Completed by RESET-10L-BL44.

## Reset stage

RESET-10L-BL44 - High-Risk Script-Era Side-Effect Cleanup Review.

## Purpose

Identify, classify, and prioritize the highest-risk remaining script-era Python files under `scripts/` based on side-effect behavior.

This review is static only. It does not migrate, archive, delete, refactor, execute, or modify script-era files. It answers which remaining scripts are most dangerous to leave active, which side effects are present, which canonical v2 boundary should eventually own useful logic, and which cleanup sprint should happen next.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_archived_script_execution_test_cleanup.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/v2_legacy_runtime_archive_validation_and_entrypoint_certification.md`
- Canonical boundary migration records from RESET-10L-BL29 through RESET-10L-BL35.
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Inspection method

Static inspection only. No script-era files and no archived scripts were executed.

Commands and inspection patterns used:

```bash
find scripts -name "*.py" \
  -not -path "*/__pycache__/*" \
  | sort

grep -R "if __name__ == .__main__." -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "scripts/" -n src tests .github \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true

grep -R "import scripts\.\|from scripts\." -n src tests .github scripts \
  --include="*.py" \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true

grep -R "open(.*w\|open(.*a\|to_csv\|to_json\|write\|writerow\|mkdir\|unlink\|remove\|rename\|replace" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "yfinance\|yf\.\|requests\|http\|urlopen\|download\|SEC\|EDGAR\|sec\.gov\|alpha\|provider\|api" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "os.environ\|getenv\|TOKEN\|API_KEY\|SECRET\|PASSWORD\|credential\|chat_id\|BOT" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "telegram\|Telegram\|send_message\|sendDocument\|bot\|notify\|notification\|webhook\|chat_id" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "report\|summary\|artifact\|daily\|telegram_message\|markdown\|html\|pdf\|xlsx\|write_report" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "portfolio\|watchlist\|position\|transaction\|allocation\|rebalance\|buy\|sell\|cash\|holdings" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "Decision\|decision\|final_decision\|BUY\|SELL\|HOLD\|conviction\|urgency\|score\|scoring\|target_price\|target price\|tradeability\|recommendation\|allocation" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "subprocess\|os.system\|Popen\|check_call\|check_output\|shell=True" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true
```

Representative high-risk files were also read directly to distinguish static text, unit-test imports, manual entrypoints, and actual side-effect code paths.

## Inventory baseline

BL44 re-ran the inventory rather than assuming BL42 results.

Remaining committed Python files under `scripts/`: 52.

Runnable script-era entrypoints with `if __name__ == "__main__"`: 43.

Directory distribution:

| directory | Python files |
|---|---:|
| `scripts/` | 2 |
| `scripts/core/` | 20 |
| `scripts/data_sources/` | 3 |
| `scripts/diagnostics/` | 1 |
| `scripts/fundamentals/` | 9 |
| `scripts/ops/` | 1 |
| `scripts/portfolio/` | 5 |
| `scripts/reporting/` | 4 |
| `scripts/telegram/` | 1 |
| `scripts/utils/` | 1 |
| `scripts/watchlist/` | 5 |

The archived runtime scripts remain absent from active `scripts/`:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

## High-risk summary

The highest-risk remaining script-era zones are:

- Telegram delivery and command processing: credential reads, Telegram API calls, network calls, offset writes, and portfolio command coupling.
- Decision Engine and final decision artifacts: final action, allocation, execution, and decision log outputs.
- Portfolio and watchlist mutation surfaces: transaction writes, positions writes, watchlist transactions, active watchlist files, and status artifacts.
- Provider/network access: yfinance access, SEC/EDGAR bulk download support, and provider-assisted prefill paths.
- Report artifact generation: `data/processed` report outputs, `data/logs` report logs, and `reports/daily/telegram_message.txt`.
- Script-era layer builders: many write processed/log CSV artifacts and still have active test imports.

No subprocess, `os.system`, `Popen`, `check_call`, `check_output`, or `shell=True` usage was found under `scripts/`.

## Side-effect type summary

| side-effect type | files or groups found |
|---|---|
| data writes | layer builders under `scripts/core/`, fundamentals builders, SEC utilities, data-source prefill helpers, portfolio/watchlist scripts, reporting scripts, diagnostics, ops capture, utility writers |
| provider/live market access | `scripts/core/data_fetcher.py`, `scripts/core/scanner.py`, `scripts/core/build_context_backfill.py`, `scripts/core/build_entry_quality_backfill.py`, `scripts/validate_scans.py` |
| SEC/EDGAR/network access | `scripts/fundamentals/sec_companyfacts_bulk_intake.py` |
| credential/env reads | `scripts/reporting/send_telegram.py`, `scripts/telegram/process_telegram_commands.py` |
| Telegram delivery/API calls | `scripts/reporting/send_telegram.py`, `scripts/telegram/process_telegram_commands.py` |
| report writes | `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `scripts/analyze_validation.py`, `scripts/core/analyze_validation.py` |
| portfolio mutation | `scripts/portfolio/portfolio_manager.py`, `scripts/portfolio/parse_trade_commands.py`, `scripts/portfolio/build_portfolio.py`, `scripts/portfolio/evaluate_positions.py` |
| watchlist mutation | `scripts/watchlist/parse_watchlist_commands.py`, `scripts/watchlist/auto_watchlist_from_scan.py`, `scripts/watchlist/build_watchlist.py`, `scripts/watchlist/update_watchlist_actions.py`, `scripts/watchlist/evaluate_watchlist.py` |
| Decision Engine/final semantics | `scripts/core/decision_engine.py`, `scripts/core/build_stability_layer.py`, `scripts/reporting/build_reporting_layer.py`, `scripts/ops/capture_historical_evidence.py` |
| active test imports | 27 active test files import script-era modules |

## Active reference summary

Active source references are static canonical-boundary metadata under `src/market_scanner/*_boundary.py`. They identify script-era files as legacy authority evidence and do not execute them.

Active tests still import script-era modules from:

- `scripts.core`
- `scripts.fundamentals`
- `scripts.reporting`
- `scripts.data_sources`
- `scripts.diagnostics`
- `scripts.ops`
- `scripts.portfolio`

No active workflow references to script-era files were found.

No test subprocess/runpy execution of script-era files was found. However, active tests import and execute functions from high-risk script-era modules, often with temporary or monkeypatched paths. Those test execution paths should be decoupled before the underlying script-era files are migrated, archived, or deleted.

## Entrypoint summary

43 of 52 remaining script-era Python files define runnable entrypoints.

Entrypoints remain in high-risk groups:

- `scripts/core/decision_engine.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/core/build_stability_layer.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/portfolio/*`
- `scripts/watchlist/*`
- `scripts/data_sources/prefill_*.py`
- layer builders and diagnostics/ops scripts

## Canonical owner mapping summary

| likely canonical owner | high-risk script-era responsibility |
|---|---|
| `src/market_scanner/scanner/` | scanner logic, yfinance OHLCV/sector access, validation scan outputs, scanner scoring, entry-quality backfill |
| `src/market_scanner/fundamentals/` | provider/source access, SEC intake/cache/transform/index, metrics, quality, governed evidence prefill |
| `src/market_scanner/analysis/` | context layer, timing state, stability-adjacent review metadata, fundamental analysis, diagnostics |
| `src/market_scanner/decision/` | final Decision Engine semantics, review/final decision separation, stability and portfolio intelligence review linkage |
| `src/market_scanner/messaging/` | Telegram summary text and human-readable report/message composition |
| `src/market_scanner/reporting/` | dashboard/log/report artifact planning and writing policy |
| `src/market_scanner/delivery/` | Telegram send and command-processing delivery behavior |
| `src/market_scanner/config/` | governed paths, thresholds, source artifact path rules, env/credential policy surfaces |
| `src/market_scanner/utils/` | generic text/CSV helpers, low-risk maintenance helpers after policy review |
| no canonical owner identified yet | portfolio transaction mutation and watchlist command surfaces need explicit ownership policy before migration |

Mixed-responsibility files are common. In particular, `scripts/telegram/process_telegram_commands.py`, `scripts/reporting/build_reporting_layer.py`, `scripts/core/build_portfolio_intelligence.py`, `scripts/core/build_stability_layer.py`, and watchlist command files span multiple canonical boundaries.

## High-risk per-file table

| file path | primary risk status | secondary tags | side-effect types | active references | entrypoint | likely canonical owner | recommended next action | priority |
|---|---|---|---|---|---:|---|---|---|
| `scripts/core/decision_engine.py` | P0_DO_NOT_TOUCH_YET | decision_engine, final_decision, investment_semantics, data_write, active_test_reference | final decisions CSV/log writes, allocation/execution semantics | tests and canonical metadata | yes | `src/market_scanner/decision/` | governance review last; do not migrate casually | P0 |
| `scripts/reporting/send_telegram.py` | P0_DO_NOT_TOUCH_YET | telegram_delivery, credential_access, network_call, report_read, active_source_reference | env reads, Telegram API POST, reads `reports/daily/telegram_message.txt` | canonical metadata | yes | `src/market_scanner/delivery/` | isolate delivery tests/policy before any migration | P0 |
| `scripts/telegram/process_telegram_commands.py` | P0_DO_NOT_TOUCH_YET | telegram_delivery, credential_access, network_call, portfolio_mutation, subprocess_absent, mixed_responsibility | Telegram get/post, env reads, offset writes, trade command processing | canonical metadata | yes | `src/market_scanner/delivery/`, no canonical owner identified yet for portfolio commands | dedicated delivery/command governance review | P0 |
| `scripts/portfolio/portfolio_manager.py` | P0_DO_NOT_TOUCH_YET | portfolio_mutation, investment_semantics | transaction writes, position writes, BUY/SELL handling | imported by `scripts/portfolio/parse_trade_commands.py` and `scripts/portfolio/test_portfolio.py` | yes | no canonical owner identified yet | define portfolio mutation authority before migration | P0 |
| `scripts/portfolio/parse_trade_commands.py` | P0_DO_NOT_TOUCH_YET | portfolio_mutation, investment_semantics, active_script_reference | parses BUY/SELL commands and logs trades | imported by Telegram command script | yes | no canonical owner identified yet | isolate from Telegram command processing before migration | P0 |
| `scripts/watchlist/parse_watchlist_commands.py` | P0_DO_NOT_TOUCH_YET | watchlist_mutation, portfolio_reference, manual_invocation_risk | appends watchlist transactions, reads portfolio/watchlist files | no active test import found | yes | no canonical owner identified yet | policy review for watchlist command authority | P0 |
| `scripts/core/data_fetcher.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | scanner, provider_call, network_call, active_source_reference | yfinance downloads/history calls | canonical metadata and script imports | no | `src/market_scanner/scanner/` | migrate provider-free contract first, then isolate live access | P1 |
| `scripts/core/scanner.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | scanner, provider_call, scoring, investment_semantics, active_source_reference | yfinance sector lookup, scoring/target fields | canonical metadata | no | `src/market_scanner/scanner/` | scanner migration review before archive | P1 |
| `scripts/core/build_context_backfill.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | analysis, provider_call, data_write, active_test_reference | yfinance downloads, processed/log writes | active tests | yes | `src/market_scanner/analysis/` | decouple tests, then migrate review-safe logic | P1 |
| `scripts/core/build_entry_quality_backfill.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | scanner, provider_call, data_write, active_test_reference | optional yfinance downloads, processed/log writes | active tests | yes | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | decouple tests, review scoring semantics | P1 |
| `scripts/validate_scans.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | scanner, provider_call, data_write | yfinance validation and CSV writes | no active test import found | yes | `src/market_scanner/scanner/` | migrate/retire after scanner validation owner exists | P1 |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | fundamentals, sec_edgar, network_call, data_write, active_test_reference | SEC download, cache writes, manifest writes | active tests | yes | `src/market_scanner/fundamentals/` | remove high-risk test execution first, then migrate SEC intake | P1 |
| `scripts/reporting/build_reporting_layer.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | reporting, messaging, report_write, final_decision, active_test_reference, mixed_responsibility | report dashboard/log writes, Telegram message file write | active tests and canonical metadata | yes | `src/market_scanner/reporting/`, `src/market_scanner/messaging/` | split report artifact planning from writes | P1 |
| `scripts/reporting/build_telegram_summary.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | messaging, reporting, report_write, telegram_artifact, active_test_reference | writes Telegram summary artifact through reporting layer | active tests and canonical metadata | yes | `src/market_scanner/messaging/`, `src/market_scanner/reporting/` | decouple from artifact writes before migration | P1 |
| `scripts/portfolio/build_portfolio.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | portfolio_mutation, data_write, active_test_reference | positions CSV writes from transactions | active tests | yes | no canonical owner identified yet | portfolio source/mutation policy review | P1 |
| `scripts/portfolio/evaluate_positions.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | portfolio, data_write | portfolio review CSV writes | no active test import found | yes | no canonical owner identified yet | portfolio review boundary decision | P1 |
| `scripts/watchlist/auto_watchlist_from_scan.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | watchlist_mutation, scanner, data_write | auto-add watchlist transactions from scan log | no active test import found | yes | `src/market_scanner/scanner/`, no canonical owner identified yet | do not archive until watchlist owner exists | P1 |
| `scripts/watchlist/build_watchlist.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | watchlist_mutation, data_write | active watchlist CSV writes | no active test import found | yes | no canonical owner identified yet | watchlist mutation policy review | P1 |
| `scripts/watchlist/update_watchlist_actions.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | watchlist_mutation, data_write | appends watchlist transaction rows | no active test import found | yes | no canonical owner identified yet | watchlist command/state review | P1 |
| `scripts/watchlist/evaluate_watchlist.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | watchlist_mutation, data_write, configuration | status CSV writes, threshold/config reads | no active test import found | yes | no canonical owner identified yet, `src/market_scanner/config/` | watchlist timing-state owner decision | P1 |
| `scripts/data_sources/common.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | fundamentals, configuration, data_write, active_test_reference | atomic governed CSV writes | active tests | no | `src/market_scanner/fundamentals/`, `src/market_scanner/config/` | migrate write policy into canonical persistence/source boundary | P1 |
| `scripts/data_sources/prefill_fundamentals.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | fundamentals, provider_call, data_write, active_test_reference | optional governed fundamentals artifact write | active tests | yes | `src/market_scanner/fundamentals/` | decouple tests and migrate source artifact policy | P1 |
| `scripts/data_sources/prefill_portfolio_metadata.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | portfolio, provider_call, data_write, active_test_reference | optional portfolio metadata write | active tests | yes | no canonical owner identified yet, `src/market_scanner/config/` | portfolio metadata owner decision | P1 |
| `scripts/ops/capture_historical_evidence.py` | P1_CRITICAL_SIDE_EFFECT_REVIEW_REQUIRED | ops, report_write, final_decision, active_test_reference | history CSV appends from decision/report artifacts | active tests | yes | `src/market_scanner/reporting/`, `src/market_scanner/decision/`, `src/market_scanner/utils/` | decide maintenance namespace and decouple tests | P1 |
| `scripts/core/build_portfolio_intelligence.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | portfolio, analysis, decision, data_write, active_test_reference, mixed_responsibility | processed/log writes, portfolio metadata validation | active tests | yes | `src/market_scanner/decision/`, no canonical owner identified yet for portfolio | migrate review-safe classification after portfolio policy | P1 |
| `scripts/core/build_stability_layer.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | decision, analysis, data_write, active_test_reference | stability CSV/log writes from final decisions | active tests | yes | `src/market_scanner/decision/`, `src/market_scanner/analysis/` | migrate only after decision/review governance | P1 |
| `scripts/core/build_validation_layer.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | scanner, analysis, data_write, active_test_reference | validation/entry-quality CSV/log writes | active tests | yes | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | migrate validation classification to canonical owner | P1 |
| `scripts/core/build_context_layer.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | analysis, data_write, active_test_reference | context CSV/log writes | active tests | yes | `src/market_scanner/analysis/` | migrate context classification | P1 |
| `scripts/core/build_timing_state_layer.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | analysis, watchlist, data_write, active_test_reference | timing-state CSV/log writes | active tests | yes | `src/market_scanner/analysis/`, no canonical owner identified yet for watchlist timing | migrate after watchlist owner decision | P1 |
| `scripts/fundamentals/build_analysis.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, analysis, data_write, active_test_reference | optional analysis CSV write; forbidden semantic column checks | active tests and canonical metadata | yes | `src/market_scanner/analysis/`, `src/market_scanner/fundamentals/` | migrate governed analysis into canonical boundary | P1 |
| `scripts/fundamentals/build_metrics.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, data_write, active_test_reference | optional metrics CSV write | active tests and canonical metadata | yes | `src/market_scanner/fundamentals/` | migrate metrics evidence logic | P1 |
| `scripts/fundamentals/build_quality.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, analysis, data_write, active_test_reference | quality/log writes by default | active tests and canonical metadata | yes | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | migrate quality/readiness logic | P1 |
| `scripts/fundamentals/build_history_intake.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, report_write, active_test_reference | optional JSON validation report write | active tests | yes | `src/market_scanner/fundamentals/` | migrate validation/report separation | P1 |
| `scripts/fundamentals/sec_companyfacts_transform.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, sec_edgar, data_write, active_test_reference | local transform CSV writes | active tests | yes | `src/market_scanner/fundamentals/` | migrate SEC transform into canonical fundamentals | P1 |
| `scripts/fundamentals/sec_ticker_cik_index.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, sec_edgar, data_write, active_test_reference | coverage CSV writes | active tests | yes | `src/market_scanner/fundamentals/` | migrate index/coverage mapping | P1 |
| `scripts/fundamentals/run_sec_transformation_review.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | fundamentals, sec_edgar, data_write, active_test_reference | review CSV writes from local files | active tests | yes | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | migrate review harness or retire | P1 |
| `scripts/diagnostics/audit_data_coverage.py` | P1_CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE | diagnostics, portfolio, watchlist, active_test_reference | reads portfolio/watchlist/fundamentals artifacts, prints audit | active tests | yes | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/`, no canonical owner identified yet | define diagnostics/maintenance owner | P1 |
| `scripts/analyze_validation.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | analysis, report_write, duplicate_canonical_boundary | Excel summary write | no active test import found | yes | `src/market_scanner/analysis/`, `src/market_scanner/reporting/` | confirm duplicate with core validation summary | P2 |
| `scripts/core/analyze_validation.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | analysis, data_write | validation summary CSV write | no active test import found | yes | `src/market_scanner/analysis/`, `src/market_scanner/reporting/` | migrate or archive after confirmation | P2 |
| `scripts/core/log_scans.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | scanner, data_write | scan log appends | no active test import found | yes | `src/market_scanner/scanner/`, `src/market_scanner/reporting/` | archive after scanner logging policy exists | P2 |
| `scripts/core/validate_scans.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | scanner, validation, data_write | validation results CSV write | no active test import found | yes | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | reconcile with `scripts/validate_scans.py` | P2 |
| `scripts/core/validator.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | scanner, utility, data_write | ticker/log bootstrap writes | no active test import found | no | `src/market_scanner/scanner/`, `src/market_scanner/config/` | migrate path bootstrap policy or archive | P2 |
| `scripts/core/indicators.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | scanner, utility | indicator calculations | no active test import found | no | `src/market_scanner/scanner/`, `src/market_scanner/utils/` | migrate only after scanner review | P2 |
| `scripts/core/build_fundamental_analysis.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | wrapper, fundamentals, active_test_reference | delegates to script-era analysis | active tests | yes | `src/market_scanner/analysis/` | remove after tests target canonical owner | P2 |
| `scripts/core/build_fundamental_layer.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | wrapper, fundamentals, active_test_reference | delegates to script-era quality | active tests | yes | `src/market_scanner/fundamentals/` | remove after tests target canonical owner | P2 |
| `scripts/core/build_fundamental_metrics.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | wrapper, fundamentals, active_test_reference | delegates to script-era metrics | active tests | yes | `src/market_scanner/fundamentals/` | remove after tests target canonical owner | P2 |
| `scripts/core/build_fundamentals_history_intake.py` | P2_DUPLICATE_RUNTIME_RESPONSIBILITY | wrapper, fundamentals, active_test_reference | delegates to script-era history intake | active tests | yes | `src/market_scanner/fundamentals/` | remove after tests target canonical owner | P2 |

## Group-by-directory findings

### `scripts/core/`

Highest-risk files:

- `scripts/core/decision_engine.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`
- `scripts/core/build_context_backfill.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/core/build_stability_layer.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_timing_state_layer.py`

Side-effect types found: yfinance access, data writes, processed/log artifact writes, final decision semantics, allocation/execution outputs, scanner scoring, context/timing outputs, and portfolio intelligence outputs.

Active references: many active tests import these modules; canonical boundary metadata statically names scanner and decision legacy authorities.

Likely canonical owners: scanner, analysis, decision, fundamentals, config.

Recommended next action: remove or isolate high-risk test execution paths before migrating logic into canonical owners.

### `scripts/fundamentals/`

Highest-risk files:

- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_history_intake.py`

Side-effect types found: SEC/EDGAR URL support, local download/cache writes, manifest writes, output CSV writes, optional report writes, and active test imports.

Active references: active fundamentals and core tests import these files heavily.

Likely canonical owner: `src/market_scanner/fundamentals/`, with `src/market_scanner/analysis/` for review outputs.

Recommended next action: after high-risk test execution is isolated, prioritize fundamentals/provider side-effect migration review.

### `scripts/portfolio/`

Highest-risk files:

- `scripts/portfolio/portfolio_manager.py`
- `scripts/portfolio/parse_trade_commands.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/portfolio/evaluate_positions.py`

Side-effect types found: portfolio transaction writes, position writes, review writes, BUY/SELL command handling, and position mutation.

Active references: portfolio tests and Telegram command script import portfolio modules.

Likely canonical owner: no canonical owner identified yet for portfolio mutation authority.

Recommended next action: policy decision before migration. Do not touch casually.

### `scripts/reporting/`

Highest-risk files:

- `scripts/reporting/send_telegram.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`

Side-effect types found: credential reads, Telegram API POST, report/dashboard/log writes, `reports/daily/telegram_message.txt` writes, final decision representation.

Active references: reporting tests and canonical metadata.

Likely canonical owners: reporting, messaging, delivery.

Recommended next action: separate report artifact writing, message composition, and Telegram delivery before migration.

### `scripts/telegram/`

Highest-risk file:

- `scripts/telegram/process_telegram_commands.py`

Side-effect types found: credential reads, Telegram getUpdates/sendMessage calls, offset writes, trade command processing, portfolio mutation coupling.

Active references: canonical delivery metadata.

Likely canonical owners: delivery plus an unresolved portfolio command owner.

Recommended next action: dedicated Telegram command/delivery governance review.

### `scripts/watchlist/`

Highest-risk files:

- `scripts/watchlist/parse_watchlist_commands.py`
- `scripts/watchlist/auto_watchlist_from_scan.py`
- `scripts/watchlist/build_watchlist.py`
- `scripts/watchlist/update_watchlist_actions.py`
- `scripts/watchlist/evaluate_watchlist.py`

Side-effect types found: watchlist transaction writes, active watchlist writes, status writes, portfolio checks, threshold/config reads.

Active references: no active test imports found in the static import search, but files remain runnable.

Likely canonical owner: no canonical owner identified yet; scanner and config may own inputs only.

Recommended next action: define watchlist mutation/timing-state ownership before migration.

## P0 do-not-touch files

- `scripts/core/decision_engine.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/portfolio/portfolio_manager.py`
- `scripts/portfolio/parse_trade_commands.py`
- `scripts/watchlist/parse_watchlist_commands.py`

These files contain the riskiest authority surfaces: final/allocation semantics, Telegram credential/network delivery, portfolio mutation, and command-driven watchlist/portfolio behavior.

## P1 critical side-effect review files

- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`
- `scripts/core/build_context_backfill.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/validate_scans.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/portfolio/evaluate_positions.py`
- `scripts/watchlist/auto_watchlist_from_scan.py`
- `scripts/watchlist/build_watchlist.py`
- `scripts/watchlist/update_watchlist_actions.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/data_sources/common.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/data_sources/prefill_portfolio_metadata.py`
- `scripts/ops/capture_historical_evidence.py`

These files perform or can trigger live/provider/network access, data writes, report writes, portfolio/watchlist mutation, or history artifact writes.

## P1 canonical migration required files

- `scripts/core/build_portfolio_intelligence.py`
- `scripts/core/build_stability_layer.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_timing_state_layer.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/diagnostics/audit_data_coverage.py`

These files contain useful domain logic and active test coverage, but they should not remain permanent script-era owners.

## P2 duplicate runtime responsibility files

- `scripts/analyze_validation.py`
- `scripts/core/analyze_validation.py`
- `scripts/core/log_scans.py`
- `scripts/core/validate_scans.py`
- `scripts/core/validator.py`
- `scripts/core/indicators.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`

These files duplicate canonical or script-era responsibilities and should be reviewed after P0/P1 dependency and migration risks are reduced.

## P3 low-risk archive candidates

- `scripts/core/regime.py`
- `scripts/fundamentals/__init__.py`
- `scripts/reporting/reporter.py`
- `scripts/utils/utils.py`

These files appear lower-risk than the P0/P1/P2 set, but should not be archived until active imports and operator usage are confirmed.

## P4 delete candidates

- `scripts/portfolio/test_portfolio.py`

This appears to be a script-era manual/test utility under `scripts/`, not the active pytest suite. It should be confirmed before deletion or archive.

## Recommended cleanup sequence

1. Remove or isolate active tests that import and execute functions from high-risk script-era modules, especially fundamentals/provider, reporting, portfolio, data-source, diagnostics, and Decision Engine modules.
2. Review and migrate fundamentals/provider side effects into the canonical fundamentals boundary, with provider/network access kept dependency-injected and dry-run safe.
3. Review and migrate reporting, message composition, and Telegram/delivery side effects into separate canonical reporting, messaging, and delivery boundaries.
4. Review portfolio and watchlist mutation surfaces and decide canonical ownership before moving any command or transaction behavior.
5. Review Decision Engine/final semantics last under strict governance, preserving the doctrine that Decision Engine is the only allocation authority.
6. Archive or delete P2/P3/P4 candidates only after dependencies are removed and canonical replacements or archive-only conclusions are validated.

## Recommended next sprint

RESET-10L-BL45 - Remove High-Risk Script-Era Test Execution.

Rationale: no direct subprocess/runpy execution of script-era files was found, but active tests still import and exercise many high-risk script-era modules. Before migrating or archiving those files, the project should decouple test execution from the old side-effect owners and move coverage toward canonical v2 boundaries or static legacy-policy assertions.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
- No script-era files executed.
- No archived scripts executed.
- No provider calls made.
- No production pipeline executed.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery.
- No network calls.
- No credentials read.
- No portfolio/watchlist updates.
- No Decision Engine behavior changed.
- No BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior added.

## Known limitations

This review uses static inspection and direct file reads only. It does not prove every reachable branch of every script-era file, and it does not replace future migration-specific tests.

Some active tests may execute script-era functions without invoking script files as subprocesses. BL44 documents this as the next cleanup concern rather than changing tests.

Portfolio and watchlist mutation ownership remains unresolved. Those files should not be migrated or archived until ownership policy is explicit.
