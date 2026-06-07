# Script-Era Python Cleanup Inventory

## Status

Completed by RESET-10L-BL42.

## Reset stage

RESET-10L-BL42 - Script-Era Python Cleanup Inventory.

## Purpose

Inventory and classify the remaining script-era Python files under `scripts/` after the primary legacy runtime scripts were archived and `src/market_scanner/app.py` was certified as the active runtime entrypoint.

This sprint is review-only. No Python files, tests, workflows, data files, report files, archived scripts, portfolio/watchlist files, or runtime behavior were changed.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/v2_legacy_runtime_archive_validation_and_entrypoint_certification.md`
- Canonical boundary migration records from RESET-10L-BL29 through RESET-10L-BL35.
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

Policy application:

- A legacy Python file that still has tests or useful logic is not approved for long-term retention.
- Script-era files with useful logic should be migrated into canonical v2 owners before archive or deletion.
- Script-era files with side effects, provider/network access, credential use, Telegram delivery, portfolio/watchlist mutation, or Decision Engine behavior require separate controlled review before migration or archive.

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

grep -R "scanner\|scan\|universe\|ticker\|candidate" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "analysis\|analy\|fundamental\|quality\|decision\|review\|portfolio_intelligence\|final_decision" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "report\|summary\|message\|telegram\|Telegram\|send\|delivery\|notify" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "portfolio\|watchlist\|position\|transaction" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "open(.*w\|to_csv\|write\|requests\|http\|post\|get\|os.environ\|TOKEN\|API\|SECRET\|credential\|yfinance\|yf\.\|alpha\|provider" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "BUY\|SELL\|HOLD\|allocation\|conviction\|urgency\|score\|scoring\|target_price\|target price\|tradeability\|recommendation" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true
```

## Inventory scope

Included:

- all committed `scripts/**/*.py` files;
- `scripts/*.py` files;
- Python files under `scripts/core/`, `scripts/fundamentals/`, `scripts/portfolio/`, `scripts/reporting/`, `scripts/telegram/`, `scripts/watchlist/`, `scripts/data_sources/`, `scripts/diagnostics/`, `scripts/ops/`, and `scripts/utils/`.

Excluded:

- `archive/`;
- `.venv/`;
- `venv/`;
- `__pycache__/`;
- non-Python files.

## Script-era Python file count

Remaining script-era Python files under `scripts/`: 52.

Runnable script-era entrypoints with `if __name__ == "__main__"`: 43.

The primary legacy runtime entrypoints from BL40 are no longer active under `scripts/`:

- `scripts/run_scan.py`: absent;
- `scripts/run_full_pipeline.py`: absent.

## Responsibility summary

| Responsibility | Representative files | Summary |
|---|---|---|
| runtime / orchestration | `scripts/core/*`, `scripts/analyze_validation.py`, `scripts/validate_scans.py` | Many old layer builders still define runnable entrypoints and write processed/log artifacts. |
| scanner / universe | `scripts/core/scanner.py`, `scripts/core/data_fetcher.py`, `scripts/core/indicators.py`, `scripts/core/regime.py` | Legacy scanner logic, yfinance sector access, scoring, target/entry fields, and universe support remain outside canonical scanner. |
| fundamentals / provider / evidence | `scripts/fundamentals/*`, `scripts/data_sources/*` | Active tests still import fundamentals and data-source scripts; some files write governed artifacts or perform SEC download support. |
| analysis | `scripts/fundamentals/build_analysis.py`, `scripts/core/build_fundamental_analysis.py` | Legacy analysis logic remains tested and should migrate into `src/market_scanner/analysis/` or fundamentals evidence owners. |
| decision / review | `scripts/core/decision_engine.py`, `scripts/core/build_portfolio_intelligence.py`, `scripts/core/build_stability_layer.py` | Decision Engine and review-adjacent layers remain high-risk because they include allocation/final action semantics. |
| portfolio | `scripts/portfolio/*` | Portfolio scripts write transactions, positions, and reviews and include BUY/SELL command handling. |
| watchlist | `scripts/watchlist/*` | Watchlist scripts write transactions/status/active artifacts and contain Dutch operator-facing strings that need language-governance cleanup before migration. |
| messaging / report formatting | `scripts/reporting/build_telegram_summary.py`, `scripts/reporting/reporter.py` | Message formatting duplicates canonical messaging/reporting boundaries. |
| report artifact generation | `scripts/reporting/build_reporting_layer.py` | Writes reporting dashboard/log artifacts and `reports/daily/telegram_message.txt`. |
| Telegram / delivery | `scripts/reporting/send_telegram.py`, `scripts/telegram/process_telegram_commands.py` | Reads env/credentials, performs Telegram API calls, and sends messages. |
| configuration / environment | `scripts/data_sources/common.py`, `scripts/watchlist/evaluate_watchlist.py` | Loads governed paths, thresholds, and output rules outside canonical config ownership. |
| utilities / maintenance | `scripts/diagnostics/audit_data_coverage.py`, `scripts/ops/capture_historical_evidence.py`, `scripts/utils/utils.py` | Maintenance and evidence utilities remain outside canonical utility/config ownership. |

## Entrypoint summary

Runnable entrypoints remain in 43 of 52 files. These include old validation, fundamentals, SEC intake/transform, reporting, Telegram, portfolio, watchlist, diagnostics, and layer-builder commands.

Files without `__main__` entrypoints are mostly support modules or wrappers:

- `scripts/core/data_fetcher.py`
- `scripts/core/indicators.py`
- `scripts/core/regime.py`
- `scripts/core/validator.py`
- `scripts/data_sources/common.py`
- `scripts/fundamentals/__init__.py`
- `scripts/reporting/reporter.py`
- `scripts/utils/utils.py`

## Active reference summary

Active test imports remain for many script-era files:

- `tests/core/*` imports context, validation, timing, stability, fundamentals, portfolio intelligence, Decision Engine, and backfill scripts.
- `tests/fundamentals/*` imports SEC transformation, ticker/CIK, bulk intake, history, metrics, quality, and analysis scripts.
- `tests/reporting/*` imports reporting layer and Telegram summary builders.
- `tests/portfolio/*` imports portfolio builders and portfolio intelligence.
- `tests/data_sources/*` imports governed prefill helpers.
- `tests/diagnostics/*` and `tests/ops/*` import maintenance utilities.

Canonical v2 boundary tests and metadata also statically reference some script-era paths as legacy authority evidence. Those references are not runtime execution, but they confirm these files remain migration targets.

No active workflow references to `scripts/` were found during BL42.

## Side-effect risk summary

Side-effect risks found by static inspection:

- data writes through `to_csv`, `write_text`, and file handles in validation, layer, fundamentals, reporting, portfolio, watchlist, data-source, diagnostics, and ops scripts;
- network/provider access through `yfinance` in `scripts/core/scanner.py` and SEC download support in `scripts/fundamentals/sec_companyfacts_bulk_intake.py`;
- credential/environment access and Telegram API calls in `scripts/reporting/send_telegram.py` and `scripts/telegram/process_telegram_commands.py`;
- report artifact creation in `scripts/reporting/build_reporting_layer.py` and `scripts/reporting/build_telegram_summary.py`;
- portfolio/watchlist mutation in `scripts/portfolio/*` and `scripts/watchlist/*`;
- Decision Engine/final decision semantics in `scripts/core/decision_engine.py`;
- allocation/final-action reporting representation in `scripts/reporting/build_reporting_layer.py`;
- investment command parsing and transaction mutation in `scripts/portfolio/parse_trade_commands.py` and Telegram command processing.

## Canonical owner mapping summary

| Canonical owner | Script-era logic to review |
|---|---|
| `src/market_scanner/scanner/` | scanner, data fetcher, indicators, regime, validation scanner support, watchlist auto-from-scan inputs. |
| `src/market_scanner/fundamentals/` | fundamentals history intake, metrics, quality, SEC CIK/index/transform/review, data-source prefill, provider artifact validation. |
| `src/market_scanner/analysis/` | fundamentals analysis, context/validation/timing/stability analysis surfaces, diagnostic coverage analysis. |
| `src/market_scanner/decision/` | legacy Decision Engine, portfolio intelligence, final decisions, review state boundaries. |
| `src/market_scanner/messaging/` | Telegram summary text builder and old report message formatting. |
| `src/market_scanner/reporting/` | reporting dashboard/log/artifact layer and reporter helper. |
| `src/market_scanner/delivery/` | Telegram send and Telegram command-processing delivery paths. |
| `src/market_scanner/config/` | thresholds, governed path rules, data-source path validation, settings coupling. |
| `src/market_scanner/utils/` | generic file utility helpers and maintenance evidence utilities after policy review. |
| no canonical owner identified yet | portfolio/watchlist command surfaces and operator maintenance utilities need a dedicated policy decision before migration. |

## Per-file classification table

| file path | primary status | secondary tags | active references | entrypoint | side-effect risk | likely canonical owner | recommended next action | priority bucket |
|---|---|---|---|---|---|---|---|---|
| `scripts/analyze_validation.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | validation, reporting, data_write, duplicate_responsibility | no active test import found | yes | writes validation summary workbook/CSV | `src/market_scanner/analysis/`, `src/market_scanner/reporting/` | confirm whether superseded by `scripts/core/analyze_validation.py`, then migrate or archive | P2 |
| `scripts/core/analyze_validation.py` | CANONICAL_MIGRATION_REQUIRED | validation, analysis, data_write, duplicate_responsibility | no active test import found | yes | writes validation summary CSV | `src/market_scanner/analysis/`, `src/market_scanner/reporting/` | migrate validation summary logic or archive after replacement | P2 |
| `scripts/core/build_context_backfill.py` | CANONICAL_MIGRATION_REQUIRED | analysis, utility, data_write, entrypoint | test import | yes | writes historical context/backfill outputs | `src/market_scanner/analysis/` | migrate tested context backfill behavior before archive | P2 |
| `scripts/core/build_context_layer.py` | CANONICAL_MIGRATION_REQUIRED | analysis, data_write, active_reference | test import | yes | writes context layer/log artifacts | `src/market_scanner/analysis/` | migrate context classification logic into canonical analysis owner | P1 |
| `scripts/core/build_entry_quality_backfill.py` | CANONICAL_MIGRATION_REQUIRED | scanner, analysis, data_write, active_reference | test import | yes | writes historical entry quality artifacts | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | migrate or retire entry-quality backfill logic with tests | P2 |
| `scripts/core/build_fundamental_analysis.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | fundamentals, analysis, compatibility_wrapper, active_reference | test import | yes | delegates to script-era analysis main | `src/market_scanner/analysis/` | remove wrapper after tests target canonical owner | P2 |
| `scripts/core/build_fundamental_layer.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | fundamentals, compatibility_wrapper, active_reference | test import | yes | delegates to script-era quality main | `src/market_scanner/fundamentals/` | remove wrapper after canonical fundamentals migration | P2 |
| `scripts/core/build_fundamental_metrics.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | fundamentals, compatibility_wrapper, active_reference | test import | yes | delegates to script-era metrics main | `src/market_scanner/fundamentals/` | remove wrapper after canonical metrics migration | P2 |
| `scripts/core/build_fundamentals_history_intake.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | fundamentals, compatibility_wrapper, active_reference | test import | yes | delegates to script-era history intake main | `src/market_scanner/fundamentals/` | remove wrapper after canonical history intake migration | P2 |
| `scripts/core/build_portfolio_intelligence.py` | CANONICAL_MIGRATION_REQUIRED | portfolio, decision_engine, analysis, data_write, active_reference, investment_semantics | test import | yes | writes portfolio intelligence/log and validates forbidden semantics | `src/market_scanner/decision/`, no canonical owner identified yet for portfolio policy | migrate review-safe portions and isolate portfolio authority | P1 |
| `scripts/core/build_stability_layer.py` | CANONICAL_MIGRATION_REQUIRED | decision_engine, analysis, data_write, active_reference, investment_semantics | test import | yes | writes stability state/log and consumes final decisions | `src/market_scanner/decision/`, `src/market_scanner/analysis/` | review Decision Engine coupling before migration | P1 |
| `scripts/core/build_timing_state_layer.py` | CANONICAL_MIGRATION_REQUIRED | scanner, analysis, watchlist, data_write, active_reference | test import | yes | writes timing-state artifacts | `src/market_scanner/analysis/`, no canonical owner identified yet for watchlist timing | migrate timing-state classification with watchlist policy review | P1 |
| `scripts/core/build_validation_layer.py` | CANONICAL_MIGRATION_REQUIRED | scanner, analysis, validation, data_write, active_reference | test import | yes | writes validation layer/log artifacts | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | migrate validation classification into canonical scanner/analysis boundary | P1 |
| `scripts/core/data_fetcher.py` | CANONICAL_MIGRATION_REQUIRED | scanner, provider_call, network_call, active_reference | canonical metadata | no | fetches market data through legacy data access | `src/market_scanner/scanner/` | design canonical market-data/source boundary before migration | P1 |
| `scripts/core/decision_engine.py` | CANONICAL_MIGRATION_REQUIRED | decision_engine, investment_semantics, data_write, active_reference | test import and canonical metadata | yes | writes final decisions/log and owns allocation/final action semantics | `src/market_scanner/decision/` | separate certified Decision Engine authority from review boundary before migration | P0 |
| `scripts/core/indicators.py` | CANONICAL_MIGRATION_REQUIRED | scanner, utility, duplicate_responsibility | no active test import found | no | low direct write risk; computes indicators | `src/market_scanner/scanner/`, `src/market_scanner/utils/` | migrate indicator calculations only after scanner architecture review | P2 |
| `scripts/core/log_scans.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | scanner, data_write, utility | no active test import found | yes | writes scan log artifacts | `src/market_scanner/scanner/`, `src/market_scanner/reporting/` | archive after scan logging policy is replaced | P2 |
| `scripts/core/regime.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | scanner, utility | no active test import found | no | low direct write risk | `src/market_scanner/scanner/` | migrate simple regime classification or archive if unused | P3 |
| `scripts/core/scanner.py` | CANONICAL_MIGRATION_REQUIRED | scanner, provider_call, network_call, scoring, investment_semantics, active_reference | canonical metadata | no | yfinance sector access and scanner scoring/target fields | `src/market_scanner/scanner/` | high-risk scanner migration review before archive | P1 |
| `scripts/core/validate_scans.py` | CANONICAL_MIGRATION_REQUIRED | scanner, validation, data_write, duplicate_responsibility | no active test import found | yes | writes validation outputs | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | reconcile with `scripts/validate_scans.py` before migration | P2 |
| `scripts/core/validator.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | scanner, utility, data_write | no active test import found | no | creates ticker/log files | `src/market_scanner/scanner/`, `src/market_scanner/config/` | migrate file bootstrap rules or archive if obsolete | P2 |
| `scripts/data_sources/common.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider_call, configuration, data_write, active_reference | test import | no | governed artifact write helpers | `src/market_scanner/fundamentals/`, `src/market_scanner/config/` | migrate governed prefill helpers into canonical source/persistence boundary | P1 |
| `scripts/data_sources/prefill_fundamentals.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider_call, data_write, active_reference | test import | yes | optional governed write to `data/raw/fundamentals.csv` | `src/market_scanner/fundamentals/` | migrate or retire provider-assisted prefill path | P1 |
| `scripts/data_sources/prefill_portfolio_metadata.py` | CANONICAL_MIGRATION_REQUIRED | portfolio, provider_call, data_write, active_reference | test import | yes | optional governed write to portfolio metadata | no canonical owner identified yet | define portfolio metadata owner before migration | P1 |
| `scripts/diagnostics/audit_data_coverage.py` | HISTORICAL_OR_OPERATOR_UTILITY_REVIEW_REQUIRED | diagnostics, fundamentals, portfolio, watchlist, active_reference | test import | yes | reads data artifacts; reports audit summary | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/`, no canonical owner identified yet | decide whether diagnostics get a canonical maintenance namespace | P2 |
| `scripts/fundamentals/__init__.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | fundamentals, package_marker, active_reference | test package import | no | none apparent | `src/market_scanner/fundamentals/` | remove only after script-era fundamentals imports are migrated | P3 |
| `scripts/fundamentals/build_analysis.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, analysis, data_write, active_reference, duplicate_responsibility | test import and canonical metadata | yes | optional write to fundamental analysis output | `src/market_scanner/analysis/`, `src/market_scanner/fundamentals/` | migrate governed analysis logic into canonical analysis/evidence owners | P1 |
| `scripts/fundamentals/build_history_intake.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider, data_write, active_reference | test import | yes | optional report write | `src/market_scanner/fundamentals/` | migrate history intake validation into canonical fundamentals | P1 |
| `scripts/fundamentals/build_metrics.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, data_write, active_reference | test import and canonical metadata | yes | optional metrics CSV write | `src/market_scanner/fundamentals/` | migrate metric derivation into canonical fundamentals evidence | P1 |
| `scripts/fundamentals/build_quality.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, analysis, data_write, active_reference | test import and canonical metadata | yes | writes quality/log artifacts by default | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | migrate quality/readiness logic into canonical evidence boundary | P1 |
| `scripts/fundamentals/run_sec_transformation_review.py` | HISTORICAL_OR_OPERATOR_UTILITY_REVIEW_REQUIRED | fundamentals, provider, data_write, active_reference | test import | yes | optional local review CSV write | `src/market_scanner/fundamentals/` | review as controlled SEC maintenance utility before migration/archive | P2 |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider_call, network_call, data_write, active_reference | test import | yes | SEC download and manifest writes | `src/market_scanner/fundamentals/` | migrate controlled provider intake into canonical provider/source boundary | P1 |
| `scripts/fundamentals/sec_companyfacts_transform.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider, data_write, active_reference | test import | yes | local Company Facts transform and optional CSV write | `src/market_scanner/fundamentals/` | migrate transform logic into canonical normalization boundary | P1 |
| `scripts/fundamentals/sec_ticker_cik_index.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, provider, data_write, active_reference | test import | yes | ticker/CIK index and optional coverage writes | `src/market_scanner/fundamentals/` | migrate CIK mapping into canonical source metadata boundary | P1 |
| `scripts/ops/capture_historical_evidence.py` | HISTORICAL_OR_OPERATOR_UTILITY_REVIEW_REQUIRED | utility, reporting, decision_engine, data_write, active_reference | test import | yes | appends historical evidence CSVs | `src/market_scanner/utils/`, `src/market_scanner/reporting/`, `src/market_scanner/decision/` | decide maintenance namespace and evidence-retention policy | P1 |
| `scripts/portfolio/build_portfolio.py` | CANONICAL_MIGRATION_REQUIRED | portfolio, data_write, active_reference | test import | yes | writes portfolio positions | no canonical owner identified yet | define canonical portfolio state owner before migration | P1 |
| `scripts/portfolio/evaluate_positions.py` | CANONICAL_MIGRATION_REQUIRED | portfolio, data_write, analysis | no active test import found | yes | writes portfolio review | no canonical owner identified yet | review portfolio classification boundary before migration | P1 |
| `scripts/portfolio/parse_trade_commands.py` | DO_NOT_TOUCH_YET | portfolio, investment_semantics, data_write, operator_utility | no active test import found; imported by Telegram script | yes | logs BUY/SELL transactions | no canonical owner identified yet | leave untouched until portfolio command policy exists | P0 |
| `scripts/portfolio/portfolio_manager.py` | DO_NOT_TOUCH_YET | portfolio, investment_semantics, data_write, operator_utility | imported by portfolio command/test script | yes | writes transactions and positions | no canonical owner identified yet | leave untouched until portfolio command policy exists | P0 |
| `scripts/portfolio/test_portfolio.py` | DELETE_CANDIDATE_AFTER_CONFIRMATION | portfolio, test_like_script, data_write | no active test import found | yes | executes temporary portfolio test flow under temp dir | delete candidate | confirm no operator need, then delete/archive through controlled sprint | P4 |
| `scripts/reporting/build_reporting_layer.py` | CANONICAL_MIGRATION_REQUIRED | reporting, messaging, data_write, investment_semantics, active_reference | test import and canonical metadata | yes | writes dashboard/log and Telegram message artifact | `src/market_scanner/reporting/`, `src/market_scanner/messaging/` | migrate reporting representation without delivery or allocation authority | P1 |
| `scripts/reporting/build_telegram_summary.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | messaging, reporting, telegram, data_write, active_reference | test import and canonical metadata | yes | writes Telegram summary through legacy reporting layer | `src/market_scanner/messaging/`, `src/market_scanner/reporting/` | remove after canonical message/report tests own coverage | P1 |
| `scripts/reporting/reporter.py` | ARCHIVE_CANDIDATE_NOW | reporting, messaging, duplicate_responsibility | no active reference found | no | low direct risk; formats old legacy markdown report | archive only | propose low-risk archive review after BL43 | P3 |
| `scripts/reporting/send_telegram.py` | CANONICAL_MIGRATION_REQUIRED | telegram, delivery, credential_access, network_call, active_reference | canonical metadata and static tests | yes | reads `.env`/env and sends Telegram API requests | `src/market_scanner/delivery/` | isolate credentials/network behavior before archive | P1 |
| `scripts/telegram/process_telegram_commands.py` | DO_NOT_TOUCH_YET | telegram, delivery, portfolio, credential_access, network_call, investment_semantics | canonical metadata and static tests | yes | reads Telegram updates, sends messages, calls trade parser | `src/market_scanner/delivery/`, no canonical owner identified yet for portfolio commands | leave untouched until delivery and portfolio command policies are separated | P0 |
| `scripts/utils/utils.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | utility, data_write | no active reference found | no | generic text/CSV writes | `src/market_scanner/utils/` | migrate only if still needed; otherwise archive candidate | P3 |
| `scripts/validate_scans.py` | ARCHIVE_CANDIDATE_AFTER_MIGRATION | scanner, validation, duplicate_responsibility | no active test import found | yes | validates scan logs; no direct write found in search beyond read/output behavior | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | reconcile with `scripts/core/validate_scans.py` and archive duplicate | P2 |
| `scripts/watchlist/auto_watchlist_from_scan.py` | CANONICAL_MIGRATION_REQUIRED | watchlist, scanner, data_write | no active test import found | yes | writes watchlist transactions from scan output | no canonical owner identified yet | define watchlist timing-state owner before migration | P1 |
| `scripts/watchlist/build_watchlist.py` | CANONICAL_MIGRATION_REQUIRED | watchlist, data_write | no active test import found | yes | writes active watchlist | no canonical owner identified yet | define watchlist state owner before migration | P1 |
| `scripts/watchlist/evaluate_watchlist.py` | DO_NOT_TOUCH_YET | watchlist, analysis, configuration, data_write, language_cleanup_required | no active test import found | yes | writes watchlist status and contains Dutch operator strings | no canonical owner identified yet | separate language cleanup and watchlist policy before migration | P0 |
| `scripts/watchlist/parse_watchlist_commands.py` | DO_NOT_TOUCH_YET | watchlist, portfolio, data_write, operator_utility, language_cleanup_required | no active test import found | yes | appends watchlist commands and contains Dutch operator strings | no canonical owner identified yet | leave untouched until watchlist command policy exists | P0 |
| `scripts/watchlist/update_watchlist_actions.py` | CANONICAL_MIGRATION_REQUIRED | watchlist, data_write | no active test import found | yes | writes watchlist transaction updates from status | no canonical owner identified yet | define watchlist transition owner before migration | P1 |

## Responsibility-grouped findings

### runtime / orchestration

The two former whole-application runtime scripts are archived, but many lower-level script-era files remain runnable. Most runtime-like files are layer builders with direct CSV writes. They should not become permanent alternate entrypoints.

### scanner / universe

`scripts/core/scanner.py`, `scripts/core/data_fetcher.py`, `scripts/core/indicators.py`, `scripts/core/regime.py`, `scripts/core/validator.py`, and scanner validation/log scripts still contain legacy scanner and data-fetch behavior. The canonical owner is `src/market_scanner/scanner/`, but migration requires a separate market-data/source policy because `scripts/core/scanner.py` uses `yfinance` and includes scoring/target fields.

### fundamentals / provider / evidence

`scripts/fundamentals/*` and `scripts/data_sources/*` are heavily tested and contain useful provider, SEC, normalization, metric, quality, and analysis logic. They should not be archived until canonical `src/market_scanner/fundamentals/` and `src/market_scanner/analysis/` own the tested behavior. SEC bulk intake is high-risk because it includes network/download behavior.

### analysis

Context, validation, timing, stability, diagnostics, and fundamentals analysis scripts overlap with the canonical `analysis/` boundary. Many write outputs and several are still imported by tests. Migration should be staged by responsibility, not done as a bulk move.

### decision / review

`scripts/core/decision_engine.py` remains the certified script-era allocation authority and includes final action/allocation semantics. It is P0 because the repository doctrine says Decision Engine is the only allocation authority. Migration or archive requires separate governance and tests that preserve allocation boundaries.

### portfolio

Portfolio scripts mutate transaction, position, and review files and parse BUY/SELL commands. No canonical portfolio command owner is identified yet. Treat these as P0/P1 until a portfolio state and command policy exists.

### watchlist

Watchlist scripts mutate watchlist transaction/status/active files and include Dutch operator-facing strings. They need both architecture review and language-governance cleanup planning before migration or archive.

### messaging / report formatting

Legacy message/report helpers duplicate canonical messaging and reporting planning boundaries. `scripts/reporting/reporter.py` appears low-risk and unreferenced, while `build_telegram_summary.py` remains coupled to report artifact writes.

### report artifact generation

`scripts/reporting/build_reporting_layer.py` writes dashboard/log files and `reports/daily/telegram_message.txt`, and represents Decision Engine allocation fields. It must be migrated only under canonical reporting/message separation.

### Telegram / delivery

`scripts/reporting/send_telegram.py` and `scripts/telegram/process_telegram_commands.py` include credential, network, Telegram, and command-processing behavior. They are high-risk and must stay separated from message composition and reporting.

### configuration / environment

Thresholds, settings, governed path rules, and source-artifact output rules are distributed across `config.settings`, `scripts/data_sources/common.py`, and watchlist scripts. A canonical `src/market_scanner/config/` owner is not yet present.

### utilities / maintenance

Diagnostics and historical evidence capture are useful but side-effectful maintenance utilities. They need a maintenance namespace and archive/delete policy before cleanup implementation.

### unclear / do not touch

Portfolio command handling, Telegram command processing, and watchlist command/status mutation are unclear or high-risk enough to avoid touching until policy boundaries are defined.

## Prioritized cleanup sequence

### P0 - Must not touch yet

- `scripts/core/decision_engine.py`
- `scripts/portfolio/parse_trade_commands.py`
- `scripts/portfolio/portfolio_manager.py`
- `scripts/telegram/process_telegram_commands.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/watchlist/parse_watchlist_commands.py`

Reason: these files contain allocation/final decision semantics, BUY/SELL command handling, Telegram command behavior, portfolio/watchlist mutation, credential/network paths, or language-governance concerns.

### P1 - High-risk side-effect scripts requiring migration review

- Scanner/provider: `scripts/core/scanner.py`, `scripts/core/data_fetcher.py`
- Fundamentals/provider: `scripts/fundamentals/build_*`, `scripts/fundamentals/sec_*`, `scripts/data_sources/*`
- Decision/review-adjacent: `scripts/core/build_portfolio_intelligence.py`, `scripts/core/build_stability_layer.py`
- Reporting/Telegram: `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `scripts/reporting/send_telegram.py`
- Portfolio/watchlist writers: `scripts/portfolio/build_portfolio.py`, `scripts/portfolio/evaluate_positions.py`, `scripts/watchlist/auto_watchlist_from_scan.py`, `scripts/watchlist/build_watchlist.py`, `scripts/watchlist/update_watchlist_actions.py`
- Maintenance writers: `scripts/ops/capture_historical_evidence.py`

Recommended handling: migrate or isolate useful logic into canonical owners with tests before archive/delete.

### P2 - Duplicate responsibility candidates for controlled archive review

- validation and analysis duplicates: `scripts/analyze_validation.py`, `scripts/core/analyze_validation.py`, `scripts/validate_scans.py`, `scripts/core/validate_scans.py`;
- compatibility wrappers: `scripts/core/build_fundamental_analysis.py`, `scripts/core/build_fundamental_layer.py`, `scripts/core/build_fundamental_metrics.py`, `scripts/core/build_fundamentals_history_intake.py`;
- older layer/log helpers: `scripts/core/log_scans.py`, `scripts/core/validator.py`, `scripts/core/build_context_backfill.py`, `scripts/core/build_entry_quality_backfill.py`;
- diagnostics utility: `scripts/diagnostics/audit_data_coverage.py`.

Recommended handling: confirm active test coverage and canonical replacements, then propose focused archive sprints.

### P3 - Low-risk archive candidates

- `scripts/core/regime.py`
- `scripts/fundamentals/__init__.py`
- `scripts/reporting/reporter.py`
- `scripts/utils/utils.py`

Recommended handling: confirm no hidden imports or operator use, then archive in a low-risk batch after BL43 removes the archived-script execution concern.

### P4 - Delete candidates after confirmation

- `scripts/portfolio/test_portfolio.py`

Recommended handling: confirm it is not part of the test suite or operator procedure, then delete or archive through a controlled cleanup sprint.

## BL41 archived-script-execution cleanup concern

BL41 documented this concern:

```text
One existing test still validates fail-closed behavior by executing an archived script.
```

BL42 confirms the concern remains outside active `scripts/`: `tests/core/test_run_full_pipeline.py` executes `archive/legacy_runtime/scripts/run_full_pipeline.py` in a subprocess to verify fail-closed behavior.

This should become a separate cleanup item before additional archive/delete work:

```text
RESET-10L-BL43 - Remove Archived Script Execution from Tests
```

The safer next step is to rewrite that test as static archive evidence or move fail-closed certification into a non-executing validation pattern. BL42 does not modify the test.

## Recommended next cleanup step

RESET-10L-BL43 - Remove Archived Script Execution from Tests.

Rationale: before archiving more script-era files, the project should remove the remaining test pattern that executes archived code. This keeps archive semantics clean and avoids repeating the BL41 validation conflict during future cleanup sprints.

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

- Classification is based on static inspection and active reference searches only.
- Some files may have operator use not visible through source/test/workflow references.
- The table identifies likely canonical owners, but it does not authorize migration, archive, deletion, or runtime replacement.
- The inventory covers `scripts/` only. It does not classify `legacy/`, `archive/`, `data/`, reports, or non-Python artifacts.
- Several files span mixed responsibilities and should be split by policy before any code migration.
