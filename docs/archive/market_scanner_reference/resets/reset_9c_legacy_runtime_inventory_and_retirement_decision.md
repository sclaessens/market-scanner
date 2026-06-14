# RESET-9C: Legacy Runtime Inventory and Retirement Decision

## 1. Purpose

RESET-9C inventories the legacy runtime surface and records a conservative retirement decision.

This is a static inventory and governance decision only. It does not archive, delete, move, refactor, migrate, or execute legacy runtime code.

## 2. Scope

In scope:

- `scripts/` runtime files;
- tests that import or reference `scripts/`;
- `.github/workflows/` workflow references;
- legacy data and report touchpoints;
- provider, network, SEC, EDGAR, and Telegram risk surfaces;
- future cleanup sequencing.

Out of scope:

- runtime cleanup execution;
- v2 implementation;
- code migration;
- provider integration;
- raw-to-normalized transformation;
- data migration;
- generated output cleanup.

## 3. Current Safe Preconditions

Observed preconditions:

- working tree was clean before inventory;
- v2 source package exists under `src/market_scanner/`;
- v2 fixture and lifecycle directories exist;
- `.github/workflows/daily-market-scan.yml` has `workflow_dispatch`;
- no active `schedule:` trigger was found in the daily scan workflow;
- no active `cron:` line was found in the daily scan workflow;
- the legacy daily scan workflow still contains manual legacy execution steps and `contents: write`.

Important note: local `find` output shows Python bytecode cache files under `scripts/`, `src/`, and `tests/`. These are runtime/generated artifacts and are not v2 source files.

## 4. Inventory Commands Run

Safety and state:

```bash
git status --short
find src/market_scanner -maxdepth 3 -type f | sort
find data/raw data/normalized data/generated data/local data/fixtures/v2 -maxdepth 2 -type f | sort
sed -n '1,80p' .github/workflows/daily-market-scan.yml
grep -n "schedule:" .github/workflows/daily-market-scan.yml || true
grep -n "cron:" .github/workflows/daily-market-scan.yml || true
grep -n "workflow_dispatch" .github/workflows/daily-market-scan.yml || true
```

Inventory:

```bash
find scripts -type f | sort
find tests -type f | sort
find .github/workflows -type f | sort
find data -maxdepth 3 -type f | sort
find reports -maxdepth 3 -type f | sort || true
```

Static references:

```bash
grep -R "scripts/" -n .github docs src tests pyproject.toml requirements.txt README.md 2>/dev/null || true
grep -R "scripts\\." -n .github docs src tests pyproject.toml requirements.txt README.md 2>/dev/null || true
grep -R "run_scan" -n .github docs src tests scripts 2>/dev/null || true
grep -R "process_telegram_commands" -n .github docs src tests scripts 2>/dev/null || true
grep -R "data/processed" -n .github docs src tests scripts 2>/dev/null || true
grep -R "data/portfolio" -n .github docs src tests scripts 2>/dev/null || true
grep -R "data/watchlist" -n .github docs src tests scripts 2>/dev/null || true
grep -R "reports/daily" -n .github docs src tests scripts 2>/dev/null || true
grep -R "telegram" -n .github docs src tests scripts 2>/dev/null || true
grep -R "SEC\\|EDGAR\\|sec" -n .github docs src tests scripts 2>/dev/null || true
```

Additional static drilldown:

```bash
find scripts -type f -name '*.py' | sort
find scripts -type f ! -name '*.py' | sort
find scripts -type f -name '*.py' | awk -F/ '{if (NF==2) group="scripts"; else group=$1"/"$2; count[group]++} END {for (g in count) print g, count[g]}' | sort
rg -n "read_csv|to_csv|write_text|open\\(|mkdir|Path\\(|urlopen|requests|yfinance|yf\\.|TELEGRAM|telegram|api\\.telegram|sec\\.gov|EDGAR|SEC|data/|reports/|\\.csv|\\.txt" scripts -g '*.py'
rg -n "from scripts|import scripts|scripts\\." scripts tests .github/workflows -g '*.py' -g '*.yml'
find tests -type f -name '*.py' | xargs grep -l "from scripts\\|import scripts\\|scripts\\." | sort
```

## 5. Legacy Runtime Overview

The legacy runtime surface remains broad:

| Group | Python file count | Role |
|---|---:|---|
| `scripts/` | 4 | top-level orchestration and validation helpers |
| `scripts/core/` | 20 | scanner, validation, context, timing, portfolio intelligence, Decision Engine, backfills, wrappers, utilities |
| `scripts/data_sources/` | 3 | source-data prefill helpers |
| `scripts/diagnostics/` | 1 | data coverage diagnostics |
| `scripts/fundamentals/` | 9 | fundamentals, SEC, source-data transformation and review |
| `scripts/ops/` | 1 | historical evidence capture |
| `scripts/portfolio/` | 5 | portfolio transaction, position, review, and command utilities |
| `scripts/reporting/` | 4 | legacy reporting, Telegram text, Telegram delivery |
| `scripts/telegram/` | 1 | inbound Telegram command polling |
| `scripts/utils/` | 1 | shared file-output helpers |
| `scripts/watchlist/` | 5 | legacy watchlist transaction/status utilities |

The old runtime is not ready for archive/delete because it is still the only complete manual fallback implementation and contains unresolved source-data, portfolio, provider, reporting, Telegram, and Decision Engine knowledge.

## 6. Per-Script Classification Table

Categories:

- `TEMPORARY_FALLBACK_KEEP`: needed to preserve manual old-pipeline fallback until v2 replacement exists.
- `KNOWLEDGE_EXTRACTION_REQUIRED`: contains behavior or domain logic that must be understood and rewritten or explicitly rejected before retirement.
- `ARCHIVE_CANDIDATE_AFTER_EXTRACTION`: likely historical or compatibility surface after knowledge extraction and replacement.
- `DELETE_CANDIDATE_AFTER_APPROVAL`: likely redundant/generated/obsolete, but no deletion in RESET-9C.
- `REAPPROVAL_REQUIRED`: unclear, sensitive, provider/network/credential-adjacent, or operationally risky.

| Path | Category | Short purpose | Known callers/references | Data paths read | Data paths written | External/provider/network/Telegram behavior | V2 replacement status | Retirement blocker | Recommended next action |
|---|---|---|---|---|---|---|---|---|---|
| `scripts/analyze_validation.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Top-level validation-summary helper | historical docs; manual use possible | `data/scans_validation.csv` | `data/validation_summary.csv` | none detected | `NO_V2_REPLACEMENT` | confirm whether still used manually | Map to diagnostics or archive after approval |
| `scripts/run_full_pipeline.py` | `TEMPORARY_FALLBACK_KEEP` | Operator wrapper around `scripts/run_scan.py` | tests; docs; manual fallback | delegated to `run_scan.py` | delegated to `run_scan.py` | delegated provider/Telegram behavior | `NO_V2_REPLACEMENT` | only complete local wrapper for old runtime | Keep until v2 orchestration is accepted |
| `scripts/run_scan.py` | `TEMPORARY_FALLBACK_KEEP` | Main legacy scan orchestration | workflow manual dispatch; wrapper; tests | tickers, processed, raw fundamentals, portfolio, generated layers | processed CSVs, logs, portfolio review, reports | yfinance via data fetcher; Telegram delivery via sender | `V2_SCAFFOLD_ONLY` | only complete old pipeline fallback | Keep untouched until v2 end-to-end replacement |
| `scripts/validate_scans.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Top-level scan validation helper | historical/manual references | scan logs and processed ticker files | validation outputs | none detected | `NO_V2_REPLACEMENT` | determine if redundant with core validator/tests | Archive after validation knowledge extraction |
| `scripts/core/analyze_validation.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Core validation analysis helper | historical/manual references | validation result CSVs | summary CSVs | none detected | `NO_V2_REPLACEMENT` | duplicate-like top-level helper exists | Compare with top-level helper before archive |
| `scripts/core/build_context_backfill.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Historical context backfill | tests; docs | scan validation/history data | `data/processed/context_strength_historical.csv` | none detected | `NO_V2_REPLACEMENT` | backfill methodology not represented in v2 | Extract backfill assumptions before retirement |
| `scripts/core/build_context_layer.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy Context Layer builder | `run_scan.py`; tests | `scanner_ranked.csv`; optional sector RS | `context_strength.csv`; context log | none detected | `V2_SCAFFOLD_ONLY` | active fallback layer | Keep until v2 context semantics exist |
| `scripts/core/build_entry_quality_backfill.py` | `REAPPROVAL_REQUIRED` | Entry-quality historical backfill | tests; docs | scan validation and optional OHLCV | entry-quality historical output/log | imports yfinance when needed | `NO_V2_REPLACEMENT` | provider/network fallback path | Owner review before archive/delete |
| `scripts/core/build_fundamental_analysis.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Compatibility wrapper to fundamentals analysis | tests; docs | delegated | delegated | none detected | `V2_SCAFFOLD_ONLY` | wrapper still verifies legacy organization | Retire only after tests and callers move |
| `scripts/core/build_fundamental_layer.py` | `TEMPORARY_FALLBACK_KEEP` | Compatibility wrapper to current fundamental quality builder | `run_scan.py`; tests | delegated raw/processed fundamentals | delegated `fundamental_quality.csv` and log | none detected directly | `V2_SCAFFOLD_ONLY` | active fallback layer | Keep until v2 fundamentals readiness/analysis replacement |
| `scripts/core/build_fundamental_metrics.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Compatibility wrapper to fundamentals metrics | tests; docs | delegated | delegated | none detected | `V2_SCAFFOLD_ONLY` | wrapper still covered by tests | Retire after tests/callers transition |
| `scripts/core/build_fundamentals_history_intake.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Compatibility wrapper to fundamentals history intake | tests; docs | delegated | delegated report optional | none detected | `V2_SCAFFOLD_ONLY` | source-data validation knowledge needed | Extract contract knowledge, then retire wrapper |
| `scripts/core/build_portfolio_intelligence.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy Portfolio Intelligence builder | `run_scan.py`; tests | `timing_state_layer.csv`; portfolio positions; metadata | `portfolio_intelligence.csv`; log | none detected | `V2_SCAFFOLD_ONLY` | active fallback downstream layer | Keep until v2 portfolio classification exists |
| `scripts/core/build_stability_layer.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Stability metadata from final decisions | tests; docs; optional reporting input | `final_decisions.csv` | `stability_state.csv`; log | none detected | `NO_V2_REPLACEMENT` | stability concept not in v2 scaffold | Extract/report whether v2 needs stability |
| `scripts/core/build_timing_state_layer.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy Timing State builder | `run_scan.py`; tests | `fundamental_quality.csv`; entry-quality metrics | `timing_state_layer.csv`; log | none detected | `V2_SCAFFOLD_ONLY` | active fallback layer | Keep until v2 timing contract exists |
| `scripts/core/build_validation_layer.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy Validation Layer builder | `run_scan.py`; tests | `scanner_ranked.csv` | `validation_layer.csv`; `entry_quality_metrics.csv`; log | none detected | `V2_SCAFFOLD_ONLY` | active fallback layer | Keep until v2 validation exists |
| `scripts/core/data_fetcher.py` | `REAPPROVAL_REQUIRED` | Ticker loading and market data fetcher | `run_scan.py`; scanner | ticker config/source | none directly | imports `yfinance`; live provider calls | `NO_V2_REPLACEMENT` | network/provider behavior | Isolate provider policy before retirement/replacement |
| `scripts/core/decision_engine.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy final-action/allocation authority | `run_scan.py`; tests; ops | `portfolio_intelligence.csv` | `final_decisions.csv`; decision log | none detected | `V2_SCAFFOLD_ONLY` | v2 Decision Engine is review-only scaffold | Keep until approved v2 Decision Engine exists |
| `scripts/core/indicators.py` | `TEMPORARY_FALLBACK_KEEP` | Technical indicator helpers | `run_scan.py`; scanner; tests | OHLCV frames | none directly | none detected | `NO_V2_REPLACEMENT` | shared old scanner dependency | Extract indicator contracts before retirement |
| `scripts/core/log_scans.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Scanner log append helper | manual/historical references | `scanner_ranked.csv`; `market_regime.csv`; scan log | `data/logs/scans_log.csv` | none detected | `NO_V2_REPLACEMENT` | unclear current caller status | Confirm active use, then archive or replace |
| `scripts/core/regime.py` | `TEMPORARY_FALLBACK_KEEP` | Market regime classification helper | `run_scan.py`; scanner | market data frames | none directly | none detected | `NO_V2_REPLACEMENT` | active old scanner dependency | Extract if v2 needs market-regime classification |
| `scripts/core/scanner.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy scanner/discovery logic | `run_scan.py`; validation docs | OHLCV data, ticker info | setup rows returned to orchestrator | yfinance ticker info via sector lookup | `V2_SCAFFOLD_ONLY` | full discovery logic not replaced | Extract discovery concepts before retirement |
| `scripts/core/validate_scans.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Core scan validation helper | manual/historical references | scan logs, processed ticker files | `validation_results.csv` | none detected | `NO_V2_REPLACEMENT` | overlap with other validation helpers | Compare before archive/delete |
| `scripts/core/validator.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Validation helper/library | likely scanner or manual validation support | scan data objects | none directly | none detected | `NO_V2_REPLACEMENT` | validation assumptions not fully mapped | Extract contracts before retirement |
| `scripts/data_sources/common.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Shared CSV/source-data helper utilities | data-source tests and helpers | source CSVs | atomic CSV writes | none detected | `NO_V2_REPLACEMENT` | useful source-data utility patterns | Extract safe patterns into v2 only after approval |
| `scripts/data_sources/prefill_fundamentals.py` | `REAPPROVAL_REQUIRED` | Provider/operator-assisted fundamentals prefill | tests; docs | operator/provider export CSV | optional fundamentals source output | provider-adjacent by purpose | `NO_V2_REPLACEMENT` | source-data approval boundary | Owner review before any reuse |
| `scripts/data_sources/prefill_portfolio_metadata.py` | `REAPPROVAL_REQUIRED` | Portfolio metadata prefill | tests; docs | operator/provider export CSV | optional `portfolio_metadata.csv` | provider-adjacent by purpose | `NO_V2_REPLACEMENT` | source-data approval boundary | Owner review before any reuse |
| `scripts/diagnostics/audit_data_coverage.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Data coverage diagnostic | tests; docs | portfolio, watchlist, scanner, metadata, raw fundamentals | diagnostic return/report only | none detected | `NO_V2_REPLACEMENT` | diagnostics may be useful but not pipeline authority | Extract as future local diagnostic if approved |
| `scripts/fundamentals/__init__.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Package marker | tests/imports | none | none | none | `V2_SCAFFOLD_ONLY` | package needed while legacy tests run | Retire with fundamentals package |
| `scripts/fundamentals/build_analysis.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Fundamental analysis classification | `run_scan.py`; tests | quality and metrics CSVs | optional `fundamental_analysis.csv` | none detected | `V2_SCAFFOLD_ONLY` | v2 has readiness only, not analysis | Extract classification concepts before retirement |
| `scripts/fundamentals/build_history_intake.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Fundamentals history validation/intake | tests; wrappers | fundamentals history CSV | optional report | none detected | `V2_SCAFFOLD_ONLY` | source-data contract knowledge | Extract into future normalized-input contracts |
| `scripts/fundamentals/build_metrics.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Fundamental metrics builder | `run_scan.py`; tests | fundamentals history CSV | optional metrics CSV | none detected | `V2_SCAFFOLD_ONLY` | metrics logic not in v2 | Extract formulas and missing-value policy |
| `scripts/fundamentals/build_quality.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy Fundamental Quality Layer | `run_scan.py`; tests | `context_strength.csv`; raw/history/metrics fundamentals | `fundamental_quality.csv`; log | none detected | `V2_SCAFFOLD_ONLY` | active fallback layer | Keep until v2 fundamentals path approved |
| `scripts/fundamentals/run_sec_transformation_review.py` | `REAPPROVAL_REQUIRED` | Controlled local SEC transform review | tests; docs | local SEC-like files | optional review CSV | SEC/local provider-adjacent | `V2_SCAFFOLD_ONLY` | SEC reintroduction not approved beyond scaffold | Keep local-only; no execution without approval |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | `REAPPROVAL_REQUIRED` | SEC Company Facts bulk cache utility | tests; docs | SEC URL/local ZIP | `data/local/sec_edgar/...` cache/manifest | `urllib.request.urlopen`; SEC network capable | `NO_V2_REPLACEMENT` | live SEC/network behavior | Do not run; reapprove before any future use |
| `scripts/fundamentals/sec_companyfacts_transform.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Local SEC Company Facts transform | tests; docs | local SEC JSON | optional raw fundamentals rows | local SEC transform only | `V2_SCAFFOLD_ONLY` | source-data transformation knowledge | Extract contracts before any v2 transform |
| `scripts/fundamentals/sec_ticker_cik_index.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | SEC ticker/CIK mapping and coverage | tests; docs | local SEC ticker/CIK source | optional coverage CSV | SEC/local provider-adjacent | `NO_V2_REPLACEMENT` | identifier mapping knowledge needed | Extract mapping contract before retirement |
| `scripts/ops/capture_historical_evidence.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Historical evidence capture | tests; docs | generated processed/reporting artifacts | historical evidence output | none detected | `NO_V2_REPLACEMENT` | audit evidence policy unresolved | Decide if v2 needs evidence capture |
| `scripts/portfolio/build_portfolio.py` | `TEMPORARY_FALLBACK_KEEP` | Build portfolio positions from transactions | `run_scan.py`; tests | `portfolio_transactions.csv` | `portfolio_positions.csv` | none detected | `V2_SCAFFOLD_ONLY` | active fallback portfolio source builder | Keep until v2 portfolio input contract |
| `scripts/portfolio/evaluate_positions.py` | `TEMPORARY_FALLBACK_KEEP` | Build descriptive portfolio review | `run_scan.py`; tests/docs | `portfolio_positions.csv`; per-ticker processed files | `portfolio_review.csv` | none detected | `NO_V2_REPLACEMENT` | active fallback support artifact | Keep until portfolio review fate decided |
| `scripts/portfolio/parse_trade_commands.py` | `REAPPROVAL_REQUIRED` | Parse inbound trade commands | Telegram command processor | Telegram command text | transactions via `log_trade` | Telegram command dependency; BUY/SELL command syntax | `NO_V2_REPLACEMENT` | execution-like user command semantics | Reapprove command model before migration/delete |
| `scripts/portfolio/portfolio_manager.py` | `REAPPROVAL_REQUIRED` | Log trades and update positions | command parser; legacy test file | portfolio transactions | portfolio transactions and positions | none detected, but command-side effects | `NO_V2_REPLACEMENT` | portfolio command side effects | Reapprove manual portfolio source policy |
| `scripts/portfolio/test_portfolio.py` | `DELETE_CANDIDATE_AFTER_APPROVAL` | Legacy in-package test/demo | direct pytest discovery risk; docs mention side effects | portfolio CSVs | portfolio CSVs | none detected | `NO_V2_REPLACEMENT` | test placed under runtime package and may dirty data | Move/retire under test cleanup batch after approval |
| `scripts/reporting/build_reporting_layer.py` | `TEMPORARY_FALLBACK_KEEP` | Legacy authoritative Reporting Layer | `run_scan.py`; tests | `final_decisions.csv`; optional stability | dashboard CSV, reporting log, Telegram message | none detected | `V2_SCAFFOLD_ONLY` | active fallback reporting | Keep until v2 reporting output replacement |
| `scripts/reporting/build_telegram_summary.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Compatibility wrapper for reporting/Telegram text | tests; docs; manual use possible | via reporting layer | `telegram_message.txt` | none directly | `V2_SCAFFOLD_ONLY` | wrapper still covered by tests | Retire after callers/tests transition |
| `scripts/reporting/reporter.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Legacy markdown report formatter | historical docs | setup groups | markdown text | none detected | `NO_V2_REPLACEMENT` | old report semantics may be superseded | Archive after confirming no caller |
| `scripts/reporting/send_telegram.py` | `REAPPROVAL_REQUIRED` | Telegram delivery sender | `run_scan.py`; manual use | `telegram_message.txt`; env vars | none | `requests.post` to Telegram API | `NO_V2_REPLACEMENT` | network/credential-adjacent delivery | Keep isolated; reapprove before v2 delivery |
| `scripts/telegram/process_telegram_commands.py` | `REAPPROVAL_REQUIRED` | Inbound Telegram command poller | workflow manual dispatch; docs | Telegram offset log; portfolio command state | `telegram_offset.txt`; portfolio transactions/positions indirectly | Telegram API polling; command side effects | `NO_V2_REPLACEMENT` | network and command semantics | Keep manual fallback only; reapprove before migration |
| `scripts/utils/utils.py` | `ARCHIVE_CANDIDATE_AFTER_EXTRACTION` | Generic file/CSV save helpers | no strong active caller found | none | arbitrary paths | none detected | `NO_V2_REPLACEMENT` | unclear dependency | Confirm no callers, then delete/archive |
| `scripts/watchlist/auto_watchlist_from_scan.py` | `REAPPROVAL_REQUIRED` | Auto-add scanner rows to watchlist | docs mention optional utility | scanner-ranked output | watchlist transactions | none detected | `NO_V2_REPLACEMENT` | hidden filtering/auto-selection risk | Do not automate; reapprove before use |
| `scripts/watchlist/build_watchlist.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Build active watchlist from transactions | watchlist utilities/docs | watchlist transactions | watchlist active | none detected | `NO_V2_REPLACEMENT` | watchlist policy not in v2 | Extract or retire after watchlist decision |
| `scripts/watchlist/evaluate_watchlist.py` | `REAPPROVAL_REQUIRED` | Evaluate watchlist status/readiness | docs; manual possible | watchlist active; processed ticker/regime/scanner files | watchlist status | none detected | `NO_V2_REPLACEMENT` | readiness/status can imply actionability | Reapprove watchlist semantics before migration |
| `scripts/watchlist/parse_watchlist_commands.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Parse watchlist commands | watchlist command utilities | command text | transaction rows returned | none detected | `NO_V2_REPLACEMENT` | command model not mapped to v2 | Extract command grammar if needed |
| `scripts/watchlist/update_watchlist_actions.py` | `KNOWLEDGE_EXTRACTION_REQUIRED` | Convert watchlist status actions to transactions | watchlist utilities | watchlist status and transactions | watchlist transactions | none detected | `NO_V2_REPLACEMENT` | action/status semantics need review | Reapprove before any future watchlist automation |

## 7. Generated Bytecode Classification

The inventory found generated Python bytecode under `scripts/**/__pycache__/`.

Classification for all listed bytecode files:

- category: `DELETE_CANDIDATE_AFTER_APPROVAL`;
- purpose: generated interpreter cache;
- known callers/references: none required for source behavior;
- v2 replacement status: not applicable;
- retirement blocker: deletion must be an explicit cleanup batch, not RESET-9C.

Files observed:

```text
scripts/__pycache__/run_full_pipeline.cpython-313.pyc
scripts/__pycache__/run_scan.cpython-313.pyc
scripts/core/__pycache__/build_context_backfill.cpython-313.pyc
scripts/core/__pycache__/build_context_layer.cpython-313.pyc
scripts/core/__pycache__/build_entry_quality_backfill.cpython-313.pyc
scripts/core/__pycache__/build_fundamental_analysis.cpython-313.pyc
scripts/core/__pycache__/build_fundamental_layer.cpython-313.pyc
scripts/core/__pycache__/build_fundamental_metrics.cpython-313.pyc
scripts/core/__pycache__/build_fundamentals_history_intake.cpython-313.pyc
scripts/core/__pycache__/build_portfolio_intelligence.cpython-313.pyc
scripts/core/__pycache__/build_stability_layer.cpython-313.pyc
scripts/core/__pycache__/build_timing_state_layer.cpython-313.pyc
scripts/core/__pycache__/build_validation_layer.cpython-313.pyc
scripts/core/__pycache__/data_fetcher.cpython-313.pyc
scripts/core/__pycache__/decision_engine.cpython-313.pyc
scripts/core/__pycache__/indicators.cpython-313.pyc
scripts/core/__pycache__/regime.cpython-313.pyc
scripts/core/__pycache__/scanner.cpython-313.pyc
scripts/data_sources/__pycache__/common.cpython-313.pyc
scripts/data_sources/__pycache__/prefill_fundamentals.cpython-313.pyc
scripts/data_sources/__pycache__/prefill_portfolio_metadata.cpython-313.pyc
scripts/diagnostics/__pycache__/audit_data_coverage.cpython-313.pyc
scripts/fundamentals/__pycache__/__init__.cpython-313.pyc
scripts/fundamentals/__pycache__/build_analysis.cpython-313.pyc
scripts/fundamentals/__pycache__/build_history_intake.cpython-313.pyc
scripts/fundamentals/__pycache__/build_metrics.cpython-313.pyc
scripts/fundamentals/__pycache__/build_quality.cpython-313.pyc
scripts/fundamentals/__pycache__/run_sec_transformation_review.cpython-313.pyc
scripts/fundamentals/__pycache__/sec_companyfacts_bulk_intake.cpython-313.pyc
scripts/fundamentals/__pycache__/sec_companyfacts_transform.cpython-313.pyc
scripts/fundamentals/__pycache__/sec_ticker_cik_index.cpython-313.pyc
scripts/ops/__pycache__/capture_historical_evidence.cpython-313.pyc
scripts/portfolio/__pycache__/build_portfolio.cpython-313.pyc
scripts/portfolio/__pycache__/evaluate_positions.cpython-313.pyc
scripts/portfolio/__pycache__/portfolio_manager.cpython-313.pyc
scripts/portfolio/__pycache__/test_portfolio.cpython-313-pytest-9.0.3.pyc
scripts/reporting/__pycache__/build_reporting_layer.cpython-313.pyc
scripts/reporting/__pycache__/build_telegram_summary.cpython-313.pyc
scripts/reporting/__pycache__/reporter.cpython-313.pyc
scripts/reporting/__pycache__/send_telegram.cpython-313.pyc
scripts/telegram/__pycache__/process_telegram_commands.cpython-313.pyc
scripts/watchlist/__pycache__/evaluate_watchlist.cpython-313.pyc
scripts/watchlist/__pycache__/parse_watchlist_commands.cpython-313.pyc
scripts/watchlist/__pycache__/update_watchlist_actions.cpython-313.pyc
```

## 8. Legacy Tests Classification

Tests importing or referencing `scripts`:

| Test group | Files | Classification | Retirement guidance |
|---|---|---|---|
| Core legacy runtime tests | `tests/core/test_build_context_backfill.py`, `tests/core/test_build_context_layer.py`, `tests/core/test_build_entry_quality_backfill.py`, `tests/core/test_build_fundamental_analysis.py`, `tests/core/test_build_fundamental_layer.py`, `tests/core/test_build_fundamental_metrics.py`, `tests/core/test_build_fundamentals_history_intake.py`, `tests/core/test_build_portfolio_intelligence.py`, `tests/core/test_build_stability_layer.py`, `tests/core/test_build_timing_state_layer.py`, `tests/core/test_build_validation_layer.py`, `tests/core/test_decision_engine.py`, `tests/core/test_entry_quality.py`, `tests/core/test_fundamentals_operational_validation.py`, `tests/core/test_fundamentals_runtime_organization.py`, `tests/core/test_run_full_pipeline.py` | legacy-runtime tests | keep until fallback retired; map each to v2 replacement tests before deletion |
| Data-source tests | `tests/data_sources/test_prefill_common.py`, `tests/data_sources/test_prefill_fundamentals.py`, `tests/data_sources/test_prefill_portfolio_metadata.py` | legacy-runtime tests | keep until source-data reapproval decides future provider/operator prefill policy |
| Diagnostics/ops tests | `tests/diagnostics/test_audit_data_coverage.py`, `tests/ops/test_capture_historical_evidence.py` | legacy-runtime tests | keep until diagnostics/evidence policy is replaced or rejected |
| SEC/fundamentals tests | `tests/fundamentals/test_run_sec_transformation_review.py`, `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`, `tests/fundamentals/test_sec_companyfacts_transform.py`, `tests/fundamentals/test_sec_ticker_cik_index.py` | legacy-runtime tests | keep until SEC/source-data contracts are approved in v2 |
| Portfolio/reporting/operator tests | `tests/portfolio/test_portfolio_source_contract.py`, `tests/reporting/test_build_reporting_layer.py`, `tests/reporting/test_build_telegram_summary.py`, `tests/test_operator_visibility.py` | legacy-runtime tests | keep while old fallback and reporting/Telegram paths remain available |
| v2 guardrail tests mentioning `scripts` | `tests/integration/test_v2_decision_engine_scaffold.py`, `tests/integration/test_v2_minimal_pipeline_core.py`, `tests/integration/test_v2_reporting_communication_scaffold.py`, `tests/integration/test_v2_source_data_readiness_scaffold.py` | v2 tests | keep; these assert v2 does not import legacy scripts |

Tests not importing or referencing `scripts`:

- `tests/contract/test_v2_data_contracts.py`
- `tests/fixtures/test_v2_fixture_contracts.py`
- `tests/unit/test_v2_decision_records.py`
- `tests/unit/test_v2_package_bootstrap.py`
- `tests/unit/test_v2_pipeline_records.py`
- `tests/unit/test_v2_reporting_records.py`
- `tests/unit/test_v2_source_data_records.py`

## 9. Workflow Classification

| Workflow | Classification | Evidence | Retirement guidance |
|---|---|---|---|
| `.github/workflows/daily-market-scan.yml` | paused legacy workflow | `workflow_dispatch` remains; no active `schedule:` or `cron:`; still runs `scripts/telegram/process_telegram_commands.py` and `scripts/run_scan.py`; still has `contents: write` and generated-file commit steps | keep paused; manual dispatch only by explicit approval; replace with v2 CI later |

## 10. Data/Report Touchpoint Inventory

Legacy runtime data/report touchpoints:

| Path | Current role | Producers/consumers observed | RESET-9C classification |
|---|---|---|---|
| `data/processed/` | legacy generated scanner/layer/Decision Engine/reporting outputs and per-ticker data | many `scripts/core`, `scripts/fundamentals`, `scripts/reporting`, `scripts/run_scan.py`; tests | generated legacy output; not v2 source-of-truth |
| `data/portfolio/portfolio_transactions.csv` | legacy portfolio transaction source candidate | portfolio builder/manager; tests; docs | candidate manual input requiring v2 reapproval |
| `data/portfolio/portfolio_positions.csv` | legacy generated positions | portfolio builder output; portfolio intelligence input | generated legacy output unless reapproved |
| `data/portfolio/portfolio_review.csv` | legacy portfolio review artifact | portfolio evaluate output | generated legacy output |
| `data/portfolio/portfolio_metadata.csv` | manually maintained/descriptive metadata candidate | portfolio intelligence, diagnostics, prefill utilities | source-data candidate requiring reapproval |
| `data/watchlist/` | legacy watchlist transactions/status/active artifacts | watchlist scripts; workflow commits; diagnostics | legacy reference requiring separate watchlist decision |
| `data/logs/` | legacy runtime logs and offsets | layer builders, Telegram command offset, run scan | generated logs/local operational state |
| `data/raw/fundamentals.csv` and backups | legacy raw/manual fundamentals source | fundamentals quality, diagnostics, docs | source-data requiring v2 reapproval |
| `data/intake/` | historical/operator intake templates and pilots | docs and possible manual use | reapproval required before v2 use |
| `reports/daily/telegram_message.txt` | generated Telegram communication artifact | reporting builder, sender, workflow commit | generated report; communication only |
| `reports/daily/market_scan_*.md` | historical legacy markdown reports | no active producer found in current run path | archive/delete candidate later, not source-of-truth |

No data or report files were modified in RESET-9C.

## 11. Provider, Network, Telegram, and SEC Risk Inventory

| Surface | Risk type | Evidence | Decision |
|---|---|---|---|
| `scripts/core/data_fetcher.py` | live market provider | imports `yfinance`; uses `yf.download` and `yf.Ticker` | `REAPPROVAL_REQUIRED`; no v2 use without provider policy |
| `scripts/core/scanner.py` | provider-derived ticker metadata | sector lookup through ticker info | extract discovery/provider assumptions before replacement |
| `scripts/core/build_entry_quality_backfill.py` | optional provider fallback | imports yfinance and can download OHLCV | `REAPPROVAL_REQUIRED` |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | live SEC/network-capable | `urllib.request.urlopen`; `sec.gov` Company Facts ZIP URL | `REAPPROVAL_REQUIRED`; do not run |
| `scripts/fundamentals/sec_companyfacts_transform.py` | local SEC transformation | local SEC JSON transform into raw fundamentals rows | knowledge extraction before v2 transformation |
| `scripts/fundamentals/sec_ticker_cik_index.py` | SEC identifier mapping | local ticker/CIK mapping and coverage | knowledge extraction before v2 identifier contract |
| `scripts/reporting/send_telegram.py` | Telegram network delivery | `requests.post` to Telegram API; env vars | `REAPPROVAL_REQUIRED`; no v2 Telegram behavior |
| `scripts/telegram/process_telegram_commands.py` | Telegram polling and command side effects | Telegram API polling; portfolio command processing; offset log | `REAPPROVAL_REQUIRED`; manual fallback only |
| `.github/workflows/daily-market-scan.yml` | workflow network and write surface | curl to Telegram API; legacy scripts; commits generated files | keep schedule paused; manual only after explicit approval |

## 12. V2 Replacement Readiness Assessment

| Runtime area | Current v2 status | Replacement readiness |
|---|---|---|
| Discovery/scanner | package skeleton only | not replaced |
| Validation | package skeleton only | not replaced |
| Context | package skeleton only | not replaced |
| Fundamentals/source-data | source-data readiness scaffold only | not replaced for analysis, metrics, SEC, or transformations |
| Timing | package skeleton only | not replaced |
| Portfolio | package skeleton only | not replaced |
| Decision Engine | review-only scaffold | not replaced for real final action/allocation |
| Reporting | in-memory communication scaffold | not replaced for generated outputs or Telegram |
| Orchestration | minimal pass-through fixture pipeline | not replaced for production pipeline |
| Data lifecycle | skeleton and contracts exist | not a migration or runtime implementation |

## 13. Retirement Decision

Decision: `DO_NOT_ARCHIVE_OR_DELETE_LEGACY_RUNTIME_YET`

Rationale:

- v2 remains scaffold-only for most runtime layers;
- the old runtime is still the only complete manual fallback path;
- the paused workflow can still be manually dispatched by explicit approval;
- legacy tests still import and protect `scripts` behavior;
- provider, SEC, Telegram, portfolio, watchlist, and reporting command surfaces require explicit owner review;
- many files contain domain and transformation knowledge that must be extracted before archive/delete;
- generated data/report touchpoints still need a separate cleanup plan.

## 14. Future Cleanup Batches

Recommended follow-up batches:

| Batch | Purpose | Execution note |
|---|---|---|
| RESET-9C1 - Legacy Runtime Knowledge Extraction Map | Extract layer-by-layer behavior, formulas, contracts, data paths, and forbidden semantics from legacy code | documentation/static analysis only |
| RESET-9C2 - Legacy Runtime Test Retirement Map | Map legacy tests to v2 replacement tests and identify test delete/archive order | no test deletion yet |
| RESET-9C3 - Legacy Data/Report Touchpoint Cleanup Plan | Classify `data/processed`, `data/portfolio`, `data/watchlist`, `data/logs`, and `reports` artifacts | no data changes yet |
| RESET-9C4 - Manual Fallback Policy | Define who may manually dispatch the legacy workflow, when, and with what output handling | workflow policy only |
| RESET-9C5 - Approved Archive/Delete Execution Batch 1 | Remove or archive lowest-risk generated bytecode/cache and obsolete helpers | only after explicit approval |
| RESET-9C6 - Provider/SEC/Telegram Reapproval Plan | Define conditions for any future provider, SEC, Telegram, or network behavior | no live calls |

## 15. Stop Conditions

Future archive/delete work must stop if:

- a file is referenced by `.github/workflows/daily-market-scan.yml`, `scripts/run_scan.py`, or `scripts/run_full_pipeline.py`;
- a test imports the file and no replacement test exists;
- a data/report path producer or consumer is unclear;
- the file contains provider, SEC, Telegram, credential-adjacent, or network behavior;
- the file contains Decision Engine, allocation, final-action, command, portfolio, or reporting semantics not yet represented in v2;
- any legacy CSV/data/report file would be modified without explicit authorization;
- cleanup would re-enable scheduled legacy workflow execution.

## 16. Validation Results

Validation run after creating this document:

```bash
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- scripts src tests data reports .github/workflows || true
```

Result:

- only RESET-9C documentation files changed;
- no runtime, code, tests, data, reports, or workflow diffs.
- `git diff --check` passed.
- `git diff -- scripts src tests data reports .github/workflows || true` produced no diff.

No pytest is required because RESET-9C changes documentation only.

## 17. Recommended Next Action

RESET-9C1 - Legacy Runtime Knowledge Extraction Map.
