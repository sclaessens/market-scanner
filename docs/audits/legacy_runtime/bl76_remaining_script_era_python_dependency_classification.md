# BL76 — Static dependency classification of remaining script-era Python files

Status: COMPLETED — STATIC CLASSIFICATION ONLY

## Purpose

BL76 classifies the remaining script-era Python files under `scripts/` after BL74 and BL75. It does not archive, delete, move, execute, or refactor runtime Python files. The output is a controlled queue for later cleanup sprints.

## Registry basis

BL76 follows the cleanup direction established by:

- `docs/audits/legacy_runtime/bl70_python_cleanup_registry_lock.md`
- `docs/audits/legacy_runtime/v2_script_era_python_cleanup_inventory.md`
- `docs/audits/legacy_runtime/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/audits/legacy_runtime/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/audits/legacy_runtime/bl74_decouple_active_tests_from_script_era_fundamentals.md`
- `docs/audits/legacy_runtime/bl75_retry_archiving_bl74_unblocked_script_era_files.md`
- `docs/active/architecture/v2_canonical_runtime_architecture.md`

## Classification labels

- `ARCHIVE_READY`: no known active runtime ownership; candidate for archive after grep validation.
- `BLOCKED_ACTIVE_REFERENCE`: likely still referenced by active docs/tests/runtime/workflows; inspect before moving.
- `BLOCKED_RUNTIME_ENTRYPOINT`: likely old executable entrypoint; do not archive until runtime ownership is proven elsewhere.
- `BLOCKED_SIDE_EFFECT_RISK`: may write data, send Telegram, change portfolio/watchlist state, fetch providers, or affect decisions.
- `MIGRATION_REQUIRED`: useful logic may still need canonical `src/market_scanner/` ownership before archive.
- `DOC_ONLY_REFERENCE`: appears primarily historical/governance referenced; archive candidate after active-reference check.
- `UNKNOWN`: insufficient evidence; inspect before action.

## Static classification

### `scripts/core/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/core/analyze_validation.py` | ARCHIVE_READY | Old validation analysis script; likely superseded by canonical validation/reporting boundaries. | BL77 archive candidate after grep. |
| `scripts/core/build_context_backfill.py` | BLOCKED_SIDE_EFFECT_RISK | Backfill naming implies data writes. | Static inspect; archive only if no active writes are required. |
| `scripts/core/build_context_layer.py` | MIGRATION_REQUIRED | Layer builder may contain useful context contract logic. | Confirm canonical `src` ownership before archive. |
| `scripts/core/build_entry_quality_backfill.py` | BLOCKED_SIDE_EFFECT_RISK | Backfill naming implies generated data mutation. | Static inspect before action. |
| `scripts/core/build_fundamental_analysis.py` | MIGRATION_REQUIRED | Legacy fundamentals analysis wrapper; BL74 decoupled tests but logic may need canonical ownership check. | Compare to canonical fundamentals modules. |
| `scripts/core/build_fundamental_layer.py` | MIGRATION_REQUIRED | Legacy layer builder; likely superseded but should be checked against canonical fundamentals provider boundary. | Confirm canonical replacement. |
| `scripts/core/build_fundamental_metrics.py` | MIGRATION_REQUIRED | Legacy metrics builder; pure logic may need migration if not already represented. | Inspect before archive. |
| `scripts/core/build_fundamentals_history_intake.py` | MIGRATION_REQUIRED | Historical intake validation may encode useful schema rules. | Confirm canonical ownership. |
| `scripts/core/build_portfolio_intelligence.py` | BLOCKED_SIDE_EFFECT_RISK | Portfolio intelligence can influence downstream state/review. | Inspect and prove no active runtime dependency. |
| `scripts/core/build_stability_layer.py` | MIGRATION_REQUIRED | Legacy layer builder; check canonical stability contract first. | Confirm canonical replacement. |
| `scripts/core/build_timing_state_layer.py` | MIGRATION_REQUIRED | Timing layer may contain still-useful logic. | Confirm canonical timing ownership. |
| `scripts/core/build_validation_layer.py` | MIGRATION_REQUIRED | Validation layer may contain still-useful contract logic. | Confirm canonical validation ownership. |
| `scripts/core/data_fetcher.py` | BLOCKED_SIDE_EFFECT_RISK | Provider/network-fetch naming. | Do not execute; static inspect only. |
| `scripts/core/decision_engine.py` | BLOCKED_SIDE_EFFECT_RISK | Decision authority file; high-governance risk. | Do not archive until canonical authority and references are proven. |
| `scripts/core/indicators.py` | MIGRATION_REQUIRED | Indicator calculations may be pure reusable logic. | Compare with canonical analysis modules. |
| `scripts/core/log_scans.py` | BLOCKED_SIDE_EFFECT_RISK | Logging/generated output risk. | Static inspect before archive. |
| `scripts/core/scanner.py` | BLOCKED_RUNTIME_ENTRYPOINT | Likely old scanner entrypoint. | Confirm canonical runtime entrypoint and workflow references. |
| `scripts/core/validate_scans.py` | MIGRATION_REQUIRED | Scan validation logic may be reusable. | Confirm canonical validation replacement. |
| `scripts/core/validator.py` | MIGRATION_REQUIRED | Generic validation utility may still encode contracts. | Inspect before archive. |

### `scripts/data_sources/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/data_sources/common.py` | MIGRATION_REQUIRED | Shared source utilities may need canonical ownership. | Inspect and migrate if needed. |
| `scripts/data_sources/prefill_fundamentals.py` | BLOCKED_SIDE_EFFECT_RISK | Prefill implies data writes. | Static inspect only. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | BLOCKED_SIDE_EFFECT_RISK | Portfolio metadata write risk. | Static inspect only. |

### `scripts/diagnostics/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/diagnostics/audit_data_coverage.py` | ARCHIVE_READY | Diagnostic/audit helper; likely not canonical runtime. | BL77 archive candidate after grep. |

### `scripts/fundamentals/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/fundamentals/build_analysis.py` | MIGRATION_REQUIRED | BL74 removed active test imports; useful analysis logic may still need canonical proof. | Inspect against canonical fundamentals modules. |
| `scripts/fundamentals/build_history_intake.py` | MIGRATION_REQUIRED | Historical schema/intake rules may need migration. | Confirm canonical replacement. |
| `scripts/fundamentals/build_metrics.py` | MIGRATION_REQUIRED | Metrics logic may be pure and reusable. | Inspect before archive. |
| `scripts/fundamentals/build_quality.py` | MIGRATION_REQUIRED | Quality-state logic may need canonical ownership. | Inspect before archive. |
| `scripts/fundamentals/run_sec_transformation_review.py` | BLOCKED_SIDE_EFFECT_RISK | SEC review runner; may read/write evidence. | Static inspect; no execution. |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | BLOCKED_SIDE_EFFECT_RISK | SEC bulk intake risk. | Static inspect; no live calls. |
| `scripts/fundamentals/sec_companyfacts_transform.py` | MIGRATION_REQUIRED | Transform logic may be pure and valuable. | Compare with canonical provider adapter. |
| `scripts/fundamentals/sec_ticker_cik_index.py` | BLOCKED_SIDE_EFFECT_RISK | SEC index/cache risk. | Static inspect before archive. |

### `scripts/ops/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/ops/capture_historical_evidence.py` | BLOCKED_SIDE_EFFECT_RISK | Evidence capture suggests data writes. | Static inspect only. |

### `scripts/portfolio/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/portfolio/build_portfolio.py` | BLOCKED_SIDE_EFFECT_RISK | Portfolio-state writer risk. | Static inspect before archive. |
| `scripts/portfolio/evaluate_positions.py` | BLOCKED_SIDE_EFFECT_RISK | Portfolio analysis risk. | Confirm canonical portfolio boundary. |
| `scripts/portfolio/parse_trade_commands.py` | BLOCKED_SIDE_EFFECT_RISK | Trade command parser is governance-sensitive. | Inspect; do not expand authority. |
| `scripts/portfolio/portfolio_manager.py` | BLOCKED_SIDE_EFFECT_RISK | Portfolio manager can mutate state. | Do not archive until ownership is clear. |

### `scripts/reporting/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/reporting/build_reporting_layer.py` | BLOCKED_SIDE_EFFECT_RISK | Report generation risk. | Static inspect before archive. |
| `scripts/reporting/build_telegram_summary.py` | BLOCKED_SIDE_EFFECT_RISK | Telegram/report content boundary. | Inspect; no sending. |
| `scripts/reporting/send_telegram.py` | BLOCKED_SIDE_EFFECT_RISK | Telegram delivery risk. | Do not execute; isolate before archive. |

### `scripts/telegram/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/telegram/process_telegram_commands.py` | BLOCKED_SIDE_EFFECT_RISK | Command handler can trigger actions. | Static inspect only. |

### `scripts/watchlist/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/watchlist/auto_watchlist_from_scan.py` | BLOCKED_SIDE_EFFECT_RISK | Watchlist mutation risk. | Static inspect before archive. |
| `scripts/watchlist/build_watchlist.py` | BLOCKED_SIDE_EFFECT_RISK | Watchlist write risk. | Static inspect before archive. |
| `scripts/watchlist/evaluate_watchlist.py` | BLOCKED_SIDE_EFFECT_RISK | Watchlist analysis risk. | Confirm canonical replacement. |
| `scripts/watchlist/parse_watchlist_commands.py` | BLOCKED_SIDE_EFFECT_RISK | Command parser can alter watchlist actions. | Static inspect only. |
| `scripts/watchlist/update_watchlist_actions.py` | BLOCKED_SIDE_EFFECT_RISK | Explicit watchlist action update risk. | Do not execute; inspect only. |

### Top-level `scripts/`

| File | Classification | Rationale | Next action |
|---|---|---|---|
| `scripts/analyze_validation.py` | ARCHIVE_READY | Duplicate/top-level validation analysis helper. | BL77 archive candidate after grep. |
| `scripts/validate_scans.py` | MIGRATION_REQUIRED | Top-level scan validation may duplicate core validation. | Confirm canonical replacement before archive. |

## BL77 candidates

Initial low-risk archive candidates for the next sprint, pending active-reference grep:

- `scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`

These should be archived only after confirming no active references in `tests/`, `src/`, `.github/`, `pyproject.toml`, and active docs.

## Explicit non-candidates for immediate archive

Do not archive in the next batch without additional static inspection:

- scanner/runtime entrypoints;
- provider/data-fetch/intake scripts;
- backfill/prefill scripts;
- portfolio/watchlist state writers;
- Telegram delivery and command handlers;
- Decision Engine authority files;
- fundamentals transform/quality/metrics files where pure logic may need migration.

## Validation

Static-only sprint. No Python runtime files were executed.

Recommended local validation before merge:

```bash
git status
pytest -q
```

## Guardrails confirmation

- No live SEC/EDGAR calls.
- No yfinance calls.
- No credentials read.
- No production data written.
- No production reports generated.
- No Telegram messages sent.
- No portfolio/watchlist production state modified.
- No Decision Engine authority changed.
- No script-era Python runtime file executed.
- No archive/delete/move performed in BL76.
