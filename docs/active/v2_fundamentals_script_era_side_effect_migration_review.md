# Fundamentals Script-Era Side-Effect Migration Review

## Status

Completed by RESET-10L-BL46.

## Reset stage

RESET-10L-BL46 - Fundamentals Script-Era Side-Effect Migration Review.

## Purpose

Review script-era fundamentals, provider, and source-data Python files to determine what useful logic must migrate into canonical v2 ownership before old files can be archived or removed.

This sprint is review-only. No Python files, tests, workflows, data files, report files, archived scripts, production artifacts, portfolio/watchlist files, or runtime behavior were changed.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_free_cash_flow_derivation_policy.md`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/v2_nvda_real_source_persistence_smoke.md`
- `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- `docs/active/v2_real_analysis_output_defect_review.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Inspection method

Static inspection only. No script-era files, archived scripts, provider clients, SEC/EDGAR clients, yfinance calls, network calls, production pipeline commands, or tests were executed.

Commands and inspection patterns used:

```bash
find scripts -name "*.py" \
  -not -path "*/__pycache__/*" \
  | sort

grep -R "fundamental\|fundamentals\|quality\|source_data\|source data\|persistence\|companyfacts\|SEC\|EDGAR\|yfinance\|yf\.\|provider\|FreeCashFlow\|OperatingCashFlow\|CapitalExpenditures\|Revenue\|NetIncome\|EPS" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "yfinance\|yf\.\|requests\|http\|urlopen\|download\|SEC\|EDGAR\|sec\.gov\|companyfacts\|alpha\|provider\|api" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "open(.*w\|open(.*a\|to_csv\|to_json\|write\|writerow\|mkdir\|unlink\|remove\|rename\|replace\|Path(.*data\|data/" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "os.environ\|getenv\|TOKEN\|API_KEY\|SECRET\|PASSWORD\|credential\|USER_AGENT\|SEC_USER_AGENT" -n scripts --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "scripts.fundamentals\|scripts.data_sources\|scripts.core.data_fetcher\|scripts.core.build_fundamental_analysis\|scripts.core.fundamental_layer\|scripts.core.scanner\|scripts.core.validate_scan_data\|scripts.core.context_builder" -n tests src .github \
  --include="*.py" \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true

grep -R "if __name__ == .__main__." -n scripts/fundamentals scripts/data_sources scripts/core --include="*.py" \
  --exclude-dir=__pycache__ || true

grep -R "BUY\|SELL\|HOLD\|allocation\|conviction\|urgency\|score\|scoring\|target_price\|target price\|tradeability\|recommendation" -n scripts/fundamentals scripts/data_sources scripts/core --include="*.py" \
  --exclude-dir=__pycache__ || true
```

Representative script-era and canonical files were also read directly.

## Files inspected

Governance and review records:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- fundamentals governance and implementation records listed under Policies applied.

Canonical fundamentals files:

- `src/market_scanner/fundamentals/fundamental_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_normalization_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `src/market_scanner/fundamentals/source_data_readiness.py`
- `src/market_scanner/fundamentals/source_data_records.py`

Canonical tests inspected:

- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

Script-era surfaces inspected:

- `scripts/fundamentals/`
- `scripts/data_sources/`
- relevant `scripts/core/` fundamentals/provider/source-data wrappers and adjacent source-access files.

## Fundamentals script-era inventory

Relevant script-era fundamentals/provider/source-data files reviewed: 22.

Core inventory:

- `scripts/fundamentals/__init__.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/data_sources/common.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/data_sources/prefill_portfolio_metadata.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`

Adjacent source/analysis files reviewed because they touch fundamentals/source-data flow:

- `scripts/core/build_context_backfill.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_timing_state_layer.py`
- `scripts/core/build_validation_layer.py`

Files under `scripts/core/` with downstream Decision Engine, portfolio, stability, validation-summary, or scan-log responsibilities were inspected incidentally and are not classified as fundamentals migration targets in this review.

## Active reference summary

Active source references:

- Canonical boundaries reference script-era paths as legacy authority metadata only.
- `src/market_scanner/analysis/analysis_boundary.py` statically lists `scripts/fundamentals/build_analysis.py`, `scripts/fundamentals/build_metrics.py`, `scripts/fundamentals/build_quality.py`, and `scripts/core/build_fundamental_analysis.py`.
- `src/market_scanner/scanner/scanner_boundary.py` statically lists `scripts/core/data_fetcher.py` and `scripts/core/scanner.py`.

Active test source references:

- Source grep still finds legacy imports in old test files for `scripts.fundamentals` and `scripts.data_sources`.
- BL45 classifies those high-risk script-era behavior tests as inactive pytest collection blockers through `tests/conftest.py`.
- These test files remain migration evidence, not approved long-term active dependencies.

Workflow references:

- No `.github` workflow reference to fundamentals script-era modules was found.

## Entrypoint summary

Runnable entrypoints remain in these reviewed files:

- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/data_sources/prefill_fundamentals.py`
- `scripts/data_sources/prefill_portfolio_metadata.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/build_context_backfill.py`
- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_timing_state_layer.py`
- `scripts/core/build_validation_layer.py`

Non-entrypoint reviewed files include:

- `scripts/fundamentals/__init__.py`
- `scripts/data_sources/common.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`

## Side-effect summary

Side-effect risks found:

- SEC/EDGAR/network access: `scripts/fundamentals/sec_companyfacts_bulk_intake.py` uses `urlopen` against the SEC Company Facts ZIP URL when `--download` is used.
- SEC cache/local data writes: `sec_companyfacts_bulk_intake.py` creates `data/local/sec_edgar/companyfacts`, writes the ZIP, writes a manifest, uses temporary files, replaces outputs, and unlinks temporary files.
- Local SEC transformation writes: `sec_companyfacts_transform.py`, `sec_ticker_cik_index.py`, and `run_sec_transformation_review.py` write optional CSV outputs.
- Script-era fundamentals writes: `build_quality.py` writes `data/processed/fundamental_quality.csv` and `data/logs/fundamental_layer_log.csv`; `build_metrics.py`, `build_analysis.py`, and `build_history_intake.py` write optional metrics/analysis/report outputs.
- Data-source prefill writes: `scripts/data_sources/common.py` provides atomic CSV write helpers; `prefill_fundamentals.py` can write `data/raw/fundamentals.csv`; `prefill_portfolio_metadata.py` can write `data/portfolio/portfolio_metadata.csv`.
- yfinance/source access: `scripts/core/data_fetcher.py`, `scripts/core/scanner.py`, `scripts/core/build_context_backfill.py`, and `scripts/core/build_entry_quality_backfill.py` include yfinance access.
- Credential access: no fundamentals-specific credential or environment reads were found. `scripts/data_sources/common.py` records `NO_CREDENTIALS_USED`. Credential/env reads found by the broad search are Telegram/reporting files outside this review scope.
- Investment semantics: `scripts/fundamentals/build_history_intake.py`, `scripts/fundamentals/build_analysis.py`, and `scripts/data_sources/common.py` contain forbidden-term guards; `scripts/core/scanner.py` contains scoring and target/entry fields and belongs to scanner cleanup rather than fundamentals migration.

## Canonical fundamentals comparison

Canonical v2 already covers:

- source-shaped provider response contracts;
- provider category/status validation;
- raw evidence preservation;
- provenance metadata and raw payload hash fields;
- explicit missingness rather than missing-to-zero conversion;
- normalized fundamentals mapping for supported metrics;
- direct `FreeCashFlow` preservation when source-reported;
- governed derived `free_cash_flow = operating_cash_flow - capital_expenditures` with provenance and fail-closed warnings;
- prior-year growth evidence for governed comparable normalized records;
- readiness records with neutral source-data states;
- synthetic persistence validation and tmp-path-only write tests;
- real-source smoke review using injected clients or explicitly supplied responses;
- no reporting, Telegram, portfolio/watchlist, or Decision Engine behavior.

Canonical v2 does not yet cover:

- approved live provider client implementation;
- governed SEC/EDGAR fetch execution;
- SEC Company Facts bulk download/cache/manifest ownership;
- SEC ticker/CIK index ownership;
- full SEC Company Facts JSON fact-selection, conflict evidence, skipped-fact evidence, and source-row transformation parity;
- local real-source capture CLI;
- production-approved persistence path policy;
- multi-period real-source ingestion from a bulk archive;
- migration of old `fundamental_quality`, ratio metric, and descriptive analysis CSV outputs into canonical owners;
- first-class integration of governed growth evidence as the complete analysis input schema;
- EPS prior-year growth evidence policy;
- archive/delete readiness for script-era fundamentals wrappers.

## Per-file classification table

| file path | primary status | secondary tags | active references | entrypoint | side-effect risks | canonical owner | canonical parity status | recommended next action | priority |
|---|---|---|---|---:|---|---|---|---|---|
| `scripts/fundamentals/__init__.py` | ARCHIVE_AFTER_CANONICAL_PARITY | fundamentals, package_marker, active_test_reference | ignored legacy tests | no | none apparent | `src/market_scanner/fundamentals/` | package marker only | archive after script-era fundamentals imports are retired | P3 |
| `scripts/fundamentals/build_history_intake.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, source_normalization, data_write, entrypoint | ignored legacy tests | yes | optional JSON report write | `src/market_scanner/fundamentals/` | partial; canonical has source contracts but not CSV history schema parity | migrate reusable history schema validation into canonical fundamentals | P1 |
| `scripts/fundamentals/build_metrics.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, metrics, growth_evidence, data_write, entrypoint | ignored legacy tests | yes | optional CSV write | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | partial; canonical has growth evidence but not old ratio metric parity | review metric formulas, migrate only governed evidence-safe parts | P1 |
| `scripts/fundamentals/build_quality.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, quality, data_write, persistence, entrypoint | ignored legacy tests and canonical metadata | yes | default processed/log writes | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | partial; canonical has readiness, not old quality profile output parity | migrate neutral source-quality/readiness logic after production path policy | P1 |
| `scripts/fundamentals/build_analysis.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, analysis, data_write, canonical_duplicate, entrypoint | ignored legacy tests and canonical metadata | yes | optional CSV write | `src/market_scanner/analysis/`, `src/market_scanner/fundamentals/` | partial; canonical analysis boundary is review-only but old analysis table remains separate | migrate review-safe analysis inputs only after growth/EPS evidence policy | P1 |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py` | SIDE_EFFECT_REVIEW_REQUIRED | fundamentals, sec_edgar, network_call, bulk_intake, data_write, entrypoint | ignored legacy tests | yes | SEC download, cache dir creation, ZIP write, manifest write, temp replace/unlink | `src/market_scanner/fundamentals/`, `src/market_scanner/config/` | gap; no canonical live SEC fetch/cache boundary | govern live provider/SEC boundary before migration | P0 |
| `scripts/fundamentals/sec_companyfacts_transform.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, sec_edgar, source_normalization, free_cash_flow, data_write, entrypoint | ignored legacy tests | yes | reads local JSON, optional CSV write | `src/market_scanner/fundamentals/` | partial; canonical has metric mapping and FCF derivation but not SEC fact-selection parity | migrate SEC transformation logic after live/source boundary governance | P1 |
| `scripts/fundamentals/sec_ticker_cik_index.py` | CANONICAL_MIGRATION_REQUIRED | fundamentals, sec_edgar, configuration, data_write, entrypoint | ignored legacy tests | yes | local JSON read, optional coverage CSV write | `src/market_scanner/fundamentals/`, `src/market_scanner/config/` | gap; no canonical ticker/CIK mapping owner | migrate or govern ticker/CIK source mapping | P2 |
| `scripts/fundamentals/run_sec_transformation_review.py` | ARCHIVE_AFTER_CANONICAL_PARITY | fundamentals, sec_edgar, review, data_write, mixed_responsibility, entrypoint | ignored legacy tests | yes | optional review CSV write | `src/market_scanner/fundamentals/`, `src/market_scanner/analysis/` | gap; orchestration wrapper around script-era SEC transform | archive after canonical SEC transform/review parity | P2 |
| `scripts/data_sources/common.py` | CANONICAL_MIGRATION_REQUIRED | source_normalization, persistence, configuration, data_write, no_credentials_used | ignored legacy tests | no | atomic CSV write helper, governed output path checks | `src/market_scanner/fundamentals/`, `src/market_scanner/config/`, `src/market_scanner/utils/` | partial; canonical persistence has tmp-path safety but no production source prefill policy | migrate path policy and audit metadata only if still needed | P1 |
| `scripts/data_sources/prefill_fundamentals.py` | SIDE_EFFECT_REVIEW_REQUIRED | fundamentals, provider_export, data_write, persistence, entrypoint | ignored legacy tests | yes | optional governed write to `data/raw/fundamentals.csv` | `src/market_scanner/fundamentals/`, `src/market_scanner/config/` | gap; no approved production source prefill writer | govern production path policy before migration | P1 |
| `scripts/data_sources/prefill_portfolio_metadata.py` | NO_FUNDAMENTALS_RELEVANCE | portfolio, provider_export, data_write, entrypoint | ignored legacy tests | yes | optional governed write to portfolio metadata | no canonical fundamentals owner | outside fundamentals scope | move to future portfolio/source-data governance review | P2 |
| `scripts/core/build_fundamentals_history_intake.py` | ARCHIVE_AFTER_CANONICAL_PARITY | compatibility_wrapper, fundamentals, entrypoint | ignored legacy tests | yes | delegates to script-era history intake | archive only | duplicate wrapper | archive after callers target canonical owner | P2 |
| `scripts/core/build_fundamental_metrics.py` | ARCHIVE_AFTER_CANONICAL_PARITY | compatibility_wrapper, fundamentals, entrypoint | ignored legacy tests | yes | delegates to script-era metrics | archive only | duplicate wrapper | archive after callers target canonical owner | P2 |
| `scripts/core/build_fundamental_layer.py` | ARCHIVE_AFTER_CANONICAL_PARITY | compatibility_wrapper, fundamentals, entrypoint | ignored legacy tests | yes | delegates to script-era quality | archive only | duplicate wrapper | archive after callers target canonical owner | P2 |
| `scripts/core/build_fundamental_analysis.py` | ARCHIVE_AFTER_CANONICAL_PARITY | compatibility_wrapper, analysis, fundamentals, entrypoint | ignored legacy tests and canonical metadata | yes | delegates to script-era analysis | archive only | duplicate wrapper | archive after callers target canonical owner | P2 |
| `scripts/core/data_fetcher.py` | SIDE_EFFECT_REVIEW_REQUIRED | scanner, provider_call, yfinance, network_call | canonical metadata and script imports | no | yfinance downloads/history/info | `src/market_scanner/scanner/` | gap; scanner data access not canonicalized | review under scanner/source access, not fundamentals | P1 |
| `scripts/core/scanner.py` | SIDE_EFFECT_REVIEW_REQUIRED | scanner, yfinance, network_call, scoring, mixed_responsibility | canonical metadata | no | yfinance sector lookup; scoring/target fields | `src/market_scanner/scanner/` | outside canonical fundamentals; scanner parity gap | review under scanner migration before archive | P1 |
| `scripts/core/build_context_backfill.py` | SIDE_EFFECT_REVIEW_REQUIRED | analysis, yfinance, network_call, data_write, entrypoint | ignored legacy tests | yes | yfinance OHLCV download, processed/log writes | `src/market_scanner/analysis/`, `src/market_scanner/scanner/` | outside canonical fundamentals | review under scanner/analysis side-effect migration | P1 |
| `scripts/core/build_entry_quality_backfill.py` | SIDE_EFFECT_REVIEW_REQUIRED | scanner, yfinance, network_call, data_write, entrypoint | ignored legacy tests | yes | yfinance OHLCV download, processed/log writes | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | outside canonical fundamentals | review under scanner/analysis side-effect migration | P1 |
| `scripts/core/build_timing_state_layer.py` | ARCHIVE_AFTER_CANONICAL_PARITY | analysis, consumes_fundamental_quality, data_write, entrypoint | ignored legacy tests | yes | processed/log writes | `src/market_scanner/analysis/` | downstream of old quality artifact | defer until fundamentals quality parity is resolved | P2 |
| `scripts/core/build_validation_layer.py` | NO_FUNDAMENTALS_RELEVANCE | scanner, validation, data_write, entrypoint | ignored legacy tests | yes | processed/log writes | `src/market_scanner/scanner/`, `src/market_scanner/analysis/` | outside fundamentals scope | handle under scanner/validation cleanup | P2 |

## High-risk file findings

### `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

This is the highest-risk fundamentals file. It can call SEC/EDGAR through `urlopen`, requires a descriptive User-Agent for downloads, creates a local cache under `data/local/sec_edgar/companyfacts`, writes a ZIP and manifest, and performs temp-file replacement. Useful logic includes SEC URL validation, User-Agent requirement, local ZIP validation, hash/manifest metadata, and fail-closed local inspection. The live download and cache write authority must not be migrated until a canonical live provider/SEC boundary is governed.

### `scripts/fundamentals/sec_companyfacts_transform.py`

This file contains useful SEC Company Facts transformation logic not fully represented in canonical v2. It maps SEC tags, validates units, handles conflicting facts, preserves skipped-fact evidence, derives total debt and FreeCashFlow only when components are present, and emits review notes. Canonical v2 already handles generalized provider field mapping and governed FreeCashFlow derivation, but not full SEC fact-selection parity or skipped-fact evidence.

### `scripts/data_sources/prefill_fundamentals.py` and `scripts/data_sources/common.py`

These files contain useful governed prefill and path-validation ideas, but they can write `data/raw/fundamentals.csv`. Canonical persistence currently allows controlled synthetic/tmp-path writes and blocks production roots. A production-approved source artifact write path needs a separate policy before this logic can migrate.

### `scripts/fundamentals/build_quality.py`

This file is the old quality/profile builder. It writes processed/log artifacts by default and mixes source readiness, profile labels, freshness checks, and compatibility inputs. Canonical v2 already has neutral readiness and missingness, but not parity with these profile outputs. Migration should avoid importing scoring, eligibility, or decision semantics and should preserve evidence limitations.

## Canonical migration candidates

Highest-value migration candidates after governance:

1. SEC URL/User-Agent validation and local ZIP manifest metadata from `sec_companyfacts_bulk_intake.py`, without live download execution.
2. SEC Company Facts tag mapping, unit validation, conflict/skipped-fact evidence, and local transformation from `sec_companyfacts_transform.py`.
3. Ticker/CIK source mapping from `sec_ticker_cik_index.py`.
4. Fundamentals history schema validation from `build_history_intake.py`.
5. Ratio/growth metric formulas from `build_metrics.py`, only where they do not duplicate already-governed canonical growth evidence.
6. Governed source prefill audit/path policy from `scripts/data_sources/common.py` and `prefill_fundamentals.py`, only after production path policy is approved.

## Archive candidates

Archive after canonical parity:

- `scripts/fundamentals/__init__.py`
- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/fundamentals/run_sec_transformation_review.py`

These are wrappers or orchestration/review files that should not remain permanent authorities once canonical owners and tests exist.

## Delete candidates

No fundamentals/provider file is recommended for deletion now.

Deletion may be considered later only after:

- canonical parity is proven;
- active references are absent;
- archive necessity is reviewed;
- no useful migration evidence remains.

## Do-not-touch files

No fundamentals-specific file was classified as `DO_NOT_TOUCH_YET`.

However, `scripts/fundamentals/sec_companyfacts_bulk_intake.py` is a P0 side-effect review file and must not be modified until live provider/SEC governance is approved.

## Canonical parity gaps

Canonical parity gaps before archive/delete:

- no governed live SEC/provider client;
- no canonical SEC Company Facts bulk archive/cache/manifest owner;
- no canonical ticker/CIK index owner;
- no canonical SEC Company Facts local JSON transformation with conflict/skipped-fact evidence parity;
- no canonical local real-source capture CLI;
- no approved production persistence/write path for source artifacts;
- no full old `fundamental_quality.csv` profile parity;
- no final decision on whether old ratio metrics belong in fundamentals or analysis;
- no first-class analysis ingestion of governed growth evidence records;
- no governed EPS prior-year growth evidence policy.

## Recommended migration sequence

1. Govern the canonical live provider/SEC boundary before moving any fetch, download, cache, User-Agent, or network behavior.
2. Separate provider-fetch execution from normalization and persistence. Start with local/static SEC shape validation and manifest metadata only.
3. Migrate reusable source-shape and normalization logic into `src/market_scanner/fundamentals/` only where canonical parity is not already present.
4. Keep live provider calls behind explicit approval, dependency injection, dry-run/smoke guardrails, and no-production-write defaults.
5. Migrate SEC Company Facts local transformation and ticker/CIK mapping only after the live/source boundary is governed.
6. Review old metrics/quality/profile outputs separately and decide which belong in fundamentals versus analysis.
7. Migrate or retire bulk intake and data-write scripts only after production path policy is governed.
8. Archive script-era fundamentals files only after canonical parity, dependency checks, and safe tests prove no active legacy ownership remains.

## Recommended next sprint

Proceed to `RESET-10L-BL47 - Govern Canonical Fundamentals Live Provider Boundary`.

Canonical v2 currently supports injected provider responses and controlled smoke reviews, but it does not yet govern live provider/SEC fetch execution, credential/User-Agent rules, cache/write behavior, or production path restrictions. That governance should happen before any migration from `sec_companyfacts_bulk_intake.py`.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
- No script-era files executed.
- No archived scripts executed.
- No SEC/EDGAR calls made.
- No yfinance/provider calls made.
- No network calls.
- No credentials read.
- No production data writes.
- No raw payloads written.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery.
- No portfolio/watchlist updates.
- No Decision Engine behavior changed.
- No BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior added.

## Known limitations

- This review used static inspection only and did not prove functional parity.
- Script-era tests remain inactive blockers after BL45; their historical assertions need domain-by-domain migration review.
- The review does not approve live provider calls, SEC/EDGAR calls, production writes, or archive/delete actions.
- Portfolio metadata prefill was found through data-source searches but is outside canonical fundamentals ownership.
