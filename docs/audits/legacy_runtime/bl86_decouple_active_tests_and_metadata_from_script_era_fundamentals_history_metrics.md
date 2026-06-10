### BL86 — Decouple active tests and metadata references from script-era fundamentals history and metrics files

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL86 decoupled active tests and metadata references from:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Updated source metadata:

* `src/market_scanner/analysis/analysis_boundary.py`

Updated tests:

* `tests/unit/test_v2_canonical_analysis.py`
* `tests/core/test_fundamentals_runtime_organization.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `tests/core/test_build_fundamental_metrics.py`
* `tests/core/test_build_fundamentals_history_intake.py`

Result:

* Active analysis metadata no longer lists `build_history_intake.py` or `build_metrics.py` as legacy analysis authorities.
* Canonical analysis metadata now tracks migrated fundamentals contract authorities:

  * `src/market_scanner/fundamentals/fundamental_contracts.py`
  * `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`
* Active tests no longer depend on script-era history/metrics file paths.
* Core history/metrics tests now validate canonical fundamentals contracts instead of script-era modules.
* Remaining grep hits are negative guardrail assertions only.

Validation:

* focused tests: `19 passed in 0.03s`
* full suite: `550 passed in 0.57s`

Archive decision after BL86:

* `scripts/fundamentals/build_history_intake.py`: `INTERNAL_SCRIPT_DEPENDENCY_BLOCKED`
* `scripts/fundamentals/build_metrics.py`: `INTERNAL_SCRIPT_DEPENDENCY_BLOCKED`

Remaining blocker:

* internal script-era imports still exist from other `scripts/fundamentals/` files.

Recommended next sprint:

* BL87 — Review internal script-era dependencies on fundamentals history and metrics modules

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted
