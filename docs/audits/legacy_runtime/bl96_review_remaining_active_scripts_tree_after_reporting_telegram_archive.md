# BL96 — Review remaining active scripts tree after reporting and Telegram archive

Status: COMPLETED

## Purpose

BL96 reviews the remaining active `scripts/` tree after:

* BL92 archived the full `scripts/fundamentals/` runtime cluster;
* BL95 archived the active `scripts/reporting/` and `scripts/telegram/` runtime modules.

BL96 is a review-only sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

## Scope

Reviewed:

* remaining active `scripts/**/*.py` files;
* active `src`, `tests`, and `.github` imports from `scripts`;
* active/static `scripts/` path references in `src`, `tests`, `.github`, `docs/active`, and `docs/audits`;
* side-effect and runtime-risk markers under `scripts`;
* full regression suite status.

Out of scope:

* archiving any remaining script-era files;
* editing script-era runtime modules;
* executing script-era runtime modules directly;
* live provider calls;
* production data writes;
* production report generation;
* Telegram delivery;
* portfolio/watchlist mutation;
* Decision Engine authority changes.

## Remaining active script-era Python files

BL96 confirmed 28 active Python files remain under `scripts/`:

```text
scripts/core/build_context_backfill.py
scripts/core/build_context_layer.py
scripts/core/build_entry_quality_backfill.py
scripts/core/build_portfolio_intelligence.py
scripts/core/build_stability_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_validation_layer.py
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/indicators.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
scripts/core/validator.py
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
scripts/ops/capture_historical_evidence.py
scripts/portfolio/build_portfolio.py
scripts/portfolio/evaluate_positions.py
scripts/portfolio/parse_trade_commands.py
scripts/portfolio/portfolio_manager.py
scripts/validate_scans.py
scripts/watchlist/auto_watchlist_from_scan.py
scripts/watchlist/build_watchlist.py
scripts/watchlist/evaluate_watchlist.py
scripts/watchlist/parse_watchlist_commands.py
scripts/watchlist/update_watchlist_actions.py
```

Confirmed no active Python files remain under:

```text
scripts/fundamentals/
scripts/reporting/
scripts/telegram/
```

## Active imports from src/tests/.github

BL96 found active test imports from `scripts` in several remaining domains.

### Core layer builders and Decision Engine

```text
tests/core/test_build_entry_quality_backfill.py imports scripts.core.build_entry_quality_backfill
tests/core/test_build_stability_layer.py imports scripts.core.build_stability_layer
tests/core/test_build_context_layer.py imports scripts.core.build_context_layer
tests/core/test_entry_quality.py imports scripts.core.build_validation_layer
tests/core/test_build_timing_state_layer.py imports scripts.core.build_timing_state_layer
tests/core/test_build_validation_layer.py imports scripts.core.build_validation_layer
tests/core/test_build_context_backfill.py imports scripts.core.build_context_backfill
tests/core/test_decision_engine.py imports scripts.core.decision_engine
tests/core/test_build_portfolio_intelligence.py imports scripts.core.build_portfolio_intelligence
```

### Diagnostics, portfolio, ops, and data sources

```text
tests/diagnostics/test_audit_data_coverage.py imports scripts.diagnostics.audit_data_coverage
tests/portfolio/test_portfolio_source_contract.py imports scripts.portfolio.build_portfolio
tests/portfolio/test_portfolio_source_contract.py imports scripts.core.build_portfolio_intelligence
tests/ops/test_capture_historical_evidence.py imports scripts.ops.capture_historical_evidence
tests/data_sources/test_prefill_common.py imports scripts.data_sources.common
tests/data_sources/test_prefill_fundamentals.py imports scripts.data_sources.prefill_fundamentals
tests/data_sources/test_prefill_portfolio_metadata.py imports scripts.data_sources.prefill_portfolio_metadata
```

Interpretation:

* broad active test coupling remains outside the already archived fundamentals/reporting/telegram domains;
* the remaining active `scripts/` tree is not archive-ready as a whole;
* future cleanup must continue by domain-specific decoupling sprints.

## Active/static source metadata references

BL96 found canonical boundary metadata references to script-era files:

```text
src/market_scanner/decision/decision_boundary.py:
  scripts/core/decision_engine.py

src/market_scanner/scanner/scanner_boundary.py:
  scripts/core/data_fetcher.py
  scripts/core/scanner.py
```

Interpretation:

* these references are canonical metadata, not necessarily active runtime imports;
* however, they remain positive script-era file-path references and should be decoupled before archive of those domains;
* Decision Engine and scanner/provider files remain high-governance-risk and should not be archived casually.

## Side-effect and risk review

BL96 found side-effect markers across the remaining scripts tree, including:

* `argparse`
* `if __name__ == "__main__"`
* `main`
* `Path(...)`
* `mkdir`
* `pd.read_csv(...)`
* `to_csv(...)`
* `write_text(...)`
* `write(...)`
* `urllib`
* `requests`
* `yfinance`

The side-effect output confirms remaining risk clusters in:

* core layer builders;
* scanner/provider access;
* Decision Engine;
* data-source prefill utilities;
* portfolio state;
* watchlist state;
* ops evidence capture;
* diagnostics.

## Domain classification after BL96

| Domain                                        | Active imports/references? | Risk level  | BL96 decision                          |
| --------------------------------------------- | -------------------------: | ----------- | -------------------------------------- |
| `scripts/core/build_*_layer.py`               |                        yes | high        | `CORE_LAYER_DECOUPLING_REQUIRED`       |
| `scripts/core/decision_engine.py`             |                        yes | very high   | `DO_NOT_ARCHIVE_YET`                   |
| `scripts/core/data_fetcher.py` / `scanner.py` |        metadata references | high        | `SCANNER_PROVIDER_REVIEW_REQUIRED`     |
| `scripts/data_sources/`                       |                        yes | high        | `DATA_SOURCE_DECOUPLING_REQUIRED`      |
| `scripts/portfolio/`                          |                        yes | very high   | `PORTFOLIO_POLICY_REQUIRED`            |
| `scripts/watchlist/`                          |           side-effect risk | very high   | `WATCHLIST_POLICY_REQUIRED`            |
| `scripts/ops/`                                |                        yes | high        | `OPS_EVIDENCE_CAPTURE_REVIEW_REQUIRED` |
| `scripts/diagnostics/`                        |                        yes | medium/high | `DIAGNOSTICS_OWNER_REVIEW_REQUIRED`    |
| `scripts/validate_scans.py`                   |           side-effect risk | medium/high | `SCANNER_VALIDATION_REVIEW_REQUIRED`   |

## Validation

Full regression suite:

```bash
pytest -q
```

Result:

```text
560 passed in 0.57s
```

## Decision

BL96 decision:

```text
REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE
```

Reasons:

* 28 active script-era Python files remain;
* active tests still import script-era modules in core, data-source, portfolio, ops, and diagnostics domains;
* canonical boundary metadata still references script-era Decision Engine and scanner/provider files;
* side-effect markers remain across core, scanner, Decision Engine, data-source, portfolio, watchlist, ops, and diagnostics surfaces.

## Recommended next sprint

Recommended next sprint:

```text
BL97 — Decouple active data-source tests from script-era data_sources modules
```

Candidate targets:

```text
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

Candidate active tests:

```text
tests/data_sources/test_prefill_common.py
tests/data_sources/test_prefill_fundamentals.py
tests/data_sources/test_prefill_portfolio_metadata.py
```

Reason:

* the data-source cluster is small and well bounded;
* it still has active test imports;
* it contains data-write and prefill behavior, so it should be decoupled before archive;
* it is safer to decouple before touching broader core, scanner, portfolio, watchlist, or Decision Engine domains.

Likely follow-up:

```text
BL98 — Archive data_sources script-era modules after final no-active-reference check
```

Only proceed to BL98 if BL97 proves archive-readiness.

## Guardrails

* No live provider calls were run.
* No yfinance calls were run.
* No SEC/EDGAR calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist state was modified.
* No Decision Engine authority was changed.
* No script-era runtime module was archived.
* No script-era runtime module was edited.
* No script-era runtime module was executed directly.
