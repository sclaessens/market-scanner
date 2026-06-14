# BL93 — Review remaining active scripts tree after fundamentals archive

Status: COMPLETED

## Purpose

BL93 reviews the remaining active `scripts/` tree after BL92 archived the full `scripts/fundamentals/` runtime cluster.

BL93 is a review-only sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

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

BL93 confirmed 32 active Python files remain under `scripts/`:

```text id="1a36fq"
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
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
scripts/validate_scans.py
scripts/watchlist/auto_watchlist_from_scan.py
scripts/watchlist/build_watchlist.py
scripts/watchlist/evaluate_watchlist.py
scripts/watchlist/parse_watchlist_commands.py
scripts/watchlist/update_watchlist_actions.py
```

The active `scripts/fundamentals/` runtime cluster is no longer present after BL92.

## Active imports from src/tests/.github

BL93 found active test imports from `scripts` in several domains.

### Reporting

```text id="01eun1"
tests/reporting/test_build_telegram_summary.py imports scripts.reporting.build_reporting_layer
tests/reporting/test_build_telegram_summary.py imports scripts.reporting.build_telegram_summary
tests/reporting/test_build_reporting_layer.py imports scripts.reporting.build_reporting_layer
```

### Core layer builders and Decision Engine

```text id="mlwkih"
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

```text id="4zd7yc"
tests/diagnostics/test_audit_data_coverage.py imports scripts.diagnostics.audit_data_coverage
tests/portfolio/test_portfolio_source_contract.py imports scripts.portfolio.build_portfolio
tests/portfolio/test_portfolio_source_contract.py imports scripts.core.build_portfolio_intelligence
tests/ops/test_capture_historical_evidence.py imports scripts.ops.capture_historical_evidence
tests/data_sources/test_prefill_common.py imports scripts.data_sources.common
tests/data_sources/test_prefill_fundamentals.py imports scripts.data_sources.prefill_fundamentals
tests/data_sources/test_prefill_portfolio_metadata.py imports scripts.data_sources.prefill_portfolio_metadata
```

Interpretation:

* the remaining active `scripts/` tree still has broad active test coupling;
* it is not archive-ready as a whole;
* future cleanup must proceed by domain-specific decoupling sprints.

## Active/static source metadata references

BL93 found canonical boundary metadata references to script-era files:

```text id="y57y89"
src/market_scanner/reporting/report_boundary.py:
  scripts/reporting/build_reporting_layer.py
  scripts/reporting/build_telegram_summary.py
  scripts/reporting/send_telegram.py

src/market_scanner/delivery/delivery_boundary.py:
  scripts/reporting/send_telegram.py
  scripts/telegram/process_telegram_commands.py

src/market_scanner/decision/decision_boundary.py:
  scripts/core/decision_engine.py

src/market_scanner/scanner/scanner_boundary.py:
  scripts/core/data_fetcher.py
  scripts/core/scanner.py

src/market_scanner/messaging/message_boundary.py:
  scripts/reporting/build_reporting_layer.py
  scripts/reporting/build_telegram_summary.py
  scripts/reporting/send_telegram.py
```

Interpretation:

* some references are canonical boundary metadata, not necessarily active runtime imports;
* however, they are still positive references to script-era file paths and should be decoupled before archive of those domains;
* reporting/messaging/delivery has the broadest metadata footprint.

## Side-effect and risk review

BL93 found side-effect markers across the remaining scripts tree, including:

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

* reporting and Telegram delivery;
* scanner/provider access;
* core layer builders and Decision Engine;
* data-source prefill utilities;
* portfolio and watchlist mutation scripts;
* ops evidence capture.

## Domain classification after BL93

| Domain                                        | Active imports/references? | Risk level  | BL93 decision                          |
| --------------------------------------------- | -------------------------: | ----------- | -------------------------------------- |
| `scripts/reporting/`                          |                        yes | high        | `DECOUPLE_NEXT`                        |
| `scripts/telegram/`                           |                        yes | high        | `DECOUPLE_WITH_DELIVERY`               |
| `scripts/core/decision_engine.py`             |                        yes | very high   | `DO_NOT_ARCHIVE_YET`                   |
| `scripts/core/data_fetcher.py` / `scanner.py` |        metadata references | high        | `SCANNER_PROVIDER_REVIEW_REQUIRED`     |
| `scripts/core/build_*_layer.py`               |                        yes | high        | `CORE_LAYER_DECOUPLING_REQUIRED`       |
| `scripts/data_sources/`                       |                        yes | high        | `DATA_SOURCE_DECOUPLING_REQUIRED`      |
| `scripts/portfolio/`                          |                        yes | very high   | `PORTFOLIO_POLICY_REQUIRED`            |
| `scripts/watchlist/`                          |           side-effect risk | very high   | `WATCHLIST_POLICY_REQUIRED`            |
| `scripts/ops/`                                |                        yes | high        | `OPS_EVIDENCE_CAPTURE_REVIEW_REQUIRED` |
| `scripts/validate_scans.py`                   |           side-effect risk | medium/high | `SCANNER_VALIDATION_REVIEW_REQUIRED`   |

## Validation

Full regression suite:

```bash id="3kkxm4"
pytest -q
```

Result:

```text id="lnrb18"
553 passed in 0.58s
```

## Decision

BL93 decision:

```text id="su9aol"
REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE
```

Reasons:

* 32 active script-era Python files remain;
* active tests still import multiple script-era modules;
* canonical boundary metadata still references script-era reporting, delivery, messaging, decision, and scanner files;
* side-effect markers remain across reporting, delivery, scanner, core layers, portfolio, watchlist, data-source, and ops domains.

## Recommended next sprint

Recommended next sprint:

```text id="y13gux"
BL94 — Decouple active reporting, messaging, and delivery tests from script-era reporting and Telegram modules
```

Candidate targets:

* `scripts/reporting/build_reporting_layer.py`
* `scripts/reporting/build_telegram_summary.py`
* `scripts/reporting/send_telegram.py`
* `scripts/telegram/process_telegram_commands.py`

Candidate active tests/source references:

* `tests/reporting/test_build_telegram_summary.py`
* `tests/reporting/test_build_reporting_layer.py`
* `tests/unit/test_v2_canonical_reporting.py`
* `tests/unit/test_v2_canonical_messaging.py`
* `tests/unit/test_v2_canonical_delivery.py`
* `src/market_scanner/reporting/report_boundary.py`
* `src/market_scanner/messaging/message_boundary.py`
* `src/market_scanner/delivery/delivery_boundary.py`

Goal:

* remove active positive imports and static file reads from script-era reporting/Telegram modules;
* preserve canonical reporting/messaging/delivery governance;
* avoid sending Telegram messages;
* avoid report artifact writes;
* avoid credential reads;
* avoid executing script-era modules.

Likely follow-up:

```text id="yj5lb3"
BL95 — Archive reporting/messaging/delivery script-era modules after final no-active-reference check
```

Only proceed to BL95 if BL94 proves archive-readiness.

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
