# BL102 — Review remaining active scripts tree after ops archive

Status: COMPLETED

## Purpose

BL102 reviews the remaining active `scripts/` tree after:

* BL92 archived `scripts/fundamentals/`;
* BL95 archived `scripts/reporting/` and `scripts/telegram/`;
* BL98 archived `scripts/data_sources/`;
* BL101 archived `scripts/ops/`.

BL102 is a review-only sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

## Remaining active script-era Python files

BL102 confirmed that 24 active Python files remain under `scripts/`:

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

The following active script-era domains no longer contain Python runtime files:

```text
scripts/fundamentals/
scripts/reporting/
scripts/telegram/
scripts/data_sources/
scripts/ops/
```

## Active import findings

BL102 found active test imports from remaining script-era modules.

### Core tests

```text
tests/core/test_build_entry_quality_backfill.py
tests/core/test_build_stability_layer.py
tests/core/test_build_context_layer.py
tests/core/test_entry_quality.py
tests/core/test_build_timing_state_layer.py
tests/core/test_build_validation_layer.py
tests/core/test_build_context_backfill.py
tests/core/test_decision_engine.py
tests/core/test_build_portfolio_intelligence.py
```

These still import modules from `scripts.core`.

### Portfolio test

```text
tests/portfolio/test_portfolio_source_contract.py
```

This still imports:

```text
scripts.portfolio.build_portfolio
scripts.core.build_portfolio_intelligence
```

### Negative/static guardrail references

Some remaining `scripts` references are negative/static guardrails only, for example:

```text
tests/unit/test_v2_canonical_app.py
tests/test_operator_visibility.py
tests/diagnostics/test_audit_data_coverage.py
tests/ops/test_capture_historical_evidence.py
```

These do not represent active script-era runtime imports.

## Canonical metadata findings

BL102 found canonical metadata references to script-era files:

```text
src/market_scanner/decision/decision_boundary.py:
  scripts/core/decision_engine.py

src/market_scanner/scanner/scanner_boundary.py:
  scripts/core/data_fetcher.py
  scripts/core/scanner.py
```

Interpretation:

* these are metadata/static references, not necessarily active runtime imports;
* they remain positive script-era file references;
* Decision Engine and scanner/provider files remain high-risk and should not be archived casually.

## Side-effect and runtime-risk findings

BL102 found side-effect and runtime-risk markers across the remaining active scripts tree, including:

```text
argparse
if __name__ == "__main__"
Path(...)
mkdir
pd.read_csv(...)
to_csv(...)
write_text(...)
write(...)
urllib
requests
yfinance
```

Key risk clusters:

* `scripts/core/data_fetcher.py` uses yfinance/provider access.
* `scripts/core/scanner.py` remains scanner/provider-sensitive.
* `scripts/core/decision_engine.py` remains Decision Engine authority-sensitive.
* `scripts/core/build_entry_quality_backfill.py` includes yfinance-related behavior and output/log writes.
* `scripts/validate_scans.py` includes yfinance and CSV output behavior.
* `scripts/portfolio/*` remains portfolio-state and transaction-sensitive.
* `scripts/watchlist/*` remains watchlist-state and command/action-sensitive.

## Domain classification after BL102

| Domain                                                     |                     Active dependency status | Risk level             | BL102 decision                        |
| ---------------------------------------------------------- | -------------------------------------------: | ---------------------- | ------------------------------------- |
| `scripts/core/build_context_layer.py`                      |                           active test import | medium/high            | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_validation_layer.py`                   |                           active test import | medium/high            | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_timing_state_layer.py`                 |                           active test import | high                   | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_stability_layer.py`                    |                           active test import | high                   | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_context_backfill.py`                   |                           active test import | high/provider-adjacent | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_entry_quality_backfill.py`             |           active test import + yfinance risk | high                   | `DECOUPLE_TESTS_FIRST`                |
| `scripts/core/build_portfolio_intelligence.py`             | active test import + portfolio/decision risk | very high              | `DO_NOT_ARCHIVE_YET`                  |
| `scripts/core/decision_engine.py`                          |      active test import + metadata reference | very high              | `DO_NOT_ARCHIVE_YET`                  |
| `scripts/core/data_fetcher.py` / `scripts/core/scanner.py` |          metadata references + provider risk | very high              | `SCANNER_PROVIDER_REVIEW_REQUIRED`    |
| `scripts/portfolio/*`                                      |              active test import / state risk | very high              | `PORTFOLIO_AUTHORITY_POLICY_REQUIRED` |
| `scripts/watchlist/*`                                      |                            state/action risk | very high              | `WATCHLIST_AUTHORITY_POLICY_REQUIRED` |
| `scripts/validate_scans.py`                                |                         yfinance/output risk | high                   | `SCANNER_VALIDATION_REVIEW_REQUIRED`  |

## Validation

Full regression suite:

```bash
pytest -q
```

Result:

```text
581 passed in 0.58s
```

## Decision

BL102 decision:

```text
REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE
```

Reasons:

* 24 active script-era Python files remain.
* Active tests still import script-era modules from `scripts.core` and `scripts.portfolio`.
* Canonical boundary metadata still references script-era Decision Engine and scanner/provider files.
* Side-effect markers remain across core, scanner/provider access, Decision Engine, portfolio, watchlist, and scan validation surfaces.
* Portfolio, watchlist, scanner/provider, and Decision Engine files are high-risk and require dedicated authority policy before archive.

## Recommended next sprint

Recommended next sprint:

```text
BL103 — Decouple selected core layer tests from script-era modules
```

Candidate tests:

```text
tests/core/test_build_context_layer.py
tests/core/test_build_validation_layer.py
tests/core/test_entry_quality.py
tests/core/test_build_timing_state_layer.py
tests/core/test_build_stability_layer.py
```

Candidate script-era modules:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

Reason:

* this is safer than touching Decision Engine, scanner/provider, portfolio, or watchlist;
* it reduces active script-era test coupling;
* it can convert behavior tests into static/canonical contract checks;
* it should not archive files yet.

High-risk areas to avoid in BL103:

```text
scripts/core/decision_engine.py
scripts/core/data_fetcher.py
scripts/core/scanner.py
scripts/core/build_portfolio_intelligence.py
scripts/portfolio/*
scripts/watchlist/*
scripts/validate_scans.py
```

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

