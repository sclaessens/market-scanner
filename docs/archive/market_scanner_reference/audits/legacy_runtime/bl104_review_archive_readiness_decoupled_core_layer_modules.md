# BL104 — Review archive-readiness of decoupled core layer modules

Status: COMPLETED

## Purpose

BL104 reviews archive-readiness of the selected core layer modules that were decoupled from active tests in BL103.

Targeted script-era modules:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

BL104 is a review-only sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

## Precondition

BL103 decoupled the active tests for these core layer contracts:

```text
tests/core/test_build_context_layer.py
tests/core/test_build_validation_layer.py
tests/core/test_entry_quality.py
tests/core/test_build_timing_state_layer.py
tests/core/test_build_stability_layer.py
```

The tests now validate static/canonical contracts instead of importing the script-era modules directly.

## Active reference check

BL104 searched for references to:

```text
build_context_layer
build_validation_layer
build_timing_state_layer
build_stability_layer
```

Scope:

```text
src
tests
.github
```

Result:

* references remain only in the static BL103 contract tests;
* references are path constants and negative import guardrail assertions;
* no active source or workflow invocation was found for the four targeted modules.

## Active import check

BL104 checked active imports from `scripts.core`.

Remaining positive imports:

```text
tests/core/test_build_entry_quality_backfill.py
tests/core/test_build_context_backfill.py
tests/core/test_decision_engine.py
tests/core/test_build_portfolio_intelligence.py
tests/portfolio/test_portfolio_source_contract.py
```

Interpretation:

* no positive imports remain for the four BL104-targeted modules;
* remaining imports are outside BL104 scope;
* remaining imports still block broader `scripts/core/` archive readiness.

## Side-effect and runnable-surface findings

BL104 found side-effect/runnable markers in all four targeted modules.

### `scripts/core/build_context_layer.py`

Findings include:

```text
Path("data/processed/scanner_ranked.csv")
Path("data/processed/sector_relative_strength.csv")
Path("data/processed/context_strength.csv")
Path("data/logs/context_layer_log.csv")
pd.read_csv(...)
mkdir(...)
to_csv(...)
if __name__ == "__main__"
```

Interpretation:

* reads processed scanner and sector-relative-strength artifacts;
* writes context output and log artifacts;
* remains manually runnable.

### `scripts/core/build_validation_layer.py`

Findings include:

```text
Path("data/processed/scanner_ranked.csv")
Path("data/processed/validation_layer.csv")
Path("data/processed/entry_quality_metrics.csv")
Path("data/logs/validation_layer_log.csv")
pd.read_csv(...)
mkdir(...)
to_csv(...)
if __name__ == "__main__"
```

Interpretation:

* reads scanner output;
* writes validation, entry-quality, and log artifacts;
* remains manually runnable.

### `scripts/core/build_timing_state_layer.py`

Findings include:

```text
Path("data/processed/fundamental_quality.csv")
Path("data/processed/entry_quality_metrics.csv")
Path("data/processed/timing_state_layer.csv")
Path("data/logs/timing_state_layer_log.csv")
pd.read_csv(...)
mkdir(...)
to_csv(...)
if __name__ == "__main__"
```

Interpretation:

* reads fundamental quality and entry-quality artifacts;
* writes timing-state and log artifacts;
* remains manually runnable.

### `scripts/core/build_stability_layer.py`

Findings include:

```text
Path(__file__)
pd.read_csv(...)
mkdir(...)
to_csv(...)
main()
if __name__ == "__main__"
```

Interpretation:

* reads final-decision-style input artifacts;
* writes stability and log artifacts;
* remains manually runnable through a `main()` entrypoint.

## Archive-readiness decision by module

| Module                                     | Active positive import? |   Active/static references? | Side-effect markers? | Runnable surface? | BL104 status                                  |
| ------------------------------------------ | ----------------------: | --------------------------: | -------------------: | ----------------: | --------------------------------------------- |
| `scripts/core/build_context_layer.py`      |                      no | static test references only |                  yes |               yes | `NOT_ARCHIVE_READY_MANUAL_RUN_AND_WRITE_RISK` |
| `scripts/core/build_validation_layer.py`   |                      no | static test references only |                  yes |               yes | `NOT_ARCHIVE_READY_MANUAL_RUN_AND_WRITE_RISK` |
| `scripts/core/build_timing_state_layer.py` |                      no | static test references only |                  yes |               yes | `NOT_ARCHIVE_READY_MANUAL_RUN_AND_WRITE_RISK` |
| `scripts/core/build_stability_layer.py`    |                      no | static test references only |                  yes |               yes | `NOT_ARCHIVE_READY_MANUAL_RUN_AND_WRITE_RISK` |

## Validation

Focused suite:

```bash
pytest tests/core/test_build_context_layer.py \
       tests/core/test_build_validation_layer.py \
       tests/core/test_entry_quality.py \
       tests/core/test_build_timing_state_layer.py \
       tests/core/test_build_stability_layer.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
35 passed in 0.07s
```

Full suite:

```bash
pytest -q
```

Result:

```text
610 passed in 0.62s
```

## Decision

BL104 decision:

```text
DECOUPLED_CORE_LAYER_MODULES_NOT_ARCHIVE_READY_DUE_TO_MANUAL_RUN_AND_WRITE_RISK
```

Although the selected modules are no longer active positive test imports, they remain risky to archive immediately because they still contain:

* fixed production-like `data/processed` and `data/logs` paths;
* CSV read/write behavior;
* log/output write behavior;
* manual runnable surfaces;
* historical runtime responsibilities that have not yet been fully retired from active `scripts/` paths.

## Recommended next sprint

Recommended next sprint:

```text
BL105 — Fail-close or de-run selected decoupled core layer modules before archive
```

Goal:

* remove or guard manual execution risk from the four selected modules;
* preserve historical implementation for later archive;
* avoid changing canonical runtime behavior;
* avoid executing script-era modules;
* rerun focused and full tests.

Candidate actions for BL105:

* convert `if __name__ == "__main__"` blocks to fail-closed behavior; or
* move runnable behavior behind explicit non-production guardrails; or
* document and prepare a controlled archive sprint if fail-closed behavior is not needed.

BL105 should not touch:

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
