# BL98 — Archive data_sources script-era modules

Status: COMPLETED

## Purpose

BL98 archives the script-era `scripts/data_sources` modules after BL97 decoupled active tests from them.

Targeted script-era modules:

```text
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

BL98 is an archive sprint. It moves the targeted files into `archive/legacy_runtime/` and does not modify their runtime behavior.

## Pre-archive checks

### Target files before archive

BL98 confirmed the active script-era data-source files before archive:

```text
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

### Active import and path check

BL98 checked for active imports and path references:

```bash
grep -RIn \
  "from scripts.data_sources\|import scripts.data_sources\|scripts/data_sources" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result:

```text
No output
```

Interpretation:

* no active positive imports remain from `scripts.data_sources`;
* no active positive path references remain to `scripts/data_sources` in `src`, `tests`, or `.github`.

## Archived files

BL98 moved the targeted files:

```text
scripts/data_sources/common.py -> archive/legacy_runtime/scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py -> archive/legacy_runtime/scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py -> archive/legacy_runtime/scripts/data_sources/prefill_portfolio_metadata.py
```

The archive location now contains:

```text
archive/legacy_runtime/scripts/data_sources/common.py
archive/legacy_runtime/scripts/data_sources/prefill_fundamentals.py
archive/legacy_runtime/scripts/data_sources/prefill_portfolio_metadata.py
```

## Post-archive active folder check

BL98 confirmed that active `scripts/data_sources/` no longer contains Python files.

Result:

```text
No active scripts/data_sources/*.py files remain.
```

## Validation

Focused suite:

```bash
pytest tests/data_sources/test_prefill_common.py \
       tests/data_sources/test_prefill_fundamentals.py \
       tests/data_sources/test_prefill_portfolio_metadata.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
15 passed in 0.03s
```

Full suite:

```bash
pytest -q
```

Result:

```text
569 passed in 0.57s
```

## Decision

BL98 decision:

```text
ARCHIVED
```

The targeted script-era data-source modules are now archived under:

```text
archive/legacy_runtime/scripts/data_sources/
```

## Impact

After BL98:

* active `scripts/data_sources/*.py` no longer exists;
* active tests no longer import `scripts.data_sources`;
* active `src`, `tests`, and `.github` no longer reference `scripts/data_sources`;
* historical script-era data-source implementation remains preserved under `archive/legacy_runtime/`.

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
* No script-era data-source module was modified.
* No script-era data-source module was executed.
* Files were archived, not deleted.

## Recommended next sprint

Recommended next sprint:

```text
BL99 — Review remaining active scripts tree after data_sources archive
```

Goal:

* inspect remaining active `scripts/**/*.py` files after BL92, BL95, and BL98;
* confirm that `scripts/fundamentals`, `scripts/reporting`, `scripts/telegram`, and `scripts/data_sources` are now archived;
* identify the safest next decoupling domain;
* avoid runtime execution;
* avoid production writes or provider calls.

Likely next candidates:

```text
scripts/ops/capture_historical_evidence.py
scripts/diagnostics/audit_data_coverage.py
```

or a bounded review of selected `scripts/core` layer builders.
