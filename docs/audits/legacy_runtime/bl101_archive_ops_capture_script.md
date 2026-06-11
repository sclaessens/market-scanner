# BL101 — Archive ops capture script

Status: COMPLETED

## Purpose

BL101 archives the remaining active ops script-era module after BL100 decoupled active ops and diagnostics tests from script-era modules.

Targeted script-era module:

```text
scripts/ops/capture_historical_evidence.py
```

BL101 is an archive sprint. It moves the targeted file into `archive/legacy_runtime/` and does not modify its runtime behavior.

## Pre-archive checks

### Active ops files before archive

BL101 confirmed the active ops script-era file before archive:

```text
scripts/ops/capture_historical_evidence.py
```

### Active import check

BL101 checked for real active imports:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.ops|import scripts\.ops)" \
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

* no active positive imports remain from `scripts.ops`;
* the ops capture script is no longer an active test dependency.

### Active positive path/reference check

BL101 checked for references to:

```text
scripts/ops/capture_historical_evidence.py
capture_historical_evidence
```

Scope:

```text
src
tests
.github
```

Result:

```text
tests/ops/test_capture_historical_evidence.py: static legacy path reference
tests/ops/test_capture_historical_evidence.py: negative import guardrail assertion
```

Interpretation:

* remaining references are static/negative test guardrails only;
* no active runtime import, workflow invocation, or source dependency remains.

## Archived file

BL101 moved:

```text
scripts/ops/capture_historical_evidence.py -> archive/legacy_runtime/scripts/ops/capture_historical_evidence.py
```

The archive location now contains:

```text
archive/legacy_runtime/scripts/ops/capture_historical_evidence.py
```

## Post-archive active folder check

BL101 confirmed that active `scripts/ops/` no longer contains Python files.

Result:

```text
No active scripts/ops/*.py files remain.
```

## Validation

Focused suite:

```bash
pytest tests/ops/test_capture_historical_evidence.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
12 passed in 0.03s
```

Full suite:

```bash
pytest -q
```

Result:

```text
581 passed in 0.58s
```

## Decision

BL101 decision:

```text
ARCHIVED
```

The targeted script-era ops capture module is now archived under:

```text
archive/legacy_runtime/scripts/ops/
```

## Impact

After BL101:

* active `scripts/ops/*.py` no longer exists;
* active tests no longer import `scripts.ops`;
* active `src`, `tests`, and `.github` no longer contain positive runtime imports from `scripts.ops`;
* historical ops evidence-capture implementation remains preserved under `archive/legacy_runtime/`.

## Remaining cleanup status

The following script-era domains have now been fully archived from active Python runtime paths:

```text
scripts/fundamentals/
scripts/reporting/
scripts/telegram/
scripts/data_sources/
scripts/ops/
```

Remaining active script-era domains still include:

```text
scripts/core/
scripts/portfolio/
scripts/watchlist/
scripts/validate_scans.py
```

## Recommended next sprint

Recommended next sprint:

```text
BL102 — Review remaining active scripts tree after ops archive
```

Goal:

* inspect remaining active `scripts/**/*.py` files after BL92, BL95, BL98, and BL101;
* confirm that `scripts/fundamentals`, `scripts/reporting`, `scripts/telegram`, `scripts/data_sources`, and `scripts/ops` are now archived;
* identify the safest next decoupling domain;
* avoid runtime execution;
* avoid production writes or provider calls.

Likely next candidates:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
scripts/core/build_entry_quality_backfill.py
scripts/core/build_context_backfill.py
```

Decision Engine, scanner/provider, portfolio, and watchlist files should remain high-risk and should not be archived casually.

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
* No script-era ops module was modified.
* No script-era ops module was executed.
* Files were archived, not deleted.
