# BL126 — Archive fail-closed logging/validation core helpers

Status: completed
Sprint type: controlled archive
Archive action performed: yes

## Scope

BL126 archived the fail-closed script-era logging/validation helpers:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`

to:

* `archive/legacy_runtime/scripts/core/log_scans.py`
* `archive/legacy_runtime/scripts/core/validate_scans.py`

using `git mv`.

The archived file bodies were not modified during this sprint.

## Out of scope

This sprint did not touch:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/decision_engine.py`
* scanner/provider runtime
* yfinance execution
* live provider calls
* SEC/EDGAR integrations
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* Decision Engine authority
* trade command parser
* portfolio command processing

## Background

BL125 fail-closed:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`

and archived:

* `scripts/core/validator.py`

to:

* `archive/legacy_runtime/scripts/core/validator.py`

BL125 confirmed that `log_scans.py` and `validate_scans.py` no longer expose active write-capable execution paths. BL126 performs the controlled archive of those two fail-closed modules.

## Commands executed

### Branch setup

```bash
git checkout main
git pull origin main
git status
git checkout -b bl126-archive-fail-closed-logging-validation-core-helpers
git status
```

### Pre-archive checks

```bash
test -f scripts/core/log_scans.py
test -f scripts/core/validate_scans.py
test ! -f archive/legacy_runtime/scripts/core/log_scans.py
test ! -f archive/legacy_runtime/scripts/core/validate_scans.py

grep -RInE \
  "scripts\.core\.(log_scans|validate_scans)|from scripts\.core import (log_scans|validate_scans)|import scripts\.core\.(log_scans|validate_scans)" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc" || true
```

Result summary:

* both active files existed before archive;
* no archive-path conflict existed;
* no active import from `src`, `tests`, `.github`, or `scripts` referenced:

  * `scripts.core.log_scans`
  * `scripts.core.validate_scans`

## Fail-close verification before archive

```bash
grep -nE "FAIL_CLOSED|SystemExit|legacy|disabled" scripts/core/log_scans.py scripts/core/validate_scans.py

.venv/bin/python scripts/core/log_scans.py || true
.venv/bin/python scripts/core/validate_scans.py || true
```

Result summary:

* `scripts/core/log_scans.py` contained:

  * `FAIL_CLOSED_MESSAGE`
  * preserved `_legacy_log_scans_impl()`
  * `SystemExit(FAIL_CLOSED_MESSAGE)`
* `scripts/core/validate_scans.py` contained:

  * `FAIL_CLOSED_MESSAGE`
  * preserved `_legacy_validate_scans_impl()`
  * preserved `_legacy_main_impl()`
  * `SystemExit(FAIL_CLOSED_MESSAGE)`

Direct execution results:

```text
FAIL_CLOSED: scripts/core/log_scans.py is a legacy script-era logging module. Manual/runtime execution is disabled; the historical implementation is preserved only for legacy audit and controlled archive purposes.
```

```text
FAIL_CLOSED: scripts/core/validate_scans.py is a legacy script-era validation module. Manual/runtime execution is disabled; the historical implementation is preserved only for legacy audit and controlled archive purposes.
```

## Archive action

```bash
mkdir -p archive/legacy_runtime/scripts/core
git mv scripts/core/log_scans.py archive/legacy_runtime/scripts/core/log_scans.py
git mv scripts/core/validate_scans.py archive/legacy_runtime/scripts/core/validate_scans.py
```

## Post-archive checks

```bash
test ! -f scripts/core/log_scans.py
test ! -f scripts/core/validate_scans.py
test -f archive/legacy_runtime/scripts/core/log_scans.py
test -f archive/legacy_runtime/scripts/core/validate_scans.py

find scripts/core -maxdepth 1 -type f -name "*.py" | sort

grep -RInE \
  "scripts\.core\.(log_scans|validate_scans)|from scripts\.core import (log_scans|validate_scans)|import scripts\.core\.(log_scans|validate_scans)" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc" || true
```

Result summary:

* `scripts/core/log_scans.py` no longer exists.
* `scripts/core/validate_scans.py` no longer exists.
* `archive/legacy_runtime/scripts/core/log_scans.py` exists.
* `archive/legacy_runtime/scripts/core/validate_scans.py` exists.
* no active import references either archived module.

## Remaining active scripts/core inventory

After BL126, active `scripts/core/` contains only:

```text
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/scanner.py
```

## Validation

Operator visibility:

```text
8 passed
```

Full suite:

```text
667 passed
```

## Decision

BL126 completed the controlled archive successfully.

```text
BL126 archive fail-closed logging/validation core helpers: PASSED
```

## Recommended next sprint

The remaining active `scripts/core/` files are now:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/decision_engine.py`

Recommended next sprint:

```text
BL127 — Review remaining scanner/provider core modules after logging validation archive
```

Recommended BL127 scope:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

Decision Engine remains P0 and out of scope until the scanner/provider cluster is resolved.
