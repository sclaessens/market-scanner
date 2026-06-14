# BL123 — Controlled archive of script-era indicators helper

Status: completed
Sprint type: controlled archive
Archive action performed: yes

## Scope

BL123 archived only:

* `scripts/core/indicators.py`

to:

* `archive/legacy_runtime/scripts/core/indicators.py`

using `git mv`.

The archived file body was not modified.

## Out of scope

This sprint did not touch:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/decision_engine.py`
* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`
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
* scan validation runtime
* Decision Engine authority
* trade command parser
* portfolio command processing
* any runtime behavior

## Background

BL122 reviewed archive-readiness for:

* `scripts/core/indicators.py`

BL122 found that:

* no active import from `src`, `tests`, `.github`, or `scripts` references `scripts.core.indicators`;
* the only remaining active positive `scripts.core` import is `tests/core/test_decision_engine.py` importing `scripts.core.decision_engine`;
* `scripts/core/indicators.py` has no yfinance/provider/network markers;
* `scripts/core/indicators.py` has no runtime entrypoint;
* `scripts/core/indicators.py` has no production data/log write behavior;
* `scripts/core/indicators.py` is a pure pandas helper that computes moving averages, ATR14, 20-day high/low, and 20-day average volume.

BL122 approved controlled archive in BL123.

## Commands executed

### Branch setup

```bash
git checkout main
git pull origin main
git status
git checkout -b bl123-controlled-archive-script-era-indicators-helper
git status
```

### Pre-archive checks

```bash
test -f scripts/core/indicators.py
test ! -f archive/legacy_runtime/scripts/core/indicators.py

grep -RInE \
  "scripts\.core\.indicators|from scripts\.core import indicators|import scripts\.core\.indicators" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc" || true
```

Result summary:

* `scripts/core/indicators.py` existed before the archive.
* `archive/legacy_runtime/scripts/core/indicators.py` did not exist before the archive.
* No active import from `src`, `tests`, `.github`, or `scripts` referenced `scripts.core.indicators`.

## Archive action

```bash
mkdir -p archive/legacy_runtime/scripts/core
git mv scripts/core/indicators.py archive/legacy_runtime/scripts/core/indicators.py
```

The file was moved with `git mv`, preserving historical source content.

## Post-archive checks

```bash
test ! -f scripts/core/indicators.py
test -f archive/legacy_runtime/scripts/core/indicators.py

find scripts/core -maxdepth 1 -type f -name "*.py" | sort

grep -RInE \
  "scripts\.core\.indicators|from scripts\.core import indicators|import scripts\.core\.indicators" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc" || true
```

Result summary:

* `scripts/core/indicators.py` no longer exists.
* `archive/legacy_runtime/scripts/core/indicators.py` exists.
* No active import from `src`, `tests`, `.github`, or `scripts` references `scripts.core.indicators`.

## Remaining active scripts/core inventory

After the archive, the active `scripts/core/` tree contains 6 Python files:

```text
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
scripts/core/validator.py
```

## Archived file verification

The archived file still contains the original pure pandas helper:

* `MIN_REQUIRED_ROWS = 20`
* `add_indicators(df: pd.DataFrame) -> pd.DataFrame`

It computes:

* `MA20`
* `MA50`
* `MA200`
* `ATR14`
* `20D_HIGH`
* `20D_LOW`
* `AVG_VOL_20`

No archived file body changes were made.

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

BL123 completed the controlled archive successfully.

Decision:

```text
BL123 controlled archive of script-era indicators helper: PASSED
```

## Recommended next sprint

The remaining active `scripts/core/` tree now contains:

* Decision Engine authority:

  * `scripts/core/decision_engine.py`

* scanner/provider-risk modules:

  * `scripts/core/data_fetcher.py`
  * `scripts/core/scanner.py`

* logging/validation/bootstrap modules:

  * `scripts/core/log_scans.py`
  * `scripts/core/validate_scans.py`
  * `scripts/core/validator.py`

Recommended next sprint:

```text
BL124 — Review logging/validation/bootstrap core helpers after indicators archive
```

Recommended BL124 scope:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL124 should be review-only.

## Final BL123 result

```text
BL124 logging/validation/bootstrap review: APPROVED
```
