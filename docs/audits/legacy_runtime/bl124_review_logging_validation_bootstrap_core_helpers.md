# BL124 — Review logging/validation/bootstrap core helpers after indicators archive

Status: completed
Sprint type: review-only
Archive action performed: no
Runtime code changed: no

## Scope

BL124 reviewed the remaining logging/validation/bootstrap helpers under `scripts/core/`:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL124 did not archive, move, delete, fail-close, or modify runtime code.

## Out of scope

This sprint did not touch:

* archive moves
* runtime behavior
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

BL123 archived:

* `scripts/core/indicators.py`

to:

* `archive/legacy_runtime/scripts/core/indicators.py`

using `git mv`.

After BL123, the remaining active `scripts/core/` files were:

* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/log_scans.py`
* `scripts/core/scanner.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL124 reviewed the logging/validation/bootstrap cluster.

## Commands executed

### Branch setup

```bash id="vyhwev"
git checkout main
git pull origin main
git status
git checkout -b bl124-review-logging-validation-bootstrap-core-helpers
git status
```

### Existence and active import check

```bash id="d61zyd"
test -f scripts/core/log_scans.py
test -f scripts/core/validate_scans.py
test -f scripts/core/validator.py

grep -RInE \
  "scripts\.core\.(log_scans|validate_scans|validator)|from scripts\.core import (log_scans|validate_scans|validator)|import scripts\.core\.(log_scans|validate_scans|validator)|from scripts\.core|import scripts\.core" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* all three target files exist;
* no active import references:

  * `scripts.core.log_scans`
  * `scripts.core.validate_scans`
  * `scripts.core.validator`
* the only active positive `scripts.core` import remains:

```text id="a4x5pf"
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Other matches were static string assertions preventing script-era imports.

## Broad reference findings

The broad reference scan found only:

* direct definitions inside the target files;
* test-local contract helpers not importing these modules;
* historical audit/backlog references;
* references to the separate root-level `scripts/validate_scans.py`.

No active runtime import or test import of the three BL124 target modules was found.

## Runtime / write-risk findings

### `scripts/core/log_scans.py`

Findings:

* reads `data/processed/scanner_ranked.csv`;
* reads `data/processed/market_regime.csv`;
* writes/appends to `data/logs/scans_log.csv`;
* creates log directory via `mkdir(...)`;
* creates a log file with `to_csv(...)`;
* appends rows via `to_csv(..., mode="a", header=False, index=False)`;
* has direct execution via `if __name__ == "__main__"`.

Classification:

```text id="zc376l"
FAIL_CLOSE_REQUIRED_BEFORE_ARCHIVE
```

Rationale:

Although no active import was found, the file has direct execution and production log-write behavior. It should be fail-closed before archive.

### `scripts/core/validate_scans.py`

Findings:

* reads configured scan log data;
* reads ticker price/indicator files from processed data;
* writes `data/processed/validation_results.csv`;
* creates output directory via `mkdir(...)`;
* defines `validate_scans()`;
* defines `main()`;
* has direct execution via `if __name__ == "__main__"`.

Classification:

```text id="vnncb3"
FAIL_CLOSE_REQUIRED_BEFORE_ARCHIVE
```

Rationale:

Although no active import was found, the file has direct execution and production processed-data write behavior. It should be fail-closed before archive.

### `scripts/core/validator.py`

Findings:

* imports configuration paths;
* creates directories via `mkdir(...)`;
* checks the tickers file;
* creates `SCANS_LOG_FILE` using `open(..., "w", ...)` and `csv.writer(...)`;
* defines `validate_inputs()`;
* has no direct `main()`;
* has no direct `if __name__ == "__main__"`;
* no active import was found.

Classification:

```text id="bc4l9a"
CONTROLLED_ARCHIVE_APPROVED
```

Rationale:

The file contains bootstrap/write behavior, but no active import and no direct entrypoint. It may be archived with `git mv`, provided historical source is preserved and tests remain green.

## Validation

Operator visibility:

```text id="pgjz6j"
8 passed
```

Full suite:

```text id="izmyyu"
667 passed
```

## Decision

BL124 does not approve direct archive of `log_scans.py` or `validate_scans.py`.

BL124 approves one cleanup execution sprint.

Decision:

```text id="m9d6bd"
BL125 cleanup execution sprint: APPROVED
```

Recommended BL125 scope:

* fail-close `scripts/core/log_scans.py`;
* fail-close `scripts/core/validate_scans.py`;
* controlled archive `scripts/core/validator.py`.

## Required next sprint

Approved next sprint:

```text id="e3v67g"
BL125 — Clean up logging/validation/bootstrap core helpers
```

BL125 must:

* preserve historical source bodies for fail-closed modules;
* make public/manual execution fail closed for `log_scans.py`;
* make public/manual execution fail closed for `validate_scans.py`;
* move `scripts/core/validator.py` to `archive/legacy_runtime/scripts/core/validator.py` using `git mv`;
* not modify scanner/provider runtime;
* not modify Decision Engine behavior;
* not perform live provider calls;
* not perform production data writes.

## Final BL124 result

```text id="u5b1e7"
BL125 cleanup execution sprint: APPROVED
```
