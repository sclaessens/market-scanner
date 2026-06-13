# BL125 — Clean up logging/validation/bootstrap core helpers

## Sprint Status

Completed.

## Files Changed

- `scripts/core/log_scans.py`
- `scripts/core/validate_scans.py`
- `scripts/core/validator.py`
- `archive/legacy_runtime/scripts/core/validator.py`
- `docs/audits/legacy_runtime/bl125_cleanup_logging_validation_bootstrap_core_helpers.md`
- `docs/active/project/backlog.md`

## Scope

- Fail-close `scripts/core/log_scans.py`.
- Fail-close `scripts/core/validate_scans.py`.
- Archive `scripts/core/validator.py` to `archive/legacy_runtime/scripts/core/validator.py` using `git mv`.

## Explicit Out of Scope

- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`
- `scripts/core/decision_engine.py`
- scanner/provider runtime
- yfinance execution
- live provider calls
- SEC/EDGAR
- credentials
- production data writes
- report generation
- Telegram delivery
- portfolio state
- watchlist state
- trade command parser
- Decision Engine behavior

## Pre-Change Active Import Check

Command:

```bash
grep -RInE "scripts\.core\.(log_scans|validate_scans|validator)|from scripts\.core import (log_scans|validate_scans|validator)|import scripts\.core\.(log_scans|validate_scans|validator)" tests src .github scripts --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude="*.pyc"
```

Result:

```text
No matches.
```

## Fail-Close Behavior for `log_scans.py`

- Added `FAIL_CLOSED_MESSAGE`.
- Preserved the historical logging body as `_legacy_log_scans_impl()`.
- Public `log_scans()` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- Direct execution through `if __name__ == "__main__"` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- Historical write markers remain only inside the preserved legacy implementation path.

Direct execution check:

```bash
.venv/bin/python scripts/core/log_scans.py
```

Result:

```text
Exit code: 1
FAIL_CLOSED: scripts/core/log_scans.py is a legacy script-era logging module. Manual/runtime execution is disabled; the historical implementation is preserved only for legacy audit and controlled archive purposes.
```

This exits before the legacy implementation can write to `data/logs/scans_log.csv`.

## Fail-Close Behavior for `validate_scans.py`

- Added `FAIL_CLOSED_MESSAGE`.
- Preserved the historical validation body as `_legacy_validate_scans_impl()`.
- Preserved the historical CLI body as `_legacy_main_impl()`.
- Public `validate_scans()` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- Public `main()` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- Direct execution through `if __name__ == "__main__"` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- Historical write markers remain only inside the preserved legacy implementation path.

Direct execution check:

```bash
.venv/bin/python scripts/core/validate_scans.py
```

Result:

```text
Exit code: 1
FAIL_CLOSED: scripts/core/validate_scans.py is a legacy script-era validation module. Manual/runtime execution is disabled; the historical implementation is preserved only for legacy audit and controlled archive purposes.
```

This exits before the legacy implementation can write to `data/processed/validation_results.csv`.

## Archive Action for `validator.py`

Command:

```bash
git mv scripts/core/validator.py archive/legacy_runtime/scripts/core/validator.py
```

Result:

- `scripts/core/validator.py` no longer exists.
- `archive/legacy_runtime/scripts/core/validator.py` exists.
- The archived `validator.py` body was not modified.

## Post-Change Active Import Check

Command:

```bash
grep -RInE "scripts\.core\.(log_scans|validate_scans|validator)|from scripts\.core import (log_scans|validate_scans|validator)|import scripts\.core\.(log_scans|validate_scans|validator)" tests src .github scripts --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude="*.pyc"
```

Result:

```text
No matches.
```

## Remaining Active `scripts/core` Inventory

```text
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
```

## Validation Commands and Results

```bash
test ! -f scripts/core/validator.py
test -f archive/legacy_runtime/scripts/core/validator.py
find scripts/core -maxdepth 1 -type f -name "*.py" | sort
.venv/bin/python -m pytest tests/test_operator_visibility.py -q
.venv/bin/python -m pytest -q
```

Results:

```text
operator visibility: 8 passed in 0.02s
full suite: 667 passed in 1.16s
```

## Next Recommended Sprint

`BL126 — Archive-readiness review for fail-closed logging and validation helpers`

BL126 should be review-only. It should verify that `scripts/core/log_scans.py` and `scripts/core/validate_scans.py` have no active imports, fail closed on public/manual execution, preserve historical implementations, and are either approved or blocked for a later controlled archive sprint.
