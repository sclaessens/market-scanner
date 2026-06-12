# BL112 — Archive-readiness review for fail-closed historical backfill modules

Status: completed
Sprint type: review-only / archive-readiness
Archive action performed: no

## Scope

This review covered only the following fail-closed historical backfill modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

## Out of scope

This sprint did not modify or touch:

* archive moves
* runtime code
* tests
* Decision Engine authority
* scanner/provider runtime
* SEC/EDGAR integrations
* yfinance calls
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* scan validation runtime
* portfolio intelligence
* trade command parser
* portfolio source contract

## Background

BL109 decoupled active historical backfill tests from the script-era modules.

BL110 reviewed archive-readiness and found that both modules were not archive-ready because they still exposed manual-run/write-risk markers.

BL111 made both modules fail-closed:

* `FAIL_CLOSED_MESSAGE` is present in both modules.
* Public `main()` raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
* Direct script execution via `if __name__ == "__main__"` raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
* Historical `main()` bodies are preserved under `_legacy_main_impl(...)`.
* Historical read/write markers remain only in preserved legacy logic.

## Commands executed

### Reference inventory

```bash
grep -RInE \
  "build_entry_quality_backfill|build_context_backfill" \
  src tests .github docs scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Active import check

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core\.build_entry_quality_backfill|import scripts\.core\.build_entry_quality_backfill|from scripts\.core\.build_context_backfill|import scripts\.core\.build_context_backfill)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Fail-closed / write-risk marker check

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|def _legacy_main_impl|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
```

### Direct execution check

```bash
.venv/bin/python scripts/core/build_entry_quality_backfill.py || true
.venv/bin/python scripts/core/build_context_backfill.py || true
```

### Focused tests

```bash
.venv/bin/python -m pytest \
  tests/core/test_build_entry_quality_backfill.py \
  tests/core/test_build_context_backfill.py \
  tests/test_operator_visibility.py \
  -q
```

### Full suite

```bash
.venv/bin/python -m pytest -q
```

## Findings

### Active imports

The active import grep returned no matches from:

* `src`
* `tests`
* `.github`

No active runtime or active test path imports:

* `scripts.core.build_entry_quality_backfill`
* `scripts.core.build_context_backfill`

### Fail-closed state

Both target modules contain a `FAIL_CLOSED_MESSAGE`.

Both public `main()` functions now fail closed immediately.

Both direct execution blocks now fail closed immediately.

The preserved historical write-risk markers are no longer reachable through direct script execution.

### Direct execution result

Both scripts exited immediately with the fail-closed message:

```text
FAIL_CLOSED: This legacy historical backfill module is fail-closed and must not be executed manually. It is retained only for historical review pending controlled archive governance.
```

No production data writes were performed.

No provider calls were performed.

No credentials were read.

### Focused test result

```text
24 passed
```

### Full test result

```text
628 passed
```

## Decision

BL112 approves the next sprint as a controlled archive sprint.

Approved next sprint:

```text
BL113 — Controlled archive of fail-closed historical backfill modules
```

This is approval for a scoped archive sprint only.

BL112 itself did not archive, move, delete, or modify runtime code.

## BL113 scope recommendation

BL113 may archive only:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

Expected archive destination:

* `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`
* `archive/legacy_runtime/scripts/core/build_context_backfill.py`

BL113 must preserve history with `git mv`.

BL113 must not broaden scope to other `scripts/` files.

BL113 must run focused and full tests after archive.

## Final BL112 result

```text
BL113 controlled archive sprint: APPROVED
```
