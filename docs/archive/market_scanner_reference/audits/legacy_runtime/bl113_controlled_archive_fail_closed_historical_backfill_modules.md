# BL113 — Controlled archive of fail-closed historical backfill modules

Sprint type: Controlled archive sprint.

BL113 was explicitly approved by BL112. This sprint moved only the two fail-closed historical backfill modules from the active `scripts/core/` tree into the legacy runtime archive. Historical file content was preserved.

## Scope

Archived source paths:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

Archive destination:

* `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`
* `archive/legacy_runtime/scripts/core/build_context_backfill.py`

Archive method:

* `git mv` was used for both files.

## Explicit out of scope

* Decision Engine
* portfolio intelligence
* portfolio source contract
* trade command parser
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* credentials
* production data writes
* report generation
* Telegram
* watchlist state
* portfolio state
* scan validation runtime
* any other `scripts/` cleanup candidate

## Commands executed

Pre-archive reference scan:

```bash
grep -RInE \
  "build_entry_quality_backfill|build_context_backfill" \
  src tests .github docs scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Relevant pre-archive output:

```text
tests/core/test_build_context_backfill.py:162:def _build_context_backfill_contract(
tests/test_operator_visibility.py:23:    "core/test_build_context_backfill.py",
tests/test_operator_visibility.py:24:    "core/test_build_entry_quality_backfill.py",
docs/... historical, audit, backlog, and legacy references
scripts/core/build_context_backfill.py:211:def build_context_backfill(
scripts/core/build_context_backfill.py:351:    output = build_context_backfill(
```

Pre-archive active import scan:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core\.build_entry_quality_backfill|import scripts\.core\.build_entry_quality_backfill|from scripts\.core\.build_context_backfill|import scripts\.core\.build_context_backfill)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
No matches.
```

Pre-archive fail-closed marker scan:

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|def _legacy_main_impl|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
```

Relevant pre-archive output:

```text
scripts/core/build_entry_quality_backfill.py:73:FAIL_CLOSED_MESSAGE = (
scripts/core/build_entry_quality_backfill.py:486:def _legacy_main_impl(argv: Iterable[str] | None = None) -> int:
scripts/core/build_entry_quality_backfill.py:518:def main(argv: Iterable[str] | None = None) -> int:
scripts/core/build_entry_quality_backfill.py:519:    raise SystemExit(FAIL_CLOSED_MESSAGE)
scripts/core/build_entry_quality_backfill.py:522:if __name__ == "__main__":
scripts/core/build_entry_quality_backfill.py:523:    raise SystemExit(FAIL_CLOSED_MESSAGE)
scripts/core/build_context_backfill.py:65:FAIL_CLOSED_MESSAGE = (
scripts/core/build_context_backfill.py:349:def _legacy_main_impl() -> None:
scripts/core/build_context_backfill.py:360:def main() -> None:
scripts/core/build_context_backfill.py:361:    raise SystemExit(FAIL_CLOSED_MESSAGE)
scripts/core/build_context_backfill.py:364:if __name__ == "__main__":
scripts/core/build_context_backfill.py:365:    raise SystemExit(FAIL_CLOSED_MESSAGE)
```

Pre-archive grep conclusion:

* active imports from `src`, `tests`, and `.github` were absent for both target modules;
* both active modules were fail-closed before archive;
* historical read/write markers remained preserved inside historical functions/helpers, not through direct execution;
* remaining documentation references were audit, backlog, historical, or legacy evidence.

## Archive action performed

Commands:

```bash
mkdir -p archive/legacy_runtime/scripts/core

git mv scripts/core/build_entry_quality_backfill.py \
       archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py

git mv scripts/core/build_context_backfill.py \
       archive/legacy_runtime/scripts/core/build_context_backfill.py
```

Result:

* `scripts/core/build_entry_quality_backfill.py` was moved to `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`.
* `scripts/core/build_context_backfill.py` was moved to `archive/legacy_runtime/scripts/core/build_context_backfill.py`.
* No historical code inside either file was modified.

## Post-archive verification

File existence check:

```bash
test ! -f scripts/core/build_entry_quality_backfill.py
test ! -f scripts/core/build_context_backfill.py
test -f archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py
test -f archive/legacy_runtime/scripts/core/build_context_backfill.py
```

Result:

```text
Passed.
```

Post-archive active import scan:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core\.build_entry_quality_backfill|import scripts\.core\.build_entry_quality_backfill|from scripts\.core\.build_context_backfill|import scripts\.core\.build_context_backfill)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
No matches.
```

Post-archive active tree check:

```bash
find scripts/core -maxdepth 1 -type f | sort
```

Output:

```text
scripts/core/build_portfolio_intelligence.py
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/indicators.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
scripts/core/validator.py
```

Conclusion: the two target files are no longer in the active `scripts/core/` tree.

Archive content check:

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|def _legacy_main_impl|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py \
  archive/legacy_runtime/scripts/core/build_context_backfill.py
```

Relevant output:

```text
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:73:FAIL_CLOSED_MESSAGE = (
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:486:def _legacy_main_impl(argv: Iterable[str] | None = None) -> int:
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:518:def main(argv: Iterable[str] | None = None) -> int:
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:519:    raise SystemExit(FAIL_CLOSED_MESSAGE)
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:522:if __name__ == "__main__":
archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py:523:    raise SystemExit(FAIL_CLOSED_MESSAGE)
archive/legacy_runtime/scripts/core/build_context_backfill.py:65:FAIL_CLOSED_MESSAGE = (
archive/legacy_runtime/scripts/core/build_context_backfill.py:349:def _legacy_main_impl() -> None:
archive/legacy_runtime/scripts/core/build_context_backfill.py:360:def main() -> None:
archive/legacy_runtime/scripts/core/build_context_backfill.py:361:    raise SystemExit(FAIL_CLOSED_MESSAGE)
archive/legacy_runtime/scripts/core/build_context_backfill.py:364:if __name__ == "__main__":
archive/legacy_runtime/scripts/core/build_context_backfill.py:365:    raise SystemExit(FAIL_CLOSED_MESSAGE)
```

Conclusion: archived files preserve fail-closed state and historical bodies.

## Validation

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/core/test_build_entry_quality_backfill.py \
  tests/core/test_build_context_backfill.py \
  tests/test_operator_visibility.py \
  -q
```

Result:

```text
24 passed in 0.35s
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
628 passed in 1.23s
```

## Final decision

`BL113 completed. Fail-closed historical backfill modules archived successfully.`

## Recommended next sprint

`BL114 — Review remaining active scripts/core tree after historical backfill archive`

BL114 should be review-only. It should classify remaining active script-era dependencies, active imports, side-effect risks, and candidate next sprints. It must not archive anything.
