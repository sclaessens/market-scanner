# BL111 — Fail-close decoupled historical backfill modules

Sprint type: Technical safety sprint.

This was not an archive sprint, migration sprint, or runtime behavior enhancement sprint. No files were moved to `archive/`, no script-era modules were deleted, and no canonical runtime behavior was changed.

## Scope

Target modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

## Explicit out of scope

* archiving
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

## Commands executed

Initial reference scan:

```bash
grep -RInE \
  "build_entry_quality_backfill|build_context_backfill" \
  src tests .github docs scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Relevant initial output:

```text
tests/core/test_build_context_backfill.py:162:def _build_context_backfill_contract(
tests/test_operator_visibility.py:23:    "core/test_build_context_backfill.py",
tests/test_operator_visibility.py:24:    "core/test_build_entry_quality_backfill.py",
docs/... historical, audit, backlog, and legacy references
scripts/core/build_context_backfill.py:205:def build_context_backfill(
scripts/core/build_context_backfill.py:345:    output = build_context_backfill(
```

Initial target import scan:

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

Initial side-effect marker scan:

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
```

Relevant initial output:

```text
scripts/core/build_entry_quality_backfill.py:8:- data/processed/entry_quality_metrics_historical.csv
scripts/core/build_entry_quality_backfill.py:117:    df = normalize_columns(pd.read_csv(path))
scripts/core/build_entry_quality_backfill.py:460:    log_path.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_entry_quality_backfill.py:503:    output.to_csv(out_path, index=False)
scripts/core/build_entry_quality_backfill.py:512:if __name__ == "__main__":
scripts/core/build_context_backfill.py:7:    data/processed/context_strength_historical.csv
scripts/core/build_context_backfill.py:211:    scans = validate_scans_input(pd.read_csv(scans_path))
scripts/core/build_context_backfill.py:281:    output_path.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_context_backfill.py:283:    output.to_csv(output_path, index=False)
scripts/core/build_context_backfill.py:343:def main() -> None:
scripts/core/build_context_backfill.py:354:if __name__ == "__main__":
```

Initial grep conclusion:

* active imports from `src`, `tests`, and `.github` were absent for both target modules;
* both target modules still exposed manual-run/write-risk markers through `main()`, direct `if __name__ == "__main__"` entrypoints, fixed `data/processed` and `data/logs` paths, `pd.read_csv(...)`, `mkdir(...)`, and `to_csv(...)`;
* remaining documentation references are audit, backlog, historical, or legacy evidence.

## Changes made

`scripts/core/build_entry_quality_backfill.py`:

* added `FAIL_CLOSED_MESSAGE`;
* renamed the historical `main(argv)` body to `_legacy_main_impl(argv)` to preserve the historical implementation text;
* replaced `main(argv)` with immediate `SystemExit(FAIL_CLOSED_MESSAGE)`;
* replaced direct execution with immediate `SystemExit(FAIL_CLOSED_MESSAGE)`.

`scripts/core/build_context_backfill.py`:

* added `FAIL_CLOSED_MESSAGE`;
* renamed the historical `main()` body to `_legacy_main_impl()` to preserve the historical implementation text;
* replaced `main()` with immediate `SystemExit(FAIL_CLOSED_MESSAGE)`;
* replaced direct execution with immediate `SystemExit(FAIL_CLOSED_MESSAGE)`.

No pure helper functions, historical computation logic, default path definitions, provider-related internals, or output-writing helper bodies were refactored.

## Post-change grep conclusions

Target import scan:

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

Side-effect marker scan:

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|def _legacy_main_impl|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
```

Relevant post-change output:

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

Conclusion: `FAIL_CLOSED` markers are present in both modules, direct execution is fail-closed, and historical read/write markers remain only inside preserved historical functions/helpers that are not called by `main()` or direct execution.

## Direct execution fail-closed check

```bash
.venv/bin/python scripts/core/build_entry_quality_backfill.py
.venv/bin/python scripts/core/build_context_backfill.py
```

Result:

```text
FAIL_CLOSED: This legacy historical backfill module is fail-closed and must not be executed manually. It is retained only for historical review pending controlled archive governance.
```

Both commands exited immediately with code `1`. No production data writes, provider calls, report generation, Telegram delivery, or state changes were performed.

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
24 passed in 0.37s
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
628 passed in 0.99s
```

## Decision

Decision: `BL112 archive-readiness review approved for fail-closed historical backfill modules`

This decision approves only a review-only archive-readiness sprint for:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

It does not approve archive, deletion, file moves, runtime behavior changes, provider calls, production writes, report generation, Telegram delivery, or state changes.

Blockers: none for BL111 fail-close.
