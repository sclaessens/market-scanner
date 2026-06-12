# BL109 — Decouple historical backfill tests from script-era modules

Sprint type: Technical test-decoupling sprint.

This was not an archive sprint. No files were moved to `archive/`, no script-era modules were deleted, and no runtime script modules were modified.

## Scope

Script-era modules reviewed:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

Target tests updated:

* `tests/core/test_build_entry_quality_backfill.py`
* `tests/core/test_build_context_backfill.py`

Required test-harness updates:

* `tests/conftest.py`
* `tests/test_operator_visibility.py`

## Explicit out of scope

* Decision Engine
* portfolio intelligence
* portfolio source contract
* trade command parser
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* Telegram
* production data writes
* reports
* credentials
* portfolio state
* watchlist state
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
tests/conftest.py:9:    "core/test_build_context_backfill.py",
tests/conftest.py:10:    "core/test_build_entry_quality_backfill.py",
tests/core/test_build_entry_quality_backfill.py:5:from scripts.core.build_entry_quality_backfill import (
tests/core/test_build_context_backfill.py:9:from scripts.core import build_context_backfill as b
tests/test_operator_visibility.py:17:    "core/test_build_context_backfill.py",
tests/test_operator_visibility.py:18:    "core/test_build_entry_quality_backfill.py",
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
tests/core/test_build_entry_quality_backfill.py:5:from scripts.core.build_entry_quality_backfill import (
```

Additional dependency observed during inspection:

```text
tests/core/test_build_context_backfill.py:9:from scripts.core import build_context_backfill as b
```

This import shape was not matched by the stricter target-import grep, but it was identified by the broad initial reference scan and removed during BL109.

Script side-effect marker scan:

```bash
grep -RInE \
  "if __name__|def main|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
```

Relevant output:

```text
scripts/core/build_entry_quality_backfill.py:8:- data/processed/entry_quality_metrics_historical.csv
scripts/core/build_entry_quality_backfill.py:9:- data/logs/entry_quality_backfill_log.csv
scripts/core/build_entry_quality_backfill.py:117:    df = normalize_columns(pd.read_csv(path))
scripts/core/build_entry_quality_backfill.py:460:    log_path.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_entry_quality_backfill.py:503:    output.to_csv(out_path, index=False)
scripts/core/build_entry_quality_backfill.py:512:if __name__ == "__main__":
scripts/core/build_context_backfill.py:7:    data/processed/context_strength_historical.csv
scripts/core/build_context_backfill.py:8:    data/logs/context_backfill_log.csv
scripts/core/build_context_backfill.py:211:    scans = validate_scans_input(pd.read_csv(scans_path))
scripts/core/build_context_backfill.py:281:    output_path.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_context_backfill.py:283:    output.to_csv(output_path, index=False)
scripts/core/build_context_backfill.py:343:def main() -> None:
scripts/core/build_context_backfill.py:354:if __name__ == "__main__":
```

Initial grep conclusion:

* the two target tests actively depended on the historical script-era backfill modules;
* the target modules retain script-era side-effect markers including provider-adjacent behavior, `read_csv`, `to_csv`, `mkdir`, fixed `data/processed` and `data/logs` paths, and manual entrypoints;
* active script-era dependency existed only in the target tests for BL109 scope;
* historical, audit, backlog, and legacy documentation references remain evidence only.

## Changes made

`tests/core/test_build_entry_quality_backfill.py`:

* removed imports from `scripts.core.build_entry_quality_backfill`;
* added test-local contract helpers for scan input normalization, duplicate key validation, point-in-time indicators, entry-quality calculation, schema validation, enum validation, and NaN failure behavior;
* preserved the existing behavioral assertions without importing or executing the historical script module.

`tests/core/test_build_context_backfill.py`:

* removed `from scripts.core import build_context_backfill as b`;
* added test-local contract helpers for scan input validation, no-future-candle return alignment, context classification, percentile classification, output validation, and synthetic row-preserving backfill output;
* preserved tmp-path-only output and log writes for the historical output contract;
* used injected synthetic price fixtures only, with no provider calls.

`tests/conftest.py`:

* removed the two decoupled backfill tests from `_HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS` so they are active in the full suite.

`tests/test_operator_visibility.py`:

* removed the two decoupled backfill tests from the inactive high-risk blocker list;
* added a guard proving the historical backfill tests are active and do not import `scripts.*`.

## Post-change grep conclusion

Command:

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

Additional active `scripts.core` import scan after the change:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
tests/core/test_build_portfolio_intelligence.py:9:from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py:8:from scripts.core import build_portfolio_intelligence
```

Conclusion: target backfill imports are gone from active tests. Remaining active `scripts.core` imports are outside BL109 scope.

## Validation

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/core/test_build_entry_quality_backfill.py \
  tests/core/test_build_context_backfill.py \
  -q
```

Result:

```text
17 passed in 0.53s
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
628 passed in 1.13s
```

The full-suite count increased from the prior `610 passed` baseline because BL109 reactivated the two previously ignored historical backfill tests and added one operator-visibility guard confirming their decoupled active status.

## Decision

Decision: `BL110 archive-readiness review approved for the two backfill modules`

This decision approves a review-only archive-readiness sprint for:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

It does not approve archive, deletion, file moves, runtime behavior changes, provider calls, production writes, or report generation.

Blockers: none for BL109 test decoupling.
