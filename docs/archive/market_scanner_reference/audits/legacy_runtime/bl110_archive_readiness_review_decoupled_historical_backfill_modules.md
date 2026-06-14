# BL110 — Archive-readiness review for decoupled historical backfill modules

## Sprint type

Review-only archive-readiness sprint.

No code changes, no archive moves, no runtime changes, no provider calls, no production data writes, no report generation, no Telegram delivery, and no Decision Engine / portfolio / watchlist / scanner-provider changes were in scope.

## Scope

Target modules:

- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_context_backfill.py`

Related tests reviewed:

- `tests/core/test_build_entry_quality_backfill.py`
- `tests/core/test_build_context_backfill.py`
- `tests/test_operator_visibility.py`

## Context

BL109 decoupled the historical backfill tests from script-era imports. The purpose of BL110 was to determine whether the two decoupled historical backfill modules are ready for a controlled archive sprint.

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
Active import check
grep -RInE \
  "^[[:space:]]*(from scripts\.core\.build_entry_quality_backfill|import scripts\.core\.build_entry_quality_backfill|from scripts\.core\.build_context_backfill|import scripts\.core\.build_context_backfill)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
Manual-run/write-risk check
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs" \
  scripts/core/build_entry_quality_backfill.py \
  scripts/core/build_context_backfill.py
Focused tests
.venv/bin/python -m pytest \
  tests/core/test_build_entry_quality_backfill.py \
  tests/core/test_build_context_backfill.py \
  tests/test_operator_visibility.py \
  -q
Full suite
.venv/bin/python -m pytest -q
Findings
Active imports

The active import grep returned no matches.

Conclusion: BL109 successfully removed active test imports from:

scripts.core.build_entry_quality_backfill
scripts.core.build_context_backfill

Remaining references are test-local contract names, operator visibility references, historical documentation, backlog references, audit references, and the script files themselves.

Manual-run/write-risk

The two target modules are not archive-ready yet.

build_entry_quality_backfill.py still contains:

fixed output/log references under data/processed and data/logs;
pd.read_csv(...);
mkdir(...);
to_csv(...);
CLI argument defaults pointing at production-like processed/log paths;
main(...);
if __name__ == "__main__".

build_context_backfill.py still contains:

fixed output/log references under data/processed and data/logs;
pd.read_csv(...);
mkdir(...);
to_csv(...);
CLI argument defaults pointing at production-like processed/log paths;
main(...);
if __name__ == "__main__".

No fail-closed marker was detected for either module.

Test results

Focused suite:

24 passed

Full suite:

628 passed
BL110 decision
BL111 archive sprint: NOT APPROVED

The two historical backfill modules are decoupled from active tests, but they still expose manual execution and write-risk behavior.

Required next sprint
BL111 — Fail-close decoupled historical backfill modules

BL111 must remain limited to:

scripts/core/build_entry_quality_backfill.py
scripts/core/build_context_backfill.py

BL111 should:

make manual execution fail-closed;
preserve historical function bodies;
avoid archiving;
avoid provider calls;
avoid production data writes;
keep canonical runtime untouched;
run focused and full tests.
Out of scope
Archiving the two modules
Decision Engine
portfolio intelligence
portfolio source contract
trade command parser
scanner/provider runtime
SEC/EDGAR
yfinance calls
credentials
production data writes
report generation
Telegram delivery
watchlist state
portfolio state
