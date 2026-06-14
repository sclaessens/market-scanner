# BL118 — Archive-readiness review for fail-closed portfolio intelligence module

Status: completed
Sprint type: review-only / archive-readiness
Archive action performed: no

## Scope

This review covered only:

* `scripts/core/build_portfolio_intelligence.py`

BL118 did not modify runtime code, tests, production data, or archived files.

## Out of scope

This sprint did not touch:

* archive moves
* `scripts/core/decision_engine.py`
* Decision Engine authority
* scanner/provider runtime
* SEC/EDGAR integrations
* yfinance execution
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* scan validation runtime
* trade command parser
* portfolio command processing
* any other `scripts/core/` cleanup candidate

## Background

BL115 decoupled active tests from:

* `scripts.core.build_portfolio_intelligence`

BL116 reviewed archive-readiness and concluded that the module was not yet archive-ready because manual/runtime execution and write-risk markers remained.

BL117 then fail-closed the decoupled script-era module:

* added `FAIL_CLOSED_MESSAGE`;
* preserved the historical runtime body as `_legacy_build_portfolio_intelligence_impl()`;
* made public `build_portfolio_intelligence()` raise `SystemExit(FAIL_CLOSED_MESSAGE)`;
* made direct script execution raise `SystemExit(FAIL_CLOSED_MESSAGE)`.

BL118 reviewed whether the fail-closed module is now ready for controlled archive.

## Commands executed

### Branch setup

```bash id="d6txp7"
git checkout main
git pull origin main
git status
git checkout -b bl118-archive-readiness-review-fail-closed-portfolio-intelligence-module
git status
```

### Active import check

```bash id="vln4p2"
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result:

```text id="lmbnjr"
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Conclusion: there is no active import from `src`, `tests`, or `.github` to `scripts.core.build_portfolio_intelligence`.

### Fail-closed and historical-body verification

```bash id="n0shj5"
grep -nE \
  "FAIL_CLOSED|def _legacy_build_portfolio_intelligence_impl|def build_portfolio_intelligence|if __name__|read_csv|to_csv|mkdir|data/processed|data/logs|portfolio_positions|portfolio_metadata|timing_state" \
  scripts/core/build_portfolio_intelligence.py
```

Result summary:

* `FAIL_CLOSED_MESSAGE` is present.
* fixed legacy input/output paths are still preserved as historical source content.
* `pd.read_csv(...)`, `mkdir(...)`, and `to_csv(...)` remain only inside the preserved historical implementation body.
* `_legacy_build_portfolio_intelligence_impl()` is present.
* public `build_portfolio_intelligence()` raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
* direct execution via `if __name__ == "__main__"` raises `SystemExit(FAIL_CLOSED_MESSAGE)`.

### Direct execution check

```bash id="4bzfh3"
.venv/bin/python scripts/core/build_portfolio_intelligence.py || true
```

Result:

```text id="3oxk9e"
FAIL_CLOSED: scripts/core/build_portfolio_intelligence.py is a legacy script-era portfolio intelligence module. Active tests were decoupled in BL115 and archive-readiness was reviewed in BL116. Manual/runtime execution is disabled pending controlled archive review.
```

Conclusion: direct execution fails closed and does not execute the historical data-write body.

### Reference scan

```bash id="lyf48y"
grep -RInE \
  "build_portfolio_intelligence|portfolio_intelligence" \
  src tests .github scripts docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

References remain in:

* canonical portfolio source contract enum text;
* Decision Engine tests that use `portfolio_intelligence.csv` as an input/provenance artifact;
* test-local BL115 contract helpers;
* operator visibility guard;
* the fail-closed legacy module itself;
* historical audit and backlog documentation.

No active import of `scripts.core.build_portfolio_intelligence` was found.

### Focused tests

```bash id="rc7ki8"
.venv/bin/python -m pytest \
  tests/core/test_build_portfolio_intelligence.py \
  tests/portfolio/test_portfolio_source_contract.py \
  tests/test_operator_visibility.py \
  -q
```

Result:

```text id="3lz2js"
46 passed
```

### Full suite

```bash id="h5ndfa"
.venv/bin/python -m pytest -q
```

Result:

```text id="281c3i"
667 passed
```

## Archive-readiness classification

`scripts/core/build_portfolio_intelligence.py` is archive-ready for a controlled archive sprint.

Reasons:

* active tests no longer import `scripts.core.build_portfolio_intelligence`;
* `src` and `.github` do not import `scripts.core.build_portfolio_intelligence`;
* manual/public execution is fail-closed;
* direct script execution is fail-closed;
* historical runtime implementation is preserved for audit trail;
* direct execution does not write production outputs;
* focused tests remain green;
* full suite remains green.

## Decision

BL118 approves a controlled archive sprint.

Decision:

```text id="2xclp7"
BL119 controlled archive: APPROVED
```

## Required next sprint

Approved next sprint:

```text id="hahg59"
BL119 — Controlled archive of fail-closed portfolio intelligence module
```

BL119 must move only:

* `scripts/core/build_portfolio_intelligence.py`

to:

* `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`

BL119 must use `git mv`.

BL119 must not modify runtime behavior or archive any other module.

## BL119 scope recommendation

BL119 may modify only:

* `scripts/core/build_portfolio_intelligence.py`
* `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`
* BL119 audit documentation
* `docs/active/project/backlog.md`

BL119 must not touch:

* `scripts/core/decision_engine.py`
* Decision Engine authority
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* credentials
* production data
* reports
* Telegram
* portfolio state
* watchlist state
* trade command parser
* portfolio command processing
* scan validation runtime
* any other `scripts/core/` module

## Final BL118 result

```text id="owim1i"
BL119 controlled archive: APPROVED
```
