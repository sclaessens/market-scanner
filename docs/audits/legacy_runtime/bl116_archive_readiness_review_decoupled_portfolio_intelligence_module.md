# BL116 — Archive-readiness review for decoupled portfolio intelligence module

Status: completed
Sprint type: review-only / archive-readiness
Archive action performed: no

## Scope

This review covered only:

* `scripts/core/build_portfolio_intelligence.py`

BL116 did not modify runtime code, tests, production data, or archived files.

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

BL115 decoupled the active portfolio intelligence tests from:

* `scripts.core.build_portfolio_intelligence`

BL115 confirmed that the only remaining active positive `scripts.core` import is:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

BL116 reviewed whether `scripts/core/build_portfolio_intelligence.py` is ready for controlled archive.

## Commands executed

### Active import check

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Manual-run / write-risk marker check

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|argparse|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs|portfolio_intelligence|portfolio_metadata|portfolio_positions|timing_state" \
  scripts/core/build_portfolio_intelligence.py
```

### Reference scan

```bash
grep -RInE \
  "build_portfolio_intelligence|portfolio_intelligence" \
  src tests .github scripts docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Focused tests

```bash
.venv/bin/python -m pytest \
  tests/core/test_build_portfolio_intelligence.py \
  tests/portfolio/test_portfolio_source_contract.py \
  tests/test_operator_visibility.py \
  -q
```

### Full suite

```bash
.venv/bin/python -m pytest -q
```

## Findings

### Active imports

The active import grep returned only:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

There is no remaining active import from `src`, `tests`, or `.github` to:

```text
scripts.core.build_portfolio_intelligence
```

### Manual-run / write-risk markers

`scripts/core/build_portfolio_intelligence.py` is not archive-ready yet.

It still contains:

* fixed input/output paths:

  * `data/processed/timing_state_layer.csv`
  * `data/portfolio/portfolio_positions.csv`
  * `data/portfolio/portfolio_metadata.csv`
  * `data/processed/portfolio_intelligence.csv`
  * `data/logs/portfolio_intelligence_log.csv`
* `pd.read_csv(...)`
* `mkdir(...)`
* `to_csv(...)`
* direct execution via `if __name__ == "__main__"`
* direct call to `build_portfolio_intelligence()` from the script entrypoint

No `FAIL_CLOSED` marker is currently present.

### Validation

Focused tests:

```text
46 passed
```

Full suite:

```text
667 passed
```

## Decision

BL116 does not approve an archive sprint.

Decision:

```text
BL117 archive sprint: NOT APPROVED
BL117 fail-close sprint: APPROVED
```

## Required next sprint

Approved next sprint:

```text
BL117 — Fail-close decoupled portfolio intelligence module
```

BL117 must make manual execution fail closed while preserving historical runtime body/content.

BL117 must not archive the module.

## BL117 scope recommendation

BL117 may modify only:

* `scripts/core/build_portfolio_intelligence.py`
* BL117 audit documentation
* `docs/active/project/backlog.md`

BL117 must not touch:

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
* any other `scripts/core/` module

## Final BL116 result

```text
BL117 fail-close sprint: APPROVED
```
