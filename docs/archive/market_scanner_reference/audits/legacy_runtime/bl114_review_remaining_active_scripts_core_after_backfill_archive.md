# BL114 — Review remaining active scripts/core tree after historical backfill archive

Status: completed
Sprint type: review-only / inventory
Archive action performed: no

## Scope

This review inspected the remaining active `scripts/core/` tree after BL113 archived the fail-closed historical backfill modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

BL114 did not archive, move, delete, or modify runtime code.

## Out of scope

This sprint did not touch:

* Decision Engine authority
* portfolio intelligence behavior
* portfolio source contract behavior
* trade command parser
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
* any archive action
* any test rewrite

## Commands executed

### Remaining active files

```bash
find scripts/core -maxdepth 1 -type f | sort
```

```bash
find scripts/core -maxdepth 1 -type f -name "*.py" | sort
```

### Active `scripts.core` imports

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Entrypoint / side-effect marker scan

```bash
grep -RInE \
  "if __name__|def main|argparse|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs|yfinance|yf\.|requests|telegram|send_message|os\.environ|dotenv" \
  scripts/core \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Runtime/reference scan

```bash
grep -RInE \
  "scripts/core|scripts\.core" \
  src tests .github scripts docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Validation

```bash
.venv/bin/python -m pytest tests/test_operator_visibility.py -q
```

```bash
.venv/bin/python -m pytest -q
```

## Remaining active scripts/core files

BL114 confirmed that the active `scripts/core/` tree now contains 8 Python files:

* `scripts/core/build_portfolio_intelligence.py`
* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/indicators.py`
* `scripts/core/log_scans.py`
* `scripts/core/scanner.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

The two historical backfill modules archived in BL113 are no longer present in `scripts/core/`.

## Active scripts.core imports

BL114 found three remaining active positive imports from `scripts.core`:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
tests/core/test_build_portfolio_intelligence.py:9:from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py:8:from scripts.core import build_portfolio_intelligence
```

These imports block broad archive-readiness for the remaining `scripts/core/` tree.

## Side-effect and risk findings

### `scripts/core/decision_engine.py`

Classification: `P0_DO_NOT_TOUCH_YET`

Reasons:

* still actively imported by `tests/core/test_decision_engine.py`;
* owns Decision Engine / final decision semantics;
* contains processed/log writes;
* defines `main()`;
* must not be moved, archived, or refactored casually.

### `scripts/core/build_portfolio_intelligence.py`

Classification: `DECOUPLE_TESTS_FIRST`

Reasons:

* actively imported by:

  * `tests/core/test_build_portfolio_intelligence.py`
  * `tests/portfolio/test_portfolio_source_contract.py`
* reads portfolio and timing-state inputs;
* writes `data/processed/portfolio_intelligence.csv`;
* writes `data/logs/portfolio_intelligence_log.csv`;
* contains portfolio/review-adjacent behavior;
* must not be archived until active tests are decoupled and portfolio-source contract behavior is preserved.

### `scripts/core/data_fetcher.py`

Classification: `SCANNER_PROVIDER_REVIEW_REQUIRED`

Reasons:

* contains yfinance provider access;
* statically referenced by canonical scanner boundary metadata;
* must not be executed or archived without a dedicated scanner/source-access review.

### `scripts/core/scanner.py`

Classification: `SCANNER_PROVIDER_REVIEW_REQUIRED`

Reasons:

* contains yfinance access;
* includes scanner scoring / target fields;
* statically referenced by canonical scanner boundary metadata;
* must be handled under scanner/provider governance, not generic cleanup.

### `scripts/core/log_scans.py`

Classification: `SIDE_EFFECT_REVIEW_REQUIRED`

Reasons:

* reads/writes scan logs;
* creates log paths;
* has direct execution block;
* not currently shown as actively imported by tests, but still production-data-write adjacent.

### `scripts/core/validate_scans.py`

Classification: `SIDE_EFFECT_REVIEW_REQUIRED`

Reasons:

* reads scan logs;
* writes validation results;
* defines `main()`;
* has direct execution block;
* must be reconciled with root-level validation/runtime validation ownership before archive.

### `scripts/core/validator.py`

Classification: `SIDE_EFFECT_REVIEW_REQUIRED`

Reasons:

* creates directories;
* utility/bootstrap behavior may still encode path policy;
* no active test import found in BL114.

### `scripts/core/indicators.py`

Classification: `REVIEW_FOR_PURE_LOGIC_MIGRATION_OR_ARCHIVE`

Reasons:

* no active test import found in BL114;
* no write-risk markers found in the BL114 grep output;
* likely pure indicator logic, but still belongs to scanner/runtime cleanup and should not be archived without focused review.

## Test results

Operator visibility:

```text
7 passed
```

Full suite:

```text
628 passed
```

## Decision

BL114 does not approve any archive sprint.

The remaining active `scripts/core/` tree is not archive-ready because active imports remain for:

* `scripts.core.decision_engine`
* `scripts.core.build_portfolio_intelligence`

Decision Engine is explicitly blocked from cleanup.

The safest next sprint is a focused test-decoupling sprint for portfolio intelligence only.

Approved next sprint:

```text
BL115 — Decouple active portfolio intelligence tests from script-era module
```

## BL115 scope recommendation

BL115 should target only:

* `tests/core/test_build_portfolio_intelligence.py`
* `tests/portfolio/test_portfolio_source_contract.py`

and their dependency on:

* `scripts/core/build_portfolio_intelligence.py`

BL115 must not archive anything.

BL115 must not touch:

* `scripts/core/decision_engine.py`
* Decision Engine authority
* scanner/provider runtime
* yfinance
* SEC/EDGAR
* production data
* reports
* Telegram
* portfolio state
* watchlist state
* trade command parser

## Final BL114 result

```text
BL115 archive sprint: NOT APPROVED
BL115 portfolio-intelligence test decoupling sprint: APPROVED
```
