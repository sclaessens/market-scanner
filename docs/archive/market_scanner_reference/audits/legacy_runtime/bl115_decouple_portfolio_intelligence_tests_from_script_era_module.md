# BL115 — Decouple active portfolio intelligence tests from script-era module

Sprint type: Test decoupling / legacy dependency removal sprint.

This was not an archive sprint, runtime migration sprint, portfolio behavior enhancement sprint, or Decision Engine sprint.

## Scope

Updated tests:

* `tests/core/test_build_portfolio_intelligence.py`
* `tests/portfolio/test_portfolio_source_contract.py`

Required test-harness updates:

* `tests/conftest.py`
* `tests/test_operator_visibility.py`

Explicit target dependency:

* `scripts.core.build_portfolio_intelligence`

## Explicit out of scope

* archiving
* `scripts/core/build_portfolio_intelligence.py` runtime behavior
* `scripts/core/decision_engine.py`
* Decision Engine authority
* `scripts/portfolio/build_portfolio.py`
* `scripts/portfolio/parse_trade_commands.py`
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* credentials
* production data writes
* reports
* Telegram
* portfolio state
* watchlist state
* scan validation runtime

## Initial grep findings

Command:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|from scripts\.core import decision_engine|from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
tests/core/test_build_portfolio_intelligence.py:9:from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py:8:from scripts.core import build_portfolio_intelligence
```

Conclusion: BL115 target imports were present only in the two portfolio-intelligence tests. The Decision Engine import remains explicitly out of scope.

Additional inspection confirmed:

* `tests/core/test_build_portfolio_intelligence.py` covered timing input validation, portfolio source handling, metadata status/reason handling, row preservation, output/log schema, forbidden semantic absence, deterministic output, and approved tmp-path writes.
* `tests/portfolio/test_portfolio_source_contract.py` covered that portfolio transaction `last_action` fields do not become portfolio intelligence authority fields.
* `tests/conftest.py` and `tests/test_operator_visibility.py` listed the two portfolio-intelligence tests as high-risk script-era blockers because they imported `scripts.core`.

## Changes made

`tests/core/test_build_portfolio_intelligence.py`:

* removed `from scripts.core import build_portfolio_intelligence as portfolio_module`;
* added test-local portfolio intelligence contract helpers using synthetic DataFrames and `tmp_path` files only;
* preserved existing assertions for required input handling, row/order preservation, portfolio source handling, metadata validation, forbidden semantic absence, output/log schema, deterministic output, and approved tmp-path writes;
* replaced source-inspection checks with inspection of the test-local contract helper.

`tests/portfolio/test_portfolio_source_contract.py`:

* removed `from scripts.core import build_portfolio_intelligence`;
* added a small test-local portfolio intelligence contract helper for the existing `last_action` non-authority assertion;
* preserved the existing portfolio source contract tests that use `scripts.portfolio.build_portfolio`.

`tests/conftest.py`:

* removed the two decoupled portfolio-intelligence tests from `_HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS`.

`tests/test_operator_visibility.py`:

* removed the two decoupled portfolio-intelligence tests from the inactive blocker list;
* added a guard proving the two tests are active and do not import `scripts.core`.

No production or runtime module was modified.

## Post-change grep findings

Command:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core import decision_engine|from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Conclusion: the only remaining positive `scripts.core` import is the out-of-scope Decision Engine test import. There is no active import of `scripts.core.build_portfolio_intelligence`.

## Validation

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/core/test_build_portfolio_intelligence.py \
  tests/portfolio/test_portfolio_source_contract.py \
  tests/test_operator_visibility.py \
  -q
```

Result:

```text
46 passed in 0.53s
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
667 passed in 1.13s
```

Full-suite count note:

* the full-suite count increased from the prior `628 passed` baseline because the two previously ignored portfolio-intelligence tests are active again and operator visibility now includes one additional decoupling guard.

## Final decision

`BL116 archive-readiness review for decoupled portfolio intelligence module: APPROVED`

BL116 must be review-only. BL115 does not approve archive, file moves, runtime behavior changes, portfolio behavior changes, Decision Engine changes, provider calls, production writes, reports, Telegram delivery, or state changes.

Blockers: none for BL115 test decoupling.
