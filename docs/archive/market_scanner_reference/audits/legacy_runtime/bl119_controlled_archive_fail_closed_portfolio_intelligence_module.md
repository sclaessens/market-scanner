# BL119 — Controlled archive of fail-closed portfolio intelligence module

## Scope

- `scripts/core/build_portfolio_intelligence.py`

## Archive Action

- Moved `scripts/core/build_portfolio_intelligence.py` to `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`.
- The move was performed with `git mv`.
- No source body edits were made to the archived module.

## Explicit Out of Scope

- `scripts/core/decision_engine.py`
- Decision Engine authority
- `scripts/portfolio/build_portfolio.py`
- `scripts/portfolio/parse_trade_commands.py`
- scanner/provider runtime
- SEC/EDGAR
- yfinance
- credentials
- production data writes
- reports
- Telegram
- portfolio state
- watchlist state
- scan validation runtime
- any other `scripts/core/` module

## Commands Executed

```bash
git checkout main
git pull origin main
git status --short --branch
git checkout -b bl119-controlled-archive-fail-closed-portfolio-intelligence-module
test -f scripts/core/build_portfolio_intelligence.py
test ! -f archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py
grep -RInE "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" tests src .github --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude="*.pyc"
grep -nE "FAIL_CLOSED|def _legacy_build_portfolio_intelligence_impl|def build_portfolio_intelligence|if __name__" scripts/core/build_portfolio_intelligence.py
mkdir -p archive/legacy_runtime/scripts/core
git mv scripts/core/build_portfolio_intelligence.py archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py
test ! -f scripts/core/build_portfolio_intelligence.py
test -f archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py
grep -nE "FAIL_CLOSED|def _legacy_build_portfolio_intelligence_impl|def build_portfolio_intelligence|if __name__" archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py
grep -RInE "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" tests src .github --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ --exclude="*.pyc"
find scripts/core -maxdepth 1 -type f -name "*.py" | sort
.venv/bin/python -m pytest tests/core/test_build_portfolio_intelligence.py tests/portfolio/test_portfolio_source_contract.py tests/test_operator_visibility.py -q
.venv/bin/python -m pytest -q
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

## Initial Archive-Readiness Findings

- `scripts/core/build_portfolio_intelligence.py` existed before the archive.
- `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py` did not exist before the archive.
- The active import check returned only:
  - `tests/core/test_decision_engine.py:9:from scripts.core import decision_engine`
- No active import from `src`, `tests`, or `.github` referenced `scripts.core.build_portfolio_intelligence`.
- The pre-archive fail-closed marker check confirmed:
  - `FAIL_CLOSED_MESSAGE` existed;
  - `_legacy_build_portfolio_intelligence_impl()` existed;
  - public `build_portfolio_intelligence()` raised `SystemExit(FAIL_CLOSED_MESSAGE)`;
  - direct execution raised `SystemExit(FAIL_CLOSED_MESSAGE)`.

## Exact `git mv` Action

```bash
git mv scripts/core/build_portfolio_intelligence.py archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py
```

## Post-Archive File Existence Check

- `scripts/core/build_portfolio_intelligence.py` no longer exists.
- `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py` exists.

## Archived File Fail-Closed Marker Check

- The archived file still contains `FAIL_CLOSED_MESSAGE`.
- The archived file still contains `_legacy_build_portfolio_intelligence_impl()`.
- The archived public `build_portfolio_intelligence()` still raises `SystemExit(FAIL_CLOSED_MESSAGE)`.
- The archived `if __name__ == "__main__"` block still raises `SystemExit(FAIL_CLOSED_MESSAGE)`.

## Post-Change Active Import Check

- The active import check returned only:
  - `tests/core/test_decision_engine.py:9:from scripts.core import decision_engine`
- No active import from `src`, `tests`, or `.github` references `scripts.core.build_portfolio_intelligence`.

## Remaining Active `scripts/core` Inventory

```text
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/indicators.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
scripts/core/validator.py
```

`scripts/core/build_portfolio_intelligence.py` is no longer present in the active `scripts/core/` tree.

## Focused Test Result

```text
46 passed in 0.88s
```

## Full Suite Result

```text
667 passed in 1.19s
```

## Governance Grep Result

The mandatory governance greps were run before PR. They reported pre-existing out-of-scope matches in portfolio command/manager files and Python bytecode cache files. BL119 did not modify those files.

## Final Decision

`BL120 remaining active scripts/core review: APPROVED`
