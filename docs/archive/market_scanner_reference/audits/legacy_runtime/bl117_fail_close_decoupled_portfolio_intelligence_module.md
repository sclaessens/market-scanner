# BL117 — Fail-close decoupled portfolio intelligence module

Sprint type: Controlled fail-close sprint.

This was not an archive sprint, runtime migration sprint, portfolio behavior enhancement sprint, or Decision Engine sprint.

## Scope

Target runtime file:

* `scripts/core/build_portfolio_intelligence.py`

Explicit target:

* disable manual/runtime execution of the script-era module while preserving historical implementation content for audit purposes.

## Explicit out of scope

* archiving
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

## Initial active import check

Command:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" \
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

Conclusion: no active `src`, `tests`, or `.github` path imports `scripts.core.build_portfolio_intelligence`. The remaining `scripts.core` import is the out-of-scope Decision Engine test import.

## Initial manual-run/write-risk marker check

Command:

```bash
grep -nE \
  "FAIL_CLOSED|def build_portfolio_intelligence|if __name__|read_csv|to_csv|mkdir|data/processed|data/logs|portfolio_positions|portfolio_metadata|timing_state" \
  scripts/core/build_portfolio_intelligence.py
```

Relevant output:

```text
11:INPUT_PATH = Path("data/processed/timing_state_layer.csv")
12:PORTFOLIO_PATH = Path("data/portfolio/portfolio_positions.csv")
13:PORTFOLIO_METADATA_PATH = Path("data/portfolio/portfolio_metadata.csv")
14:OUTPUT_PATH = Path("data/processed/portfolio_intelligence.csv")
15:LOG_PATH = Path("data/logs/portfolio_intelligence_log.csv")
156:        df = pd.read_csv(path)
168:        return pd.read_csv(path), "AVAILABLE"
177:        metadata_df = pd.read_csv(path, dtype=str, keep_default_na=False)
660:def build_portfolio_intelligence() -> pd.DataFrame:
696:    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
697:    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
698:    output_df.to_csv(OUTPUT_PATH, index=False)
699:    log_df.to_csv(LOG_PATH, index=False)
705:if __name__ == "__main__":
```

Conclusion: before BL117, the module had no `FAIL_CLOSED` marker and still exposed fixed production-like paths, read/write calls, directory creation, and direct execution.

## Implementation summary

Changes made in `scripts/core/build_portfolio_intelligence.py`:

* added `FAIL_CLOSED_MESSAGE` near the top of the module;
* renamed the historical public `build_portfolio_intelligence()` body to `_legacy_build_portfolio_intelligence_impl()`;
* added a new public `build_portfolio_intelligence()` function that raises `SystemExit(FAIL_CLOSED_MESSAGE)`;
* changed the `if __name__ == "__main__"` block to raise `SystemExit(FAIL_CLOSED_MESSAGE)`.

Historical helper functions, path constants, read/write code, output schema, and portfolio intelligence logic were preserved.

## Post-change grep result

Active import check:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core import build_portfolio_intelligence|import scripts\.core\.build_portfolio_intelligence|from scripts\.core|import scripts\.core)" \
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

Fail-closed marker check:

```bash
grep -nE \
  "FAIL_CLOSED|def _legacy_build_portfolio_intelligence_impl|def build_portfolio_intelligence|if __name__" \
  scripts/core/build_portfolio_intelligence.py
```

Output:

```text
18:FAIL_CLOSED_MESSAGE = (
19:    "FAIL_CLOSED: scripts/core/build_portfolio_intelligence.py is a legacy "
667:def _legacy_build_portfolio_intelligence_impl() -> pd.DataFrame:
712:def build_portfolio_intelligence() -> pd.DataFrame:
713:    raise SystemExit(FAIL_CLOSED_MESSAGE)
716:if __name__ == "__main__":
717:    raise SystemExit(FAIL_CLOSED_MESSAGE)
```

Conclusion: public/manual execution is fail-closed, and the historical implementation is preserved under a private legacy name.

## Direct execution result

Command:

```bash
.venv/bin/python scripts/core/build_portfolio_intelligence.py
```

Result:

```text
FAIL_CLOSED: scripts/core/build_portfolio_intelligence.py is a legacy script-era portfolio intelligence module. Active tests were decoupled in BL115 and archive-readiness was reviewed in BL116. Manual/runtime execution is disabled pending controlled archive review.
```

The command exited non-zero with code `1` before any production read/write behavior.

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
46 passed in 0.51s
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
667 passed in 1.06s
```

## Final decision

`BL118 archive-readiness review for fail-closed portfolio intelligence module: APPROVED`

BL118 must be review-only. BL117 does not approve archive, file moves, runtime migration, portfolio behavior changes, Decision Engine changes, provider calls, production writes, reports, Telegram delivery, or state changes.
