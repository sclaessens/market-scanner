# BL74 — Decouple active tests from script-era fundamentals

## Status

Done.

## Purpose

Remove active test dependency on the script-era fundamentals package so `tests/`, `src/`, and `.github/` no longer import `scripts.fundamentals`.

This sprint addresses the BL73 blocker that prevented future cleanup from retrying archival of `scripts/fundamentals/__init__.py`.

## Registry basis

BL74 follows the BL70 canonical cleanup registry and the BL73 carry-forward blocker.

BL73 left `scripts/fundamentals/__init__.py` blocked because active tests still imported `scripts.fundamentals`. BL74 targeted only those active test dependencies and the archive pytest collection behavior needed for future script-era moves.

## Removed, replaced, or migrated script-era imports

Removed script-era imports from:

- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_sec_ticker_cik_index.py`
- `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- `tests/core/test_build_fundamental_analysis.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/core/test_build_fundamental_layer.py`
- `tests/core/test_fundamentals_operational_validation.py`
- `tests/core/test_build_fundamental_metrics.py`
- `tests/core/test_build_fundamentals_history_intake.py`

No `scripts.fundamentals` imports were preserved for backward compatibility.

No old script-era fundamentals module is executed from active tests.

No runtime logic was migrated because the affected tests validated retired script-era behavior rather than a canonical `src/market_scanner` implementation of the same build pipeline.

## Handling by affected test

- `tests/fundamentals/test_sec_companyfacts_transform.py`: converted to static legacy CompanyFacts transform evidence.
- `tests/fundamentals/test_sec_ticker_cik_index.py`: converted to static legacy ticker-CIK mapping evidence.
- `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`: converted to static legacy bulk-intake policy evidence.
- `tests/fundamentals/test_run_sec_transformation_review.py`: converted to static legacy review-only policy evidence.
- `tests/core/test_build_fundamental_analysis.py`: converted to static legacy analysis classification evidence.
- `tests/core/test_fundamentals_runtime_organization.py`: converted to static evidence that active tests no longer certify script import compatibility.
- `tests/core/test_build_fundamental_layer.py`: converted to static legacy quality classification evidence.
- `tests/core/test_fundamentals_operational_validation.py`: converted to static legacy operational-flow policy evidence.
- `tests/core/test_build_fundamental_metrics.py`: converted to static legacy metrics contract evidence.
- `tests/core/test_build_fundamentals_history_intake.py`: converted to static legacy history-intake contract evidence.

The converted tests were removed from `tests/conftest.py` collection ignore so pytest now collects them as active static evidence tests.

## Canonical tests added or adjusted

Adjusted `tests/test_operator_visibility.py` so it distinguishes:

- unresolved high-risk script-era tests that remain inactive migration blockers; and
- BL74 static legacy fundamentals evidence tests that are active and decoupled from script imports.

No canonical runtime behavior tests were added because the removed imports covered retired script-era behavior, not canonical `src/market_scanner` runtime behavior.

## Archive pytest collection

Archive collection was fixed in `pyproject.toml`:

- `norecursedirs = ["archive"]`

This ensures pytest does not recurse into `archive/`, including future archived paths such as `archive/legacy_runtime/scripts/portfolio/test_portfolio.py`.

## Remaining blockers

No BL74 blocker remains for active `scripts.fundamentals` imports in `tests/`, `src/`, or `.github/`.

Future cleanup can retry archiving `scripts/fundamentals/__init__.py` and any already archived script-era test files if focused validation passes.

## Validation commands and results

Initial discovery:

```bash
grep -R "from scripts.fundamentals\|import scripts.fundamentals" -n tests src .github \
  --include="*.py" \
  --include="*.yml" \
  --include="*.yaml"
```

Result: found the ten known test files listed above.

Final import check:

```bash
grep -R "from scripts.fundamentals\|import scripts.fundamentals" -n tests src .github \
  --include="*.py" \
  --include="*.yml" \
  --include="*.yaml" || true
```

Result: no output.

Pytest:

```bash
source .venv/bin/activate && pytest -q
```

Result:

```text
522 passed in 0.78s
```

Note: the bare `pytest -q` command was attempted first, but `pytest` was not on the shell PATH until the workspace virtual environment was activated.

## Guardrails confirmation

BL74 did not:

- run live SEC or EDGAR calls;
- run yfinance calls;
- write production data;
- send Telegram messages;
- modify portfolio or watchlist state;
- change Decision Engine authority;
- change allocation, tradeability, conviction, BUY, SELL, REMOVE, urgency, or gating logic;
- change runtime source code;
- change active workflow behavior.

BL74 preserved:

- classification upstream, allocation downstream;
- Decision Engine as the only allocation authority;
- reporting as communication only;
- row-preservation policy as static legacy evidence;
- deterministic active test collection behavior.
