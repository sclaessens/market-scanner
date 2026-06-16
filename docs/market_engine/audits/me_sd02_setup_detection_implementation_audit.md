# ME-SD02 — Setup Detection implementation audit

## Status

COMPLETED BY ME-SD02

## Sprint

ME-SD02 — Implement first Setup Detection layer

## Branch

me-sd02-implement-setup-detection-layer

## Sprint goal

Implement the first non-actionable Setup Detection runtime layer from approved Market Engine observation inputs.

The implementation must remain inside the Setup Detection job family and must not introduce Analysis Review behavior, Recommendation Review behavior, Portfolio Review behavior, Delivery / Reporting behavior, Telegram behavior, provider behavior, portfolio/watchlist mutation, or Decision Engine behavior.

## Files added

Runtime:

* `src/market_engine/setup_detection/__init__.py`
* `src/market_engine/setup_detection/sec_companyfacts_setup_detection.py`

Tests:

* `tests/market_engine/setup_detection/test_sec_companyfacts_setup_detection.py`

Documentation:

* `docs/market_engine/audits/me_sd02_setup_detection_implementation_audit.md`

## Files changed

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract implemented

Implemented output contract:

* `sec-companyfacts-setup-detection-v1`

Implemented builder:

* `build_sec_companyfacts_setup_detection(...)`

Implemented persistence helper:

* `persist_sec_companyfacts_setup_detection(...)`

## Input contracts

Approved input contracts:

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`

Unsupported input contracts fail closed with `ValueError`.

## Output contract

Output:

* ticker;
* CIK;
* provider name;
* setup detection format version;
* fundamental observation format version;
* derived observation format version;
* source context format version;
* source context state;
* source refresh snapshot metadata;
* setup detection run ID;
* input contract references;
* setup items;
* non-actionable boundary;
* warnings.

## Categories implemented

Implemented setup categories:

* `cash_generation_setup`
* `fundamental_availability_setup`
* `profitability_evidence_setup`
* `revenue_evidence_setup`
* `balance_sheet_evidence_setup`
* `data_limitation_setup`

`not_assessed_setup` is exported as an approved category and remains available for future use.

## States implemented

Implemented setup states:

* `setup_detected`
* `setup_partially_detected`
* `setup_not_detected`
* `setup_conflicted`
* `setup_blocked_by_missing_data`
* `setup_not_assessed`

## Persistence behavior

Persistence writes deterministic JSON to:

```text
data/market_engine/setup_detections/<setup_detection_run_id>/<ticker>/setup_detection.json
```

Tests use temporary directories only.

Persistence refuses overwrite with `FileExistsError`.

## Tests added

ME-SD02 tests cover:

* complete positive setup evidence produces setup detection output;
* partial evidence produces `setup_partially_detected`;
* missing required observations produce `setup_blocked_by_missing_data`;
* conflicted evidence produces `setup_conflicted`;
* unsupported input contracts fail closed;
* numeric zero is preserved and not treated as missing;
* source and derived references are preserved;
* forbidden action-authority terms are not emitted in normal setup text;
* persistence writes JSON under a temporary root;
* persistence refuses overwrite;
* active tests do not import legacy `scripts` or old `market_scanner`.

## Validation commands and results

Initial command:

```bash
pytest tests/market_engine/setup_detection -q
```

Result:

```text
zsh:1: command not found: pytest
```

Repository virtualenv command:

```bash
.venv/bin/pytest tests/market_engine/setup_detection -q
```

Result:

```text
11 passed
```

Full Market Engine command:

```bash
.venv/bin/pytest tests/market_engine -q
```

Result:

```text
147 passed
```

Repository validation:

```bash
git diff --check
git status --short
git diff --stat
git diff --name-only
grep -R "from scripts\|import scripts\|from market_scanner\|import market_scanner" src/market_engine tests/market_engine || true
```

Results:

* `git diff --check` passed.
* `git status --short` showed only the planned Setup Detection runtime, test, backlog, roadmap, and audit changes.
* `git diff --stat` and `git diff --name-only` showed documentation-only updates plus the new active Setup Detection runtime and tests.
* The legacy dependency grep found only pre-existing negative assertion strings in Recommendation Review tests and no active imports from legacy `scripts` or old `market_scanner`.

## Boundaries preserved

Confirmed:

* no live provider calls were introduced;
* no SEC calls were introduced;
* no EDGAR calls were introduced;
* no yfinance calls were introduced;
* no production data writes were introduced;
* no portfolio mutation was introduced;
* no watchlist mutation was introduced;
* no Telegram behavior was introduced;
* no reporting output was introduced;
* no Decision Engine behavior was introduced;
* no Analysis Review behavior was changed;
* no Recommendation Review behavior was changed;
* no Portfolio Review behavior was changed;
* no Delivery / Reporting behavior was changed;
* no legacy `scripts` imports were introduced;
* no old `market_scanner` imports were introduced.

## Next recommended sprint

Recommended next sprint:

```text
ME-AR03 — Extend Analysis Review contract for Setup Detection input
```
