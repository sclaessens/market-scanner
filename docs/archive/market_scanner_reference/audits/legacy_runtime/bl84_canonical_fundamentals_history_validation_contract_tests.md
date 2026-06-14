# BL84 — Add canonical fundamentals history validation contract tests

Status: COMPLETED

## Purpose

BL84 adds canonical fundamentals history validation contract coverage based on the contracts extracted in BL81 and mapped in BL82.

BL84 complements BL83. BL83 added canonical metrics formula and missing-data coverage. BL84 adds canonical history validation coverage.

The goal is to prove the BL81 history-intake validation contracts inside the canonical `src/market_scanner/fundamentals/` namespace, without relying on script-era modules.

## Scope

Updated canonical module:

* `src/market_scanner/fundamentals/fundamental_contracts.py`

Added contract tests:

* `tests/contract/test_v2_fundamental_history_validation_contracts.py`

Reviewed existing related coverage:

* `tests/contract/test_v2_fundamental_contracts.py`

Out of scope:

* Archiving script-era files.
* Editing `scripts/fundamentals/build_history_intake.py`.
* Editing `scripts/fundamentals/build_metrics.py`.
* Editing provider, portfolio, watchlist, Telegram, reporting, scanner, or Decision Engine runtime.
* Provider calls.
* Production data writes.

## Background

BL81 extracted stable contracts from:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL82 mapped those contracts to existing canonical modules and concluded:

```text
PARTIAL_CANONICAL_COVERAGE_WITH_TEST_GAPS
```

BL83 closed the largest metrics gap by adding canonical coverage for:

* ratio formulas;
* YoY formulas;
* prior-year lookup;
* missing-input policy;
* zero-denominator policy;
* metric status;
* no-file-side-effect guardrails.

BL84 closes the core history-validation gap.

## Canonical module updated

### `src/market_scanner/fundamentals/fundamental_contracts.py`

BL84 extends the canonical fundamentals contract module with pure, side-effect-free history validation helpers.

The module now exposes:

* numeric history fields;
* date history fields;
* supported fiscal periods;
* fiscal-year validation;
* fiscal-period validation;
* date validation;
* duplicate-key validation;
* batch-level history contract validation.

It remains intentionally pure and does not:

* read files;
* write files;
* call providers;
* import script-era modules;
* import network libraries;
* calculate investment scores;
* rank securities;
* generate recommendations;
* touch portfolio/watchlist state;
* send Telegram messages.

## Contract coverage added

BL84 adds explicit tests for the following BL81 history validation contracts.

### Numeric fields

The tests confirm canonical ownership for:

```text
revenue
gross_profit
operating_income
net_income
diluted_eps
total_debt
total_equity
free_cash_flow
```

### Date fields

The tests confirm canonical ownership for:

```text
period_end_date
report_date
source_freshness_date
extraction_date
```

### Supported fiscal periods

The tests confirm canonical support for:

```text
FY
Q1
Q2
Q3
Q4
TTM
```

Fiscal period validation is case-insensitive.

### Fiscal-year validation

The tests confirm that `fiscal_year` must:

* parse as an integer;
* be within the canonical range:

```text
1900 <= fiscal_year <= 2200
```

Invalid years are reported explicitly without scoring or quality inference.

### Date validation

The tests confirm that non-empty date fields must parse as ISO dates.

Invalid date values are reported per column.

### Duplicate-key validation

The tests confirm the canonical duplicate key:

```text
ticker + fiscal_year + fiscal_period
```

Duplicate detection normalizes:

* `ticker` to uppercase;
* `fiscal_period` to uppercase;
* `fiscal_year` to integer.

Duplicate keys produce a `DUPLICATE_HISTORY_KEY` contract issue.

### Required-value separation

The tests confirm that missing required values remain separate from invalid-value checks.

### Numeric validation continuity

The tests confirm that existing numeric validation still reports non-numeric values explicitly.

### Forbidden investment-authority fields

The tests confirm that forbidden authority fields remain blocked in history validation.

### No legacy/network/provider dependency

The tests confirm that the canonical fundamental contract source does not import:

* `scripts`;
* `requests`;
* `urllib`;
* `httpx`;
* `aiohttp`;
* `yfinance`;
* `EDGAR`.

### No file side effects

The tests confirm that importing and executing the canonical history validation helpers creates no files or directories, including:

* `data/raw`;
* `data/processed`;
* `data/local`;
* `reports`.

## Validation

Focused new test:

```bash
pytest tests/contract/test_v2_fundamental_history_validation_contracts.py -q
```

Result:

```text
13 passed in 0.02s
```

Related focused tests:

```bash
pytest tests/contract/test_v2_fundamental_contracts.py \
       tests/contract/test_v2_fundamental_history_validation_contracts.py -q
```

Result:

```text
27 passed in 0.02s
```

Full suite:

```bash
pytest -q
```

Result:

```text
547 passed in 0.59s
```

## Archive readiness decision

BL84 improves canonical coverage, but it does not archive script-era files.

Current status:

| File                                           | Status after BL84                                   | Reason                                                                                                                                                    |
| ---------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py` | `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK` | Core BL81 history validation contracts now have canonical contract coverage, but a final active-reference/parity review is still required before archive. |
| `scripts/fundamentals/build_metrics.py`        | `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK` | Core BL81 metrics formulas and missing-data behavior were covered by BL83; final active-reference/parity review remains required.                         |

## Decision

BL84 closes the core BL82 canonical test gap for fundamentals history validation.

The project now has explicit canonical coverage for:

* metrics formulas and missing-data behavior via BL83;
* history validation contracts via BL84.

No script-era runtime file is archived by BL84.

## Recommended next sprint

Recommended next sprint:

```text
BL85 — Review archive-readiness of script-era fundamentals history and metrics modules
```

BL85 should be documentation-first and must verify active references before any archive action.

Candidate files:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL85 must check:

* active runtime imports;
* tests that still depend on script-era modules;
* docs/static references;
* canonical coverage from BL83 and BL84;
* whether the script-era files can be archived or need one final parity note.

Alternative safer sprint:

```text
BL85 — Static active-reference review for build_history_intake.py and build_metrics.py
```

## Guardrails

* No live SEC/EDGAR calls were run.
* No yfinance calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist production state was modified.
* No Decision Engine authority was changed.
* No script-era Python runtime files were executed.

## Final status

BL84 completed canonical fundamentals history validation contract coverage.

The repository now has explicit canonical tests for the BL81 history validation contracts.

No legacy fundamentals script is archived by BL84.
