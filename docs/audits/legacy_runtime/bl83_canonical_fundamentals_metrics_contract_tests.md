# BL83 — Add canonical fundamentals history and metrics contract tests

Status: COMPLETED

## Purpose

BL83 adds canonical fundamentals metrics contract coverage based on the contracts extracted in BL81 and mapped in BL82.

This sprint introduces a pure canonical metrics contract module and focused contract tests.

The goal is to prove the BL81 metrics formulas and missing-data policies inside the canonical `src/market_scanner/fundamentals/` namespace, without relying on script-era modules.

## Scope

Added canonical module:

* `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`

Added contract tests:

* `tests/contract/test_v2_fundamentals_metrics_contracts.py`

Reviewed existing related coverage:

* `tests/contract/test_v2_fundamental_contracts.py`
* `tests/contract/test_v2_fundamentals_normalization_contracts.py`
* `tests/unit/test_v2_fundamentals_normalization_adapter.py`

Out of scope:

* Archiving script-era files.
* Editing `scripts/fundamentals/build_history_intake.py`.
* Editing `scripts/fundamentals/build_metrics.py`.
* Provider calls.
* Production data writes.
* Decision Engine behavior.
* Portfolio/watchlist state.
* Telegram delivery.

## Background

BL81 extracted stable contracts from:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL82 mapped those contracts to existing canonical modules and concluded:

```text
PARTIAL_CANONICAL_COVERAGE_WITH_TEST_GAPS
```

The most important identified gap was the absence of an explicit canonical fundamentals metrics contract for:

* ratio formulas;
* YoY formulas;
* prior-year lookup;
* missing-input policy;
* zero-denominator policy;
* metric status;
* output/write separation.

BL83 addresses that gap.

## Canonical module added

### `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`

This module is intentionally pure and deterministic.

It provides:

* canonical metrics identity fields;
* canonical metrics input fields;
* canonical derived metric fields;
* canonical helper/status fields;
* forbidden investment-authority fields;
* deterministic in-memory metrics contract generation.

It does not:

* read files;
* write files;
* call providers;
* import script-era modules;
* import network libraries;
* score investments;
* rank securities;
* generate recommendations;
* touch portfolio/watchlist state;
* send Telegram messages.

## Contract coverage added

BL83 adds explicit tests for the following BL81 contracts.

### Identity, input, metric, and helper fields

The tests confirm canonical ownership for:

```text
ticker
fiscal_year
fiscal_period
period_end_date
report_date
currency
source_name
source_reference
source_freshness_date
extraction_date
```

Input fields:

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

Derived metric fields:

```text
gross_margin
operating_margin
net_margin
free_cash_flow_margin
debt_to_equity
return_on_equity
revenue_yoy_growth
eps_yoy_growth
free_cash_flow_yoy_growth
```

Helper fields:

```text
metric_status
metric_missing_inputs
metric_warnings
```

### Ratio formulas

The tests confirm:

```text
gross_margin = gross_profit / revenue
operating_margin = operating_income / revenue
net_margin = net_income / revenue
free_cash_flow_margin = free_cash_flow / revenue
debt_to_equity = total_debt / total_equity
return_on_equity = net_income / total_equity
```

### YoY formulas

The tests confirm:

```text
revenue_yoy_growth = (current_revenue - prior_year_revenue) / abs(prior_year_revenue)
eps_yoy_growth = (current_diluted_eps - prior_year_diluted_eps) / abs(prior_year_diluted_eps)
free_cash_flow_yoy_growth = (current_free_cash_flow - prior_year_free_cash_flow) / abs(prior_year_free_cash_flow)
```

The prior-year lookup uses:

```text
ticker + (fiscal_year - 1) + fiscal_period
```

### Missing-input policy

The tests confirm:

* missing numerator values produce missing metrics;
* missing denominator values produce missing metrics;
* missing inputs are reported in `metric_missing_inputs`;
* missing values are not converted to zero;
* partial metrics produce `metric_status = partial`.

### Zero-denominator policy

The tests confirm:

* zero denominators do not raise runtime errors;
* zero denominators produce missing metric values;
* zero denominator evidence is captured in `metric_missing_inputs` or `metric_warnings`;
* partial metrics produce `metric_status = partial`.

### Missing prior-year policy

The tests confirm:

* missing prior-year rows produce missing YoY metrics;
* missing prior-year rows produce the warning:

```text
yoy_growth:missing_prior_year
```

### Absolute prior-year denominator policy

The tests confirm that YoY growth uses `abs(prior)` as denominator.

### Investment-authority guardrail

The tests confirm that the canonical metrics contract does not introduce forbidden investment authority fields such as:

* buy;
* sell;
* action;
* final_action;
* decision;
* allocation;
* position_size;
* urgency;
* conviction;
* tradeability;
* ranking;
* score;
* target;
* target_price;
* recommendation;
* investment_quality.

### No legacy/network/provider dependency

The tests confirm that the canonical metrics contract source does not import:

* `scripts`;
* `requests`;
* `urllib`;
* `httpx`;
* `aiohttp`;
* `yfinance`;
* `EDGAR`.

### No file side effects

The tests confirm that importing and executing the canonical metrics contract helpers creates no files or directories, including:

* `data/raw`;
* `data/processed`;
* `data/local`;
* `reports`.

## Validation

Focused new test:

```bash
pytest tests/contract/test_v2_fundamentals_metrics_contracts.py -q
```

Result:

```text
12 passed in 0.02s
```

Related focused tests:

```bash
pytest tests/contract/test_v2_fundamental_contracts.py \
       tests/contract/test_v2_fundamentals_normalization_contracts.py \
       tests/unit/test_v2_fundamentals_normalization_adapter.py \
       tests/contract/test_v2_fundamentals_metrics_contracts.py -q
```

Result:

```text
54 passed in 0.03s
```

Full suite:

```bash
pytest -q
```

Result:

```text
534 passed in 0.56s
```

## Archive readiness decision

BL83 improves canonical coverage, but it does not archive script-era files.

Current status:

| File                                           | Status after BL83                                   | Reason                                                                                                                                                           |
| ---------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py` | `NOT_ARCHIVE_READY`                                 | History validation parity still requires explicit canonical coverage beyond metrics formulas.                                                                    |
| `scripts/fundamentals/build_metrics.py`        | `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK` | Core BL81 metrics formulas and missing-data behavior now have canonical contract coverage, but a final reference/parity review is still required before archive. |

## Decision

BL83 closes the largest BL82 canonical test gap: fundamentals metrics formulas and missing-data behavior now have explicit canonical ownership under `src/market_scanner/fundamentals/`.

No script-era runtime file is archived by BL83.

## Recommended next sprint

Recommended next sprint:

```text
BL84 — Add canonical fundamentals history validation contract tests
```

BL84 should focus on the remaining history-intake contract gaps:

* duplicate key policy;
* supported fiscal periods;
* fiscal-year range;
* date parsing;
* required-value behavior;
* validation result shape.

Alternative follow-up:

```text
BL84 — Review archive-readiness of scripts/fundamentals/build_metrics.py after BL83 parity coverage
```

This alternative should be documentation-first and must verify active references before any archive action.

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

BL83 completed canonical fundamentals metrics contract coverage.

The repository now has explicit canonical tests for the BL81 metrics formula and missing-data contracts.

No legacy fundamentals script is archived by BL83.
