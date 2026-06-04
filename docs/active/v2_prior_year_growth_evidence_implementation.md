# V2 Prior-Year Growth Evidence Implementation

Status: COMPLETED
Reset stage: RESET-10L-BL25

## Files Changed

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/backlog.md`

## Implementation Summary

Governed prior-year growth evidence is implemented in the existing v2 fundamentals provider boundary.

The implementation adds a neutral `ProviderPriorYearGrowthEvidenceRecord` and pure adapter functions that compare current and prior normalized fundamentals records in memory:

- `build_prior_year_growth_evidence`
- `build_prior_year_growth_evidence_record`

The boundary preserves current and prior values, current and prior period references, fiscal context, currency, unit, source references, source record identities, source field names, formula, growth status, growth rate, and validation warnings.

No provider calls, network calls, file writes, production pipeline hooks, report hooks, Telegram hooks, portfolio/watchlist hooks, or Decision Engine behavior were added.

## Growth Evidence States Supported

The implementation supports explicit states:

- `growth_available`
- `growth_missing_prior_period`
- `growth_missing_current_period`
- `growth_invalid_current_period`
- `growth_invalid_prior_period`
- `growth_not_parseable`
- `growth_not_comparable`
- `growth_period_mismatch`
- `growth_currency_mismatch`
- `growth_unit_mismatch`
- `growth_provenance_gap`

Metric mismatch fails closed as `growth_not_comparable` with a visible `growth_metric_mismatch` warning.

## Formula

For valid comparable values:

```text
growth_rate = (current_value - prior_value) / abs(prior_value)
```

If the prior value is zero, missing, invalid, not parseable, not comparable, or provenance-linked evidence is incomplete, the implementation returns `growth_rate = None` and an explicit fail-closed growth status.

## Validation / Fail-Closed Behavior

Growth evidence is produced only when:

- current and prior records exist;
- metric names match;
- values are valid and parseable;
- prior value is non-zero;
- fiscal period and fiscal quarter match;
- current fiscal year is exactly one year after prior fiscal year;
- currency matches;
- unit matches;
- current and prior source references are present;
- current and prior source record identities are present;
- current and prior source field names are present.

Missing, invalid, not-parseable, zero-prior, mismatched, or provenance-gap inputs fail closed. Missing values are not converted to zero.

## NVDA Impact

A synthetic NVDA-shaped test now proves that source-derived FreeCashFlow can produce governed prior-year growth evidence when both current and prior comparable FreeCashFlow values are valid and provenance-linked.

The test preserves source-derived FreeCashFlow provenance from:

- `NetCashProvidedByUsedInOperatingActivities`
- `PaymentsToAcquirePropertyPlantAndEquipment`

The NVDA-shaped growth evidence returns `growth_available` for `free_cash_flow` without producing investment behavior.

## Tests Added / Updated

Updated tests cover:

- valid revenue growth evidence;
- valid FreeCashFlow growth evidence;
- absolute-prior formula behavior;
- missing current period;
- missing prior period;
- invalid current period;
- invalid prior period;
- not-parseable current period;
- not-parseable prior period;
- zero prior value;
- currency mismatch;
- unit mismatch;
- period mismatch;
- fiscal year mismatch;
- metric mismatch;
- current provenance gap;
- prior provenance gap;
- NVDA-shaped source-derived FreeCashFlow growth evidence;
- no investment authority fields on the growth evidence contract.

## Python File Creation Policy Result

No new Python files were created.

The implementation updated existing fundamentals provider contract and adapter modules, plus existing tests.

## Guardrails

No credentials committed.

No raw live payloads committed.

No production data writes.

No reports generated.

No Telegram artifacts generated.

No unsafe production pipeline execution.

No portfolio/watchlist updates.

No final BUY/SELL/HOLD recommendation.

No missing values converted to zero.

No one-off ticker-specific Python files created.

## Known Limitations

The implementation creates governed growth evidence records. It does not yet re-run NVDA real analysis to confirm whether the analysis state moves beyond `LIMITED_ANALYSIS`.

The implementation does not add production data persistence, reporting, Telegram delivery, portfolio/watchlist behavior, or Decision Engine investment behavior.

## Next Recommended Step

Proceed to `RESET-10L-BL26 — Re-run NVDA Real Analysis with Governed Growth Evidence`.
