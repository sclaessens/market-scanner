# V2 Governed FreeCashFlow Derivation Implementation

## Status

Completed by RESET-10L-BL22.

## Reset Stage

RESET-10L-BL22 — Implement Governed FreeCashFlow Derivation.

## Files Changed

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_dry_run_fixture_review.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/backlog.md`

## Implementation Summary

The v2 provider adapter now supports governed FreeCashFlow normalization under the RESET-10L-BL21 Option C policy.

Direct source `FreeCashFlow` remains source-reported when present and valid.

When direct `FreeCashFlow` is absent or explicitly missing, the adapter may derive:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

The derived record is visibly marked as derived, preserves both source input field names, records the derivation formula, and exposes neutral validation/readiness warnings.

## States Supported

The `free_cash_flow` normalized record now supports:

- `source_reported`
- `source_derived`
- `missing`
- `invalid`
- `not_parseable`
- `not_derivable`

## Derivation Conditions

Derived FreeCashFlow is produced only when:

- direct `FreeCashFlow` is absent or explicitly missing;
- operating cash flow is present;
- capital expenditures is present;
- both values parse safely as numeric values;
- both values have matching currency;
- both values have matching unit;
- both values have matching reported period;
- both values have compatible fiscal year and fiscal quarter context;
- both source field names are present;
- capital expenditures is represented as a non-negative outflow amount.

Negative capital expenditures fail closed as `not_derivable` with `free_cash_flow:sign_convention_ambiguous`.

## Tests Added Or Updated

Updated existing tests only:

- provider adapter unit tests now cover source-reported, source-derived, missing input, invalid input, not-parseable input, currency mismatch, unit mismatch, period mismatch, fiscal context mismatch, missing provenance, ambiguous sign convention, and NVDA-shaped derivation;
- real-source smoke tests now expect valid governed FreeCashFlow derivation when operating cash flow and capital expenditures are available;
- provider dry-run fixture review now expects governed derived FreeCashFlow while preserving explicit missingness for unrelated missing fields;
- provider-to-persistence integration contracts now prove NVDA-shaped derived FreeCashFlow can reach the persistence boundary and write only under `tmp_path`.

## NVDA Impact

Synthetic NVDA-shaped input with missing direct `FreeCashFlow`, valid operating cash flow, and valid capital expenditures now produces:

- `free_cash_flow` value: derived;
- `normalization_status`: `source_derived`;
- readiness warning: `free_cash_flow:source_derived`;
- persistence metric value status: `source_derived`.

This does not produce any final investment recommendation.

## Python File Creation Policy Result

No new Python files were created.

The implementation updated the existing provider contract and provider adapter modules because they already own provider/source normalization and readiness behavior.

No one-off ticker-specific Python files were created.

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

No allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior added.

## Known Limitations

This implementation supports governed FreeCashFlow derivation only in the v2 provider adapter boundary.

It does not re-run the NVDA real fundamental analysis review, does not execute live source capture, and does not modify legacy scripts or production pipeline behavior.

## Next Recommended Step

Proceed to RESET-10L-BL23 — Re-run NVDA One-Ticker Real Fundamental Analysis with Derived FreeCashFlow.
