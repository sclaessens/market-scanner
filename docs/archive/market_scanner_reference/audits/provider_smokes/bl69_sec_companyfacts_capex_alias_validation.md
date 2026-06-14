# BL69 — SEC CompanyFacts capex alias validation and one-ticker live source-readiness result

## Status

Passed.

## Purpose

Document the completed SEC CompanyFacts capex alias validation after BL68B and the controlled BL68C live retry.

This document records that the live SEC CompanyFacts one-ticker smoke for NVDA now reaches source-readiness `available` with all required canonical fields present, including source-derived free cash flow.

## Scope

This was a documentation-only governance task.

No code was changed in BL69.

No live request was executed as part of BL69. This document records the already completed BL68C controlled live retry result.

## Preceding implementation

BL68B added `PaymentsToAcquireProductiveAssets` as a safe SEC CompanyFacts alias for canonical `capital_expenditures`.

The alias was added in two required places:

* the controlled SEC CompanyFacts live smoke concept allowlist;
* the SEC CompanyFacts canonical metric mapping for `capital_expenditures`.

The canonical mapping now accepts:

```text
capital_expenditures:
- PaymentsToAcquirePropertyPlantAndEquipment
- PaymentsToAcquireProductiveAssets
```

The change intentionally does not add acquisition or intangible concepts as capex aliases.

The following concepts remain excluded from the capex alias set:

* `PaymentsToAcquireBusinessesNetOfCashAcquired`
* `PaymentsToAcquireIntangibleAssets`
* `PaymentsToAcquireBusinessesNetOfCashAcquiredAndIntangibleAssets`
* other acquisition/intangible concepts

## Validation basis

BL68C was executed as a controlled one-ticker live SEC CompanyFacts retry.

Controlled live retry constraints:

* ticker: `NVDA`
* CIK: `0001045810`
* one live SEC request
* HTTP 2xx required
* no production writes
* no raw SEC payload persistence
* no provider cache
* no CSV writes
* no report generation
* no Telegram delivery
* no portfolio/watchlist impact
* no Decision Engine impact
* repository must remain clean after execution

## BL68C live retry result

The controlled live retry produced:

```json
{
  "canonical_fields_found": [
    "revenue",
    "net_income",
    "operating_income",
    "operating_cash_flow",
    "capital_expenditures",
    "free_cash_flow"
  ],
  "canonical_fields_missing": [],
  "cik": "0001045810",
  "failure_category": "",
  "free_cash_flow_status": "derived",
  "growth_evidence_status": "available",
  "http_status_category": "2xx",
  "readiness_state": "available",
  "request_count": 1,
  "request_executed": true,
  "status": "passed",
  "ticker": "NVDA"
}
```

The live retry warnings included:

```text
free_cash_flow:source_derived
revenue:growth_available
revenue:governed_prior_year_growth
free_cash_flow:growth_available
free_cash_flow:governed_prior_year_growth
net_income:growth_available
net_income:governed_prior_year_growth
operating_income:growth_available
operating_income:governed_prior_year_growth
```

These warnings are evidence/status annotations, not blocking issues.

## Resolved blocker

The previous live blocker was:

```text
capital_expenditures:missing_fact
```

After BL68B and the BL68C retry, this blocker is resolved for the controlled NVDA one-ticker live smoke.

`capital_expenditures` is now present in `canonical_fields_found`.

`canonical_fields_missing` is empty.

## Free cash flow status

Free cash flow was not source-reported as a direct canonical field.

It was successfully source-derived from:

```text
operating_cash_flow - capital_expenditures
```

The live smoke summary reports this as:

```text
free_cash_flow_status: derived
```

The underlying warning confirms the governed derivation path:

```text
free_cash_flow:source_derived
```

## Source-readiness conclusion

For the controlled one-ticker NVDA SEC CompanyFacts live smoke:

```text
readiness_state: available
status: passed
growth_evidence_status: available
canonical_fields_missing: []
```

Therefore, the SEC CompanyFacts live provider source-readiness is marked as passed for the governed one-ticker NVDA smoke.

## Governance interpretation

This validates the provider path for the approved one-ticker NVDA smoke only.

It does not mean that all tickers are now validated.

Correct interpretation:

```text
The SEC CompanyFacts provider now supports the productive-assets capex alias generally.
The controlled live validation proves this path for NVDA / CIK 0001045810.
Broader ticker validation remains out of scope.
```

## Guardrails confirmed

BL68C confirmed:

* exactly one live SEC request was executed;
* the HTTP status category was `2xx`;
* no production writes occurred;
* no raw SEC payload was stored;
* no provider cache was introduced;
* no CSV/report/Telegram output was produced;
* no portfolio/watchlist or Decision Engine path was touched;
* `git status` remained clean after the retry.

## Local validation before documentation

After BL68B was merged into `main`, the following checks passed:

```text
pytest tests/unit/test_v2_sec_companyfacts_smoke_boundary.py -q
28 passed

pytest tests/unit/test_v2_sec_companyfacts_live_smoke.py -q
17 passed

pytest -q
501 passed
```

The alias was confirmed on `main` in:

```text
src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py
src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py
```

## BL69 result

BL69 records the SEC CompanyFacts capex alias validation and marks the controlled one-ticker NVDA live provider source-readiness as passed.

No runtime behavior was changed by BL69.
